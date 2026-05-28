import torch
from contextlib import contextmanager


# ─────────────────────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────────────────────

class AttentionStorage:
    """Accumulates per-block attention maps during a forward pass."""
    def __init__(self):
        self.maps = []   # list[dict]

    def clear(self):
        self.maps.clear()

    def by_block(self, block_idx):
        """Return all stored maps for a given block index."""
        return [m for m in self.maps if m.get("block") == block_idx]


# ─────────────────────────────────────────────────────────────────────────────
# LaBraM
# ─────────────────────────────────────────────────────────────────────────────

def _register_labram_hooks(model, storage):
    """Hook model.blocks[i].attn.attn_drop to capture attention before dropout.

    LaBraM's Attention.forward:
        attn = (q @ k.T).softmax(-1)
        attn = self.attn_drop(attn)   ← hook fires here, inp[0] = attn
    """
    handles = []
    for block_idx, block in enumerate(model.blocks):
        def make_hook(idx):
            def hook(m, inp, out):
                storage.maps.append({
                    "block": idx,
                    "type": "self",
                    "attn": inp[0].detach().float().cpu(),  # (B, H, N, N)
                })
            return hook
        h = block.attn.attn_drop.register_forward_hook(make_hook(block_idx))
        handles.append(h)
    return handles


# ─────────────────────────────────────────────────────────────────────────────
# NeuroRVQ
# ─────────────────────────────────────────────────────────────────────────────

def _register_neurorvq_hooks(model, storage):
    """Same hook as LaBraM.  NeuroRVQ processes 4 branches sequentially through
    the same model.blocks, so each block fires 4 times per forward pass.
    A per-block call counter assigns branch indices 0–3.
    """
    handles = []
    for block_idx, block in enumerate(model.blocks):
        counter = [0]  # mutable for closure

        def make_hook(idx, ctr):
            def hook(m, inp, out):
                branch = ctr[0] % 4
                ctr[0] += 1
                storage.maps.append({
                    "block": idx,
                    "branch": branch,
                    "type": "self",
                    "attn": inp[0].detach().float().cpu(),  # (B, H, N, N)
                })
            return hook

        h = block.attn.attn_drop.register_forward_hook(make_hook(block_idx, counter))
        handles.append(h)
    return handles


# ─────────────────────────────────────────────────────────────────────────────
# REVE
# ─────────────────────────────────────────────────────────────────────────────

def _patch_reve(model, storage):
    """Replace ClassicalAttention.forward with an explicit SDPA-free version
    that computes softmax(QKᵀ/√d) and saves the attention matrix.
    """
    from einops import rearrange
    patched = []

    block_idx = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ != "ClassicalAttention":
            continue

        original_forward = module.forward

        def make_patched(mod, idx):
            def patched_forward(qkv):
                q, k, v = qkv.chunk(3, dim=-1)
                q, k, v = (
                    rearrange(t, "batch seq (heads dim) -> batch heads seq dim",
                              heads=mod.heads)
                    for t in (q, k, v)
                )
                scale = q.shape[-1] ** -0.5
                dots = torch.matmul(q, k.transpose(-1, -2)) * scale
                attn = dots.softmax(dim=-1)  # (B, H, N, N)

                storage.maps.append({
                    "block": idx,
                    "type": "self",
                    "attn": attn.detach().float().cpu(),
                })

                out = torch.matmul(attn, v)
                out = rearrange(out, "batch heads seq dim -> batch seq (heads dim)")
                return out

            return patched_forward

        module.forward = make_patched(module, block_idx)
        patched.append((module, original_forward))
        block_idx += 1

    return patched


def _unpatch_reve(patched):
    for module, orig in patched:
        module.forward = orig


# ─────────────────────────────────────────────────────────────────────────────
# Context manager
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def attention_context(model_name, model):
    """Context manager that installs hooks/patches on *model* to capture
    attention weight tensors during inference.

    Parameters
    ----------
    model_name : str
        One of "LaBraM", "NeuroRVQ", "REVE".
    model : nn.Module
        The unwrapped model (output of ``unwrap_model(wrapper)``).

    Yields
    ------
    storage : AttentionStorage
        Call ``storage.maps`` after the forward pass to get the collected maps.
    """
    storage = AttentionStorage()
    handles = []
    patched_reve = []

    try:
        if model_name == "LaBraM":
            handles = _register_labram_hooks(model, storage)
        elif model_name == "NeuroRVQ":
            handles = _register_neurorvq_hooks(model, storage)
        elif model_name == "REVE":
            patched_reve = _patch_reve(model, storage)
        else:
            raise ValueError(
                f"Unsupported model for attention extraction: {model_name}. "
                "Supported: LaBraM, NeuroRVQ, REVE."
            )
        yield storage

    finally:
        for h in handles:
            h.remove()
        _unpatch_reve(patched_reve)
