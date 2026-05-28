import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Activation identity rules
# ─────────────────────────────────────────────────────────────────────────────

class GELUIdentityFn(torch.autograd.Function):
    """Identity rule for GELU: forward computes gelu, backward passes relevance through."""

    @staticmethod
    def forward(ctx, x):
        return F.gelu(x)

    @staticmethod
    def backward(ctx, out_relevance):
        return out_relevance


class GEGLUIdentityFn(torch.autograd.Function):
    """Identity rule for GEGLU: routes all relevance to value half, zeros gate half."""

    @staticmethod
    def forward(ctx, x):
        x_val, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x_val

    @staticmethod
    def backward(ctx, out_relevance):
        zeros = torch.zeros_like(out_relevance)
        return torch.cat([out_relevance, zeros], dim=-1)


class ELUIdentityFn(torch.autograd.Function):
    """Identity rule for ELU: forward computes elu, backward passes relevance through."""

    @staticmethod
    def forward(ctx, x):
        return F.elu(x)

    @staticmethod
    def backward(ctx, out_relevance):
        return out_relevance


# ─────────────────────────────────────────────────────────────────────────────
# Module-level forward patch functions (bound as methods via types.MethodType)
# ─────────────────────────────────────────────────────────────────────────────


def gelu_identity_forward(self, x):
    """Identity rule replacement for nn.GELU.forward."""
    return GELUIdentityFn.apply(x)


def geglu_identity_forward(self, x):
    """Identity rule replacement for GEGLU.forward."""
    return GEGLUIdentityFn.apply(x)


def elu_identity_forward(self, x):
    """Identity rule replacement for nn.ELU.forward."""
    return ELUIdentityFn.apply(x)
