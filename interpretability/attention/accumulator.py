import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Accumulator — reduces raw (B, H, N, N) tensors per batch into compact stats
# ─────────────────────────────────────────────────────────────────────────────

class AttentionAccumulator:
    """Incrementally reduces raw ``(B, H, N, N)`` attention maps to compact
    statistics so the raw tensors can be discarded after each batch.

    After ``consume()`` for every batch, retrieve via:
      * ``get_mean_attn()``    → ``{block: (H, N, N)}``                  — for heads grid
      * ``get_chan_scores()``  → ``{branch: {block: (N_total, C)}}``     — channel topomap
      * ``get_cls_row()``      → ``{branch: {block: (N_total, C)}}``     — CLS attends to …
      * ``get_cls_col()``      → ``{branch: {block: (N_total, C)}}``     — … attends to CLS
      * ``get_y()``            → ``np.ndarray (N_total,)``
    """

    def __init__(self, n_chans, n_time_patches, has_cls):
        self.n_chans = n_chans
        self.n_time_patches = n_time_patches
        self.has_cls = has_cls
        self._expected = n_chans * n_time_patches

        # Per-block sum across samples; divide by count once in get_mean_attn()
        self._attn_sum   = {}  # {block: (H, N, N)}
        self._attn_count = {}  # {block: int (samples)}

        # Per-sample channel scores — concatenated at retrieval time
        self._chan_lists    = {}  # {(branch, block): list[(B, C)]}
        self._cls_row_lists = {}  # {(branch, block): list[(B, C)]}
        self._cls_col_lists = {}  # {(branch, block): list[(B, C)]}

        self._y_list = []

    def consume(self, storage_maps, y_batch):
        """Process one batch of raw maps from ``AttentionStorage`` and discard."""
        B = len(y_batch)
        for m in storage_maps:
            block = m["block"]
            branch = m.get("branch", 0)
            attn = m["attn"].numpy()
            self._update_mean_attn(block, attn)
            self._update_chan_scores(branch, block, attn, B)
        self._y_list.append(y_batch)

    def _update_mean_attn(self, block, attn):
        # attn: (B, H, N, N) — accumulate sum across batch dim
        sum_b = attn.sum(axis=0)  # (H, N, N)
        if block not in self._attn_sum:
            self._attn_sum[block] = sum_b
            self._attn_count[block] = attn.shape[0]
        else:
            self._attn_sum[block] += sum_b
            self._attn_count[block] += attn.shape[0]

    def _update_chan_scores(self, branch, block, attn, B):
        H = attn.shape[1]
        offset = 1 if self.has_cls else 0
        token_attn = attn[:, :, offset:, offset:]
        if token_attn.shape[-1] != self._expected:
            return
        C, A = self.n_chans, self.n_time_patches
        key = (branch, block)

        received = token_attn.sum(axis=-2) # received attention per token: (B, H, C*A)
        by_chan = received.reshape(B, H, C, A).mean(axis=(1, 3)) # (B, C)
        self._chan_lists.setdefault(key, []).append(by_chan)

        if self.has_cls:
            cls_r = attn[:, :, 0, 1:]
            cls_c = attn[:, :, 1:, 0]
            if cls_r.shape[-1] == self._expected:
                r = cls_r.reshape(B, H, C, A).mean(axis=(1, 3))
                c = cls_c.reshape(B, H, C, A).mean(axis=(1, 3))
                self._cls_row_lists.setdefault(key, []).append(r)
                self._cls_col_lists.setdefault(key, []).append(c)

    def get_mean_attn(self):
        return {b: s / self._attn_count[b] for b, s in self._attn_sum.items()}

    def _unpack_branch_block(self, lists_dict):
        result = {}
        for (branch, block), vecs in lists_dict.items():
            result.setdefault(branch, {})[block] = np.concatenate(vecs, axis=0)
        return result

    def get_chan_scores(self):
        return self._unpack_branch_block(self._chan_lists)

    def get_cls_row(self):
        return self._unpack_branch_block(self._cls_row_lists)

    def get_cls_col(self):
        return self._unpack_branch_block(self._cls_col_lists)

    def get_y(self):
        return np.concatenate(self._y_list) if self._y_list else None

    def n_blocks(self):
        return max(self._attn_sum) + 1 if self._attn_sum else 0
