"""Per-layer top-k attention routing with adaptive feedback.

Each layer has an independent CovarianceRouter that predicts which
KV blocks are important for a given query. No multi-stage scan,
no delta correction, no Triton.

    Prefill:  Q,K,V → FlashAttention (full, causal)
              K → Router.update_prefill(K) → finalize()

    Decode:   scores = Router.route(Q)           # O(r*d + r*N)
              top_k  = scores.topk(budget)       # O(N)
              K,V    = gather(cache, top_k)      # index select
              output = FlashAttention(Q, K, V)   # CUDA C++
              stats.update(top_k)                # free feedback
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch

try:
    from hip_attn.v1_2.topk.k_importance import CovarianceRouter
    from hip_attn.v1_2.topk.running_stats import RunningBlockStats
except ImportError:
    import importlib.util as _ilu
    import os as _os
    _p = _os.path.join(_os.path.dirname(__file__), "..", "topk", "k_importance.py")
    _s = _ilu.spec_from_file_location("k_importance", _os.path.abspath(_p))
    _m = _ilu.module_from_spec(_s)
    _s.loader.exec_module(_m)
    CovarianceRouter = _m.CovarianceRouter

    _p2 = _os.path.join(_os.path.dirname(__file__), "..", "topk", "running_stats.py")
    _s2 = _ilu.spec_from_file_location("running_stats", _os.path.abspath(_p2))
    _m2 = _ilu.module_from_spec(_s2)
    _s2.loader.exec_module(_m2)
    RunningBlockStats = _m2.RunningBlockStats


@dataclass
class RouterConfig:
    """Per-layer router configuration."""
    rank: int = 8                 # SVD rank for covariance decomposition
    budget: int = 128             # max blocks to attend per query
    block_size: int = 64          # tokens per block
    sink_blocks: int = 4          # always-attend prefix blocks
    window_blocks: int = 8        # always-attend recent blocks


@dataclass
class LayerRouter:
    """Per-layer state: router + running stats."""
    router: CovarianceRouter
    stats: RunningBlockStats
    finalized: bool = False


class RouterAttentionPipeline:
    """Per-layer top-k routing → FlashAttention. That's it.

    Each of the N layers gets:
      - CovarianceRouter: rank-r approximation of attention pattern
      - RunningBlockStats: EMA feedback from actual attention
    """

    def __init__(
        self,
        num_layers: int,
        num_heads_kv: int,
        head_dim: int,
        config: RouterConfig,
        device: torch.device,
    ):
        self.num_heads_kv = num_heads_kv
        self.head_dim = head_dim
        self.config = config
        self.device = device
        self.layers: Dict[int, LayerRouter] = {}

    def _get_layer(self, layer_id: int) -> LayerRouter:
        if layer_id not in self.layers:
            router = CovarianceRouter(
                rank=self.config.rank,
                dim=self.head_dim,
                block_size=self.config.block_size,
                budget=self.config.budget,
                sink_blocks=self.config.sink_blocks,
                window_blocks=self.config.window_blocks,
            )
            max_blocks = 1024 * 1024 // self.config.block_size
            stats = RunningBlockStats(max_blocks=max_blocks,
                                      num_heads=self.num_heads_kv)
            self.layers[layer_id] = LayerRouter(router=router, stats=stats)
        return self.layers[layer_id]

    # ------------------------------------------------------------------
    # Prefill: full attention + build router
    # ------------------------------------------------------------------

    def forward_prefill(
        self,
        q: torch.Tensor,          # [B, T_q, H_q, D]
        k: torch.Tensor,          # [B, T_k, H_kv, D]
        v: torch.Tensor,          # [B, T_k, H_kv, D_v]
        layer_id: int,
        sm_scale: float,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        """Full causal attention for prefill. Router learns K distribution."""
        from sgl_kernel.flash_attn import flash_attn_varlen_func

        output = flash_attn_varlen_func(
            q.reshape(-1, q.shape[-2], q.shape[-1]),
            k.reshape(-1, k.shape[-2], k.shape[-1]),
            v.reshape(-1, v.shape[-2], v.shape[-1]),
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            softmax_scale=sm_scale,
            causal=True,
        )

        # Build router from K (O(d²) per token, hidden during prefill)
        layer = self._get_layer(layer_id)
        layer.router.update_prefill(k.permute(0, 2, 1, 3))

        return output.reshape(q.shape[0], -1, q.shape[-2], q.shape[-1])

    def finalize_layer(self, layer_id: int):
        """Eigendecompose after prefill. O(d³) once per layer."""
        layer = self.layers.get(layer_id)
        if layer is not None and not layer.finalized:
            layer.router.finalize()
            layer.finalized = True

    def finalize_all(self):
        for lid in self.layers:
            self.finalize_layer(lid)

    # ------------------------------------------------------------------
    # Decode: route → gather → flash attention → update stats
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        q: torch.Tensor,            # [B, 1, H_q, D]
        k_new: torch.Tensor,        # [B, 1, H_kv, D]
        v_new: torch.Tensor,        # [B, 1, H_kv, D_v]
        k_cache: torch.Tensor,      # [N_pages, page_size, H_kv, D]
        v_cache: torch.Tensor,      # [N_pages, page_size, H_kv, D_v]
        block_table: torch.Tensor,  # [B, max_blocks_per_seq]
        seq_lens: torch.Tensor,     # [B]
        layer_id: int,
        sm_scale: float,
        num_heads_q: int,
        k_descale: Optional[torch.Tensor] = None,
        v_descale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-layer top-k routing → FlashAttention.

        5 steps, no Triton, no delta, no multi-stage scan:
          1. route(q)            → scores       O(r*d + r*N)
          2. topk(scores)        → block_mask   O(N)
          3. gather(cache, mask) → K, V         index select
          4. flash_attn(q, K, V) → output       CUDA C++
          5. stats.update(mask)  → feedback      free
        """
        from sgl_kernel.flash_attn import flash_attn_varlen_func

        layer = self._get_layer(layer_id)
        assert layer.finalized, "Call finalize_layer() after prefill"

        B = q.shape[0]
        page_size = k_cache.shape[1]
        H_kv = k_cache.shape[2]
        D = k_cache.shape[3]
        D_v = v_cache.shape[3]
        gqa_ratio = num_heads_q // H_kv

        # --- Step 1: Route ---
        q_squeezed = q.squeeze(1)  # [B, H_q, D]
        if gqa_ratio > 1:
            q_for_route = q_squeezed.reshape(B, H_kv, gqa_ratio, D).mean(2)
        else:
            q_for_route = q_squeezed

        n_blocks_max = seq_lens.max().item() // page_size + 1
        boost = layer.stats.get_boost(n_blocks_max)

        # --- Step 2: Top-k selection ---
        block_mask = layer.router.route(q_for_route, empirical_boost=boost)

        # Update router with new K token
        layer.router.update_decode(k_new.squeeze(1))

        # --- Step 3: Gather selected K/V ---
        all_q, all_k, all_v = [], [], []
        cu_q, cu_k = [0], [0]

        for b in range(B):
            seq_len = seq_lens[b].item()
            n_pages = (seq_len + page_size - 1) // page_size

            # Union across heads, always include last page
            mask_b = block_mask[b, :, :n_pages]
            selected = mask_b.any(dim=0)
            selected[-1] = True
            idx = selected.nonzero(as_tuple=True)[0]

            # Physical page gather
            pages = block_table[b, idx]
            k_g = k_cache[pages].reshape(-1, H_kv, D)
            v_g = v_cache[pages].reshape(-1, H_kv, D_v)

            # FP8 descaling
            if k_descale is not None:
                k_g = k_g.to(torch.float16) * k_descale[b, None, :, None]
                v_g = v_g.to(torch.float16) * v_descale[b, None, :, None]

            all_q.append(q[b, 0:1])       # [1, H_q, D]
            all_k.append(k_g)
            all_v.append(v_g)
            cu_q.append(cu_q[-1] + 1)
            cu_k.append(cu_k[-1] + k_g.shape[0])

        # --- Step 4: FlashAttention ---
        q_cat = torch.cat(all_q)
        k_cat = torch.cat(all_k)
        v_cat = torch.cat(all_v)

        cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=q.device)
        cu_k_t = torch.tensor(cu_k, dtype=torch.int32, device=q.device)
        max_k = max(cu_k[i+1] - cu_k[i] for i in range(B))

        output = flash_attn_varlen_func(
            q_cat, k_cat, v_cat,
            cu_q_t, cu_k_t,
            1, max_k,
            softmax_scale=sm_scale,
            causal=False,  # RoPE already applied in cache
        )

        # --- Step 5: Update stats (free) ---
        for b in range(B):
            n_pages = (seq_lens[b].item() + page_size - 1) // page_size
            selected = block_mask[b, :, :n_pages].any(dim=0).nonzero(as_tuple=True)[0]
            layer.stats.update(selected, attention_lse=None, block_size=page_size)

        return output.unsqueeze(1)  # [B, 1, H_q, D_v]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all layer states (new request batch)."""
        self.layers.clear()

    def reset_layer(self, layer_id: int):
        self.layers.pop(layer_id, None)


# ------------------------------------------------------------------
# Delta correction: fixes distributional shift from sparse attention
# ------------------------------------------------------------------

def apply_delta_correction(
    context_sparse: torch.Tensor,
    context_dense: torch.Tensor,
    sample_indices: torch.Tensor,
    gamma: int,
    smooth: bool = True,
) -> torch.Tensor:
    """Correct distributional shift between sparse and full attention.

    Even with perfect top-k routing, sparse attention has a different
    softmax denominator than full attention → distributional shift.
    Delta correction samples a few positions, computes dense attention
    there, and interpolates the correction across the sequence.

    This is independent of router quality — it's a mathematical
    property of the sparse softmax approximation.

    Args:
        context_sparse: [B, T, H, D] sparse attention output
        context_dense: [B, N_samples, H, D] dense-recomputed at sample positions
        sample_indices: [N_samples] positions where dense was computed
        gamma: interpolation block width (1 sample per gamma tokens)
        smooth: linear interpolation between samples (vs. blockwise copy)

    Returns:
        Corrected output: [B, T, H, D]
    """
    N_samples = sample_indices.shape[0]
    T = context_sparse.shape[1]

    # Delta at sample points: dense - sparse
    sparse_at_samples = context_sparse[:, sample_indices]
    delta = context_dense - sparse_at_samples  # [B, N_samples, H, D]

    # Expand delta to cover gamma positions each
    delta_expanded = delta.repeat_interleave(gamma, dim=1)

    if smooth and N_samples > 1:
        # Linear interpolation between consecutive delta values
        delta_next = torch.roll(delta_expanded, -gamma, 1)
        delta_next[:, -gamma:] = delta_expanded[:, -gamma:]

        t = torch.arange(delta_expanded.shape[1], device=delta.device)
        alpha = (t % gamma).float() / gamma
        delta_expanded = (
            delta_expanded * (1.0 - alpha[None, :, None, None])
            + delta_next * alpha[None, :, None, None]
        )

    # Apply correction
    context = context_sparse.clone()
    n_apply = min(delta_expanded.shape[1], T)
    context[:, :n_apply] += delta_expanded[:, :n_apply]

    # Overwrite sample points with exact dense values
    context[:, sample_indices] = context_dense

    return context
