"""Triton-free attention pipeline using CovarianceRouter + FlashAttention.

Replaces the 3-stage Triton hierarchical scan + Triton BSA with:
1. CovarianceRouter for block selection (PyTorch native, O(r*d + r*N))
2. FlashAttention for compute (CUDA C++ via sgl_kernel)
3. PyTorch native delta correction (optional, no Triton)

Architecture:
    Prefill:  K → CovarianceRouter.update_prefill(K) → finalize()
              Q,K,V → FlashAttention (full, causal) → output

    Decode:   Q → CovarianceRouter.route(Q) → block_mask
              block_mask → gather K/V from paged cache
              Q, K_gathered, V_gathered → FlashAttention → output
              (optional) delta correction via PyTorch native ops
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from hip_attn.v1_2.topk.k_importance import CovarianceRouter
except ImportError:
    # Allow direct file import for testing
    import importlib.util as _ilu
    import os as _os
    _p = _os.path.join(_os.path.dirname(__file__), "..", "topk", "k_importance.py")
    _s = _ilu.spec_from_file_location("k_importance", _os.path.abspath(_p))
    _m = _ilu.module_from_spec(_s)
    _s.loader.exec_module(_m)
    CovarianceRouter = _m.CovarianceRouter


@dataclass
class RouterConfig:
    """Configuration for CovarianceRouter-based attention."""
    rank: int = 8
    budget: int = 128
    block_size: int = 64
    sink_blocks: int = 4
    window_blocks: int = 8
    # Delta correction
    delta_enabled: bool = True
    delta_gamma: int = 16
    delta_smooth: bool = True
    delta_sample_ratio: float = 0.0625  # 1/16 positions recomputed


@dataclass
class RouterState:
    """Per-layer router state."""
    router: CovarianceRouter
    finalized: bool = False


class RouterAttentionPipeline:
    """Triton-free sparse attention pipeline.

    Uses CovarianceRouter for O(r*N) block selection and
    FlashAttention (CUDA C++) for attention compute.
    No Triton kernels anywhere in the pipeline.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads_kv: int,
        head_dim: int,
        config: RouterConfig,
        device: torch.device,
    ):
        self.num_layers = num_layers
        self.num_heads_kv = num_heads_kv
        self.head_dim = head_dim
        self.config = config
        self.device = device

        # Per-layer router states
        self.states: Dict[int, RouterState] = {}

    def _get_or_create_router(self, layer_id: int) -> RouterState:
        if layer_id not in self.states:
            router = CovarianceRouter(
                rank=self.config.rank,
                dim=self.head_dim,
                block_size=self.config.block_size,
                budget=self.config.budget,
                sink_blocks=self.config.sink_blocks,
                window_blocks=self.config.window_blocks,
            )
            self.states[layer_id] = RouterState(router=router)
        return self.states[layer_id]

    def forward_prefill(
        self,
        q: torch.Tensor,       # [B, T_q, H_q, D]
        k: torch.Tensor,       # [B, T_k, H_kv, D]
        v: torch.Tensor,       # [B, T_k, H_kv, D_v]
        layer_id: int,
        sm_scale: float,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool = True,
    ) -> torch.Tensor:
        """Full attention prefill + router update.

        During prefill we run full FlashAttention (no sparsity needed)
        and simultaneously build the CovarianceRouter for decode.
        """
        from sgl_kernel.flash_attn import flash_attn_varlen_func

        # 1. Full FlashAttention for prefill (CUDA C++)
        # q, k, v need to be [total_tokens, heads, dim] for varlen
        q_flat = q.reshape(-1, q.shape[-2], q.shape[-1])
        k_flat = k.reshape(-1, k.shape[-2], k.shape[-1])
        v_flat = v.reshape(-1, v.shape[-2], v.shape[-1])

        output = flash_attn_varlen_func(
            q_flat, k_flat, v_flat,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            softmax_scale=sm_scale,
            causal=causal,
        )

        # 2. Update router with K (PyTorch native, O(d²) per token)
        state = self._get_or_create_router(layer_id)
        # k shape: [B, T_k, H_kv, D] → need [B, H_kv, T_k, D] for router
        k_for_router = k.permute(0, 2, 1, 3)
        state.router.update_prefill(k_for_router)

        return output.reshape(q.shape[0], -1, q.shape[-2], q.shape[-1])

    def finalize_layer(self, layer_id: int):
        """Finalize router after prefill (eigendecompose, O(d³) once)."""
        state = self.states.get(layer_id)
        if state is not None and not state.finalized:
            state.router.finalize()
            state.finalized = True

    def finalize_all(self):
        """Finalize all layer routers."""
        for layer_id in self.states:
            self.finalize_layer(layer_id)

    def forward_decode(
        self,
        q: torch.Tensor,           # [B, 1, H_q, D]
        k_new: torch.Tensor,       # [B, 1, H_kv, D]
        v_new: torch.Tensor,       # [B, 1, H_kv, D_v]
        k_cache: torch.Tensor,     # [N_pages, page_size, H_kv, D]
        v_cache: torch.Tensor,     # [N_pages, page_size, H_kv, D_v]
        block_table: torch.Tensor, # [B, max_blocks_per_seq]
        seq_lens: torch.Tensor,    # [B]
        layer_id: int,
        sm_scale: float,
        num_heads_q: int,
        k_descale: Optional[torch.Tensor] = None,
        v_descale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sparse attention decode using CovarianceRouter + FlashAttention.

        1. Route query → block mask (PyTorch native, O(r*d + r*N))
        2. Gather selected K/V from paged cache (torch.index_select)
        3. FlashAttention on gathered K/V (CUDA C++)
        4. Update router state with new K token
        """
        from sgl_kernel.flash_attn import flash_attn_varlen_func

        state = self._get_or_create_router(layer_id)
        assert state.finalized, "Router must be finalized before decode"

        B = q.shape[0]
        page_size = k_cache.shape[1]
        H_kv = k_cache.shape[2]
        D = k_cache.shape[3]
        D_v = v_cache.shape[3]

        # GQA ratio
        gqa_ratio = num_heads_q // H_kv

        # 1. Route: get block mask
        # q: [B, 1, H_q, D] → [B, H_kv, D] (average over GQA groups)
        q_squeezed = q.squeeze(1)  # [B, H_q, D]
        if gqa_ratio > 1:
            q_for_route = q_squeezed.reshape(B, H_kv, gqa_ratio, D).mean(dim=2)
        else:
            q_for_route = q_squeezed

        block_mask = state.router.route(q_for_route)  # [B, H_kv, N_blocks]

        # 2. Update router with new K
        k_for_router = k_new.squeeze(1)  # [B, H_kv, D]
        state.router.update_decode(k_for_router)

        # 3. Gather selected K/V and run FlashAttention
        # For each request, gather selected blocks from paged cache
        all_q = []
        all_k = []
        all_v = []
        cu_seqlens_q_list = [0]
        cu_seqlens_k_list = [0]

        for b in range(B):
            seq_len = seq_lens[b].item()
            n_pages = (seq_len + page_size - 1) // page_size

            # Union of selected blocks across KV heads
            mask_b = block_mask[b, :, :n_pages]  # [H_kv, n_pages]
            selected = mask_b.any(dim=0)  # [n_pages] — union across heads

            # Always include the last page (contains new token)
            selected[-1] = True

            selected_indices = selected.nonzero(as_tuple=True)[0]  # [n_selected]
            n_selected = selected_indices.shape[0]

            # Map to physical pages via block_table
            physical_pages = block_table[b, selected_indices]  # [n_selected]

            # Gather K/V from cache
            k_gathered = k_cache[physical_pages]  # [n_selected, page_size, H_kv, D]
            v_gathered = v_cache[physical_pages]  # [n_selected, page_size, H_kv, D_v]

            # Handle FP8 descaling
            if k_descale is not None:
                k_gathered = k_gathered.to(torch.float16) * k_descale[b:b+1, None, :, None]
                v_gathered = v_gathered.to(torch.float16) * v_descale[b:b+1, None, :, None]

            # Flatten to [n_selected * page_size, H_kv, D]
            k_flat = k_gathered.reshape(-1, H_kv, D)
            v_flat = v_gathered.reshape(-1, H_kv, D_v)

            # Trim to actual token count in last page
            total_gathered_tokens = n_selected * page_size
            # Don't exceed seq_len worth of tokens from selected blocks
            # (last page may be partially filled)

            # Q for this request: expand for GQA
            q_b = q[b, 0:1]  # [1, H_q, D]

            all_q.append(q_b)
            all_k.append(k_flat)
            all_v.append(v_flat)
            cu_seqlens_q_list.append(cu_seqlens_q_list[-1] + 1)
            cu_seqlens_k_list.append(cu_seqlens_k_list[-1] + total_gathered_tokens)

        # Concatenate all requests
        q_cat = torch.cat(all_q, dim=0)  # [B, H_q, D]
        k_cat = torch.cat(all_k, dim=0)  # [total_k, H_kv, D]
        v_cat = torch.cat(all_v, dim=0)  # [total_k, H_kv, D_v]

        cu_seqlens_q_t = torch.tensor(cu_seqlens_q_list, dtype=torch.int32, device=q.device)
        cu_seqlens_k_t = torch.tensor(cu_seqlens_k_list, dtype=torch.int32, device=q.device)

        max_seqlen_q = 1
        max_seqlen_k = max(cu_seqlens_k_list[i+1] - cu_seqlens_k_list[i] for i in range(B))

        # 4. FlashAttention on gathered sparse K/V (CUDA C++)
        # causal=False because positions already encoded via RoPE in cache
        output = flash_attn_varlen_func(
            q_cat, k_cat, v_cat,
            cu_seqlens_q_t, cu_seqlens_k_t,
            max_seqlen_q, max_seqlen_k,
            softmax_scale=sm_scale,
            causal=False,
        )

        # output: [B, H_q, D_v]
        return output.unsqueeze(1)  # [B, 1, H_q, D_v]

    def reset(self):
        """Reset all router states (e.g., for new request)."""
        self.states.clear()

    def reset_layer(self, layer_id: int):
        """Reset router state for a specific layer."""
        self.states.pop(layer_id, None)


def apply_delta_native(
    context_dense: torch.Tensor,
    context_sparse: torch.Tensor,
    idx: torch.Tensor,
    num_last_dense: int,
    gamma: int,
    smooth: bool = True,
) -> torch.Tensor:
    """PyTorch-native delta correction (no Triton).

    Args:
        context_dense: [B, N_DELTA + num_last_dense, H, D] dense-recomputed at sample points
        context_sparse: [B, T, H, D] sparse attention output
        idx: [N_DELTA + num_last_dense] sample point indices
        num_last_dense: number of trailing tokens computed fully dense
        gamma: block width for interpolation
        smooth: use linear interpolation between deltas

    Returns:
        Corrected context: [B, T + num_last_dense, H, D]
    """
    # Split dense into delta samples and trailing dense tokens
    context_dense_main = context_dense[:, :-num_last_dense] if num_last_dense > 0 else context_dense
    last_context_dense = context_dense[:, -num_last_dense:] if num_last_dense > 0 else None
    idx_main = idx[:-num_last_dense] if num_last_dense > 0 else idx

    # Compute delta at sample points
    sparse_at_samples = context_sparse[:, idx_main]  # [B, N_DELTA, H, D]
    delta = context_dense_main - sparse_at_samples     # [B, N_DELTA, H, D]

    # Expand delta to cover gamma positions each
    delta_expanded = delta.repeat_interleave(gamma, dim=1)  # [B, N_DELTA*gamma, H, D]

    if smooth:
        # Linear interpolation between consecutive delta values
        delta_next = torch.roll(delta_expanded, -gamma, 1)
        delta_next[:, -gamma:] = delta_expanded[:, -gamma:]  # clamp at sequence end

        t = torch.arange(delta_expanded.shape[1], device=delta.device)
        alpha = (t % gamma).float() / gamma
        delta_expanded = (
            delta_expanded * (1.0 - alpha[None, :, None, None])
            + delta_next * alpha[None, :, None, None]
        )

    # Apply correction to sparse output
    T = context_sparse.shape[1]
    context = context_sparse.clone()
    n_apply = min(delta_expanded.shape[1], T)
    context[:, :n_apply] += delta_expanded[:, :n_apply]

    # Overwrite sample points with exact dense values
    context[:, idx_main] = context_dense_main

    # Append trailing dense tokens
    if last_context_dense is not None:
        context = torch.cat([context, last_context_dense], dim=1)

    return context
