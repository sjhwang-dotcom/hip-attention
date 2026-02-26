"""Experimental delta attention paths.

These are gated behind the ``HIP_EXPERIMENTAL=1`` environment variable.

- ``forward_delta_exp``: Exponential delta attention with sliding window + sparse merge.
- ``forward_iter_corr``: Iterative correction with hierarchical block refinement.
"""

import os
import warnings

import torch
import triton

from hip_attn.v1_2.attention_extend import (
    dual_stage_quadratic_hip_attention,
    get_block_sparse_backend,
)
from hip_attn.v1_2.attention_metadata import (
    HiPAttentionArgs,
    HiPAttentionOutputMetadata,
)
from hip_attn.v1_2.config.delta_config import DeltaAttentionConfig
from hip_attn.v1_2.query_sparse_attention import query_sparse_attention

try:
    from sglang.srt.distributed import get_tensor_model_parallel_rank

    SGLANG_DIST_ACTIVATED = True
except ImportError:
    SGLANG_DIST_ACTIVATED = False


def _is_experimental_enabled() -> bool:
    return os.getenv("HIP_EXPERIMENTAL", "0") == "1"


def get_local_rank() -> int:
    if SGLANG_DIST_ACTIVATED:
        return get_tensor_model_parallel_rank()
    else:
        return 0


def forward_delta_exp(
    query: torch.Tensor,
    sm_scale: float,
    k: torch.Tensor,
    v: torch.Tensor,
    args: HiPAttentionArgs,
    cached_metadata: HiPAttentionOutputMetadata,
    config: DeltaAttentionConfig,
):
    """Exponential delta attention path (experimental).

    Exact copy of the ``delta_exp`` branch (original lines 987-1178).
    """
    if not _is_experimental_enabled():
        raise RuntimeError(
            "forward_delta_exp requires HIP_EXPERIMENTAL=1 environment variable"
        )

    delta_exp_w = config.exp_w
    delta_exp_bk = 16
    delta_exp_k = 0
    delta_exp_window = config.exp_window
    delta_exp_sink = config.exp_sink
    delta_merge_strategy = "delta"  # replace / delta
    if delta_exp_k == 0:
        delta_exp_bk = 64

    bsa_fn = get_block_sparse_backend(query, args.disable_flashdecode)

    BSZ, TDST, HEAD, HID = query.shape

    args_sw = args.clone()
    if args_sw.rope_range is None:
        args_sw.rope_range = (0, HID)
    args_sw.block_size_q = args_sw.block_sparse_block_size_q
    args_sw.block_size_k = delta_exp_bk
    args_sw.second_stage_k = delta_exp_k
    args_sw.sink_token_size = delta_exp_sink
    args_sw.sliding_window_size = delta_exp_window
    args_sw.sliding_window_indices = None

    BDST = triton.cdiv(TDST, args_sw.block_size_q)
    BH = BSZ * HEAD

    if delta_exp_k == 0:
        indices = torch.zeros(
            (BH, BDST, delta_exp_k // delta_exp_bk),
            dtype=torch.int64,
            device=query.device,
        )
        ks = torch.zeros((BH, BDST), dtype=torch.int64, device=query.device)
        ks_count = ks.unsqueeze(-1)
        ks_start_end = torch.zeros(
            (BH, BDST, 2), dtype=torch.int64, device=query.device
        )
        ks_start_end[:, :, 1:] = ks[:, :, None]
    else:
        indices = torch.rand(
            (BH, BDST, delta_exp_k // delta_exp_bk), device=query.device
        )
        indices = (
            indices
            * args_sw.position_ids[
                :, :: args_sw.block_size_q
            ].repeat_interleave(HEAD, dim=0)[:, :, None]
        )
        indices = indices.to(torch.int64) // delta_exp_bk * delta_exp_bk

        indices, _ = indices.sort(dim=-1)
        indices = indices // args_sw.block_size_k * args_sw.block_size_k

        unique_mask = torch.roll(indices, shifts=1, dims=-1) != indices
        indices = torch.where(
            unique_mask, indices, torch.iinfo(indices.dtype).max
        )
        indices, _ = indices.sort(dim=-1)
        active_mask = indices < (
            args_sw.position_ids[
                :, :: args_sw.block_size_q, None
            ].repeat_interleave(HEAD, 0)
            + args_sw.block_size_q
        )
        ks = active_mask.int().sum(-1)
        ks_count = ks.unsqueeze(-1)
        ks_start_end = torch.zeros(
            (ks.shape[0], ks.shape[1], 2),
            dtype=torch.int32,
            device=query.device,
        )
        ks_start_end[:, :, -1] = ks

    context_sw = bsa_fn(
        q=(query * sm_scale).to(query.dtype),
        k=k,
        v=v,
        seq_lens=args_sw.position_ids + 1,
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        access_counter=None,
        cache_miss_counter=None,
        EXTEND_BACKEND=args_sw.sa_extend_backend,
        model_context_length=args_sw.model_context_length,
        extend_context_length=args_sw.extend_context_length,
        offload_update_cache=False,
        args=args_sw,
    )
    context_sw = context_sw.to(query.dtype)

    args_sparse = args.clone()
    query_sparse = query[:, ::delta_exp_w].contiguous()
    args_sparse.position_ids = args.position_ids[:, ::delta_exp_w].contiguous()
    args_sparse.query_for_landmark = query
    args_sparse.position_ids_for_landmark = args.position_ids

    context_sparse, metadata = dual_stage_quadratic_hip_attention(
        q=(query_sparse * sm_scale).to(query.dtype),
        k=k,
        v=v,
        args=args_sparse,
        cached_metadata=cached_metadata,
    )
    context_sparse = context_sparse.to(query.dtype)

    if delta_merge_strategy == "delta":
        context_sw_for_sparse = context_sw[:, ::delta_exp_w]
        delta_sparse = context_sparse - context_sw_for_sparse

        delta_sparse = delta_sparse.repeat_interleave(delta_exp_w, dim=1)

        if config.smooth:
            # (exp) linear interpolate diff
            delta_sparse_shift = torch.roll(delta_sparse, -delta_exp_w, 1)
            delta_sparse_shift[:, -delta_exp_w:] = delta_sparse[:, -1:]

            idx = torch.arange(
                0, delta_sparse.shape[1], device=delta_sparse.device
            )
            idx = (idx % delta_exp_w).float() / delta_exp_w
            delta_sparse = (
                delta_sparse
                + (delta_sparse_shift - delta_sparse) * idx[None, :, None, None]
            )

        context_sparse = context_sw + delta_sparse[:, : context_sw.shape[1]]
    elif delta_merge_strategy == "replace":
        context_sw[:, ::delta_exp_w] = context_sparse
        context_sparse = context_sw
    else:
        raise Exception()

    return context_sparse, metadata


def forward_iter_corr(
    context_sparse: torch.Tensor,
    query: torch.Tensor,
    sm_scale: float,
    args: HiPAttentionArgs,
    config: DeltaAttentionConfig,
    k_descale: torch.Tensor = None,
    v_descale: torch.Tensor = None,
):
    """Iterative correction path (experimental).

    Exact copy of the ``iter_corr`` branch (original lines 1248-1409).
    """
    if not _is_experimental_enabled():
        raise RuntimeError(
            "forward_iter_corr requires HIP_EXPERIMENTAL=1 environment variable"
        )

    w_size = config.gamma * 2

    num_queries = query.shape[1]
    num_dense_first = max(128, w_size)
    num_dense_last = num_queries % w_size + max(128, w_size)
    num_sparse = num_queries - num_dense_first - num_dense_last

    # iteratively correction errors

    def perform_correction(
        context_sparse: torch.Tensor,
        context_sparse_raw: torch.Tensor,
        block_start_indices: torch.Tensor,
        block_size: int,
    ):
        assert block_start_indices.ndim == 1
        assert context_sparse.ndim == 4
        assert context_sparse_raw.shape == context_sparse.shape
        assert not (args.need_apply_rope and args.using_extend)

        assert args.using_paged_cache

        if False:
            context_sparse_raw = context_sparse

        query_for_recomp = query[:, block_start_indices, :, :]
        k_cache = args.get_k_cache()
        v_cache = args.get_v_cache()

        assert args.position_ids.shape[0] == 1
        if get_local_rank() == 0:
            print(
                "recomp_attn shapes",
                query_for_recomp.shape,
                block_start_indices.shape,
            )
        context_dense = (
            query_sparse_attention(
                query_for_recomp.permute(0, 2, 1, 3).contiguous(),
                None,
                None,
                args.position_ids[:, block_start_indices],
                sm_scale,
                k_cache,
                v_cache,
                args.block_table,
                k_descale=k_descale,
                v_descale=v_descale,
                softmax_sink=args.softmax_sink,
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # type: torch.Tensor
        assert context_dense.shape[-2:] == query.shape[-2:]

        if block_size > 1:
            assert not config.smooth
            block_diff = diff = (
                context_dense - context_sparse_raw[:, block_start_indices]
            )
            diff = diff.repeat_interleave(block_size, 1)

            token_indices = (
                block_start_indices[:, None]
                + torch.arange(0, block_size, device=context_sparse.device)[None, :]
            )
            token_indices = token_indices.view(-1)

            context_sparse_new = diff + context_sparse_raw[:, token_indices]
            context_sparse.index_copy_(
                dim=1, index=token_indices, source=context_sparse_new
            )
        else:
            context_sparse.index_copy_(
                dim=1, index=block_start_indices, source=context_dense
            )
            block_diff = None

        return context_sparse, block_diff

    block_start_indices = torch.arange(
        num_dense_first,
        num_dense_first + num_sparse,
        step=w_size,
        device=query.device,
    )
    assert (num_dense_first % w_size) == 0
    assert ((num_dense_first + num_sparse) % w_size) == 0

    split = 2

    def block_diff_to_score(block_diff: torch.Tensor):
        return (
            block_diff.squeeze(0)
            .norm(dim=-1, keepdim=False)
            .sum(dim=-1, keepdim=False)
        )

    context_sparse_raw = context_sparse.clone()

    context_sparse, block_diff = perform_correction(
        context_sparse,
        context_sparse_raw,
        block_start_indices,
        w_size,
    )
    # [T,]
    block_diff_scores_parent, block_diff_indices = block_diff_to_score(
        block_diff
    ).topk(k=block_diff.shape[1] // split, dim=0, sorted=False)
    block_start_indices_parent = block_start_indices[block_diff_indices]
    block_start_indices_parent, tind = block_start_indices_parent.sort()
    block_diff_scores_parent = block_diff_scores_parent[tind]

    depth = 0
    max_iter = 4
    while (w_size // split) > 0 and (depth < max_iter):
        depth += 1
        block_start_indices_child = block_start_indices_parent + w_size // split
        w_size = w_size // split

        context_sparse, block_diff = perform_correction(
            context_sparse,
            context_sparse_raw,
            block_start_indices_child,
            w_size,
        )
        if (w_size // split) > 0:
            block_diff_scores_child = block_diff_to_score(block_diff)
            block_diff_scores_parent, next_blocks_location = torch.cat(
                [block_diff_scores_parent, block_diff_scores_child]
            ).topk(k=block_diff_scores_parent.shape[0] // 2, sorted=False)
            block_start_indices_parent = torch.cat(
                [block_start_indices_parent, block_start_indices_child]
            )[next_blocks_location]
            block_start_indices_parent, tind = block_start_indices_parent.sort()
            block_diff_scores_parent = block_diff_scores_parent[tind]

    # fill dense for first and last part
    dense_indices = torch.cat(
        [
            torch.arange(0, num_dense_first, device=query.device),
            torch.arange(
                num_dense_first + num_sparse,
                num_queries,
                device=query.device,
            ),
        ]
    )
    context_sparse, _ = perform_correction(
        context_sparse,
        context_sparse_raw,
        dense_indices,
        1,
    )

    context = context_sparse
    return context
