"""Three-kernel pipeline: HiP Sparse -> QSA Dense Recompute -> Delta Correction.

Extracted from ``_forward_delta_attn`` in ``paged_hip.py`` (original lines 831-2034).
Zero behavioral change -- every computation produces identical results.
"""

import os

import cv2
import numba
import numpy as np
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


def get_local_rank() -> int:
    if SGLANG_DIST_ACTIVATED:
        return get_tensor_model_parallel_rank()
    else:
        return 0


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@numba.njit(parallel=False)
def convert_qsa_mask_to_img(
    bsa_indices: np.ndarray,
    bsa_scores,
    seq_len: np.ndarray,
    tdst: np.ndarray,
    TDST: int,
    TSRC: int,
    POOL_SIZE: int,
):
    N_SPARSE_Q = bsa_indices.shape[0]
    N_BLOCK = bsa_indices.shape[1]
    img = np.zeros((TDST // POOL_SIZE, TSRC // POOL_SIZE, 3), dtype=np.int32)
    img_cnt = np.zeros((TDST // POOL_SIZE, TSRC // POOL_SIZE, 1), dtype=np.int32)

    px_cnt = 0

    for i_q in numba.prange(N_SPARSE_Q):
        for k in range(N_BLOCK):
            pty = tdst[i_q]
            ptx = bsa_indices[i_q, k]
            if (ptx // POOL_SIZE) < img.shape[1] and (pty // POOL_SIZE) < img.shape[0]:
                if bsa_scores is not None:
                    score = bsa_scores[i_q, k]
                else:
                    score = 1.0
                img[pty // POOL_SIZE, ptx // POOL_SIZE, 0] += 255
                img[pty // POOL_SIZE, ptx // POOL_SIZE, 1] += int(255 * score)
                img[pty // POOL_SIZE, ptx // POOL_SIZE, 2] += int(255 * (1 - score))
                img_cnt[pty // POOL_SIZE, ptx // POOL_SIZE] += 1
                px_cnt += 1

    for i in numba.prange(img.shape[0]):
        for j in range(img.shape[1]):
            c = img_cnt[i, j]
            if c > 0:
                img[i, j] = (img[i, j] / c).astype(np.int32)

    img = img.astype(np.uint8)

    return img


class DeltaPipeline:
    """Three-kernel pipeline: HiP Sparse -> QSA Dense Recompute -> Delta Correction."""

    def __init__(self, config: DeltaAttentionConfig):
        self.config = config

    def forward(
        self,
        query: torch.Tensor,
        sm_scale: float,
        k: torch.Tensor,
        v: torch.Tensor,
        args: HiPAttentionArgs,
        cached_metadata: HiPAttentionOutputMetadata,
        k_descale: torch.Tensor = None,
        v_descale: torch.Tensor = None,
        rope_cos: torch.Tensor = None,
        rope_sin: torch.Tensor = None,
    ):
        assert not torch.cuda.is_current_stream_capturing()

        test_qsa_masking = os.getenv("HIP_DEBUG_DELTA_QSA", "0") == "1"
        delta_pool_q = os.getenv("DELTA_POOL_Q", "0") == "1"

        # Stage 1: Sparse context
        context_sparse, metadata, sparse_mx, sparse_nc = self._compute_sparse(
            query, sm_scale, k, v, args, cached_metadata,
            test_qsa_masking,
            k_descale, v_descale,
        )

        if self.config.just_return:
            return context_sparse, metadata

        if self.config.iter_corr:
            from hip_attn.v1_2.pipeline.experimental import forward_iter_corr
            context = forward_iter_corr(
                context_sparse, query, sm_scale, args, self.config,
                k_descale=k_descale, v_descale=v_descale,
            )
            return context, metadata

        # Stage 2: Recompute indices
        num_sparse, num_last_dense, idx, idx_sparse, query_for_dense, \
            query_for_dense_non_pooled, context_sparse_raw, context_sparse, \
            sparse_mx, sparse_nc = self._select_recompute_queries(
                query, args, context_sparse,
                sparse_mx, sparse_nc,
                test_qsa_masking, delta_pool_q,
            )

        # Stage 3: Dense recomputation
        context_dense, context_dense_non_pooled, dense_mx, dense_nc, \
            context_sparse, context_sparse_raw, metadata = self._compute_dense(
                query, query_for_dense, query_for_dense_non_pooled,
                idx, idx_sparse, sm_scale, args,
                k, v,
                k_descale, v_descale, rope_cos, rope_sin,
                test_qsa_masking, delta_pool_q,
                num_sparse, num_last_dense,
                context_sparse, context_sparse_raw,
                metadata,
                sparse_mx, sparse_nc,
            )

        # Stage 4: Delta correction
        context = self._apply_delta(
            context_sparse, context_sparse_raw,
            context_dense, context_dense_non_pooled,
            idx, num_sparse, num_last_dense, query,
            delta_pool_q,
        )

        return context, metadata

    def _compute_sparse(
        self,
        query: torch.Tensor,
        sm_scale: float,
        k: torch.Tensor,
        v: torch.Tensor,
        args: HiPAttentionArgs,
        cached_metadata: HiPAttentionOutputMetadata,
        test_qsa_masking: bool,
        k_descale: torch.Tensor = None,
        v_descale: torch.Tensor = None,
    ):
        """Stage 1: Compute sparse context attention.

        Original lines 953-1243.
        """
        config = self.config
        delta_attention_args_window = config.base_window_size
        delta_attention_args_adjust_norm_const = config.adjust_norm_const

        sparse_mx = None
        sparse_nc = None

        assert isinstance(delta_attention_args_window, int)
        if delta_attention_args_window == 0:
            assert delta_attention_args_window == 0

            delta_exp = config.exp_enabled

            if delta_exp:
                from hip_attn.v1_2.pipeline.experimental import forward_delta_exp
                context_sparse, metadata = forward_delta_exp(
                    query, sm_scale, k, v, args, cached_metadata, config,
                )
                context_sparse = context_sparse.to(query.dtype)
                context_sparse = context_sparse[:, -query.shape[1]:, :, :].contiguous()
                return context_sparse, metadata, sparse_mx, sparse_nc
            else:
                args.bsa_return_running_statistics = delta_attention_args_adjust_norm_const

                if test_qsa_masking:
                    context_sparse = torch.zeros_like(query)
                else:
                    context_sparse, metadata = dual_stage_quadratic_hip_attention(
                        q=(query * sm_scale).to(query.dtype),
                        k=k,
                        v=v,
                        args=args,
                        cached_metadata=cached_metadata,
                    )

                if delta_attention_args_adjust_norm_const:
                    context_sparse, (sparse_mx, sparse_nc) = context_sparse

                context_sparse = context_sparse.to(query.dtype)
                context_sparse = context_sparse[:, -query.shape[1]:, :, :].contiguous()

                if delta_attention_args_adjust_norm_const:
                    sparse_mx = sparse_mx[:, -query.shape[1]:].contiguous()
                    sparse_nc = sparse_nc[:, -query.shape[1]:].contiguous()

                if test_qsa_masking:
                    metadata = None
        else:
            assert delta_attention_args_window > 0
            bsa_fn = get_block_sparse_backend(query, args.disable_flashdecode)

            BSZ, TDST, HEAD, HID = query.shape

            args_sw = args.clone()
            if args_sw.rope_range is None:
                args_sw.rope_range = (0, HID)
            args_sw.block_size_q = args_sw.block_sparse_block_size_q
            args_sw.block_size_k = args_sw.stages[-1].stage_chunk_size
            args_sw.second_stage_k = 0
            # args_sw.sink_token_size = 0 #NOTE: you should inherit this value
            args_sw.sliding_window_size = delta_attention_args_window
            args_sw.sliding_window_indices = None

            if os.getenv("HIP_DEBUG_FORCE_CHUNKED_SW", "0") == "1":
                args_sw.using_chunked_sliding_window = True

            BDST = triton.cdiv(TDST, args_sw.block_size_q)
            BH = BSZ * HEAD

            indices = torch.zeros((BH, BDST, 0), dtype=torch.int64, device=query.device)
            ks = torch.zeros((BH, BDST), dtype=torch.int64, device=query.device)
            ks_count = ks.unsqueeze(-1)
            ks_start_end = torch.zeros(
                (BH, BDST, 2), dtype=torch.int64, device=query.device
            )

            context_sparse = bsa_fn(
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
                return_running_statistics=delta_attention_args_adjust_norm_const,
                args=args_sw,
            )
            if delta_attention_args_adjust_norm_const:
                context_sparse, (sparse_mx, sparse_nc) = context_sparse

            context_sparse = context_sparse.to(query.dtype)
            context_sparse = context_sparse[:, -query.shape[1]:, :, :].contiguous()
            if delta_attention_args_adjust_norm_const:
                sparse_mx = sparse_mx[:, -query.shape[1]:].contiguous()
                sparse_nc = sparse_nc[:, -query.shape[1]:].contiguous()
            metadata = None

        return context_sparse, metadata, sparse_mx, sparse_nc

    def _select_recompute_queries(
        self,
        query: torch.Tensor,
        args: HiPAttentionArgs,
        context_sparse: torch.Tensor,
        sparse_mx,
        sparse_nc,
        test_qsa_masking: bool,
        delta_pool_q: bool,
    ):
        """Stage 2: Select which queries to recompute densely.

        Original lines 1420-1492.
        Returns (num_sparse, num_last_dense, idx, idx_sparse,
                 query_for_dense, query_for_dense_non_pooled,
                 context_sparse_raw, context_sparse_trimmed,
                 sparse_mx, sparse_nc).
        """
        config = self.config
        delta_attention_args_w = config.gamma
        delta_attention_args_diff = config.diff_mode
        delta_attention_args_adjust_norm_const = config.adjust_norm_const

        num_queries = query.shape[1]
        num_last_dense = num_queries % delta_attention_args_w + max(
            128, delta_attention_args_w
        )
        num_last_dense = min(num_queries, num_last_dense)
        num_sparse = num_queries - num_last_dense

        context_sparse_raw = context_sparse
        context_sparse_trimmed = context_sparse[:, :num_sparse]
        if delta_attention_args_adjust_norm_const:
            sparse_mx = sparse_mx[:, :num_sparse]
            sparse_nc = sparse_nc[:, :num_sparse]

        query_for_dense = None
        query_for_dense_non_pooled = None
        idx = None
        idx_sparse = None

        if num_last_dense > 0:
            idx = torch.arange(
                0,
                num_sparse,
                step=delta_attention_args_w,
                device=query.device,
            )
            rolling_idx = False
            if rolling_idx:
                idx = (idx + (args.layer_id % delta_attention_args_w)).clamp_max(
                    num_sparse - 1
                )

            if delta_attention_args_adjust_norm_const:
                context_sparse_for_diff = context_sparse_trimmed[:, idx]
                sparse_mx_for_diff = sparse_mx[:, idx]
                sparse_nc_for_diff = sparse_nc[:, idx]

            idx_sparse = idx
            idx = torch.cat(
                (
                    idx,
                    torch.arange(num_sparse, num_queries, device=query.device),
                )
            )
            if (not test_qsa_masking) and (delta_attention_args_diff == 2):
                idx = torch.arange(num_sparse, num_queries, device=query.device)
                query_for_dense = query[:, idx]
            else:
                if delta_pool_q:
                    query_for_dense = torch.cat(
                        [
                            query[:, :num_sparse]
                            .reshape(
                                query.shape[0],
                                num_sparse // delta_attention_args_w,
                                delta_attention_args_w,
                                query.shape[2],
                                query.shape[3],
                            )
                            .mean(dim=2),
                            query[:, idx[idx_sparse.shape[0]:]],
                        ],
                        dim=1,
                    )
                    query_for_dense_non_pooled = query[:, idx].clone()
                else:
                    query_for_dense = query[:, idx]

        return (
            num_sparse, num_last_dense, idx, idx_sparse,
            query_for_dense, query_for_dense_non_pooled,
            context_sparse_raw, context_sparse_trimmed,
            sparse_mx, sparse_nc,
        )

    def _compute_dense(
        self,
        query: torch.Tensor,
        query_for_dense: torch.Tensor,
        query_for_dense_non_pooled,
        idx: torch.Tensor,
        idx_sparse,
        sm_scale: float,
        args: HiPAttentionArgs,
        k: torch.Tensor,
        v: torch.Tensor,
        k_descale: torch.Tensor,
        v_descale: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        test_qsa_masking: bool,
        delta_pool_q: bool,
        num_sparse: int,
        num_last_dense: int,
        context_sparse: torch.Tensor,
        context_sparse_raw: torch.Tensor,
        metadata,
        sparse_mx,
        sparse_nc,
    ):
        """Stage 3: Dense recomputation via QSA.

        Original lines 1494-1962.
        Returns (context_dense, context_dense_non_pooled, dense_mx, dense_nc,
                 context_sparse, context_sparse_raw, metadata).
        """
        config = self.config
        delta_attention_args_extend = config.extend_mode
        delta_attention_args_adjust_norm_const = config.adjust_norm_const
        delta_attention_args_w = config.gamma

        context_dense_non_pooled = None

        if (args.need_apply_rope and args.using_extend) and (
            delta_attention_args_extend == "none"
        ):
            assert delta_attention_args_extend == "none"
            # TODO: using paged attention
            repeated_k = args.gather_k_from_paged_cache(disable_gqa=True, gqa_q=query)
            repeated_v = args.gather_v_from_paged_cache(
                disable_gqa=True, gqa_q=query
            )  # B, T, H, D
            assert repeated_k.shape[2] < 128

            seq_len = args.position_ids.amax().item() + 1
            repeated_k = repeated_k[:, :seq_len]
            repeated_v = repeated_v[:, :seq_len]

            query_for_recomp = query_for_dense

            cos = args.rope_cos
            sin = args.rope_sin
            assert cos.ndim == 2, cos.shape
            assert sin.shape == cos.shape, sin.shape

            cos = cos.view(1, cos.shape[-2], 1, cos.shape[-1])
            sin = sin.view(1, sin.shape[-2], 1, sin.shape[-1])

            idx_tsrc = torch.arange(0, repeated_k.shape[1], device=cos.device)
            idx_tsrc.clamp_min_(seq_len - args.model_context_length)

            repeated_k = (
                (repeated_k.to(cos.dtype) * cos[:, idx_tsrc, :, :])
                + (rotate_half(repeated_k.to(sin.dtype)) * sin[:, idx_tsrc, :, :])
            ).to(repeated_k.dtype)

            query_for_recomp = (
                (query_for_recomp * cos[:, args.position_ids.view(-1)[idx], :, :])
                + (
                    rotate_half(query_for_recomp)
                    * sin[:, args.position_ids.view(-1)[idx], :, :]
                )
            ).to(query_for_recomp.dtype)

            assert args.position_ids.shape[0] == 1
            context_dense = (
                query_sparse_attention(
                    query_for_recomp.permute(0, 2, 1, 3).contiguous(),
                    repeated_k.permute(0, 2, 1, 3).contiguous(),
                    repeated_v.permute(0, 2, 1, 3).contiguous(),
                    args.position_ids[:, idx],
                    sm_scale,
                    None,
                    None,
                    None,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    softmax_sink=args.softmax_sink,
                )
                .permute(0, 2, 1, 3)
                .contiguous()
            )
        else:
            if args.need_apply_rope and args.using_extend:
                assert delta_attention_args_extend in ("self_extend", "nope")

            query_for_recomp = query_for_dense

            if args.using_paged_cache:
                assert args.using_paged_cache

                k_cache = args.get_k_cache()
                v_cache = args.get_v_cache()

                assert args.position_ids.shape[0] == 1

                # NOTE: using Delta 2
                test_qsa_masking = os.getenv("HIP_DEBUG_DELTA_QSA", "0") == "1"
                # NOTE: save mask image
                debug_qsa_masking = os.getenv("HIP_DEBUG_DELTA_QSA_IMSAVE", "0") == "1"
                debug_qsa_masking_state = (
                    os.getenv("HIP_DEBUG_DELTA_QSA_IMSAVE_STATE", "0") == "1"
                )
                mask_idx = args.position_ids[:, idx]
                qsa_mask_block_size_q = config.qsa_block_size_q
                qsa_mask_block_size_k = config.qsa_block_size_k
                reverse_iter = config.qsa_reverse_iter
                qsa_mask_block_top_k = config.qsa_top_k
                online_topk_method = config.online_topk_method
                exact_k = config.qsa_exact_k
                threshold_refresh_interval = config.qsa_threshold_refresh
                # using each block scores
                qsa_mask_pre_trim = 40960000
                # using sum of block scores
                qsa_mask_post_trim = config.qsa_post_trim

                context_dense = query_sparse_attention(
                    query_for_recomp.permute(0, 2, 1, 3).contiguous(),
                    None,
                    None,
                    mask_idx,
                    sm_scale,
                    k_cache,
                    v_cache,
                    args.block_table,
                    return_running_statistics=delta_attention_args_adjust_norm_const,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    extend_backend=delta_attention_args_extend,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    model_context_length=args.model_context_length,
                    self_extend_scale=args.self_extend_scale,
                    softmax_sink=args.softmax_sink,
                    bsa_top_block_k=qsa_mask_block_top_k,
                    bsa_block_size_k=qsa_mask_block_size_k,
                    bsa_mask_sink_token_size=max(1, args.sink_token_size),
                    bsa_mask_sliding_window_size=args.sliding_window_size,
                    return_bsa_indices=test_qsa_masking,
                    online_topk_method=online_topk_method,
                    reverse_iter=reverse_iter,
                    exact_k=exact_k,
                    threshold_refresh_interval=threshold_refresh_interval,
                )

                if delta_pool_q:
                    context_dense_non_pooled = query_sparse_attention(
                        query_for_dense_non_pooled.permute(0, 2, 1, 3).contiguous(),
                        None,
                        None,
                        mask_idx,
                        sm_scale,
                        k_cache,
                        v_cache,
                        args.block_table,
                        return_running_statistics=False,
                        k_descale=k_descale,
                        v_descale=v_descale,
                        extend_backend=delta_attention_args_extend,
                        rope_cos=rope_cos,
                        rope_sin=rope_sin,
                        model_context_length=args.model_context_length,
                        self_extend_scale=args.self_extend_scale,
                        softmax_sink=args.softmax_sink,
                        bsa_top_block_k=qsa_mask_block_top_k,
                        bsa_block_size_k=qsa_mask_block_size_k,
                        bsa_mask_sink_token_size=max(1, args.sink_token_size),
                        bsa_mask_sliding_window_size=args.sliding_window_size,
                        return_bsa_indices=False,
                        online_topk_method=online_topk_method,
                        reverse_iter=reverse_iter,
                        exact_k=exact_k,
                        threshold_refresh_interval=threshold_refresh_interval,
                    )

                if test_qsa_masking:
                    context_dense, (bsa_indices, bsa_block_sums) = context_dense

                    if debug_qsa_masking and (get_local_rank() == 0):
                        scores = bsa_block_sums[0, 0]
                        scores_min = scores.amin()
                        scores_max = scores.amax()
                        scores = (scores - scores_min) / (scores_max - scores_min)
                        mask = convert_qsa_mask_to_img(
                            bsa_indices[0, 0].cpu().numpy(),
                            None,
                            idx.cpu().numpy(),
                            idx.cpu().numpy(),
                            query.shape[1],
                            int(mask_idx.amax().item()) + 256,
                            256,
                        )
                        cv2.imwrite(f"dummy_qsa_mask_ilayer_{args.layer_id}.png", mask)

                    args_sparse = args.clone()
                    args_sparse.rope_range = (0, query.shape[-1])
                    args_sparse.position_ids = args_sparse.position_ids[
                        :, :-num_last_dense
                    ]
                    args_sparse.block_size_q = qsa_mask_block_size_q
                    args_sparse.block_sparse_block_size_q = args_sparse.block_size_q
                    args_sparse.block_size_k = qsa_mask_block_size_k
                    args_sparse.sliding_window_size = (
                        args_sparse.sliding_window_size + 512
                    )

                    bsa_fn = get_block_sparse_backend(
                        query,
                        args.disable_flashdecode,
                    )

                    indices = bsa_indices.flatten(0, 1)[:, :-num_last_dense, :]

                    num_union = (
                        args_sparse.block_sparse_block_size_q // delta_attention_args_w
                    )
                    assert (
                        args_sparse.block_sparse_block_size_q % delta_attention_args_w
                    ) == 0
                    if indices.shape[1] % num_union:
                        indices = torch.cat(
                            [
                                indices,
                                indices[:, -1:, :].repeat(
                                    1, num_union - indices.shape[1] % num_union, 1
                                ),
                            ],
                            dim=1,
                        )
                    indices = indices.view(
                        indices.shape[0],
                        indices.shape[1] // num_union,
                        num_union,
                        indices.shape[2],
                    )
                    indices = indices.flatten(-2, -1)

                    num_blocks_to_trim = qsa_mask_pre_trim // args_sparse.block_size_k
                    if num_blocks_to_trim < indices.shape[-1]:
                        block_scores = bsa_block_sums.flatten(0, 1)[
                            :, :-num_last_dense, :
                        ]
                        if block_scores.shape[1] % num_union:
                            block_scores = torch.cat(
                                [
                                    block_scores,
                                    block_scores[:, -1:, :].repeat(
                                        1,
                                        num_union - block_scores.shape[1] % num_union,
                                        1,
                                    ),
                                ],
                                dim=1,
                            )
                        block_scores = block_scores.view(
                            block_scores.shape[0],
                            block_scores.shape[1] // num_union,
                            num_union,
                            block_scores.shape[2],
                        )
                        block_scores = block_scores.flatten(-2, -1)

                        t_indices = torch.sort(
                            block_scores, dim=-1, descending=True
                        ).indices
                        indices = indices.gather(
                            dim=-1,
                            index=t_indices[..., :num_blocks_to_trim],
                        )

                    indices, t_sort = indices.sort(dim=-1)
                    num_blocks_to_trim = qsa_mask_post_trim // args_sparse.block_size_k
                    if num_blocks_to_trim < indices.shape[-1]:
                        block_scores = bsa_block_sums.flatten(0, 1)[
                            :, :-num_last_dense, :
                        ]
                        if block_scores.shape[1] % num_union:
                            block_scores = torch.cat(
                                [
                                    block_scores,
                                    block_scores[:, -1:, :].repeat(
                                        1,
                                        num_union - block_scores.shape[1] % num_union,
                                        1,
                                    ),
                                ],
                                dim=1,
                            )
                        block_scores = block_scores.view(
                            block_scores.shape[0],
                            block_scores.shape[1] // num_union,
                            num_union,
                            block_scores.shape[2],
                        )
                        block_scores = block_scores.flatten(-2, -1)
                        block_scores = block_scores.gather(dim=-1, index=t_sort)

                        unique_mask = torch.roll(indices, shifts=1, dims=-1) != indices
                        block_scores_cumsum = torch.exp(
                            block_scores - block_scores.amax(-1, keepdim=True)
                        ).cumsum(-1)
                        block_scores_cumsum_base = (
                            (block_scores_cumsum * unique_mask).cummax(-1).values
                        )
                        block_scores_cumsum = (
                            block_scores_cumsum
                            - block_scores_cumsum_base
                            + block_scores
                        )
                        unique_mask_last = (
                            torch.roll(indices, shifts=-1, dims=-1) != indices
                        )
                        block_scores_cumsum = torch.where(
                            unique_mask_last,
                            block_scores_cumsum,
                            torch.finfo(block_scores_cumsum.dtype).min,
                        )
                        counter_start = torch.arange(
                            0,
                            block_scores_cumsum.shape[-1],
                            device=block_scores_cumsum.device,
                        )
                        counter_end = counter_start.clone()
                        counter_start = (
                            (counter_start[None, None, :] * unique_mask)
                            .cummax(dim=-1)
                            .values
                        )
                        counter_end = (
                            (counter_end[None, None, :] * unique_mask_last)
                            .cummax(dim=-1)
                            .values
                        )
                        counter = (counter_end - counter_start + 1) * unique_mask_last
                        block_scores_cumsum = torch.where(
                            unique_mask_last,
                            block_scores_cumsum / counter,
                            torch.finfo(block_scores_cumsum.dtype).min,
                        )
                        indices = torch.where(
                            unique_mask_last, indices, torch.iinfo(indices.dtype).max
                        )
                        t_sort = block_scores_cumsum.argsort(dim=-1, descending=True)
                        indices = indices.gather(
                            index=t_sort[..., :num_blocks_to_trim], dim=-1
                        )
                    else:
                        unique_mask = torch.roll(indices, shifts=1, dims=-1) != indices
                        indices = torch.where(
                            unique_mask, indices, torch.iinfo(indices.dtype).max
                        )
                    indices, _ = indices.sort(dim=-1)

                    active_mask = indices < (
                        args_sparse.position_ids[
                            :, :: args_sparse.block_size_q, None
                        ].repeat_interleave(query.shape[2], 0)
                        + args.block_size_q
                    )
                    ks = active_mask.int().sum(-1)
                    ks_count = ks.unsqueeze(-1)
                    ks_start_end = torch.zeros(
                        (ks.shape[0], ks.shape[1], 2),
                        dtype=torch.int32,
                        device=query.device,
                    )
                    ks_start_end[:, :, -1] = ks

                    bsa_block_size_q = 128
                    if args_sparse.block_size_q > bsa_block_size_q:
                        assert (args_sparse.block_size_q % bsa_block_size_q) == 0
                        nrepeat = args_sparse.block_size_q // bsa_block_size_q
                        indices = indices.repeat_interleave(nrepeat, 1)
                        ks = ks.repeat_interleave(nrepeat, 1)
                        ks_count = ks_count.repeat_interleave(nrepeat, 1)
                        ks_start_end = ks_start_end.repeat_interleave(nrepeat, 1)
                        args_sparse.block_size_q = bsa_block_size_q
                        args_sparse.block_sparse_block_size_q = bsa_block_size_q

                    if debug_qsa_masking and (get_local_rank() == 0):
                        root = "/data/jeff/delta/datasave"
                        mask = convert_qsa_mask_to_img(
                            indices[0].cpu().numpy(),
                            None,
                            torch.arange(0, indices.shape[1]).numpy()
                            * bsa_block_size_q,
                            torch.arange(0, indices.shape[1]).numpy()
                            * bsa_block_size_q,
                            query.shape[1],
                            int(mask_idx.amax().item()) + 256,
                            256,
                        )
                        cv2.imwrite(
                            f"{root}/dummy_qsa_mask_ilayer_{args.layer_id}_bsa.png",
                            mask,
                        )

                        if debug_qsa_masking_state:
                            torch.save(
                                {
                                    "q": query[:, :-num_last_dense],
                                    "k": k,
                                    "v": k,
                                    "using_paged_cache": args.using_paged_cache,
                                    "k_paged": args.gather_k_from_paged_cache(),
                                    "v_paged": args.gather_v_from_paged_cache(),
                                    "seq_lens": args_sparse.position_ids + 1,
                                    "indices": indices,
                                    "ks": ks,
                                    "sm_scale": sm_scale,
                                },
                                f"{root}/dummy_qsa_mask_ilayer_{args.layer_id}_state.pth",
                            )

                    context_sparse = bsa_fn(
                        q=(query[:, :-num_last_dense] * sm_scale).to(query.dtype),
                        k=k,
                        v=v,
                        seq_lens=args_sparse.position_ids + 1,
                        indices=indices,
                        ks=ks,
                        ks_count=ks_count,
                        ks_start_end=ks_start_end,
                        access_counter=None,
                        cache_miss_counter=None,
                        EXTEND_BACKEND=args_sparse.sa_extend_backend,
                        model_context_length=args_sparse.model_context_length,
                        extend_context_length=args_sparse.extend_context_length,
                        offload_update_cache=False,
                        args=args_sparse,
                        k_descale=k_descale,
                        v_descale=v_descale,
                    )

                    context_sparse = torch.cat(
                        [
                            context_sparse,
                            context_dense.permute(0, 2, 1, 3)[:, -num_last_dense:],
                        ],
                        dim=1,
                    )

                    metadata = None

                    context_sparse_raw = context_sparse
                    context_sparse = context_sparse[:, :num_sparse]
            else:
                assert k is not None
                assert v is not None
                context_dense = query_sparse_attention(
                    query_for_recomp.permute(0, 2, 1, 3).contiguous(),
                    k.permute(0, 2, 1, 3).contiguous(),
                    v.permute(0, 2, 1, 3).contiguous(),
                    args.position_ids[:, idx],
                    sm_scale,
                    None,
                    None,
                    None,
                    return_running_statistics=delta_attention_args_adjust_norm_const,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    extend_backend=delta_attention_args_extend,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    model_context_length=args.model_context_length,
                    self_extend_scale=args.self_extend_scale,
                    softmax_sink=args.softmax_sink,
                )

        if delta_attention_args_adjust_norm_const:
            context_dense, (dense_mx, dense_nc) = context_dense
        else:
            dense_mx = dense_nc = None

        if dense_mx is not None:
            dense_mx = dense_mx.permute(0, 2, 1)
            dense_nc = dense_nc.permute(0, 2, 1)
        context_dense = context_dense.permute(0, 2, 1, 3).contiguous()

        return (
            context_dense, context_dense_non_pooled, dense_mx, dense_nc,
            context_sparse, context_sparse_raw, metadata,
        )

    def _apply_delta(
        self,
        context_sparse: torch.Tensor,
        context_sparse_raw: torch.Tensor,
        context_dense: torch.Tensor,
        context_dense_non_pooled,
        idx: torch.Tensor,
        num_sparse: int,
        num_last_dense: int,
        query: torch.Tensor,
        delta_pool_q: bool,
    ):
        """Stage 4: Apply delta correction.

        Original lines 1964-2033.
        """
        config = self.config
        delta_attention_args_diff = config.diff_mode
        delta_attention_args_w = config.gamma
        delta_attention_args_smooth = config.smooth
        delta_attention_args_extend = config.extend_mode
        num_queries = query.shape[1]

        if delta_attention_args_diff == 0:
            context = torch.zeros_like(query)
            context[:, :num_sparse] = context_sparse
            context[:, idx] = context_dense
        elif delta_attention_args_diff == 2:
            context = torch.zeros_like(query)
            context[:, idx] = context_dense
            context[:, :num_sparse] = context_sparse
        else:
            from hip_attn.v1_2.delta.apply_delta import apply_delta

            if delta_pool_q:
                context_dense = context_dense_non_pooled.permute(0, 2, 1, 3)

            context = apply_delta(
                context_dense,
                context_sparse,
                idx,
                num_last_dense,
                delta_attention_args_w,
                delta_attention_args_smooth,
            )

            if delta_attention_args_extend == "nope":
                context[:, idx] = context_sparse_raw[:, idx]

        return context
