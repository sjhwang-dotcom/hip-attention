"""Partial FA3 pipeline: splits prefill into dense FA3 + sparse HiP regions."""

import os
from typing import Optional

import torch

from hip_attn.v1_2.attention_metadata import (
    HiPAttentionArgs,
    HiPAttentionOutputMetadata,
)
from hip_attn.v1_2.utils import capture


@capture
def _forward_partial_fa3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    rope_is_neox_style: bool,
    cached_metadata: HiPAttentionOutputMetadata,
    is_decode: bool,
    seq_thresh_fa3: int,
    mixing_len: int,
    args: HiPAttentionArgs,
    max_context_len: int,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    inner_function_do_scale: bool,
    inner_function,
):
    query = q

    context_fa3 = None
    metadata = None

    if (not is_decode) and (seq_thresh_fa3 > 0):
        if args.using_paged_cache:
            pass
        else:
            assert k is not None
            max_context_len = min(max_context_len, k.shape[1])
        min_context_len = max(0, max_context_len - query.shape[1])

        len_query_for_fa3 = max(
            0, min(seq_thresh_fa3, max_context_len) - min_context_len
        )
        len_query_for_hip = max(
            0, max_context_len - max(min_context_len, seq_thresh_fa3 - mixing_len)
        )

        if len_query_for_fa3 > 0:
            assert not is_decode

            if args.using_paged_cache:
                k = args.gather_k_from_paged_cache(
                    seq_len=min(max_context_len, args.model_context_length)
                )
                v = args.gather_v_from_paged_cache(
                    seq_len=min(max_context_len, args.model_context_length)
                )

            query_fa3 = query[:, :len_query_for_fa3].contiguous()
            len_kv = k.shape[1] - (
                len_query_for_hip - (query.shape[1] - len_query_for_fa3)
            )  # BUG: this should be bug, because this will lose keys for len_for_mix
            k_fa3 = k[:, :len_kv].contiguous()
            v_fa3 = v[:, :len_kv].contiguous()

            is_fp8 = k.dtype in (torch.float8_e5m2,)
            if is_fp8:
                query_fa3 = query_fa3.to(torch.float16)
                k_fa3 = k_fa3.to(torch.float16)
                v_fa3 = v_fa3.to(torch.float16)

            if k.dtype == torch.float8_e4m3fn:
                query_fa3 = query_fa3.to(k.dtype)

            # Import _forward_fa3 from paged_hip (stays there as it's used elsewhere too)
            from hip_attn.v1_2.paged_hip import _forward_fa3

            context_fa3 = _forward_fa3(
                q=query_fa3,
                k=k_fa3,
                v=v_fa3,
                sm_scale=sm_scale,
                position_ids=args.position_ids[:, :len_query_for_fa3],
                using_extend=args.using_extend,
                need_apply_rope=args.need_apply_rope,
                rope_cos=args.rope_cos,
                rope_sin=args.rope_sin,
                rope_is_neox_style=rope_is_neox_style,
                k_descale=k_descale,
                v_descale=v_descale,
            )

    if args.using_paged_cache:
        k = v = None

    if context_fa3 is not None:
        if len_query_for_hip > 0:
            args_sparse = args.clone()
            args_sparse.position_ids = args_sparse.position_ids[:, -len_query_for_hip:]
            if args_sparse.q_mask is not None:
                args_sparse.q_mask = args_sparse.q_mask[:, -len_query_for_hip:]
            if args_sparse.query_for_landmark is not None:
                args_sparse.query_for_landmark = args_sparse.query_for_landmark[
                    :, -len_query_for_hip:
                ]

            yarn_scale = float(os.getenv("HIP_DEBUG_YARN_SCALE_HINT", "1"))
            if yarn_scale > 1:
                assert int(yarn_scale) == yarn_scale
                yarn_scale = int(yarn_scale)
                args_sparse.rope_cos = args_sparse.rope_cos[::yarn_scale]
                args_sparse.rope_sin = args_sparse.rope_sin[::yarn_scale]

            context_sparse, metadata = inner_function(
                q=(
                    query[:, -len_query_for_hip:]
                    * (sm_scale if inner_function_do_scale else 1)
                ).to(query.dtype),
                k=k,
                v=v,
                args=args_sparse,
                cached_metadata=cached_metadata,
            )
            if context_sparse.ndim == 3:
                context_sparse = context_sparse.unsqueeze(0)
                assert context_fa3.shape[0] == 1

            len_for_mix = (len_query_for_hip + len_query_for_fa3) - query.shape[1]

            if len_for_mix > 0:
                context_fa3_mix = context_fa3[:, -len_for_mix:]
                context_sparse_mix = context_sparse[:, :len_for_mix]

                chunk_len = min(context_sparse_mix.shape[1], len_for_mix)
                offset = min_context_len - (seq_thresh_fa3 - mixing_len)
                scale_global = (
                    torch.arange(
                        offset,
                        offset + chunk_len,
                        device=query.device,
                        dtype=torch.float32,
                    )
                    / mixing_len
                )

                len_for_spike = min(chunk_len, 32)
                scale = torch.clamp_min(
                    (
                        torch.arange(
                            0, chunk_len, device=query.device, dtype=torch.float32
                        )
                        - (chunk_len - len_for_spike)
                    )
                    / len_for_spike,
                    0,
                )

                scale = torch.maximum(scale, scale_global)

                scale = scale[None, :, None, None]
                context_mix = (
                    context_sparse_mix * scale + context_fa3_mix * (1.0 - scale)
                ).to(context_fa3_mix.dtype)

                context = torch.cat(
                    [
                        context_fa3[:, :-len_for_mix],
                        context_mix,
                        context_sparse[:, len_for_mix:],
                    ],
                    dim=1,
                )
            else:
                context = torch.cat([context_fa3, context_sparse], dim=1)
        else:
            context = context_fa3
    else:
        # no fa3
        context, metadata = inner_function(
            q=(query * (sm_scale if inner_function_do_scale else 1)).to(query.dtype),
            k=k,
            v=v,
            args=args,
            cached_metadata=cached_metadata,
        )

    context = context.to(query.dtype)

    return context, metadata
