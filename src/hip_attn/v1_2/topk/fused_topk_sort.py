"""Fused top-K selection + position sort for QSA mask post-trimming.

Replaces the 4-sort PyTorch path in ``delta_pipeline.py`` (lines 662-751)
with an optimized 2-op sequence:

    1. ``torch.topk`` -- partial select O(N) average, no full sort.
    2. ``argsort`` on only K elements -- O(K log K) instead of O(N log N).

This is the PyTorch-native fast path.  A future CUDA C++ kernel will fuse
the radix-select + bitonic-sort into a single launch.

Functions
---------
fused_topk_sort
    Select top-K blocks by score, return sorted by position.
fused_topk_sort_with_dedup
    Same as above but with cumulative-score deduplication of repeated
    block indices (mirrors the original cumsum scoring logic).
"""

from typing import Tuple

import torch


def fused_topk_sort(
    block_scores: torch.Tensor,  # [BH, Q, N_BLOCKS]
    indices: torch.Tensor,       # [BH, Q, N_BLOCKS] block positions
    k: int,                      # number of top blocks to keep
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select top-K blocks by score, return sorted by position.

    Replaces::

        sort(scores, descending) -> gather(top-k) -> sort(positions)

    with::

        topk(scores, k)  ->  sort_small(positions[top-k])

    Parameters
    ----------
    block_scores : Tensor[BH, Q, N_BLOCKS]
        Per-block attention scores (higher = more important).
    indices : Tensor[BH, Q, N_BLOCKS]
        Block position indices (integer, same shape as *block_scores*).
    k : int
        Number of top blocks to keep.

    Returns
    -------
    selected_indices : Tensor[BH, Q, K]
        Top-K block positions, sorted ascending by position.
    selected_scores : Tensor[BH, Q, K]
        Corresponding scores in position-sorted order.
    """
    n_blocks = block_scores.shape[-1]
    k_clamped = min(k, n_blocks)

    # Step 1: Partial select top-K by score -- O(N) average via nth_element
    # internally, unlike full O(N log N) sort.  ``sorted=False`` avoids the
    # extra O(K log K) sort inside topk that we don't need yet.
    top_scores, top_idx = torch.topk(
        block_scores, k_clamped, dim=-1, sorted=False,
    )

    # Step 2: Gather the position indices for the selected blocks.
    selected_indices = indices.gather(dim=-1, index=top_idx)

    # Step 3: Sort ONLY the K selected elements by position (K << N).
    sort_order = selected_indices.argsort(dim=-1)
    selected_indices = selected_indices.gather(dim=-1, index=sort_order)
    selected_scores = top_scores.gather(dim=-1, index=sort_order)

    return selected_indices, selected_scores


def fused_topk_sort_with_dedup(
    block_scores: torch.Tensor,  # [BH, Q, N_BLOCKS]
    indices: torch.Tensor,       # [BH, Q, N_BLOCKS] block positions
    k: int,                      # number of top blocks to keep
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-K with cumulative-score deduplication of duplicate block indices.

    Mirrors the original logic in ``delta_pipeline.py`` lines 670-751:

    1. Sort by position to group duplicates.
    2. Compute cumulative-score per unique group, averaging over group size.
    3. Keep only the last element of each group (the "representative").
    4. Select top-K among unique representatives by cumulative score.
    5. Final sort by position.

    This is semantically equivalent to the original 4-sort path but
    restructured for clarity and amenable to future CUDA fusion.

    Parameters
    ----------
    block_scores : Tensor[BH, Q, N_BLOCKS]
        Per-block attention scores.
    indices : Tensor[BH, Q, N_BLOCKS]
        Block position indices (may contain duplicates across the union
        of multiple query-group blocks).
    k : int
        Number of unique blocks to keep after deduplication.

    Returns
    -------
    selected_indices : Tensor[BH, Q, K]
        Top-K unique block positions, sorted ascending.
    selected_scores : Tensor[BH, Q, K]
        Corresponding cumulative scores.
    """
    n_blocks = block_scores.shape[-1]
    k_clamped = min(k, n_blocks)

    # -- Phase 1: Sort by position to group duplicates --
    pos_order = indices.argsort(dim=-1)
    sorted_indices = indices.gather(dim=-1, index=pos_order)
    sorted_scores = block_scores.gather(dim=-1, index=pos_order)

    # -- Phase 2: Cumulative score with duplicate grouping --
    # Identify group boundaries.
    unique_mask_first = torch.roll(sorted_indices, shifts=1, dims=-1) != sorted_indices
    unique_mask_first[..., 0] = True
    unique_mask_last = torch.roll(sorted_indices, shifts=-1, dims=-1) != sorted_indices
    unique_mask_last[..., -1] = True

    # Stabilised cumulative sum within each group.
    score_max = sorted_scores.amax(-1, keepdim=True)
    exp_scores = torch.exp(sorted_scores - score_max)
    cumsum = exp_scores.cumsum(-1)

    # Reset cumsum at group boundaries: subtract the cumsum value at the
    # start of each group so that each group accumulates independently.
    cumsum_base = (cumsum * unique_mask_first).cummax(-1).values
    cumsum_group = cumsum - cumsum_base + sorted_scores

    # Compute group sizes via index arithmetic.
    arange = torch.arange(n_blocks, device=indices.device)
    counter_start = (arange[None, None, :] * unique_mask_first).cummax(dim=-1).values
    counter_end = (arange[None, None, :] * unique_mask_last).cummax(dim=-1).values
    group_size = (counter_end - counter_start + 1) * unique_mask_last

    # Average cumulative score per group; only the last element of each
    # group is the representative.
    NEG_INF = torch.finfo(cumsum_group.dtype).min
    cumsum_avg = torch.where(
        unique_mask_last,
        cumsum_group / group_size.clamp(min=1),
        NEG_INF,
    )

    # Sentinel non-representative positions so they sort to the end.
    INT_MAX = torch.iinfo(sorted_indices.dtype).max
    representative_indices = torch.where(unique_mask_last, sorted_indices, INT_MAX)

    # -- Phase 3: Top-K among unique representatives --
    # Use topk on the cumulative-average scores.
    top_cum_scores, top_idx = torch.topk(
        cumsum_avg, k_clamped, dim=-1, sorted=False,
    )
    selected_indices = representative_indices.gather(dim=-1, index=top_idx)
    selected_scores = top_cum_scores

    # -- Phase 4: Final sort by position --
    sort_order = selected_indices.argsort(dim=-1)
    selected_indices = selected_indices.gather(dim=-1, index=sort_order)
    selected_scores = selected_scores.gather(dim=-1, index=sort_order)

    return selected_indices, selected_scores
