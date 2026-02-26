"""Tests for fused top-K selection + position sort.

Validates correctness against the original 4-sort PyTorch path from
``delta_pipeline.py`` and benchmarks the speedup.
"""

import os
import sys
import time
import types

import pytest
import torch

# Ensure the src directory is on the path so we can import hip_attn.
_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src")
_SRC_DIR = os.path.abspath(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def _ensure_parent_modules() -> None:
    """Stub out parent packages when GPU deps (nvtx, cupy, etc.) are missing.

    The top-level ``hip_attn/__init__.py`` imports heavy GPU-only modules.
    We only need ``hip_attn.v1_2.topk``, so we create minimal namespace
    stubs for the package chain so that subpackage resolution works.
    """
    try:
        import nvtx  # noqa: F401
        return  # GPU deps available, no stubbing needed
    except ImportError:
        pass

    pkg_root = os.path.join(_SRC_DIR, "hip_attn")
    v12_root = os.path.join(pkg_root, "v1_2")
    topk_root = os.path.join(v12_root, "topk")

    stubs = {
        "hip_attn": [pkg_root],
        "hip_attn.v1_2": [v12_root],
        "hip_attn.v1_2.topk": [topk_root],
    }
    for mod_name, path in stubs.items():
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            stub.__path__ = path
            stub.__package__ = mod_name
            sys.modules[mod_name] = stub


_ensure_parent_modules()

from hip_attn.v1_2.topk.fused_topk_sort import (  # noqa: E402
    fused_topk_sort,
    fused_topk_sort_with_dedup,
)


# ---------------------------------------------------------------------------
# Reference implementations (original 4-sort path from delta_pipeline.py)
# ---------------------------------------------------------------------------

def _reference_topk_sort(
    block_scores: torch.Tensor,
    indices: torch.Tensor,
    k: int,
):
    """Original 4-sort path: sort-descending -> gather -> sort-ascending."""
    n = block_scores.shape[-1]
    k = min(k, n)

    # Sort 1: full sort by score descending
    t_indices = torch.sort(block_scores, dim=-1, descending=True).indices
    top_idx = t_indices[..., :k]

    # Gather positions and scores
    sel_indices = indices.gather(dim=-1, index=top_idx)
    sel_scores = block_scores.gather(dim=-1, index=top_idx)

    # Sort 2: sort selected by position ascending
    sort_order = sel_indices.argsort(dim=-1)
    sel_indices = sel_indices.gather(dim=-1, index=sort_order)
    sel_scores = sel_scores.gather(dim=-1, index=sort_order)

    return sel_indices, sel_scores


def _reference_topk_sort_with_dedup(
    block_scores: torch.Tensor,
    indices: torch.Tensor,
    k: int,
):
    """Reference dedup path from delta_pipeline.py lines 670-751."""
    n = block_scores.shape[-1]
    k = min(k, n)

    # Sort by position
    pos_order = indices.argsort(dim=-1)
    sorted_indices = indices.gather(dim=-1, index=pos_order)
    sorted_scores = block_scores.gather(dim=-1, index=pos_order)

    unique_mask = torch.roll(sorted_indices, shifts=1, dims=-1) != sorted_indices
    unique_mask[..., 0] = True
    block_scores_cumsum = torch.exp(
        sorted_scores - sorted_scores.amax(-1, keepdim=True)
    ).cumsum(-1)
    block_scores_cumsum_base = (block_scores_cumsum * unique_mask).cummax(-1).values
    block_scores_cumsum = block_scores_cumsum - block_scores_cumsum_base + sorted_scores

    unique_mask_last = torch.roll(sorted_indices, shifts=-1, dims=-1) != sorted_indices
    unique_mask_last[..., -1] = True
    block_scores_cumsum = torch.where(
        unique_mask_last,
        block_scores_cumsum,
        torch.finfo(block_scores_cumsum.dtype).min,
    )

    arange = torch.arange(n, device=indices.device)
    counter_start = (arange[None, None, :] * unique_mask).cummax(dim=-1).values
    counter_end = (arange[None, None, :] * unique_mask_last).cummax(dim=-1).values
    counter = (counter_end - counter_start + 1) * unique_mask_last
    block_scores_cumsum = torch.where(
        unique_mask_last,
        block_scores_cumsum / counter,
        torch.finfo(block_scores_cumsum.dtype).min,
    )
    sorted_indices = torch.where(
        unique_mask_last, sorted_indices, torch.iinfo(sorted_indices.dtype).max,
    )

    t_sort = block_scores_cumsum.argsort(dim=-1, descending=True)
    sel_indices = sorted_indices.gather(index=t_sort[..., :k], dim=-1)
    sel_scores = block_scores_cumsum.gather(index=t_sort[..., :k], dim=-1)

    sort_order = sel_indices.argsort(dim=-1)
    sel_indices = sel_indices.gather(dim=-1, index=sort_order)
    sel_scores = sel_scores.gather(dim=-1, index=sort_order)

    return sel_indices, sel_scores


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEVICE = "cpu"


@pytest.fixture
def random_data():
    """BH=4, Q=8, N_BLOCKS=512 random scores and unique indices."""
    torch.manual_seed(42)
    BH, Q, N = 4, 8, 512
    scores = torch.randn(BH, Q, N, device=DEVICE)
    indices = torch.randint(0, 10000, (BH, Q, N), device=DEVICE, dtype=torch.int64)
    return scores, indices


@pytest.fixture
def data_with_duplicates():
    """Scores and indices where block positions repeat (simulates union)."""
    torch.manual_seed(123)
    BH, Q, N = 2, 4, 256
    scores = torch.randn(BH, Q, N, device=DEVICE)
    # Use a small range of positions to force many duplicates.
    indices = torch.randint(0, 64, (BH, Q, N), device=DEVICE, dtype=torch.int64)
    return scores, indices


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicTopkSort:
    def test_output_shapes(self, random_data):
        scores, indices = random_data
        K = 128
        sel_idx, sel_sc = fused_topk_sort(scores, indices, K)
        assert sel_idx.shape == (*scores.shape[:-1], K)
        assert sel_sc.shape == (*scores.shape[:-1], K)

    def test_sorted_by_position(self, random_data):
        scores, indices = random_data
        K = 64
        sel_idx, _ = fused_topk_sort(scores, indices, K)
        # Verify ascending order along last dim.
        diffs = sel_idx[..., 1:] - sel_idx[..., :-1]
        assert (diffs >= 0).all(), "Output must be sorted ascending by position"

    def test_top_k_correct(self, random_data):
        scores, indices = random_data
        K = 32
        sel_idx, sel_sc = fused_topk_sort(scores, indices, K)
        # The selected scores must be the K largest (as a set).
        ref_topk_vals, _ = torch.topk(scores, K, dim=-1, sorted=True)
        # Sort both sets to compare.
        assert torch.allclose(
            sel_sc.sort(dim=-1, descending=True).values,
            ref_topk_vals,
        )


class TestMatchesTorchSort:
    def test_matches_reference_unique_indices(self):
        """Bit-exact match vs reference when all block positions are unique.

        When indices are globally unique (no duplicate positions), topk and
        full-sort must agree on both indices and scores.
        """
        torch.manual_seed(42)
        BH, Q, N = 4, 8, 512
        scores = torch.randn(BH, Q, N, device=DEVICE)
        # Unique indices per (BH, Q) row: no duplicates, so no tie-breaking ambiguity.
        indices = torch.stack([
            torch.stack([
                torch.randperm(N * 20, device=DEVICE)[:N]
                for _ in range(Q)
            ])
            for _ in range(BH)
        ])
        K = 128
        ref_idx, ref_sc = _reference_topk_sort(scores, indices, K)
        our_idx, our_sc = fused_topk_sort(scores, indices, K)
        assert torch.equal(our_idx, ref_idx)
        assert torch.allclose(our_sc, ref_sc)

    def test_matches_reference_small_k(self):
        """Small K with unique indices."""
        torch.manual_seed(99)
        BH, Q, N = 4, 8, 512
        scores = torch.randn(BH, Q, N, device=DEVICE)
        indices = torch.stack([
            torch.stack([
                torch.randperm(N * 20, device=DEVICE)[:N]
                for _ in range(Q)
            ])
            for _ in range(BH)
        ])
        K = 8
        ref_idx, ref_sc = _reference_topk_sort(scores, indices, K)
        our_idx, our_sc = fused_topk_sort(scores, indices, K)
        assert torch.equal(our_idx, ref_idx)
        assert torch.allclose(our_sc, ref_sc)

    def test_semantic_equivalence_with_duplicates(self, random_data):
        """When positions repeat, the *set* of (position, score) selected must
        contain the same top-K scores, even if tie-breaking differs."""
        scores, indices = random_data
        K = 128
        ref_idx, ref_sc = _reference_topk_sort(scores, indices, K)
        our_idx, our_sc = fused_topk_sort(scores, indices, K)
        # Positions must match (sorted ascending is deterministic).
        assert torch.equal(our_idx, ref_idx)
        # Score multisets must match -- sort both and compare.
        assert torch.allclose(
            our_sc.sort(dim=-1).values,
            ref_sc.sort(dim=-1).values,
        )


class TestKLargerThanN:
    def test_k_equals_n(self, random_data):
        scores, indices = random_data
        N = scores.shape[-1]
        sel_idx, sel_sc = fused_topk_sort(scores, indices, N)
        assert sel_idx.shape[-1] == N

    def test_k_exceeds_n(self, random_data):
        scores, indices = random_data
        N = scores.shape[-1]
        sel_idx, sel_sc = fused_topk_sort(scores, indices, N + 100)
        # Should clamp to N.
        assert sel_idx.shape[-1] == N

    def test_k_one(self, random_data):
        scores, indices = random_data
        sel_idx, sel_sc = fused_topk_sort(scores, indices, 1)
        assert sel_idx.shape[-1] == 1
        # The single selected score must be the global max.
        expected_max = scores.amax(dim=-1, keepdim=True)
        assert torch.allclose(sel_sc, expected_max)


class TestWithDedup:
    def test_matches_reference_dedup(self, data_with_duplicates):
        """fused_topk_sort_with_dedup must match the original cumsum logic."""
        scores, indices = data_with_duplicates
        K = 32
        ref_idx, ref_sc = _reference_topk_sort_with_dedup(scores, indices, K)
        our_idx, our_sc = fused_topk_sort_with_dedup(scores, indices, K)
        assert torch.equal(our_idx, ref_idx)
        assert torch.allclose(our_sc, ref_sc, atol=1e-5)

    def test_dedup_reduces_duplicates(self, data_with_duplicates):
        scores, indices = data_with_duplicates
        K = 32
        sel_idx, _ = fused_topk_sort_with_dedup(scores, indices, K)
        # Positions in the output should be unique (except for INT_MAX sentinels).
        for bh in range(sel_idx.shape[0]):
            for q in range(sel_idx.shape[1]):
                row = sel_idx[bh, q]
                valid = row[row < torch.iinfo(row.dtype).max]
                assert valid.unique().shape[0] == valid.shape[0], \
                    "Dedup output must not contain duplicate positions"

    def test_dedup_sorted_ascending(self, data_with_duplicates):
        scores, indices = data_with_duplicates
        K = 16
        sel_idx, _ = fused_topk_sort_with_dedup(scores, indices, K)
        diffs = sel_idx[..., 1:] - sel_idx[..., :-1]
        assert (diffs >= 0).all(), "Dedup output must be sorted ascending"


class TestPerformance:
    """Benchmark fused_topk_sort vs the original 4-sort reference.

    These tests always pass -- they print timing info for manual inspection.
    """

    @pytest.mark.parametrize("N,K", [(4096, 128), (8192, 256), (40960, 512)])
    def test_speedup(self, N, K):
        torch.manual_seed(0)
        BH, Q = 32, 1
        scores = torch.randn(BH, Q, N, device=DEVICE)
        indices = torch.randint(0, N * 10, (BH, Q, N), device=DEVICE, dtype=torch.int64)

        # Warmup
        for _ in range(3):
            _reference_topk_sort(scores, indices, K)
            fused_topk_sort(scores, indices, K)

        ITERS = 20

        t0 = time.perf_counter()
        for _ in range(ITERS):
            _reference_topk_sort(scores, indices, K)
        ref_time = (time.perf_counter() - t0) / ITERS

        t0 = time.perf_counter()
        for _ in range(ITERS):
            fused_topk_sort(scores, indices, K)
        our_time = (time.perf_counter() - t0) / ITERS

        speedup = ref_time / our_time if our_time > 0 else float("inf")
        print(
            f"\n  N={N:>6}, K={K:>4} | "
            f"ref={ref_time*1000:.3f}ms  ours={our_time*1000:.3f}ms  "
            f"speedup={speedup:.2f}x"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
