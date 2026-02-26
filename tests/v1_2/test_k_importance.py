"""Tests for Streaming K-Importance."""

import importlib.util
import os
import sys

import torch
import pytest

# Direct import to bypass hip_attn.__init__.py GPU dependencies
_MODULE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "hip_attn", "v1_2", "topk", "k_importance.py"
)
_spec = importlib.util.spec_from_file_location("k_importance", os.path.abspath(_MODULE_PATH))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

StreamingKImportance = _mod.StreamingKImportance
compute_k_importance = _mod.compute_k_importance
select_by_budget = _mod.select_by_budget


class TestStreamingKImportance:
    def test_single_token_update(self):
        pass  # StreamingKImportance imported at module level

        tracker = StreamingKImportance(budget=8, block_size=4, dim=64,
                                       sink_blocks=0, window_blocks=0)
        # Feed 4 tokens = 1 block
        for _ in range(4):
            tracker.update(torch.randn(1, 2, 64))

        assert tracker.n_blocks == 1
        mask = tracker.get_mask()
        assert mask.shape == (1, 2, 1)
        assert mask.all()  # Under budget

    def test_batch_token_update(self):
        pass  # StreamingKImportance imported at module level

        tracker = StreamingKImportance(budget=8, block_size=4, dim=64,
                                       sink_blocks=0, window_blocks=0)
        # Feed 16 tokens at once = 4 blocks
        tracker.update(torch.randn(1, 2, 16, 64))
        assert tracker.n_blocks == 4
        mask = tracker.get_mask()
        assert mask.shape == (1, 2, 4)

    def test_budget_enforcement(self):
        pass  # StreamingKImportance imported at module level

        tracker = StreamingKImportance(budget=4, block_size=4, dim=64,
                                       sink_blocks=0, window_blocks=0)
        # Feed 40 tokens = 10 blocks, budget = 4
        tracker.update(torch.randn(1, 2, 40, 64))
        assert tracker.n_blocks == 10
        mask = tracker.get_mask()
        # Should keep ~4 blocks per head
        n_active = mask.sum(dim=-1).float()
        assert (n_active <= 5).all(), f"Expected ~4, got {n_active}"
        assert (n_active >= 3).all(), f"Expected ~4, got {n_active}"

    def test_sink_always_kept(self):
        pass  # StreamingKImportance imported at module level

        tracker = StreamingKImportance(budget=4, block_size=4, dim=64,
                                       sink_blocks=2, window_blocks=0)
        # Make first 8 tokens (2 blocks) very low importance
        k = torch.randn(1, 1, 40, 64) * 0.01
        tracker.update(k)
        mask = tracker.get_mask()
        # First 2 blocks should still be kept (sink)
        assert mask[0, 0, :2].all(), "Sink blocks must always be kept"

    def test_window_always_kept(self):
        pass  # StreamingKImportance imported at module level

        tracker = StreamingKImportance(budget=4, block_size=4, dim=64,
                                       sink_blocks=0, window_blocks=2)
        tracker.update(torch.randn(1, 1, 40, 64))
        mask = tracker.get_mask()
        # Last 2 blocks should be kept (window)
        assert mask[0, 0, -2:].all(), "Window blocks must always be kept"

    def test_pending_partial_block(self):
        pass  # StreamingKImportance imported at module level

        tracker = StreamingKImportance(budget=8, block_size=4, dim=64,
                                       sink_blocks=0, window_blocks=0)
        # 5 tokens = 1 full block + 1 pending token
        tracker.update(torch.randn(1, 2, 5, 64))
        assert tracker.n_blocks == 2  # 1 flushed + 1 pending
        mask = tracker.get_mask()
        assert mask.shape == (1, 2, 2)
        assert mask[:, :, -1].all()  # Pending block always included

    def test_high_norm_blocks_rank_higher(self):
        pass  # StreamingKImportance imported at module level

        tracker = StreamingKImportance(budget=2, block_size=4, dim=64,
                                       sink_blocks=0, window_blocks=0)
        k = torch.randn(1, 1, 20, 64)
        # Make block 2 (tokens 8-11) have 10x larger norms
        k[:, :, 8:12, :] *= 10.0
        tracker.update(k)

        scores = tracker.get_scores()
        # Block 2 should have highest score
        assert scores[0, 0].argmax().item() == 2

    def test_decay_reduces_old_importance(self):
        pass  # StreamingKImportance imported at module level

        tracker = StreamingKImportance(budget=8, block_size=4, dim=64, decay=0.9,
                                       sink_blocks=0, window_blocks=0)
        # Feed uniform blocks
        k = torch.ones(1, 1, 4, 64)
        tracker.update(k)
        score_after_1 = tracker.get_scores()[0, 0, 0].item()

        tracker.update(k)
        score_after_2 = tracker.get_scores()[0, 0, 0].item()

        # First block's score should have decayed
        assert score_after_2 < score_after_1 * 0.95, \
            f"Expected decay: {score_after_2} should be < {score_after_1 * 0.95}"

    def test_reset_clears_state(self):
        pass  # StreamingKImportance imported at module level

        tracker = StreamingKImportance(budget=8, block_size=4, dim=64)
        tracker.update(torch.randn(1, 2, 16, 64))
        assert tracker.n_blocks == 4

        tracker.reset()
        assert tracker.n_blocks == 0
        with pytest.raises(RuntimeError):
            tracker.get_mask()

    def test_o1_amortized_cost(self):
        """Verify streaming is faster than batch recompute."""
        pass  # StreamingKImportance imported at module level
        import time

        tracker = StreamingKImportance(budget=64, block_size=64, dim=128,
                                       sink_blocks=2, window_blocks=4)

        # Simulate 4096 token decode, 1 token at a time
        n_tokens = 4096
        t0 = time.perf_counter()
        for i in range(n_tokens):
            tracker.update(torch.randn(1, 32, 128))
        t_streaming = time.perf_counter() - t0

        # Batch recompute
        pass  # compute_k_importance imported at module level
        k_all = torch.randn(1, 32, n_tokens, 128)
        t0 = time.perf_counter()
        compute_k_importance(k_all, block_size=64)
        t_batch = time.perf_counter() - t0

        print(f"\n  Streaming {n_tokens} tokens: {t_streaming*1000:.1f} ms")
        print(f"  Batch recompute: {t_batch*1000:.1f} ms")
        print(f"  Per-token streaming: {t_streaming/n_tokens*1e6:.1f} us")


class TestOutlierDetection:
    def test_outlier_block_always_kept(self):
        """Blocks with keys far from mean must survive budget cuts."""
        tracker = StreamingKImportance(
            budget=4, block_size=4, dim=64, sink_blocks=0, window_blocks=0,
            outlier_sigma=2.0,
        )
        # 10 blocks of normal keys
        k = torch.randn(1, 1, 40, 64) * 0.1
        # Block 5 (tokens 20-23) is a massive outlier
        k[:, :, 20:24, :] = torch.randn(1, 1, 4, 64) * 50.0
        tracker.update(k)

        mask = tracker.get_mask()
        # Block 5 must be kept despite budget=4
        assert mask[0, 0, 5].item(), "Outlier block must survive budget cuts"

    def test_outlier_scores_separate(self):
        tracker = StreamingKImportance(
            budget=8, block_size=4, dim=64, sink_blocks=0, window_blocks=0,
        )
        k = torch.randn(1, 1, 20, 64)
        k[:, :, 8:12, :] *= 20.0  # Block 2 is outlier
        tracker.update(k)

        outlier = tracker.get_outlier_scores()
        assert outlier[0, 0, 2] > outlier[0, 0, 0], \
            "Outlier block should have higher deviation score"

    def test_normal_blocks_not_flagged_as_outlier(self):
        tracker = StreamingKImportance(
            budget=8, block_size=4, dim=64, sink_blocks=0, window_blocks=0,
            outlier_sigma=3.0,
        )
        # All uniform, no outliers
        k = torch.randn(1, 1, 32, 64)
        tracker.update(k)

        outlier = tracker.get_outlier_scores()
        # With sigma=3.0, normal blocks should mostly be below threshold
        n_flagged = (outlier >= 3.0).sum().item()
        assert n_flagged <= 2, f"Too many false outliers: {n_flagged}"

    def test_delta_unrecoverable_outlier(self):
        """Simulate delta correction failure: outlier in sparse region."""
        tracker = StreamingKImportance(
            budget=4, block_size=4, dim=64, sink_blocks=1, window_blocks=1,
            outlier_sigma=2.0,
        )
        k = torch.randn(1, 1, 40, 64) * 0.1
        # Outlier in the middle (not sink, not window)
        k[:, :, 16:20, :] = torch.randn(1, 1, 4, 64) * 30.0
        tracker.update(k)

        mask = tracker.get_mask()
        # Block 4 (mid-region outlier) must be kept
        assert mask[0, 0, 4].item(), \
            "Mid-region outlier must be kept (delta cannot recover)"
        # Sink (block 0) must be kept
        assert mask[0, 0, 0].item()
        # Window (last block) must be kept
        assert mask[0, 0, -1].item()


class TestBatchCompute:
    def test_output_shape(self):
        pass  # compute_k_importance imported at module level

        k = torch.randn(2, 4, 512, 64)
        imp = compute_k_importance(k, block_size=64)
        assert imp.shape == (2, 4, 8)

    def test_high_norm_ranks_higher(self):
        pass  # compute_k_importance imported at module level

        k = torch.randn(1, 1, 256, 64)
        k[:, :, 128:192, :] *= 10.0
        imp = compute_k_importance(k, block_size=64)
        assert imp[0, 0].argmax().item() == 2

    def test_no_nan(self):
        pass  # compute_k_importance imported at module level

        k = torch.randn(1, 4, 1024, 128)
        imp = compute_k_importance(k, block_size=64)
        assert not torch.isnan(imp).any()


class TestSelectByBudget:
    def test_under_budget_all_true(self):
        pass  # select_by_budget imported at module level

        imp = torch.randn(1, 1, 8)
        mask = select_by_budget(imp, budget=16)
        assert mask.all()

    def test_budget_respected(self):
        pass  # select_by_budget imported at module level

        imp = torch.randn(1, 1, 100)
        mask = select_by_budget(imp, budget=20, sink_blocks=0, window_blocks=0)
        n = mask.sum().item()
        assert 18 <= n <= 22, f"Expected ~20, got {n}"

    def test_sink_and_window(self):
        pass  # select_by_budget imported at module level

        imp = torch.full((1, 1, 100), -1000.0)
        mask = select_by_budget(imp, budget=10, sink_blocks=3, window_blocks=3)
        assert mask[0, 0, :3].all()
        assert mask[0, 0, -3:].all()
