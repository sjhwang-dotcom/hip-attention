"""Tests for RunningBlockStats — adaptive routing via EMA feedback."""

import importlib.util
import os

import torch
import pytest

_MODULE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "hip_attn", "v1_2"
)

def _load_module(name, rel_path):
    path = os.path.abspath(os.path.join(_MODULE_DIR, rel_path))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_stats_mod = _load_module("running_stats", "topk/running_stats.py")
RunningBlockStats = _stats_mod.RunningBlockStats
RunningStatsConfig = _stats_mod.RunningStatsConfig


class TestRunningBlockStats:
    def test_init(self):
        stats = RunningBlockStats(max_blocks=100, num_heads=4)
        assert stats.importance.shape == (4, 100)
        assert stats.hit_count.shape == (4, 100)
        assert (stats.importance == 0).all()

    def test_update_increases_importance(self):
        stats = RunningBlockStats(max_blocks=100, num_heads=2,
                                  config=RunningStatsConfig(ema_alpha=0.5))
        # Hit block 5 repeatedly
        for _ in range(10):
            stats.update(torch.tensor([5]), attention_lse=None, block_size=4)

        # Block 5 should have high importance
        assert stats.importance[0, 5] > stats.importance[0, 0]
        assert stats.importance[1, 5] > stats.importance[1, 0]

    def test_hit_count_increments(self):
        stats = RunningBlockStats(max_blocks=50, num_heads=2)
        stats.update(torch.tensor([3, 7, 12]), attention_lse=None, block_size=4)
        stats.update(torch.tensor([3, 7]), attention_lse=None, block_size=4)
        stats.update(torch.tensor([3]), attention_lse=None, block_size=4)

        assert stats.hit_count[0, 3] == 3
        assert stats.hit_count[0, 7] == 2
        assert stats.hit_count[0, 12] == 1
        assert stats.hit_count[0, 0] == 0

    def test_non_selected_blocks_decay(self):
        stats = RunningBlockStats(max_blocks=50, num_heads=1,
                                  config=RunningStatsConfig(ema_alpha=0.5))
        # Set block 0 importance manually
        stats.importance[0, 0] = 1.0

        # Update only block 10 — block 0 should decay
        stats.update(torch.tensor([10]), attention_lse=None, block_size=4)

        assert stats.importance[0, 0] < 1.0, "Non-selected block should decay"
        assert stats.importance[0, 10] > 0, "Selected block should gain importance"

    def test_get_boost_zeros_below_min_observations(self):
        stats = RunningBlockStats(max_blocks=50, num_heads=2,
                                  config=RunningStatsConfig(min_observations=5))
        # Only 2 observations — below threshold
        stats.update(torch.tensor([3, 7]), attention_lse=None, block_size=4)
        stats.update(torch.tensor([3, 7]), attention_lse=None, block_size=4)

        boost = stats.get_boost(n_blocks=50)
        assert boost[0, 3] == 0, "Below min_observations should return zero boost"

    def test_get_boost_nonzero_above_min_observations(self):
        stats = RunningBlockStats(max_blocks=50, num_heads=2,
                                  config=RunningStatsConfig(min_observations=3, ema_alpha=0.3))
        for _ in range(5):
            stats.update(torch.tensor([3]), attention_lse=None, block_size=4)

        boost = stats.get_boost(n_blocks=50)
        assert boost[0, 3] > 0, "Above min_observations should return nonzero boost"

    def test_reset_clears_state(self):
        stats = RunningBlockStats(max_blocks=50, num_heads=2)
        stats.update(torch.tensor([1, 2, 3]), attention_lse=None, block_size=4)
        assert stats.hit_count.sum() > 0

        stats.reset()
        assert (stats.importance == 0).all()
        assert (stats.hit_count == 0).all()

    def test_out_of_range_indices_ignored(self):
        stats = RunningBlockStats(max_blocks=10, num_heads=1)
        # Index 15 is out of range (max_blocks=10)
        stats.update(torch.tensor([3, 15, 20]), attention_lse=None, block_size=4)
        assert stats.hit_count[0, 3] == 1
        # Should not crash, out-of-range indices silently ignored

    def test_stats_summary(self):
        stats = RunningBlockStats(max_blocks=50, num_heads=2)
        stats.update(torch.tensor([1, 2, 3, 4, 5]), attention_lse=None, block_size=4)

        summary = stats.stats_summary()
        assert "total_blocks_observed" in summary
        assert "coverage_pct" in summary
        assert summary["total_blocks_observed"] == 10  # 5 blocks × 2 heads

    def test_update_from_scores(self):
        stats = RunningBlockStats(max_blocks=50, num_heads=2,
                                  config=RunningStatsConfig(ema_alpha=0.5))
        block_indices = torch.tensor([2, 5, 8])
        # Head 0: block 2 gets high score, head 1: block 8 gets high score
        block_scores = torch.tensor([
            [0.9, 0.1, 0.0],  # head 0
            [0.0, 0.1, 0.9],  # head 1
        ])

        stats.update_from_scores(block_indices, block_scores)

        assert stats.importance[0, 2] > stats.importance[0, 8]
        assert stats.importance[1, 8] > stats.importance[1, 2]

    def test_boost_weight_scales_output(self):
        stats = RunningBlockStats(max_blocks=50, num_heads=1,
                                  config=RunningStatsConfig(
                                      boost_weight=2.0, min_observations=1,
                                      ema_alpha=0.5))
        for _ in range(3):
            stats.update(torch.tensor([5]), attention_lse=None, block_size=4)

        boost = stats.get_boost(n_blocks=50)
        # Boost should be scaled by boost_weight=2.0
        assert boost[0, 5] > 0

    def test_empty_indices_no_crash(self):
        stats = RunningBlockStats(max_blocks=50, num_heads=2)
        stats.update(torch.tensor([], dtype=torch.long), attention_lse=None,
                     block_size=4)
        assert (stats.hit_count == 0).all()
