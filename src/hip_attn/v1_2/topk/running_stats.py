"""Running block-level attention statistics for adaptive routing.

During inference, actual attention scores are computed anyway (free).
By accumulating these as EMA statistics, the CovarianceRouter can
adaptively improve its predictions over time.

Flow:
    1. Router predicts block importance (covariance-based)
    2. FlashAttention computes actual attention (returns softmax_lse)
    3. RunningBlockStats.update() incorporates measured importance
    4. Next decode step: router uses covariance + empirical boost
"""

import torch
from dataclasses import dataclass


@dataclass
class RunningStatsConfig:
    """Configuration for running attention statistics."""
    ema_alpha: float = 0.1        # EMA decay: higher = faster adaptation
    boost_weight: float = 0.5     # Weight of empirical boost in router score
    min_observations: int = 4     # Min hits before boost is applied
    normalize: bool = True        # Normalize importance to [0, 1]


class RunningBlockStats:
    """Per-layer, per-head EMA of block attention importance.

    Tracks which KV blocks actually receive attention mass during
    inference, enabling adaptive routing that improves over time.

    Usage:
        stats = RunningBlockStats(max_blocks=1024, num_heads=8)

        # After each decode step:
        stats.update(selected_indices, attention_lse)

        # Before next routing:
        boost = stats.get_boost(n_blocks=500)
        mask = router.route(q, empirical_boost=boost)
    """

    def __init__(
        self,
        max_blocks: int,
        num_heads: int,
        config: RunningStatsConfig = None,
    ):
        self.max_blocks = max_blocks
        self.num_heads = num_heads
        self.config = config or RunningStatsConfig()

        # EMA of attention importance per block
        # Higher = this block consistently receives attention mass
        self.importance = torch.zeros(num_heads, max_blocks, dtype=torch.float32)

        # How many times each block has been observed
        self.hit_count = torch.zeros(num_heads, max_blocks, dtype=torch.int32)

        # Running max for normalization
        self._running_max = torch.ones(num_heads, 1, dtype=torch.float32)

    def update(
        self,
        selected_block_indices: torch.Tensor,
        attention_lse: torch.Tensor,
        block_size: int,
    ):
        """Update importance from FlashAttention softmax_lse.

        Args:
            selected_block_indices: [n_selected] int64 — which blocks were attended
            attention_lse: [B, H_q, T_q] float32 — log-sum-exp from FlashAttention
                For decode, T_q=1, so shape is [B, H_q, 1] or [B, H_q]
            block_size: tokens per block (for mapping lse → block scores)
        """
        alpha = self.config.ema_alpha
        device = self.importance.device
        n_selected = selected_block_indices.shape[0]

        if n_selected == 0:
            return

        # Compute per-block attention mass from lse
        # lse tells us total attention energy; blocks with more tokens
        # in the softmax denominator contributed more
        # Simple proxy: uniform credit across selected blocks,
        # weighted by 1/n_selected (more selective = higher per-block credit)
        credit = 1.0 / max(n_selected, 1)

        # Clamp indices to valid range
        valid_mask = selected_block_indices < self.max_blocks
        valid_indices = selected_block_indices[valid_mask]

        if valid_indices.numel() == 0:
            return

        # EMA update for all heads (broadcast)
        for h in range(self.num_heads):
            old_vals = self.importance[h, valid_indices]
            self.importance[h, valid_indices] = (
                (1 - alpha) * old_vals + alpha * credit
            )
            self.hit_count[h, valid_indices] += 1

        # Decay non-selected blocks slightly (they weren't needed)
        # This makes unused blocks less likely to be boosted
        decay = 1 - alpha * 0.1  # Gentle decay
        all_mask = torch.ones(self.max_blocks, dtype=torch.bool, device=device)
        all_mask[valid_indices] = False
        self.importance[:, all_mask] *= decay

    def update_from_scores(
        self,
        block_indices: torch.Tensor,
        block_scores: torch.Tensor,
    ):
        """Direct update from per-block attention scores (higher fidelity).

        Args:
            block_indices: [n_selected] int64 — which blocks
            block_scores: [H, n_selected] float32 — actual attention scores per block
        """
        alpha = self.config.ema_alpha
        n_selected = block_indices.shape[0]

        if n_selected == 0:
            return

        valid_mask = block_indices < self.max_blocks
        valid_indices = block_indices[valid_mask]

        if valid_indices.numel() == 0:
            return

        valid_scores = block_scores[:, valid_mask]

        # Normalize scores to [0, 1] range
        if valid_scores.max() > 0:
            valid_scores = valid_scores / valid_scores.max()

        old_vals = self.importance[:, valid_indices]
        self.importance[:, valid_indices] = (
            (1 - alpha) * old_vals + alpha * valid_scores
        )
        self.hit_count[:, valid_indices] += 1

    def get_boost(self, n_blocks: int) -> torch.Tensor:
        """Get empirical importance boost for router scoring.

        Args:
            n_blocks: number of blocks in current sequence

        Returns:
            [H, n_blocks] float32 — boost values to add to router scores.
            Zero for blocks with insufficient observations.
        """
        n = min(n_blocks, self.max_blocks)
        boost = self.importance[:, :n].clone()

        # Zero out blocks with too few observations
        min_obs = self.config.min_observations
        insufficient = self.hit_count[:, :n] < min_obs
        boost[insufficient] = 0.0

        # Normalize if configured
        if self.config.normalize and boost.max() > 0:
            self._running_max = torch.maximum(
                self._running_max,
                boost.max(dim=-1, keepdim=True).values,
            )
            boost = boost / self._running_max.clamp(min=1e-8)

        return boost * self.config.boost_weight

    def reset(self):
        """Reset all statistics (e.g., for new request)."""
        self.importance.zero_()
        self.hit_count.zero_()
        self._running_max.fill_(1.0)

    def stats_summary(self) -> dict:
        """Return summary statistics for monitoring."""
        observed = (self.hit_count > 0).float()
        return {
            "total_blocks_observed": observed.sum().item(),
            "mean_importance": self.importance[observed.bool()].mean().item()
            if observed.any() else 0.0,
            "max_importance": self.importance.max().item(),
            "coverage_pct": 100 * observed.mean().item(),
        }
