"""Streaming K-Importance: Budget-based Sparse Attention via Key Pre-scoring.

Maintains a fixed-budget set of "important" KV blocks in streaming fashion.
As new tokens enter the KV cache, importance scores are updated incrementally
in O(1) per token. When query arrives, the pre-computed importance mask
eliminates the need for hierarchical scan.

Theory:
    importance(k_j) = E_q[ softmax(q^T K / sqrt(d))_j ]

    Approximation: maintain running proxy (mean key) and score each block
    against it. Blocks below adaptive threshold are excluded from attention.

    Streaming guarantee: amortized O(1) per new token via:
    - Running mean update: O(1)
    - New block scoring: O(block_size * dim) every block_size tokens
    - Budget enforcement: O(1) via threshold tracking
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


class StreamingKImportance:
    """Fixed-budget streaming importance tracker for KV cache blocks.

    Maintains a budget of B blocks. As new tokens arrive:
    1. Running mean of K is updated in O(1)
    2. Every block_size tokens, new block is scored against proxy
    3. If budget exceeded, lowest-importance block is evicted from mask

    No sorting. No full recomputation. Pure streaming.

    Args:
        budget: Maximum number of blocks to attend to.
        block_size: Tokens per block.
        sink_blocks: Initial blocks always kept (sink tokens).
        window_blocks: Final blocks always kept (recent context).
        dim: Head dimension (for sm_scale).
        decay: Exponential decay for old block importance (0 < decay <= 1).
            1.0 = no decay, 0.99 = mild recency bias.
    """

    def __init__(
        self,
        budget: int = 128,
        block_size: int = 64,
        sink_blocks: int = 2,
        window_blocks: int = 4,
        dim: int = 128,
        decay: float = 1.0,
    ):
        self.budget = budget
        self.block_size = block_size
        self.sink_blocks = sink_blocks
        self.window_blocks = window_blocks
        self.sm_scale = dim ** -0.5
        self.decay = decay

        # State (initialized on first token)
        self._initialized = False
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None

        # Running proxy: Welford's online mean
        self._k_sum: Optional[torch.Tensor] = None  # [BSZ, HEAD, DIM]
        self._n_tokens: int = 0

        # Block importance scores: [BSZ, HEAD, N_BLOCKS]
        self._scores: Optional[torch.Tensor] = None
        self._n_blocks: int = 0

        # Cached mask: [BSZ, HEAD, N_BLOCKS]
        self._mask: Optional[torch.Tensor] = None
        self._threshold: Optional[torch.Tensor] = None  # [BSZ, HEAD]

        # Pending block accumulator
        self._pending_keys: Optional[torch.Tensor] = None  # [BSZ, HEAD, <=block_size, DIM]
        self._pending_count: int = 0

    def _init_state(self, k_token: torch.Tensor):
        """Initialize from first token. k_token: [BSZ, HEAD, DIM]."""
        bsz, head, dim = k_token.shape
        self._device = k_token.device
        self._dtype = k_token.dtype
        self._k_sum = torch.zeros(bsz, head, dim, device=self._device, dtype=torch.float32)
        self._scores = torch.empty(bsz, head, 0, device=self._device, dtype=torch.float32)
        self._mask = torch.empty(bsz, head, 0, dtype=torch.bool, device=self._device)
        self._threshold = torch.full((bsz, head), float("-inf"), device=self._device)
        self._pending_keys = torch.empty(
            bsz, head, 0, dim, device=self._device, dtype=self._dtype
        )
        self._initialized = True

    def update(self, k_new: torch.Tensor) -> None:
        """Process new key token(s). O(1) amortized per token.

        Args:
            k_new: New key token(s) [BSZ, HEAD, DIM] or [BSZ, HEAD, T, DIM].
                If T > 1, processes T tokens at once.
        """
        if k_new.ndim == 3:
            k_new = k_new.unsqueeze(2)  # [BSZ, HEAD, 1, DIM]

        bsz, head, n_new, dim = k_new.shape

        if not self._initialized:
            self._init_state(k_new[:, :, 0, :])

        # Update running sum (for mean proxy)
        self._k_sum += k_new.float().sum(dim=2)
        self._n_tokens += n_new

        # Accumulate pending keys
        self._pending_keys = torch.cat([self._pending_keys, k_new], dim=2)
        self._pending_count += n_new

        # Flush complete blocks
        while self._pending_count >= self.block_size:
            block_keys = self._pending_keys[:, :, :self.block_size, :]
            self._pending_keys = self._pending_keys[:, :, self.block_size:, :]
            self._pending_count -= self.block_size
            self._flush_block(block_keys)

    def _flush_block(self, block_keys: torch.Tensor) -> None:
        """Score a complete block and update budget. O(block_size * dim).

        Args:
            block_keys: [BSZ, HEAD, block_size, DIM]
        """
        # Compute proxy = running mean
        proxy = (self._k_sum / self._n_tokens).unsqueeze(2)  # [BSZ, HEAD, 1, DIM]

        # Score = max proxy attention within block
        # proxy @ block_keys^T → [BSZ, HEAD, 1, block_size]
        raw_scores = torch.matmul(proxy, block_keys.transpose(-2, -1)) * self.sm_scale
        block_score = raw_scores.squeeze(2).amax(dim=-1)  # [BSZ, HEAD]

        # Apply decay to old scores
        if self.decay < 1.0 and self._scores.shape[-1] > 0:
            self._scores *= self.decay

        # Append new block score
        self._scores = torch.cat(
            [self._scores, block_score.unsqueeze(-1)], dim=-1
        )
        self._n_blocks += 1

        # Update mask with budget enforcement
        self._update_mask()

    def _update_mask(self) -> None:
        """Recompute mask based on budget. O(N) via kthvalue, no sort."""
        n = self._n_blocks
        usable_budget = max(
            self.budget - self.sink_blocks - self.window_blocks, 1
        )

        if n <= self.budget:
            # Under budget: keep everything
            self._mask = torch.ones(
                *self._scores.shape, dtype=torch.bool, device=self._device
            )
            return

        # Middle region (excluding sink + window)
        mid_start = self.sink_blocks
        mid_end = max(n - self.window_blocks, mid_start)
        mid_scores = self._scores[:, :, mid_start:mid_end]
        mid_n = mid_scores.shape[-1]

        if mid_n <= usable_budget:
            self._mask = torch.ones_like(self._scores, dtype=torch.bool)
            return

        # Adaptive threshold: kthvalue is O(N) partial selection
        k_from_top = mid_n - usable_budget
        flat_mid = mid_scores.reshape(-1, mid_n)
        threshold, _ = flat_mid.kthvalue(k=k_from_top, dim=-1)
        self._threshold = threshold.view(self._scores.shape[0], self._scores.shape[1])

        # Build mask
        self._mask = torch.ones_like(self._scores, dtype=torch.bool)
        mid_mask = mid_scores >= self._threshold.unsqueeze(-1)
        self._mask[:, :, mid_start:mid_end] = mid_mask

    def get_mask(self) -> torch.Tensor:
        """Get current importance mask. O(1).

        Returns:
            mask: [BSZ, HEAD, N_BLOCKS] bool. True = attend to this block.
        """
        if not self._initialized or self._mask is None:
            raise RuntimeError("No tokens processed yet. Call update() first.")

        # Include pending partial block (always attend to newest tokens)
        if self._pending_count > 0:
            pending_mask = torch.ones(
                *self._mask.shape[:-1], 1,
                dtype=torch.bool, device=self._device,
            )
            return torch.cat([self._mask, pending_mask], dim=-1)

        return self._mask

    def get_scores(self) -> torch.Tensor:
        """Get raw importance scores. [BSZ, HEAD, N_BLOCKS]."""
        if self._scores is None:
            raise RuntimeError("No tokens processed yet.")
        return self._scores

    @property
    def n_blocks(self) -> int:
        return self._n_blocks + (1 if self._pending_count > 0 else 0)

    @property
    def n_active_blocks(self) -> int:
        """Number of blocks in the attention mask."""
        if self._mask is None:
            return 0
        return int(self._mask.sum(dim=-1).float().mean().item())

    def reset(self) -> None:
        """Reset all state for new sequence."""
        self._initialized = False
        self._k_sum = None
        self._n_tokens = 0
        self._scores = None
        self._n_blocks = 0
        self._mask = None
        self._threshold = None
        self._pending_keys = None
        self._pending_count = 0


# --- Batch utility for prefill ---

def compute_k_importance(
    k: torch.Tensor,
    block_size: int = 64,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Batch compute block importance for prefill (non-streaming).

    Uses mean-key proxy. O(N * dim) total.

    Args:
        k: [BSZ, HEAD, N_TOKENS, DIM]
        block_size: Tokens per block.
        sm_scale: Softmax scale.

    Returns:
        importance: [BSZ, HEAD, N_BLOCKS]
    """
    bsz, head, n_tokens, dim = k.shape
    if sm_scale is None:
        sm_scale = dim ** -0.5

    # Mean key proxy
    proxy = k.mean(dim=2, keepdim=True)  # [BSZ, HEAD, 1, DIM]
    scores = torch.matmul(proxy, k.transpose(-2, -1)).squeeze(2) * sm_scale  # [BSZ, HEAD, N]

    # Block aggregation
    n_blocks = (n_tokens + block_size - 1) // block_size
    pad_len = n_blocks * block_size - n_tokens
    if pad_len > 0:
        scores = torch.nn.functional.pad(scores, (0, pad_len), value=float("-inf"))
    return scores.view(bsz, head, n_blocks, block_size).amax(dim=-1)


def select_by_budget(
    block_importance: torch.Tensor,
    budget: int,
    sink_blocks: int = 2,
    window_blocks: int = 4,
) -> torch.Tensor:
    """Select blocks within budget via threshold. O(N), no sort.

    Args:
        block_importance: [BSZ, HEAD, N_BLOCKS]
        budget: Total blocks to keep.
        sink_blocks: Always keep first N blocks.
        window_blocks: Always keep last N blocks.

    Returns:
        mask: [BSZ, HEAD, N_BLOCKS] bool
    """
    bsz, head, n_blocks = block_importance.shape

    if n_blocks <= budget:
        return torch.ones_like(block_importance, dtype=torch.bool)

    usable = max(budget - sink_blocks - window_blocks, 1)
    mid_start = sink_blocks
    mid_end = max(n_blocks - window_blocks, mid_start)
    mid = block_importance[:, :, mid_start:mid_end]
    mid_n = mid.shape[-1]

    if mid_n <= usable:
        return torch.ones_like(block_importance, dtype=torch.bool)

    k_from_top = mid_n - usable
    threshold, _ = mid.reshape(bsz * head, -1).kthvalue(k=k_from_top, dim=-1)
    threshold = threshold.view(bsz, head, 1)

    mask = torch.ones_like(block_importance, dtype=torch.bool)
    mask[:, :, mid_start:mid_end] = mid >= threshold
    return mask
