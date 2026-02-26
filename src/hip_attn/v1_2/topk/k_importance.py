"""Streaming K-Importance: Budget-based Sparse Attention via Key Pre-scoring.

Two-signal block selection:

  1. IMPORTANCE (mean-key proxy score):
     Tokens aligned with the mean key direction. These are "popular" tokens
     that most queries will attend to. Delta correction CAN recover these
     if pruned, so they are lower priority.

  2. OUTLIER (deviation from mean):
     Tokens far from the mean key direction. These are rare but critical —
     when a query does attend to them, no nearby token can substitute.
     Delta correction CANNOT recover these, so they must be kept.

  keep_mask = (importance > τ_imp) OR (outlier > τ_out)

Theory:
    RoPE creates locality bias: cos(θ·Δpos) decays with distance.
    Most attention mass is predictable (recent + sink + high-importance).
    The hard part is finding sparse outliers in the middle region.
    Outlier detection via Welford's online variance catches exactly these.

Streaming guarantee: O(1) amortized per token via:
    - Welford mean/variance update: O(dim) per token
    - Block scoring: O(block_size * dim) every block_size tokens
    - Budget enforcement: O(N) via kthvalue (amortized per block)
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
        outlier_sigma: float = 2.0,
    ):
        self.budget = budget
        self.block_size = block_size
        self.sink_blocks = sink_blocks
        self.window_blocks = window_blocks
        self.sm_scale = dim ** -0.5
        self.decay = decay
        self.outlier_sigma = outlier_sigma  # Blocks > σ stdev from mean are outliers

        # State (initialized on first token)
        self._initialized = False
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None

        # Welford's online algorithm: running mean + M2 for variance
        # Tracks per-head statistics of key norms in hidden dim
        self._k_sum: Optional[torch.Tensor] = None     # [BSZ, HEAD, DIM]
        self._k_sq_sum: Optional[torch.Tensor] = None  # [BSZ, HEAD, DIM] sum of squares
        self._n_tokens: int = 0

        # Block-level scores: [BSZ, HEAD, N_BLOCKS]
        self._importance_scores: Optional[torch.Tensor] = None  # proxy attention score
        self._outlier_scores: Optional[torch.Tensor] = None     # deviation from mean
        self._n_blocks: int = 0

        # Cached mask: [BSZ, HEAD, N_BLOCKS]
        self._mask: Optional[torch.Tensor] = None
        self._threshold: Optional[torch.Tensor] = None

        # Pending block accumulator
        self._pending_keys: Optional[torch.Tensor] = None
        self._pending_count: int = 0

    def _init_state(self, k_token: torch.Tensor):
        """Initialize from first token. k_token: [BSZ, HEAD, DIM]."""
        bsz, head, dim = k_token.shape
        self._device = k_token.device
        self._dtype = k_token.dtype
        self._k_sum = torch.zeros(bsz, head, dim, device=self._device, dtype=torch.float32)
        self._k_sq_sum = torch.zeros(bsz, head, dim, device=self._device, dtype=torch.float32)
        self._importance_scores = torch.empty(bsz, head, 0, device=self._device, dtype=torch.float32)
        self._outlier_scores = torch.empty(bsz, head, 0, device=self._device, dtype=torch.float32)
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

        # Welford update: running sum + sum of squares (for mean + variance)
        k_float = k_new.float()
        self._k_sum += k_float.sum(dim=2)
        self._k_sq_sum += (k_float ** 2).sum(dim=2)
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
        """Score a complete block for importance + outlier. O(block_size * dim).

        Two scores per block:
        1. importance = max(proxy · k_j) — how likely any query attends here
        2. outlier = max(||k_j - mean|| / σ) — how different from typical keys
           Outliers can't be recovered by delta correction → must keep.
        """
        k_float = block_keys.float()  # [BSZ, HEAD, block_size, DIM]

        # --- Signal 1: Importance (proxy attention score) ---
        proxy = (self._k_sum / self._n_tokens).unsqueeze(2)  # [BSZ, HEAD, 1, DIM]
        raw_scores = torch.matmul(proxy, k_float.transpose(-2, -1)) * self.sm_scale
        importance = raw_scores.squeeze(2).amax(dim=-1)  # [BSZ, HEAD]

        # --- Signal 2: Outlier (deviation from running mean) ---
        mean = (self._k_sum / self._n_tokens).unsqueeze(2)  # [BSZ, HEAD, 1, DIM]
        var = (self._k_sq_sum / self._n_tokens) - (self._k_sum / self._n_tokens) ** 2
        std = var.clamp(min=1e-8).sqrt().unsqueeze(2)  # [BSZ, HEAD, 1, DIM]

        # Per-token deviation: ||( k_j - mean ) / std||_2
        normalized_dev = (k_float - mean) / std  # [BSZ, HEAD, block_size, DIM]
        token_outlier = normalized_dev.norm(dim=-1)  # [BSZ, HEAD, block_size]
        # Normalize by sqrt(dim) so score is ~1 for normal, >2 for outlier
        dim = block_keys.shape[-1]
        token_outlier = token_outlier / (dim ** 0.5)
        block_outlier = token_outlier.amax(dim=-1)  # [BSZ, HEAD]

        # Apply decay to old scores
        if self.decay < 1.0 and self._importance_scores.shape[-1] > 0:
            self._importance_scores *= self.decay
            self._outlier_scores *= self.decay

        # Append
        self._importance_scores = torch.cat(
            [self._importance_scores, importance.unsqueeze(-1)], dim=-1
        )
        self._outlier_scores = torch.cat(
            [self._outlier_scores, block_outlier.unsqueeze(-1)], dim=-1
        )
        self._n_blocks += 1

        self._update_mask()

    def _update_mask(self) -> None:
        """Recompute mask from importance OR outlier. O(N) via kthvalue.

        A block is kept if:
          (importance >= threshold) OR (outlier >= outlier_sigma)

        Outlier blocks are always kept regardless of importance budget,
        because delta correction cannot recover them.
        """
        n = self._n_blocks
        usable_budget = max(
            self.budget - self.sink_blocks - self.window_blocks, 1
        )

        if n <= self.budget:
            self._mask = torch.ones(
                *self._importance_scores.shape, dtype=torch.bool, device=self._device
            )
            return

        # Outlier mask: blocks with deviation > outlier_sigma (ALWAYS kept)
        outlier_mask = self._outlier_scores >= self.outlier_sigma

        # Middle region for importance-based selection
        mid_start = self.sink_blocks
        mid_end = max(n - self.window_blocks, mid_start)
        mid_importance = self._importance_scores[:, :, mid_start:mid_end]
        mid_outlier = outlier_mask[:, :, mid_start:mid_end]
        mid_n = mid_importance.shape[-1]

        # Count outliers already in budget (they're free — must keep)
        n_outliers_in_mid = mid_outlier.sum(dim=-1)  # [BSZ, HEAD]
        remaining_budget = (usable_budget - n_outliers_in_mid).clamp(min=1)

        if mid_n <= usable_budget:
            self._mask = torch.ones_like(self._importance_scores, dtype=torch.bool)
            return

        # For importance threshold: only consider non-outlier blocks
        # (outliers are already kept, don't waste budget on them)
        mid_importance_masked = torch.where(
            mid_outlier, torch.full_like(mid_importance, float("inf")), mid_importance
        )

        # kthvalue on importance scores (excluding outliers which are inf)
        # We want to keep remaining_budget non-outlier blocks
        n_non_outlier = mid_n - n_outliers_in_mid.long()
        k_from_top = (n_non_outlier.float() - remaining_budget.float()).clamp(min=1).long()

        # Per-head threshold via kthvalue
        flat = mid_importance.reshape(-1, mid_n)
        # Use max possible k across all heads for batched kthvalue
        max_k = k_from_top.reshape(-1).max().item()
        max_k = min(max(int(max_k), 1), mid_n)
        threshold, _ = flat.kthvalue(k=max_k, dim=-1)
        self._threshold = threshold.view(self._importance_scores.shape[0], self._importance_scores.shape[1])

        # Build mask: importance OR outlier
        self._mask = torch.ones_like(self._importance_scores, dtype=torch.bool)
        importance_mask = mid_importance >= self._threshold.unsqueeze(-1)
        self._mask[:, :, mid_start:mid_end] = importance_mask | mid_outlier

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
        """Get combined scores (importance + outlier bonus). [BSZ, HEAD, N_BLOCKS]."""
        if self._importance_scores is None:
            raise RuntimeError("No tokens processed yet.")
        return self._importance_scores

    def get_outlier_scores(self) -> torch.Tensor:
        """Get outlier deviation scores. [BSZ, HEAD, N_BLOCKS]."""
        if self._outlier_scores is None:
            raise RuntimeError("No tokens processed yet.")
        return self._outlier_scores

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
        self._k_sq_sum = None
        self._n_tokens = 0
        self._importance_scores = None
        self._outlier_scores = None
        self._n_blocks = 0
        self._mask = None
        self._threshold = None
        self._pending_keys = None
        self._pending_count = 0


class CovarianceRouter:
    """Rank-r attention router via streaming covariance decomposition.

    Theory: The top-r eigenvectors of C = K^T K / N represent the directions
    most likely to receive high attention. For any query q, the projection
    g = W @ q (where W = top-r eigenvectors) captures how well q aligns
    with the dominant attention directions.

    Pre-compute H = W @ K^T during prefill. At decode time:
        scores = (W @ q)^T @ H = O(r × N) instead of O(d × N)

    Cost breakdown:
        Prefill (per token): O(d²) covariance update (fused with KV write)
        Prefill (once): O(d³) eigendecompose (~2M ops for d=128)
        Decode (per token): O(r × d + r × N) routing (~5K ops for r=4, N=1000 blocks)

    Args:
        rank: Number of principal directions (4-8 is usually sufficient).
        dim: Head dimension.
        block_size: Tokens per block for block-level routing.
        budget: Max blocks to attend to.
        sink_blocks: Always keep first N blocks.
        window_blocks: Always keep last N blocks.
    """

    def __init__(
        self,
        rank: int = 4,
        dim: int = 128,
        block_size: int = 64,
        budget: int = 128,
        sink_blocks: int = 2,
        window_blocks: int = 4,
    ):
        self.rank = rank
        self.dim = dim
        self.block_size = block_size
        self.budget = budget
        self.sink_blocks = sink_blocks
        self.window_blocks = window_blocks
        self.sm_scale = dim ** -0.5

        # State
        self._cov = None          # [BSZ, HEAD, DIM, DIM] running covariance
        self._n_tokens = 0
        self._W = None            # [BSZ, HEAD, RANK, DIM] top-r eigenvectors (router weights)
        self._H = None            # [BSZ, HEAD, RANK, N_BLOCKS] pre-computed block projections
        self._finalized = False
        self._block_keys_sum = None  # [BSZ, HEAD, N_BLOCKS, DIM] block-level key sums
        self._n_blocks = 0
        self._pending_keys = None
        self._pending_count = 0
        self._initialized = False

    def update_prefill(self, k: torch.Tensor) -> None:
        """Process key tokens during prefill. Accumulates covariance.

        Args:
            k: [BSZ, HEAD, T, DIM] key tokens
        """
        if k.ndim == 3:
            k = k.unsqueeze(2)

        bsz, head, t, dim = k.shape
        k_float = k.float()

        if not self._initialized:
            self._cov = torch.zeros(bsz, head, dim, dim, device=k.device, dtype=torch.float32)
            self._block_keys_sum = torch.empty(bsz, head, 0, dim, device=k.device, dtype=torch.float32)
            self._pending_keys = torch.empty(bsz, head, 0, dim, device=k.device, dtype=k.dtype)
            self._initialized = True

        # Streaming covariance: C += K^T @ K (batch matmul)
        # k_float: [BSZ, HEAD, T, DIM]
        # k_float^T @ k_float: [BSZ, HEAD, DIM, DIM]
        self._cov += torch.matmul(k_float.transpose(-2, -1), k_float)
        self._n_tokens += t

        # Accumulate for block-level projections
        self._pending_keys = torch.cat([self._pending_keys, k], dim=2)
        self._pending_count += t

        while self._pending_count >= self.block_size:
            block = self._pending_keys[:, :, :self.block_size, :]
            self._pending_keys = self._pending_keys[:, :, self.block_size:, :]
            self._pending_count -= self.block_size
            # Store block mean key for later projection
            block_mean = block.float().mean(dim=2, keepdim=True)  # [BSZ, HEAD, 1, DIM]
            self._block_keys_sum = torch.cat([self._block_keys_sum, block_mean], dim=2)
            self._n_blocks += 1

    def finalize(self) -> None:
        """Compute eigenvectors and pre-project all blocks. Call after prefill.

        Cost: O(d³) eigendecompose + O(r × N_blocks × d) projection.
        For d=128, r=4, N_blocks=1000: ~2M + 512K = 2.5M ops. Negligible.
        """
        if self._n_tokens == 0:
            raise RuntimeError("No tokens processed. Call update_prefill first.")

        # Normalize covariance
        C = self._cov / self._n_tokens  # [BSZ, HEAD, DIM, DIM]

        # Eigendecompose (symmetric, so use eigh — faster than svd)
        # eigenvalues in ascending order, take last r
        bsz, head, dim, _ = C.shape
        C_flat = C.reshape(bsz * head, dim, dim)
        eigenvalues, eigenvectors = torch.linalg.eigh(C_flat)

        # Top-r eigenvectors (largest eigenvalues = last r columns)
        W = eigenvectors[:, :, -self.rank:]  # [BSZ*HEAD, DIM, RANK]
        W = W.transpose(-2, -1)  # [BSZ*HEAD, RANK, DIM]
        self._W = W.reshape(bsz, head, self.rank, dim)

        # Pre-project all block means: H = W @ block_means^T
        # _block_keys_sum: [BSZ, HEAD, N_BLOCKS, DIM]
        # W: [BSZ, HEAD, RANK, DIM]
        # H: [BSZ, HEAD, RANK, N_BLOCKS]
        self._H = torch.matmul(
            self._W, self._block_keys_sum.transpose(-2, -1)
        ) * self.sm_scale

        self._finalized = True

    def route(
        self,
        q: torch.Tensor,
        empirical_boost: torch.Tensor = None,
    ) -> torch.Tensor:
        """Route a query to important blocks. O(r × d + r × N_blocks).

        Args:
            q: [BSZ, HEAD, DIM] single query token
            empirical_boost: [HEAD, N_BLOCKS] float — running stats feedback.
                Added to covariance scores before budget selection.

        Returns:
            mask: [BSZ, HEAD, N_BLOCKS] bool — which blocks to attend to
        """
        if not self._finalized:
            raise RuntimeError("Call finalize() after prefill before routing.")

        # Project query: g = W @ q  → [BSZ, HEAD, RANK]
        g = torch.matmul(self._W, q.unsqueeze(-1)).squeeze(-1)

        # Score blocks: scores = g^T @ H → [BSZ, HEAD, N_BLOCKS]
        scores = torch.matmul(g.unsqueeze(-2), self._H).squeeze(-2)

        # Add empirical boost from running statistics (free adaptation)
        if empirical_boost is not None:
            n = min(scores.shape[-1], empirical_boost.shape[-1])
            scores[..., :n] = scores[..., :n] + empirical_boost[..., :n]

        # Budget-based selection
        return self._select_by_budget(scores)

    def _select_by_budget(self, scores: torch.Tensor) -> torch.Tensor:
        """Select top blocks within budget. O(N) via kthvalue."""
        bsz, head, n = scores.shape

        if n <= self.budget:
            return torch.ones(bsz, head, n, dtype=torch.bool, device=scores.device)

        usable = max(self.budget - self.sink_blocks - self.window_blocks, 1)
        mid_start = self.sink_blocks
        mid_end = max(n - self.window_blocks, mid_start)
        mid = scores[:, :, mid_start:mid_end]
        mid_n = mid.shape[-1]

        if mid_n <= usable:
            return torch.ones(bsz, head, n, dtype=torch.bool, device=scores.device)

        k_from_top = mid_n - usable
        threshold, _ = mid.reshape(bsz * head, -1).kthvalue(k=max(k_from_top, 1), dim=-1)
        threshold = threshold.view(bsz, head, 1)

        mask = torch.ones(bsz, head, n, dtype=torch.bool, device=scores.device)
        mask[:, :, mid_start:mid_end] = mid >= threshold
        return mask

    def update_decode(self, k_new: torch.Tensor) -> None:
        """Add new decode token to router state. O(r × d).

        Args:
            k_new: [BSZ, HEAD, DIM] new key token
        """
        if not self._finalized:
            raise RuntimeError("Call finalize() first.")

        # Project new key and append to H
        # h_new = W @ k_new → [BSZ, HEAD, RANK]
        h_new = torch.matmul(self._W, k_new.unsqueeze(-1)).squeeze(-1) * self.sm_scale
        self._H = torch.cat([self._H, h_new.unsqueeze(-1)], dim=-1)
        self._n_blocks += 1  # simplified: 1 token = partial block

    def reset(self):
        """Clear all state."""
        self._cov = None
        self._n_tokens = 0
        self._W = None
        self._H = None
        self._finalized = False
        self._block_keys_sum = None
        self._n_blocks = 0
        self._pending_keys = None
        self._pending_count = 0
        self._initialized = False


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
