"""Reusable RoPE adapter for HiP and QSA kernels.

Extracted from the inline RoPE application in ``delta_pipeline._compute_dense``
(originally ``paged_hip.py`` lines 1494-1534).  Zero behavioral change.
"""

from typing import Tuple

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RoPEAdapter:
    """Unified RoPE interface for HiP and QSA kernels.

    Handles the impedance mismatch between:
    - HiP: internal RoPE with dynamic per-stage adjustment
    - QSA: expects pre-rotated Q/K tensors

    Parameters
    ----------
    cos : torch.Tensor
        Cosine embedding table, shape ``[max_seq_len, rot_dim]``.
    sin : torch.Tensor
        Sine embedding table, shape ``[max_seq_len, rot_dim]``.
    model_context_length : int
        Maximum context length the model was trained with.  Used for
        position clamping when context extension is active.
    extend_mode : str
        One of ``"none"``, ``"self_extend"``, ``"nope"``.  When ``"nope"``,
        :meth:`prepare_for_qsa` is a no-op (returns inputs unchanged).
    """

    def __init__(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
        model_context_length: int,
        extend_mode: str = "none",
    ):
        self.cos = cos  # [max_seq_len, rot_dim]
        self.sin = sin
        self.model_context_length = model_context_length
        self.extend_mode = extend_mode

    def prepare_for_qsa(
        self,
        query: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
        idx: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to Q and K for the QSA kernel.

        Matches HiP's position clamping behaviour when context extension is
        active (``extend_mode != "nope"``).

        Parameters
        ----------
        query : torch.Tensor
            Query tensor to rotate, shape ``[B, T_q, H, D]``.
        k : torch.Tensor
            Key tensor to rotate, shape ``[B, T_k, H, D]``.
        position_ids : torch.Tensor
            Absolute position ids, shape ``[1, T_total]``.
        idx : torch.Tensor
            Index tensor selecting which query positions to rotate,
            shape ``[N_recomp]``.
        seq_len : int
            Current sequence length (``position_ids.amax() + 1``).

        Returns
        -------
        rotated_q : torch.Tensor
            Rotated query, same shape/dtype as *query*.
        rotated_k : torch.Tensor
            Rotated key, same shape/dtype as *k*.
        """
        if self.extend_mode == "nope":
            return query, k

        cos = self.cos.view(1, self.cos.shape[-2], 1, self.cos.shape[-1])
        sin = self.sin.view(1, self.sin.shape[-2], 1, self.sin.shape[-1])

        # Position clamping for context extension
        idx_tsrc = torch.arange(0, k.shape[1], device=cos.device)
        idx_tsrc.clamp_min_(seq_len - self.model_context_length)

        rotated_k = (
            k.to(cos.dtype) * cos[:, idx_tsrc, :, :]
            + rotate_half(k.to(sin.dtype)) * sin[:, idx_tsrc, :, :]
        ).to(k.dtype)

        rotated_q = (
            query * cos[:, position_ids.view(-1)[idx], :, :]
            + rotate_half(query) * sin[:, position_ids.view(-1)[idx], :, :]
        ).to(query.dtype)

        return rotated_q, rotated_k

    def should_skip_hip_rope(self) -> bool:
        """Whether HiP kernel should handle its own RoPE."""
        return True
