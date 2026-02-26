"""Tests for RoPEAdapter -- zero behavioral change verification.

Follows the same module-stubbing pattern as ``test_delta_config.py`` so that
the test can run in environments where GPU-only dependencies (nvtx, triton,
etc.) are not installed.
"""

import os
import sys
import types

import torch
import pytest

# Ensure the src directory is on the path.
_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src")
_SRC_DIR = os.path.abspath(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def _ensure_parent_modules() -> None:
    """Pre-populate sys.modules with stubs when GPU deps are missing.

    ``hip_attn/__init__.py`` imports nvtx etc. which may not be available.
    We only need the rope subpackage here, so we create minimal namespace
    stubs for the parent packages.
    """
    try:
        import hip_attn  # noqa: F401
        return  # Full package available, no stubbing needed
    except (ImportError, ModuleNotFoundError):
        pass

    pkg_root = os.path.join(_SRC_DIR, "hip_attn")
    v12_root = os.path.join(pkg_root, "v1_2")
    rope_root = os.path.join(v12_root, "rope")

    stubs = {
        "hip_attn": [pkg_root],
        "hip_attn.v1_2": [v12_root],
        "hip_attn.v1_2.rope": [rope_root],
    }
    for mod_name, path in stubs.items():
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            stub.__path__ = path
            stub.__package__ = mod_name
            sys.modules[mod_name] = stub


_ensure_parent_modules()

from hip_attn.v1_2.rope.rope_adapter import RoPEAdapter, rotate_half  # noqa: E402


@pytest.fixture
def rope_tables():
    """Create small RoPE cos/sin tables for testing."""
    max_seq = 256
    rot_dim = 64
    cos = torch.randn(max_seq, rot_dim)
    sin = torch.randn(max_seq, rot_dim)
    return cos, sin


class TestPrepareForQsaShapes:
    """Verify output shapes match input shapes."""

    def test_output_shapes_match_input(self, rope_tables):
        cos, sin = rope_tables
        adapter = RoPEAdapter(cos, sin, model_context_length=256, extend_mode="none")

        B, T_q, H, D = 1, 4, 2, 64
        T_k = 128
        query = torch.randn(B, T_q, H, D)
        k = torch.randn(B, T_k, H, D)
        position_ids = torch.arange(0, T_k).unsqueeze(0)  # [1, T_k]
        idx = torch.arange(0, T_q)
        seq_len = T_k

        rotated_q, rotated_k = adapter.prepare_for_qsa(
            query, k, position_ids, idx, seq_len
        )

        assert rotated_q.shape == query.shape
        assert rotated_k.shape == k.shape
        assert rotated_q.dtype == query.dtype
        assert rotated_k.dtype == k.dtype


class TestNopePassthrough:
    """extend_mode='nope' returns inputs unchanged."""

    def test_nope_returns_same_tensors(self, rope_tables):
        cos, sin = rope_tables
        adapter = RoPEAdapter(cos, sin, model_context_length=256, extend_mode="nope")

        B, T_q, H, D = 1, 4, 2, 64
        T_k = 128
        query = torch.randn(B, T_q, H, D)
        k = torch.randn(B, T_k, H, D)
        position_ids = torch.arange(0, T_k).unsqueeze(0)
        idx = torch.arange(0, T_q)
        seq_len = T_k

        out_q, out_k = adapter.prepare_for_qsa(query, k, position_ids, idx, seq_len)

        # Must be the exact same objects (no copy)
        assert out_q is query
        assert out_k is k


class TestPositionClamping:
    """Verify clamp_min behavior for long sequences."""

    def test_clamping_for_long_context(self, rope_tables):
        cos, sin = rope_tables
        model_ctx = 128
        adapter = RoPEAdapter(cos, sin, model_context_length=model_ctx, extend_mode="none")

        B, T_q, H, D = 1, 4, 2, 64
        T_k = 200  # longer than model_context_length
        query = torch.randn(B, T_q, H, D)
        k = torch.randn(B, T_k, H, D)
        position_ids = torch.arange(0, T_k).unsqueeze(0)
        idx = torch.arange(0, T_q)
        seq_len = T_k

        rotated_q, rotated_k = adapter.prepare_for_qsa(
            query, k, position_ids, idx, seq_len
        )

        # Positions 0..(seq_len - model_ctx - 1) should be clamped to (seq_len - model_ctx).
        # The result should still have valid shapes and no NaNs.
        assert rotated_q.shape == query.shape
        assert rotated_k.shape == k.shape
        assert not torch.isnan(rotated_q).any()
        assert not torch.isnan(rotated_k).any()

    def test_no_clamping_within_context_length(self, rope_tables):
        cos, sin = rope_tables
        model_ctx = 256
        adapter = RoPEAdapter(cos, sin, model_context_length=model_ctx, extend_mode="none")

        B, T_q, H, D = 1, 4, 2, 64
        T_k = 128  # shorter than model_context_length
        query = torch.randn(B, T_q, H, D)
        k = torch.randn(B, T_k, H, D)
        position_ids = torch.arange(0, T_k).unsqueeze(0)
        idx = torch.arange(0, T_q)
        seq_len = T_k

        rotated_q, rotated_k = adapter.prepare_for_qsa(
            query, k, position_ids, idx, seq_len
        )

        # clamp_min_(seq_len - model_ctx) = clamp_min_(-128) => no clamping
        assert rotated_q.shape == query.shape
        assert rotated_k.shape == k.shape
        assert not torch.isnan(rotated_q).any()
        assert not torch.isnan(rotated_k).any()


class TestRotateHalf:
    """Verify rotate_half helper."""

    def test_rotate_half_output_shape(self):
        x = torch.randn(2, 4, 8)
        out = rotate_half(x)
        assert out.shape == x.shape

    def test_rotate_half_values(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = rotate_half(x)
        expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
        assert torch.allclose(out, expected)
