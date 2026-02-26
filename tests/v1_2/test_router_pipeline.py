"""Tests for Triton-free RouterAttentionPipeline."""

import importlib.util
import os
import sys

import torch
import pytest

# Direct import to bypass hip_attn.__init__.py GPU dependencies
_MODULE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "hip_attn", "v1_2"
)


def _load_module(name, rel_path):
    path = os.path.abspath(os.path.join(_MODULE_DIR, rel_path))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_router_mod = _load_module("router_pipeline", "pipeline/router_pipeline.py")
RouterAttentionPipeline = _router_mod.RouterAttentionPipeline
RouterConfig = _router_mod.RouterConfig
apply_delta_native = _router_mod.apply_delta_native


class TestRouterConfig:
    def test_defaults(self):
        cfg = RouterConfig()
        assert cfg.rank == 8
        assert cfg.budget == 128
        assert cfg.delta_smooth is True

    def test_custom(self):
        cfg = RouterConfig(rank=4, budget=64, block_size=32)
        assert cfg.rank == 4
        assert cfg.block_size == 32


class TestRouterAttentionPipeline:
    def test_init(self):
        pipe = RouterAttentionPipeline(
            num_layers=32, num_heads_kv=8, head_dim=128,
            config=RouterConfig(), device=torch.device("cpu"),
        )
        assert len(pipe.states) == 0

    def test_get_or_create_router(self):
        pipe = RouterAttentionPipeline(
            num_layers=32, num_heads_kv=8, head_dim=64,
            config=RouterConfig(rank=4, budget=8, block_size=4),
            device=torch.device("cpu"),
        )
        state = pipe._get_or_create_router(0)
        assert state.router is not None
        assert not state.finalized

        # Same layer returns same state
        state2 = pipe._get_or_create_router(0)
        assert state2 is state

        # Different layer creates new state
        state3 = pipe._get_or_create_router(1)
        assert state3 is not state

    def test_router_lifecycle(self):
        """Test prefill update → finalize → decode route lifecycle."""
        pipe = RouterAttentionPipeline(
            num_layers=1, num_heads_kv=2, head_dim=64,
            config=RouterConfig(rank=4, budget=4, block_size=4,
                                sink_blocks=0, window_blocks=0),
            device=torch.device("cpu"),
        )

        # Simulate prefill: update router with K
        k = torch.randn(1, 2, 32, 64)  # [B, H_kv, T, D] — 32 tokens = 8 blocks
        state = pipe._get_or_create_router(0)
        state.router.update_prefill(k)

        # Finalize
        pipe.finalize_layer(0)
        assert pipe.states[0].finalized

        # Route a query
        q = torch.randn(1, 2, 64)  # [B, H_kv, D]
        mask = state.router.route(q)
        assert mask.shape == (1, 2, 8)
        assert mask.dtype == torch.bool

        # Budget should be ~4 blocks per head
        n_active = mask.sum(dim=-1).float()
        assert (n_active <= 5).all()

    def test_reset(self):
        pipe = RouterAttentionPipeline(
            num_layers=2, num_heads_kv=2, head_dim=64,
            config=RouterConfig(rank=4, budget=8, block_size=4),
            device=torch.device("cpu"),
        )
        pipe._get_or_create_router(0)
        pipe._get_or_create_router(1)
        assert len(pipe.states) == 2

        pipe.reset()
        assert len(pipe.states) == 0

    def test_reset_layer(self):
        pipe = RouterAttentionPipeline(
            num_layers=2, num_heads_kv=2, head_dim=64,
            config=RouterConfig(rank=4, budget=8, block_size=4),
            device=torch.device("cpu"),
        )
        pipe._get_or_create_router(0)
        pipe._get_or_create_router(1)
        pipe.reset_layer(0)
        assert 0 not in pipe.states
        assert 1 in pipe.states


class TestApplyDeltaNative:
    def test_output_shape(self):
        B, T, H, D = 2, 64, 4, 128
        gamma = 8
        n_delta = T // gamma
        num_last_dense = 4

        context_sparse = torch.randn(B, T, H, D)
        context_dense = torch.randn(B, n_delta + num_last_dense, H, D)
        idx = torch.arange(0, n_delta * gamma, gamma)
        idx = torch.cat([idx, torch.arange(T - num_last_dense, T)])

        result = apply_delta_native(
            context_dense, context_sparse, idx,
            num_last_dense=num_last_dense, gamma=gamma, smooth=True,
        )
        assert result.shape == (B, T + num_last_dense, H, D)

    def test_sample_points_exact(self):
        """Dense values at sample points should overwrite sparse."""
        B, T, H, D = 1, 32, 2, 64
        gamma = 4
        n_delta = T // gamma

        context_sparse = torch.zeros(B, T, H, D)
        context_dense = torch.ones(B, n_delta, H, D) * 42.0
        idx = torch.arange(0, T, gamma)

        result = apply_delta_native(
            context_dense, context_sparse, idx,
            num_last_dense=0, gamma=gamma, smooth=False,
        )

        # Sample points should be exactly 42.0
        for i in range(n_delta):
            pos = idx[i].item()
            assert torch.allclose(result[0, pos], torch.tensor(42.0)), \
                f"Sample point {pos} not exact"

    def test_smooth_vs_blockwise(self):
        """Smooth should produce different results than blockwise."""
        B, T, H, D = 1, 64, 2, 32
        gamma = 8
        n_delta = T // gamma

        context_sparse = torch.randn(B, T, H, D)
        context_dense = torch.randn(B, n_delta, H, D) * 5.0
        idx = torch.arange(0, T, gamma)

        result_block = apply_delta_native(
            context_dense, context_sparse, idx,
            num_last_dense=0, gamma=gamma, smooth=False,
        )
        result_smooth = apply_delta_native(
            context_dense, context_sparse, idx,
            num_last_dense=0, gamma=gamma, smooth=True,
        )

        # Should differ at non-sample-point positions
        diff = (result_block - result_smooth).abs().sum()
        assert diff > 0, "Smooth and blockwise should differ"

    def test_smooth_interpolation_continuity(self):
        """Smooth delta should vary within blocks, not be constant."""
        B, T, H, D = 1, 32, 1, 16
        gamma = 4
        n_delta = T // gamma

        context_sparse = torch.zeros(B, T, H, D)
        # Make consecutive deltas very different
        context_dense = torch.zeros(B, n_delta, H, D)
        context_dense[0, 0] = 10.0
        context_dense[0, 1] = -10.0
        idx = torch.arange(0, T, gamma)

        result = apply_delta_native(
            context_dense, context_sparse, idx,
            num_last_dense=0, gamma=gamma, smooth=True,
        )

        # Within block 0 (positions 0-3), values should ramp from +10 toward -10
        block0_vals = result[0, 0:4, 0, 0]
        # Position 0 = 10 (sample point)
        # Position 1 should be between 10 and -10
        # Position 2 should be closer to -10
        # Position 3 should be even closer
        assert block0_vals[0] > block0_vals[1] > block0_vals[2] > block0_vals[3], \
            f"Expected monotonic decrease: {block0_vals}"

    def test_no_nan(self):
        B, T, H, D = 2, 128, 4, 64
        gamma = 16
        n_delta = T // gamma

        context_sparse = torch.randn(B, T, H, D)
        context_dense = torch.randn(B, n_delta + 2, H, D)
        idx = torch.cat([
            torch.arange(0, T, gamma),
            torch.tensor([T - 2, T - 1]),
        ])

        result = apply_delta_native(
            context_dense, context_sparse, idx,
            num_last_dense=2, gamma=gamma, smooth=True,
        )
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_zero_delta_no_change(self):
        """When dense matches sparse at sample points, output == sparse."""
        B, T, H, D = 1, 32, 2, 16
        gamma = 4
        n_delta = T // gamma

        context_sparse = torch.randn(B, T, H, D)
        # Dense is same as sparse at sample points → zero delta
        idx = torch.arange(0, T, gamma)
        context_dense = context_sparse[:, idx].clone()

        result = apply_delta_native(
            context_dense, context_sparse, idx,
            num_last_dense=0, gamma=gamma, smooth=True,
        )
        assert torch.allclose(result[:, :T], context_sparse, atol=1e-6), \
            "Zero delta should not change sparse output"
