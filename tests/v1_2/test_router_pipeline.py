"""Tests for per-layer top-k attention routing pipeline."""

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


_mod = _load_module("router_pipeline", "pipeline/router_pipeline.py")
RouterAttentionPipeline = _mod.RouterAttentionPipeline
RouterConfig = _mod.RouterConfig
LayerRouter = _mod.LayerRouter


class TestRouterConfig:
    def test_defaults(self):
        cfg = RouterConfig()
        assert cfg.rank == 8
        assert cfg.budget == 128
        assert cfg.block_size == 64

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
        assert len(pipe.layers) == 0

    def test_get_layer_creates_router_and_stats(self):
        pipe = RouterAttentionPipeline(
            num_layers=32, num_heads_kv=8, head_dim=64,
            config=RouterConfig(rank=4, budget=8, block_size=4),
            device=torch.device("cpu"),
        )
        layer = pipe._get_layer(0)
        assert layer.router is not None
        assert layer.stats is not None
        assert not layer.finalized

    def test_same_layer_returns_same_state(self):
        pipe = RouterAttentionPipeline(
            num_layers=32, num_heads_kv=8, head_dim=64,
            config=RouterConfig(rank=4, budget=8, block_size=4),
            device=torch.device("cpu"),
        )
        s1 = pipe._get_layer(0)
        s2 = pipe._get_layer(0)
        assert s1 is s2

    def test_different_layers_independent(self):
        pipe = RouterAttentionPipeline(
            num_layers=32, num_heads_kv=8, head_dim=64,
            config=RouterConfig(rank=4, budget=8, block_size=4),
            device=torch.device("cpu"),
        )
        s0 = pipe._get_layer(0)
        s1 = pipe._get_layer(1)
        assert s0 is not s1

    def test_router_lifecycle(self):
        """Prefill update → finalize → route."""
        pipe = RouterAttentionPipeline(
            num_layers=1, num_heads_kv=2, head_dim=64,
            config=RouterConfig(rank=4, budget=4, block_size=4,
                                sink_blocks=0, window_blocks=0),
            device=torch.device("cpu"),
        )

        # Simulate prefill: feed K to router
        layer = pipe._get_layer(0)
        k = torch.randn(1, 2, 32, 64)  # [B, H_kv, T, D]
        layer.router.update_prefill(k)

        # Finalize
        pipe.finalize_layer(0)
        assert pipe.layers[0].finalized

        # Route a query
        q = torch.randn(1, 2, 64)
        mask = layer.router.route(q)
        assert mask.shape == (1, 2, 8)  # 32 tokens / 4 block_size = 8 blocks
        assert mask.dtype == torch.bool
        assert mask.sum(dim=-1).max() <= 5  # budget=4, some tolerance

    def test_stats_feedback_changes_routing(self):
        """Running stats should influence routing decisions."""
        pipe = RouterAttentionPipeline(
            num_layers=1, num_heads_kv=1, head_dim=64,
            config=RouterConfig(rank=4, budget=3, block_size=4,
                                sink_blocks=0, window_blocks=0),
            device=torch.device("cpu"),
        )

        layer = pipe._get_layer(0)
        k = torch.randn(1, 1, 40, 64) * 0.1  # 10 blocks, all similar
        layer.router.update_prefill(k)
        pipe.finalize_layer(0)

        # Route without stats
        q = torch.randn(1, 1, 64)
        mask_before = layer.router.route(q).clone()

        # Heavily boost block 7 via stats
        for _ in range(20):
            layer.stats.update(torch.tensor([7]), attention_lse=None, block_size=4)

        # Route with stats boost
        boost = layer.stats.get_boost(10)
        mask_after = layer.router.route(q, empirical_boost=boost)

        # Block 7 should be selected after boosting
        assert mask_after[0, 0, 7].item(), \
            "Heavily boosted block should be selected"

    def test_reset(self):
        pipe = RouterAttentionPipeline(
            num_layers=2, num_heads_kv=2, head_dim=64,
            config=RouterConfig(rank=4, budget=8, block_size=4),
            device=torch.device("cpu"),
        )
        pipe._get_layer(0)
        pipe._get_layer(1)
        assert len(pipe.layers) == 2

        pipe.reset()
        assert len(pipe.layers) == 0

    def test_reset_layer(self):
        pipe = RouterAttentionPipeline(
            num_layers=2, num_heads_kv=2, head_dim=64,
            config=RouterConfig(rank=4, budget=8, block_size=4),
            device=torch.device("cpu"),
        )
        pipe._get_layer(0)
        pipe._get_layer(1)
        pipe.reset_layer(0)
        assert 0 not in pipe.layers
        assert 1 in pipe.layers

    def test_finalize_all(self):
        pipe = RouterAttentionPipeline(
            num_layers=3, num_heads_kv=2, head_dim=64,
            config=RouterConfig(rank=4, budget=8, block_size=4),
            device=torch.device("cpu"),
        )
        for i in range(3):
            layer = pipe._get_layer(i)
            k = torch.randn(1, 2, 16, 64)
            layer.router.update_prefill(k)

        pipe.finalize_all()
        for i in range(3):
            assert pipe.layers[i].finalized
