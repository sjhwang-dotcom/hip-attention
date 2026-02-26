"""Tests for DeltaAttentionConfig dataclass.

This module tests a pure-stdlib config dataclass. When torch and other GPU
dependencies are not installed, we stub out the top-level hip_attn.__init__
to allow importing the config subpackage in isolation.
"""

import importlib
import os
import sys
import types
from unittest import mock

import pytest

# Ensure the src directory is on the path.
_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src")
_SRC_DIR = os.path.abspath(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def _ensure_parent_modules() -> None:
    """Pre-populate sys.modules with stubs when GPU deps are missing.

    ``hip_attn/__init__.py`` imports torch, nvtx, etc. which are not available
    in lightweight test environments.  We only need the config subpackage here,
    so we create minimal namespace stubs for the parent packages with correct
    filesystem paths so that subpackage resolution works.
    """
    try:
        import torch  # noqa: F401 -- just probing availability
        return  # GPU deps available, no stubbing needed
    except ImportError:
        pass

    pkg_root = os.path.join(_SRC_DIR, "hip_attn")
    v12_root = os.path.join(pkg_root, "v1_2")

    # Stub the package chain up to v1_2 so that the config import resolves
    # without triggering the real __init__.py files.
    stubs = {
        "hip_attn": [pkg_root],
        "hip_attn.v1_2": [v12_root],
        "hip_attn.v1_2.attention_metadata": [],
    }
    for mod_name, path in stubs.items():
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            stub.__path__ = path
            stub.__package__ = mod_name
            sys.modules[mod_name] = stub

    # attention_metadata needs a ScanStage stub so hip_config can import it.
    am = sys.modules["hip_attn.v1_2.attention_metadata"]
    if not hasattr(am, "ScanStage"):
        am.ScanStage = type("ScanStage", (), {})


_ensure_parent_modules()

from hip_attn.v1_2.config.delta_config import DeltaAttentionConfig  # noqa: E402


class TestFromEnv:
    def test_from_env_all_fields(self):
        env = {
            "HIP_DELTA_ATTENTION_ARGS": "window_0-diff_1-w_16-sparse_decode-smooth",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = DeltaAttentionConfig.from_env()

        assert cfg.enabled is True
        assert cfg.base_window_size == 0
        assert cfg.diff_mode == 1
        assert cfg.gamma == 16
        assert cfg.dense_decode is False
        assert cfg.smooth is True
        assert cfg.just_return is False
        assert cfg.exp_enabled is False
        assert cfg.extend_mode == "none"
        assert cfg.iter_corr is False

    def test_from_env_empty_disabled(self):
        with mock.patch.dict(os.environ, {"HIP_DELTA_ATTENTION_ARGS": ""}, clear=False):
            cfg = DeltaAttentionConfig.from_env()

        assert cfg.enabled is False
        assert cfg.gamma == 16  # default

    def test_from_env_unset_disabled(self):
        env = os.environ.copy()
        env.pop("HIP_DELTA_ATTENTION_ARGS", None)
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = DeltaAttentionConfig.from_env()

        assert cfg.enabled is False

    def test_from_env_dense_decode(self):
        env = {"HIP_DELTA_ATTENTION_ARGS": "dense_decode"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = DeltaAttentionConfig.from_env()

        assert cfg.enabled is True
        assert cfg.dense_decode is True

    def test_from_env_exp_fields(self):
        env = {
            "HIP_DELTA_ATTENTION_ARGS": "exp-expw_4-expwindow_2048-expsink_256",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = DeltaAttentionConfig.from_env()

        assert cfg.exp_enabled is True
        assert cfg.exp_w == 4
        assert cfg.exp_window == 2048
        assert cfg.exp_sink == 256

    def test_from_env_iter_corr_and_bsa_meanpool(self):
        env = {
            "HIP_DELTA_ATTENTION_ARGS": "iter_corr-bsa_meanpool",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = DeltaAttentionConfig.from_env()

        assert cfg.iter_corr is True
        assert cfg.bsa_meanpool is True

    def test_from_env_extend_self(self):
        env = {"HIP_DELTA_ATTENTION_ARGS": "extend_self"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = DeltaAttentionConfig.from_env(using_extend=True)

        assert cfg.extend_mode == "self_extend"

    def test_from_env_extend_disabled_when_not_using_extend(self):
        env = {"HIP_DELTA_ATTENTION_ARGS": "extend_self"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = DeltaAttentionConfig.from_env(using_extend=False)

        assert cfg.extend_mode == "none"


class TestBsaEnvVars:
    def test_bsa_env_vars(self):
        env = {
            "HIP_DELTA_ATTENTION_ARGS": "diff_1",
            "BSA_K": "256",
            "BSA_BLOCK_Q": "64",
            "BSA_BLOCK_K": "32",
            "BSA_EXACT_K": "16",
            "BSA_TRIM": "8192",
            "BSA_WINNER_TREE": "True",
            "REVERSE_ITER": "False",
            "BSA_THRESHOLD_REFRESH": "8",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = DeltaAttentionConfig.from_env()

        assert cfg.qsa_top_k == 256
        assert cfg.qsa_block_size_q == 64
        assert cfg.qsa_block_size_k == 32
        assert cfg.qsa_exact_k == 16
        assert cfg.qsa_post_trim == 8192
        assert cfg.qsa_winner_tree is True
        assert cfg.online_topk_method == "tree"
        assert cfg.qsa_reverse_iter is False
        assert cfg.qsa_threshold_refresh == 8


class TestFromDict:
    def test_from_dict_json(self):
        d = {
            "enabled": True,
            "gamma": 32,
            "diff_mode": 2,
            "smooth": True,
            "dense_decode": True,
            "exp_enabled": True,
            "exp_w": 4,
            "qsa_top_k": 256,
        }
        cfg = DeltaAttentionConfig.from_dict(d)
        assert cfg.enabled is True
        assert cfg.gamma == 32
        assert cfg.diff_mode == 2
        assert cfg.smooth is True
        assert cfg.dense_decode is True
        assert cfg.exp_enabled is True
        assert cfg.exp_w == 4
        assert cfg.qsa_top_k == 256

    def test_from_dict_unknown_key_warns(self):
        d = {"enabled": True, "nonexistent_key": 42}
        with pytest.warns(UserWarning, match="unknown key 'nonexistent_key'"):
            cfg = DeltaAttentionConfig.from_dict(d)
        assert cfg.enabled is True


class TestValidate:
    def test_validate_bad_diff_mode(self):
        cfg = DeltaAttentionConfig(diff_mode=5)
        with pytest.raises(ValueError, match="diff_mode"):
            cfg.validate()

    def test_validate_bad_gamma(self):
        cfg = DeltaAttentionConfig(gamma=0)
        with pytest.raises(ValueError, match="gamma"):
            cfg.validate()

    def test_validate_bad_extend_mode(self):
        cfg = DeltaAttentionConfig(extend_mode="bogus")
        with pytest.raises(ValueError, match="extend_mode"):
            cfg.validate()

    def test_validate_ok(self):
        cfg = DeltaAttentionConfig(diff_mode=3, gamma=8, extend_mode="nope")
        cfg.validate()  # should not raise


class TestOnlineTopkMethod:
    def test_default_is_online(self):
        cfg = DeltaAttentionConfig()
        assert cfg.online_topk_method == "online"

    def test_winner_tree(self):
        cfg = DeltaAttentionConfig(qsa_winner_tree=True)
        assert cfg.online_topk_method == "tree"
