"""Tests for KVAccessor unified cache access."""

import importlib.util
import os
import sys
import unittest
from unittest.mock import MagicMock

import torch

# Load kv_accessor.py directly from disk to avoid triggering
# the heavy hip_attn top-level __init__.py (which imports CUDA deps).
_KV_ACCESSOR_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir, os.pardir,
    "src", "hip_attn", "v1_2", "cache", "kv_accessor.py",
)
_KV_ACCESSOR_PATH = os.path.normpath(_KV_ACCESSOR_PATH)

spec = importlib.util.spec_from_file_location("kv_accessor", _KV_ACCESSOR_PATH)
kv_accessor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kv_accessor)
KVAccessor = kv_accessor.KVAccessor


def _make_mock_args(using_paged_cache: bool = True):
    """Create a mock HiPAttentionArgs with gather methods."""
    args = MagicMock()
    args.using_paged_cache = using_paged_cache
    args.get_k_cache.return_value = torch.randn(16, 1, 4, 64)
    args.get_v_cache.return_value = torch.randn(16, 1, 4, 64)
    args.block_table = torch.zeros(1, 16, dtype=torch.int64)

    # Each call to gather returns a *new* tensor by default (MagicMock behaviour).
    # We fix the return value so caching can be verified via identity.
    k_gathered = torch.randn(1, 16, 4, 64)
    v_gathered = torch.randn(1, 16, 4, 64)
    args.gather_k_from_paged_cache.return_value = k_gathered
    args.gather_v_from_paged_cache.return_value = v_gathered
    return args, k_gathered, v_gathered


class TestKVAccessor(unittest.TestCase):
    def test_is_paged_property(self):
        args_paged, _, _ = _make_mock_args(using_paged_cache=True)
        accessor = KVAccessor(args_paged)
        self.assertTrue(accessor.is_paged)

        args_flat, _, _ = _make_mock_args(using_paged_cache=False)
        accessor_flat = KVAccessor(args_flat)
        self.assertFalse(accessor_flat.is_paged)

    def test_get_paged_returns_cache_and_table(self):
        args, _, _ = _make_mock_args()
        accessor = KVAccessor(args)
        k_cache, v_cache, block_table = accessor.get_paged()
        self.assertIs(k_cache, args.get_k_cache())
        self.assertIs(block_table, args.block_table)

    def test_get_contiguous_caching(self):
        args, k_expected, v_expected = _make_mock_args()
        accessor = KVAccessor(args)

        k1, v1 = accessor.get_contiguous()
        k2, v2 = accessor.get_contiguous()

        # Same tensor objects must be returned (cached).
        self.assertIs(k1, k_expected)
        self.assertIs(v1, v_expected)
        self.assertIs(k1, k2)
        self.assertIs(v1, v2)

        # gather should have been called exactly once each.
        args.gather_k_from_paged_cache.assert_called_once()
        args.gather_v_from_paged_cache.assert_called_once()

    def test_get_contiguous_with_seq_len(self):
        args, k_expected, v_expected = _make_mock_args()
        accessor = KVAccessor(args)

        k, v = accessor.get_contiguous(seq_len=8)
        self.assertEqual(k.shape[1], 8)
        self.assertEqual(v.shape[1], 8)

    def test_invalidate_cache(self):
        args, k_expected, v_expected = _make_mock_args()
        accessor = KVAccessor(args)

        k1, v1 = accessor.get_contiguous()
        accessor.invalidate_cache()
        k2, v2 = accessor.get_contiguous()

        # After invalidation, gather is called again (2 total).
        self.assertEqual(args.gather_k_from_paged_cache.call_count, 2)
        self.assertEqual(args.gather_v_from_paged_cache.call_count, 2)

    def test_different_gqa_params_invalidate(self):
        args, _, _ = _make_mock_args()
        accessor = KVAccessor(args)

        q1 = torch.randn(1, 16, 8, 64)
        q2 = torch.randn(1, 16, 8, 64)

        accessor.get_contiguous(disable_gqa=True, gqa_q=q1)
        accessor.get_contiguous(disable_gqa=True, gqa_q=q2)

        # Different gqa_q tensor identity triggers re-gather.
        self.assertEqual(args.gather_k_from_paged_cache.call_count, 2)


if __name__ == "__main__":
    unittest.main()
