"""Typed configuration for HiP delta attention.

Replaces the ad-hoc string parsing of HIP_DELTA_ATTENTION_ARGS and BSA_*
environment variables with a single validated dataclass.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DeltaAttentionConfig:
    """All knobs that govern delta (incremental) attention behaviour.

    Construction paths:
        - ``DeltaAttentionConfig.from_env()``  -- mirrors the legacy env-var parsing.
        - ``DeltaAttentionConfig.from_dict(d)`` -- for JSON / programmatic configs.
        - Direct construction with keyword arguments.
    """

    # -- core delta flags --
    enabled: bool = False
    gamma: int = 16
    diff_mode: int = 1
    smooth: bool = True
    just_return: bool = False
    base_window_size: int = 0
    dense_decode: bool = False
    extend_mode: str = "none"

    # -- exponential sparse flags --
    exp_enabled: bool = False
    exp_w: int = 2
    exp_window: int = 1024
    exp_sink: int = 128

    # -- iteration / correction --
    iter_corr: bool = False
    adjust_norm_const: bool = False
    bsa_meanpool: bool = False

    # -- QSA / BSA block-sparse parameters --
    qsa_block_size_q: int = 128
    qsa_block_size_k: int = 64
    qsa_top_k: int = 128
    qsa_exact_k: int = 8
    qsa_post_trim: int = 4096
    qsa_winner_tree: bool = False
    qsa_reverse_iter: bool = True
    qsa_threshold_refresh: int = 4

    # -- property derived from qsa_winner_tree --

    @property
    def online_topk_method(self) -> str:
        return "tree" if self.qsa_winner_tree else "online"

    # -- factory methods --

    @classmethod
    def from_env(cls, *, using_extend: bool = True) -> DeltaAttentionConfig:
        """Parse ``HIP_DELTA_ATTENTION_ARGS`` and ``BSA_*`` env vars.

        The parsing logic intentionally mirrors the original code in
        ``paged_hip.py`` so that the two paths produce identical results.

        Args:
            using_extend: Passed from the outer ``HiPAttentionConfig``.  When
                ``False``, extend modes parsed from the arg string are forced
                to ``"none"``.
        """
        raw = os.getenv("HIP_DELTA_ATTENTION_ARGS", None)
        enabled = raw is not None and raw != ""

        cfg = cls(enabled=enabled)

        if enabled:
            cfg._parse_arg_string(raw, using_extend=using_extend)

        cfg._parse_bsa_env_vars()
        return cfg

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DeltaAttentionConfig:
        """Construct from a plain dictionary (e.g. loaded from JSON).

        Unknown keys are silently ignored with a warning so that forward-
        compatible configs do not break older code.
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        cfg = cls()
        for key, value in d.items():
            if key in known_fields:
                setattr(cfg, key, value)
            else:
                warnings.warn(f"DeltaAttentionConfig.from_dict: unknown key '{key}'")
        cfg.validate()
        return cfg

    # -- validation --

    def validate(self) -> None:
        """Assert that field values are within acceptable ranges."""
        if self.diff_mode not in (0, 1, 2, 3, 4):
            raise ValueError(
                f"diff_mode must be in {{0, 1, 2, 3, 4}}, got {self.diff_mode}"
            )
        if self.gamma < 1:
            raise ValueError(f"gamma must be >= 1, got {self.gamma}")
        if self.extend_mode not in ("none", "self_extend", "nope"):
            raise ValueError(
                f"extend_mode must be 'none', 'self_extend', or 'nope', "
                f"got '{self.extend_mode}'"
            )
        if self.qsa_block_size_q < 1:
            raise ValueError(
                f"qsa_block_size_q must be >= 1, got {self.qsa_block_size_q}"
            )
        if self.qsa_block_size_k < 1:
            raise ValueError(
                f"qsa_block_size_k must be >= 1, got {self.qsa_block_size_k}"
            )
        if self.qsa_top_k < 1:
            raise ValueError(f"qsa_top_k must be >= 1, got {self.qsa_top_k}")

    # -- internal helpers --

    def _parse_arg_string(self, raw: str, *, using_extend: bool) -> None:
        """Parse the dash-separated ``HIP_DELTA_ATTENTION_ARGS`` string."""
        for word in raw.split("-"):
            word = word.strip()
            if not word:
                continue

            if word == "smooth":
                self.smooth = True
            elif word == "exp":
                self.exp_enabled = True
            elif word == "JUST_RETURN":
                self.just_return = True
            elif word == "sparse_decode":
                self.dense_decode = False
            elif word == "dense_decode":
                self.dense_decode = True
            elif word == "recompute_dense":
                pass  # backward compat
            elif word == "bsa_meanpool":
                self.bsa_meanpool = True
            elif word == "iter_corr":
                self.iter_corr = True
            elif word.startswith("extend"):
                extend_kind = word.split("_")[1]
                if extend_kind == "self":
                    self.extend_mode = "self_extend"
                elif extend_kind == "nope":
                    self.extend_mode = "nope"
                else:
                    raise ValueError(f"Unknown extend mode: {extend_kind}")
                if not using_extend:
                    self.extend_mode = "none"
            elif word.startswith("window_"):
                self.base_window_size = int(word.split("_")[1])
            elif word.startswith("diff_"):
                self.diff_mode = int(word.split("_")[1])
            elif word.startswith("w_"):
                self.gamma = int(word.split("_")[1])
            elif word.startswith("expw_"):
                self.exp_w = int(word.split("_")[1])
            elif word.startswith("expsink_"):
                self.exp_sink = int(word.split("_")[1])
            elif word.startswith("expwindow_"):
                self.exp_window = int(word.split("_")[1])
            else:
                warnings.warn(f"unknown delta args: {word}")

    def _parse_bsa_env_vars(self) -> None:
        """Read BSA_* / REVERSE_ITER environment variables."""
        self.qsa_block_size_q = int(os.getenv("BSA_BLOCK_Q", str(self.qsa_block_size_q)))
        self.qsa_block_size_k = int(os.getenv("BSA_BLOCK_K", str(self.qsa_block_size_k)))
        self.qsa_top_k = int(os.getenv("BSA_K", str(self.qsa_top_k)))
        self.qsa_exact_k = int(os.getenv("BSA_EXACT_K", str(self.qsa_exact_k)))
        self.qsa_post_trim = int(os.getenv("BSA_TRIM", str(self.qsa_post_trim)))
        self.qsa_winner_tree = os.getenv("BSA_WINNER_TREE", "False") == "True"
        self.qsa_reverse_iter = os.getenv("REVERSE_ITER", "True") == "True"
        self.qsa_threshold_refresh = int(
            os.getenv("BSA_THRESHOLD_REFRESH", str(self.qsa_threshold_refresh))
        )
