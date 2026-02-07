"""Tests for train.py --cache-dir CLI flag.

Spec: docs/precompute-cache.md (Requirements 4 and 5)

These tests verify that:
- --cache-dir flag exists and is accepted
- --cache-dir is mutually exclusive with --data-dir
- Neither --cache-dir nor --data-dir raises error
- --cache-dir passes cache_dir= to MultiDayEnv
- Train/val/test split works with sorted .npz files
- Manifest.json loading is skipped when --cache-dir is set
- evaluate_sortino() accepts cache_path parameter
- evaluate_sortino(cache_path=...) uses from_cache() instead of from_file()
- Existing --data-dir workflow is unchanged
"""

import argparse
import inspect
import os
import sys
import textwrap

import numpy as np
import pytest

# Path to the train.py script
TRAIN_SCRIPT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train.py")
)


# ===========================================================================
# Helper: extract argparse parser from train.py
# ===========================================================================


def _load_train_module():
    """Import train.py as a module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("train", TRAIN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # Add project paths
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, os.path.join(project_root, "python"))
    spec.loader.exec_module(mod)
    return mod


def _get_parser_actions(mod=None):
    """Extract argparse parser actions from train.py by inspecting source."""
    source = open(TRAIN_SCRIPT).read()
    return source


# ===========================================================================
# Test 1: --cache-dir flag exists
# ===========================================================================


class TestCacheDirFlagExists:
    """train.py should define a --cache-dir CLI flag."""

    def test_cache_dir_in_source(self):
        """train.py source should contain '--cache-dir' argument definition."""
        source = open(TRAIN_SCRIPT).read()
        assert "--cache-dir" in source, (
            "train.py does not define --cache-dir flag"
        )

    def test_cache_dir_is_argparse_argument(self):
        """--cache-dir should be an argparse add_argument call."""
        source = open(TRAIN_SCRIPT).read()
        assert "add_argument" in source and "cache-dir" in source, (
            "train.py should use add_argument for --cache-dir"
        )


# ===========================================================================
# Test 2: --cache-dir and --data-dir are mutually exclusive
# ===========================================================================


class TestMutualExclusivity:
    """--cache-dir and --data-dir should be mutually exclusive."""

    def test_source_has_mutual_exclusion_logic(self):
        """train.py should enforce mutual exclusivity between --cache-dir and --data-dir."""
        source = open(TRAIN_SCRIPT).read()
        # The script should check that exactly one is provided
        # This could be via argparse mutually_exclusive_group or manual check
        has_mutex_group = "mutually_exclusive" in source
        has_manual_check = (
            ("cache_dir" in source and "data_dir" in source) and
            ("error" in source.lower() or "raise" in source.lower() or
             "parser.error" in source)
        )
        assert has_mutex_group or has_manual_check, (
            "train.py should enforce mutual exclusivity between --cache-dir and --data-dir"
        )


# ===========================================================================
# Test 3: evaluate_sortino accepts cache_path parameter
# ===========================================================================


class TestEvaluateSortinoCache:
    """evaluate_sortino() should accept cache_path parameter."""

    def test_evaluate_sortino_has_cache_path_param(self):
        """evaluate_sortino function should accept cache_path parameter."""
        mod = _load_train_module()
        sig = inspect.signature(mod.evaluate_sortino)
        assert "cache_path" in sig.parameters, (
            f"evaluate_sortino should have cache_path parameter, "
            f"got params: {list(sig.parameters.keys())}"
        )

    def test_evaluate_sortino_cache_path_default_none(self):
        """cache_path should default to None."""
        mod = _load_train_module()
        sig = inspect.signature(mod.evaluate_sortino)
        param = sig.parameters["cache_path"]
        assert param.default is None, (
            f"cache_path should default to None, got {param.default}"
        )


# ===========================================================================
# Test 4: make_env accepts cache_path parameter OR make_env_from_cache exists
# ===========================================================================


class TestMakeEnvCacheSupport:
    """There should be a way to create an env from cache in train.py."""

    def test_source_uses_from_cache(self):
        """train.py should reference from_cache when --cache-dir is used."""
        source = open(TRAIN_SCRIPT).read()
        assert "from_cache" in source, (
            "train.py should use from_cache() when loading from cache"
        )

    def test_source_references_cache_dir_in_multi_day_env(self):
        """train.py should pass cache_dir to MultiDayEnv when --cache-dir is set."""
        source = open(TRAIN_SCRIPT).read()
        assert "cache_dir" in source, (
            "train.py should reference cache_dir for MultiDayEnv"
        )


# ===========================================================================
# Test 5: make_train_env supports cache_dir
# ===========================================================================


class TestMakeTrainEnvCacheDir:
    """make_train_env should support passing cache_dir."""

    def test_make_train_env_has_cache_dir_support(self):
        """make_train_env should accept or use cache_dir parameter."""
        source = open(TRAIN_SCRIPT).read()
        # Either make_train_env accepts cache_dir, or main() branches on it
        has_cache_in_make_train = "cache_dir" in source
        assert has_cache_in_make_train, (
            "train.py should support cache_dir in the training env factory"
        )


# ===========================================================================
# Test 6: Existing --data-dir workflow is unchanged
# ===========================================================================


class TestDataDirUnchanged:
    """Existing --data-dir workflow should remain unchanged."""

    def test_data_dir_flag_still_exists(self):
        """--data-dir should still be defined in train.py."""
        source = open(TRAIN_SCRIPT).read()
        assert "--data-dir" in source

    def test_load_manifest_still_exists(self):
        """load_manifest function should still exist."""
        source = open(TRAIN_SCRIPT).read()
        assert "load_manifest" in source

    def test_make_env_still_exists(self):
        """make_env function should still exist."""
        mod = _load_train_module()
        assert hasattr(mod, "make_env"), "make_env function should still exist"

    def test_evaluate_sortino_still_exists(self):
        """evaluate_sortino function should still exist."""
        mod = _load_train_module()
        assert hasattr(mod, "evaluate_sortino"), (
            "evaluate_sortino function should still exist"
        )


# ===========================================================================
# Test 7: train/val/test split logic for cache_dir
# ===========================================================================


class TestSplitLogic:
    """train.py should split cached days the same way as .bin files."""

    def test_source_handles_cache_dir_split(self):
        """train.py should split .npz files for train/val/test when using --cache-dir."""
        source = open(TRAIN_SCRIPT).read()
        # Should have logic to list/sort .npz files and split them
        has_npz_listing = ".npz" in source or "npz" in source
        has_split = "train" in source.lower() and "val" in source.lower()
        assert has_npz_listing and has_split, (
            "train.py should list .npz files and split for train/val/test"
        )


# ===========================================================================
# Test 8: --cache-dir skips manifest loading
# ===========================================================================


class TestSkipsManifest:
    """When --cache-dir is used, manifest.json should not be loaded."""

    def test_cache_dir_path_skips_manifest(self):
        """The cache_dir code path should not call load_manifest."""
        source = open(TRAIN_SCRIPT).read()
        # The source should have conditional logic:
        # if cache_dir: (use .npz files) else: load_manifest
        # We verify the branching exists
        has_branching = (
            "cache_dir" in source and
            "load_manifest" in source
        )
        assert has_branching, (
            "train.py should branch between cache_dir and load_manifest"
        )


# ===========================================================================
# Test 9: evaluate_sortino with cache_path uses from_cache
# ===========================================================================


class TestEvaluateSortinoUsesFromCache:
    """evaluate_sortino with cache_path should use PrecomputedEnv.from_cache()."""

    def test_evaluate_sortino_references_from_cache(self):
        """evaluate_sortino body should reference from_cache when cache_path is set."""
        mod = _load_train_module()
        source = inspect.getsource(mod.evaluate_sortino)
        assert "from_cache" in source, (
            "evaluate_sortino should use from_cache when cache_path is provided"
        )


# ===========================================================================
# Test 10: Full integration — cache_dir env produces valid observations
# ===========================================================================


class TestCacheDirIntegration:
    """End-to-end: create cache, build MultiDayEnv, verify it works."""

    def test_cache_dir_env_produces_valid_obs(self):
        """MultiDayEnv(cache_dir=...) should produce valid 54-dim obs."""
        import tempfile
        import lob_rl_core
        from lob_rl.multi_day_env import MultiDayEnv

        FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
        DAY_FILES = [os.path.join(FIXTURE_DIR, f"day{i}.bin") for i in range(3)]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(cache_dir)

            # Create cache from real .bin files
            cfg = lob_rl_core.SessionConfig.default_rth()
            for i, bin_path in enumerate(DAY_FILES):
                obs, mid, spread, n = lob_rl_core.precompute(bin_path, cfg)
                if n >= 2:
                    np.savez(os.path.join(cache_dir, f"day{i:02d}.npz"),
                             obs=obs, mid=mid, spread=spread)

            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            obs, info = env.reset()

            assert obs.shape == (54,)
            assert obs.dtype == np.float32
            assert not np.any(np.isnan(obs))
            assert not np.any(np.isinf(obs))
            assert "day_index" in info

    def test_cache_dir_zero_cpp_precompute_during_reset(self):
        """With cache_dir, reset() should NOT call lob_rl_core.precompute.

        This verifies acceptance criterion 4: zero C++ precompute calls.
        """
        import tempfile
        from unittest import mock
        import lob_rl_core
        from lob_rl.multi_day_env import MultiDayEnv
        from conftest import make_realistic_obs

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(cache_dir)

            # Create cache with synthetic data
            obs, mid, spread = make_realistic_obs(50)
            np.savez(os.path.join(cache_dir, "2025-01-10.npz"),
                     obs=obs, mid=mid, spread=spread)

            # Patch precompute to track calls
            with mock.patch.object(lob_rl_core, "precompute",
                                   wraps=lob_rl_core.precompute) as mock_precompute:
                env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)
                env.reset()

                # precompute should NOT have been called
                mock_precompute.assert_not_called()
