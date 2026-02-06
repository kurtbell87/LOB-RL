"""Tests for Python binding of precompute().

Spec: docs/precompute-binding.md

These tests verify that:
- lob_rl_core.precompute(path, cfg) is callable from Python
- Returns a tuple of (obs, mid, spread, num_steps)
- obs is a numpy ndarray with shape (N, 43) and dtype float32
- mid is a numpy ndarray with shape (N,) and dtype float64
- spread is a numpy ndarray with shape (N,) and dtype float64
- num_steps is an int matching array shapes
- All values are finite; mid and spread are positive
- Invalid inputs raise exceptions
"""

import pathlib

import numpy as np
import pytest

import lob_rl_core

# Fixture path: the precompute_rth.bin file is in the C++ test fixtures directory
FIXTURE_DIR = pathlib.Path(__file__).parent.parent.parent / "tests" / "fixtures"
PRECOMPUTE_FIXTURE = str(FIXTURE_DIR / "precompute_rth.bin")


# ===========================================================================
# Test 1: precompute returns tuple of 4 elements
# ===========================================================================


class TestPrecomputeReturnType:
    """precompute() should return a tuple of exactly 4 elements."""

    def test_precompute_returns_tuple(self):
        """precompute(path, cfg) should return a tuple."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        result = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert isinstance(result, tuple), (
            f"Expected tuple, got {type(result).__name__}"
        )

    def test_precompute_returns_4_elements(self):
        """The returned tuple should have exactly 4 elements."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        result = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert len(result) == 4, (
            f"Expected 4 elements, got {len(result)}"
        )


# ===========================================================================
# Test 2: obs is numpy array with correct shape and dtype
# ===========================================================================


class TestObsArray:
    """obs should be a numpy ndarray with shape (N, 43) and dtype float32."""

    def test_obs_is_numpy_array(self):
        """obs (first element) should be a numpy ndarray."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        obs, _, _, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert isinstance(obs, np.ndarray), (
            f"Expected numpy.ndarray, got {type(obs).__name__}"
        )

    def test_obs_dtype_is_float32(self):
        """obs should have dtype float32."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        obs, _, _, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert obs.dtype == np.float32, (
            f"Expected float32, got {obs.dtype}"
        )

    def test_obs_is_2d(self):
        """obs should be 2-dimensional."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        obs, _, _, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert obs.ndim == 2, (
            f"Expected 2D array, got {obs.ndim}D"
        )

    def test_obs_has_43_columns(self):
        """obs should have exactly 43 columns."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        obs, _, _, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert obs.shape[1] == 43, (
            f"Expected 43 columns, got {obs.shape[1]}"
        )

    def test_obs_has_positive_num_rows(self):
        """obs should have N > 0 rows for the fixture file."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        obs, _, _, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert obs.shape[0] > 0, "Expected N > 0 rows from fixture"


# ===========================================================================
# Test 3: mid is numpy array with correct shape and dtype
# ===========================================================================


class TestMidArray:
    """mid should be a numpy ndarray with shape (N,) and dtype float64."""

    def test_mid_is_numpy_array(self):
        """mid (second element) should be a numpy ndarray."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        _, mid, _, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert isinstance(mid, np.ndarray), (
            f"Expected numpy.ndarray, got {type(mid).__name__}"
        )

    def test_mid_dtype_is_float64(self):
        """mid should have dtype float64."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        _, mid, _, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert mid.dtype == np.float64, (
            f"Expected float64, got {mid.dtype}"
        )

    def test_mid_is_1d(self):
        """mid should be 1-dimensional."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        _, mid, _, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert mid.ndim == 1, (
            f"Expected 1D array, got {mid.ndim}D"
        )

    def test_mid_shape_matches_obs_rows(self):
        """mid.shape[0] should match obs.shape[0]."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        obs, mid, _, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert mid.shape[0] == obs.shape[0], (
            f"mid length {mid.shape[0]} != obs rows {obs.shape[0]}"
        )


# ===========================================================================
# Test 4: spread is numpy array with correct shape and dtype
# ===========================================================================


class TestSpreadArray:
    """spread should be a numpy ndarray with shape (N,) and dtype float64."""

    def test_spread_is_numpy_array(self):
        """spread (third element) should be a numpy ndarray."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        _, _, spread, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert isinstance(spread, np.ndarray), (
            f"Expected numpy.ndarray, got {type(spread).__name__}"
        )

    def test_spread_dtype_is_float64(self):
        """spread should have dtype float64."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        _, _, spread, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert spread.dtype == np.float64, (
            f"Expected float64, got {spread.dtype}"
        )

    def test_spread_is_1d(self):
        """spread should be 1-dimensional."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        _, _, spread, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert spread.ndim == 1, (
            f"Expected 1D array, got {spread.ndim}D"
        )

    def test_spread_shape_matches_obs_rows(self):
        """spread.shape[0] should match obs.shape[0]."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        obs, _, spread, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert spread.shape[0] == obs.shape[0], (
            f"spread length {spread.shape[0]} != obs rows {obs.shape[0]}"
        )


# ===========================================================================
# Test 5: num_steps matches array shapes
# ===========================================================================


class TestNumSteps:
    """num_steps should be an int matching all array dimensions."""

    def test_num_steps_is_int(self):
        """num_steps (fourth element) should be an int."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        _, _, _, num_steps = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert isinstance(num_steps, int), (
            f"Expected int, got {type(num_steps).__name__}"
        )

    def test_num_steps_matches_obs_rows(self):
        """num_steps should equal obs.shape[0]."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        obs, _, _, num_steps = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert num_steps == obs.shape[0], (
            f"num_steps={num_steps} != obs.shape[0]={obs.shape[0]}"
        )

    def test_num_steps_matches_mid_length(self):
        """num_steps should equal mid.shape[0]."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        _, mid, _, num_steps = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert num_steps == mid.shape[0], (
            f"num_steps={num_steps} != mid.shape[0]={mid.shape[0]}"
        )

    def test_num_steps_matches_spread_length(self):
        """num_steps should equal spread.shape[0]."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        _, _, spread, num_steps = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert num_steps == spread.shape[0], (
            f"num_steps={num_steps} != spread.shape[0]={spread.shape[0]}"
        )

    def test_all_sizes_consistent(self):
        """num_steps, obs rows, mid length, and spread length should all match."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        obs, mid, spread, num_steps = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert num_steps == obs.shape[0] == mid.shape[0] == spread.shape[0], (
            f"Inconsistent sizes: num_steps={num_steps}, "
            f"obs={obs.shape[0]}, mid={mid.shape[0]}, spread={spread.shape[0]}"
        )


# ===========================================================================
# Test 6: all obs values are finite
# ===========================================================================


class TestObsFiniteness:
    """All observation values should be finite (no NaN, no inf)."""

    def test_all_obs_finite(self):
        """Every element of obs should be finite."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        obs, _, _, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert np.all(np.isfinite(obs)), (
            f"Found non-finite obs values: "
            f"NaN count={np.sum(np.isnan(obs))}, "
            f"Inf count={np.sum(np.isinf(obs))}"
        )

    def test_no_nan_in_obs(self):
        """obs should contain no NaN values."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        obs, _, _, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert not np.any(np.isnan(obs)), "obs contains NaN values"

    def test_no_inf_in_obs(self):
        """obs should contain no infinite values."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        obs, _, _, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert not np.any(np.isinf(obs)), "obs contains infinite values"


# ===========================================================================
# Test 7: all mid and spread values are finite and positive
# ===========================================================================


class TestMidSpreadValues:
    """mid and spread values should be finite and positive."""

    def test_mid_all_finite(self):
        """All mid values should be finite."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        _, mid, _, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert np.all(np.isfinite(mid)), "mid contains non-finite values"

    def test_mid_all_positive(self):
        """All mid values should be positive (prices > 0)."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        _, mid, _, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert np.all(mid > 0), (
            f"mid has non-positive values; min={np.min(mid)}"
        )

    def test_spread_all_finite(self):
        """All spread values should be finite."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        _, _, spread, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert np.all(np.isfinite(spread)), "spread contains non-finite values"

    def test_spread_all_positive(self):
        """All spread values should be positive (spread > 0)."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        _, _, spread, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert np.all(spread > 0), (
            f"spread has non-positive values; min={np.min(spread)}"
        )

    def test_spread_less_than_mid(self):
        """Spread should be much smaller than mid (sanity check)."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        _, mid, spread, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        # For any reasonable financial data, spread < mid
        assert np.all(spread < mid), (
            "Spread should be smaller than mid price"
        )


# ===========================================================================
# Test 8: SessionConfig.default_rth() works for precompute
# ===========================================================================


class TestDefaultRthUsage:
    """The standard usage pattern with default_rth() should work."""

    def test_default_rth_does_not_crash(self):
        """precompute(path, SessionConfig.default_rth()) should not crash."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        result = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        # Just verify it returns without exception
        assert result is not None

    def test_default_rth_produces_results(self):
        """precompute with default_rth() should produce N > 0 steps."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        _, _, _, num_steps = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert num_steps > 0, "Expected at least 1 step from fixture"


# ===========================================================================
# Test 9: custom SessionConfig works
# ===========================================================================


class TestCustomSessionConfig:
    """precompute should accept custom SessionConfig values."""

    def test_custom_config_does_not_crash(self):
        """precompute with a custom SessionConfig should not crash."""
        cfg = lob_rl_core.SessionConfig()
        cfg.rth_open_ns = 13 * 3_600_000_000_000 + 30 * 60_000_000_000
        cfg.rth_close_ns = 20 * 3_600_000_000_000
        cfg.warmup_messages = 10
        result = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert result is not None

    def test_custom_config_returns_valid_arrays(self):
        """Custom config should still produce valid numpy arrays."""
        cfg = lob_rl_core.SessionConfig()
        cfg.rth_open_ns = 13 * 3_600_000_000_000 + 30 * 60_000_000_000
        cfg.rth_close_ns = 20 * 3_600_000_000_000
        cfg.warmup_messages = 10
        obs, mid, spread, num_steps = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert isinstance(obs, np.ndarray)
        assert isinstance(mid, np.ndarray)
        assert isinstance(spread, np.ndarray)
        assert isinstance(num_steps, int)

    def test_custom_config_with_zero_warmup(self):
        """precompute with warmup_messages=0 should work (skips all pre-market)."""
        cfg = lob_rl_core.SessionConfig()
        cfg.rth_open_ns = 13 * 3_600_000_000_000 + 30 * 60_000_000_000
        cfg.rth_close_ns = 20 * 3_600_000_000_000
        cfg.warmup_messages = 0
        result = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        obs, mid, spread, num_steps = result
        # With no warmup, results may differ but should still be structurally valid
        assert obs.ndim == 2
        if num_steps > 0:
            assert obs.shape[1] == 43

    def test_narrower_rth_window_produces_fewer_or_equal_steps(self):
        """A narrower RTH window should produce <= steps vs default_rth()."""
        default_cfg = lob_rl_core.SessionConfig.default_rth()
        _, _, _, default_steps = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, default_cfg)

        narrow_cfg = lob_rl_core.SessionConfig()
        # Narrow window: 14:00 to 19:00 UTC (subset of 13:30-20:00)
        narrow_cfg.rth_open_ns = 14 * 3_600_000_000_000
        narrow_cfg.rth_close_ns = 19 * 3_600_000_000_000
        narrow_cfg.warmup_messages = -1
        _, _, _, narrow_steps = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, narrow_cfg)

        assert narrow_steps <= default_steps, (
            f"Narrower window should produce <= steps: "
            f"narrow={narrow_steps} > default={default_steps}"
        )


# ===========================================================================
# Test 10: invalid file path raises exception
# ===========================================================================


class TestInvalidInputs:
    """Invalid inputs should raise appropriate exceptions."""

    def test_nonexistent_path_raises(self):
        """precompute with a nonexistent file path should raise RuntimeError."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        with pytest.raises(RuntimeError):
            lob_rl_core.precompute("/nonexistent/path/no_such_file.bin", cfg)

    def test_empty_path_raises(self):
        """precompute with an empty string path should raise RuntimeError."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        with pytest.raises(RuntimeError):
            lob_rl_core.precompute("", cfg)

    def test_directory_path_raises(self):
        """precompute with a directory path (not a file) should raise RuntimeError."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        with pytest.raises(RuntimeError):
            lob_rl_core.precompute(str(FIXTURE_DIR), cfg)


# ===========================================================================
# Additional: precompute function exists and is callable
# ===========================================================================


class TestPrecomputeExists:
    """The precompute function should be accessible on the module."""

    def test_precompute_attribute_exists(self):
        """lob_rl_core should have a precompute attribute."""
        assert hasattr(lob_rl_core, "precompute"), (
            "lob_rl_core module does not have 'precompute' attribute"
        )

    def test_precompute_is_callable(self):
        """lob_rl_core.precompute should be callable."""
        assert callable(lob_rl_core.precompute), (
            "lob_rl_core.precompute is not callable"
        )


# ===========================================================================
# Additional: obs array memory layout
# ===========================================================================


class TestObsMemoryLayout:
    """obs array should be C-contiguous for efficient numpy operations."""

    def test_obs_is_c_contiguous(self):
        """obs should be C-contiguous (row-major) for numpy compatibility."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        obs, _, _, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert obs.flags["C_CONTIGUOUS"], "obs array should be C-contiguous"

    def test_obs_is_writeable(self):
        """obs array should be writeable (owned data, not a read-only view)."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        obs, _, _, _ = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert obs.flags["WRITEABLE"], "obs array should be writeable"


# ===========================================================================
# Additional: deterministic results
# ===========================================================================


class TestDeterminism:
    """Calling precompute twice with same inputs should produce identical results."""

    def test_two_calls_produce_same_results(self):
        """precompute called twice should return identical arrays."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        obs1, mid1, spread1, n1 = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        obs2, mid2, spread2, n2 = lob_rl_core.precompute(PRECOMPUTE_FIXTURE, cfg)
        assert n1 == n2, f"num_steps differ: {n1} vs {n2}"
        np.testing.assert_array_equal(obs1, obs2, err_msg="obs arrays differ")
        np.testing.assert_array_equal(mid1, mid2, err_msg="mid arrays differ")
        np.testing.assert_array_equal(spread1, spread2, err_msg="spread arrays differ")
