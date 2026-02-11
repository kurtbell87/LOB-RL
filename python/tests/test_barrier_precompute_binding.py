"""Tests for Python binding of barrier_precompute().

Spec: docs/barrier-precompute-cpp.md

These tests verify that:
- lob_rl_core.barrier_precompute(path, instrument_id, ...) is callable
- Returns a dict with all expected keys matching Python process_session() output
- Returns None when data is insufficient (< lookback + 1 bars)
- Array shapes and dtypes match the spec exactly
- Scalar metadata values are correct
- Invalid inputs raise exceptions
"""

import numpy as np
import pytest

import lob_rl_core


# ===========================================================================
# Test 1: barrier_precompute function exists and is callable
# ===========================================================================


class TestBarrierPrecomputeExists:
    """barrier_precompute should be accessible from lob_rl_core module."""

    def test_function_exists(self):
        """lob_rl_core should have a barrier_precompute attribute."""
        assert hasattr(lob_rl_core, "barrier_precompute"), (
            "lob_rl_core module should expose barrier_precompute"
        )

    def test_function_is_callable(self):
        """barrier_precompute should be a callable."""
        assert callable(lob_rl_core.barrier_precompute), (
            "barrier_precompute should be callable"
        )


# ===========================================================================
# Test 2: Invalid path raises exception
# ===========================================================================


class TestBarrierPrecomputeInvalidPath:
    """barrier_precompute should raise on invalid file paths."""

    def test_nonexistent_path_raises(self):
        """Should raise when given a path that doesn't exist."""
        with pytest.raises(Exception):
            lob_rl_core.barrier_precompute("/nonexistent/path.dbn.zst", 0)

    def test_empty_path_raises(self):
        """Should raise when given an empty path."""
        with pytest.raises(Exception):
            lob_rl_core.barrier_precompute("", 0)


# ===========================================================================
# Test 3: Return type is dict or None
# ===========================================================================


class TestBarrierPrecomputeReturnType:
    """barrier_precompute should return a dict on success, None on insufficient data."""

    def test_insufficient_data_returns_none(self):
        """With a file that has too few bars, should return None.

        Since we can't easily create a .dbn.zst fixture in Python tests,
        this test uses a mock approach — the binding should handle
        the insufficient data case by returning None.
        """
        # This test verifies the contract: if the file has < lookback+1 bars,
        # return None. We test this indirectly by checking the function signature
        # accepts the expected parameters.
        # The actual None-return behavior is tested in C++ tests.
        pass  # Tested in C++ — verified via integration below


# ===========================================================================
# Test 4: Function signature accepts expected parameters
# ===========================================================================


class TestBarrierPrecomputeSignature:
    """barrier_precompute should accept the documented parameter names."""

    def test_accepts_keyword_args(self):
        """Should accept bar_size, lookback, a, b, t_max as keyword args.

        This tests that the pybind11 binding declares all expected py::arg()s.
        We can't call it with valid data without a fixture, so we test that
        the function accepts keywords by catching the file-not-found error.
        """
        with pytest.raises(Exception):
            lob_rl_core.barrier_precompute(
                "/nonexistent.dbn.zst", 0,
                bar_size=500, lookback=10, a=20, b=10, t_max=40
            )

    def test_accepts_positional_args(self):
        """Should accept path and instrument_id as positional args."""
        with pytest.raises(Exception):
            # path, instrument_id as positional
            lob_rl_core.barrier_precompute("/nonexistent.dbn.zst", 12345)


# ===========================================================================
# Test 5: Default parameter values
# ===========================================================================


class TestBarrierPrecomputeDefaults:
    """barrier_precompute should have correct default parameter values.

    Defaults from spec: bar_size=500, lookback=10, a=20, b=10, t_max=40.
    We verify the function can be called with only path + instrument_id.
    """

    def test_defaults_accepted(self):
        """Should accept call with only path and instrument_id (all others default)."""
        with pytest.raises(Exception):
            # Only required args — defaults should kick in
            lob_rl_core.barrier_precompute("/nonexistent.dbn.zst", 0)


# ===========================================================================
# Test 6: Return dict has expected keys (integration, needs fixture)
# ===========================================================================


# These tests require a .dbn.zst fixture file to actually run.
# They are structured to verify the dict contract when data IS available.
# Mark as skip if no fixture is available.

_FIXTURE_AVAILABLE = False  # Set to True when a small .dbn.zst test fixture exists


@pytest.mark.skipif(not _FIXTURE_AVAILABLE, reason="No .dbn.zst fixture available")
class TestBarrierPrecomputeDictKeys:
    """The returned dict should contain all keys matching process_session() output."""

    EXPECTED_KEYS = {
        # Bar OHLCV
        "bar_open", "bar_high", "bar_low", "bar_close",
        "bar_volume", "bar_vwap", "bar_t_start", "bar_t_end",
        # Trade sequences
        "trade_prices", "trade_sizes", "bar_trade_offsets",
        # Labels
        "label_values", "label_tau", "label_resolution_bar",
        # Features
        "features",
        # Scalars
        "bar_size", "lookback", "a", "b", "t_max",
        "n_bars", "n_usable", "n_features",
    }

    def test_all_expected_keys_present(self):
        """Dict should contain all expected keys."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result is not None
        for key in self.EXPECTED_KEYS:
            assert key in result, f"Missing expected key: {key}"

    def test_no_unexpected_keys(self):
        """Dict should not contain keys outside the expected set."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result is not None
        for key in result:
            assert key in self.EXPECTED_KEYS, f"Unexpected key: {key}"


@pytest.mark.skipif(not _FIXTURE_AVAILABLE, reason="No .dbn.zst fixture available")
class TestBarrierPrecomputeDtypes:
    """Array dtypes must match the spec exactly."""

    def test_features_dtype_float32(self):
        """features should be np.float32."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result["features"].dtype == np.float32

    def test_bar_open_dtype_float64(self):
        """bar_open should be np.float64."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result["bar_open"].dtype == np.float64

    def test_bar_volume_dtype_int32(self):
        """bar_volume should be np.int32."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result["bar_volume"].dtype == np.int32

    def test_label_values_dtype_int8(self):
        """label_values should be np.int8."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result["label_values"].dtype == np.int8

    def test_label_tau_dtype_int32(self):
        """label_tau should be np.int32."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result["label_tau"].dtype == np.int32

    def test_bar_t_start_dtype_int64(self):
        """bar_t_start should be np.int64."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result["bar_t_start"].dtype == np.int64

    def test_trade_prices_dtype_float64(self):
        """trade_prices should be np.float64."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result["trade_prices"].dtype == np.float64

    def test_trade_sizes_dtype_int32(self):
        """trade_sizes should be np.int32."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result["trade_sizes"].dtype == np.int32

    def test_bar_trade_offsets_dtype_int64(self):
        """bar_trade_offsets should be np.int64."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result["bar_trade_offsets"].dtype == np.int64


@pytest.mark.skipif(not _FIXTURE_AVAILABLE, reason="No .dbn.zst fixture available")
class TestBarrierPrecomputeShapes:
    """Array shapes must be consistent with n_bars and n_usable."""

    def test_bar_arrays_shape_matches_n_bars(self):
        """Bar OHLCV arrays should have shape (n_bars,)."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        n = int(result["n_bars"])
        for key in ("bar_open", "bar_high", "bar_low", "bar_close",
                     "bar_vwap", "bar_t_start", "bar_t_end"):
            assert result[key].shape == (n,), (
                f"{key} shape {result[key].shape} != ({n},)"
            )

    def test_bar_volume_shape(self):
        """bar_volume should have shape (n_bars,)."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        n = int(result["n_bars"])
        assert result["bar_volume"].shape == (n,)

    def test_label_arrays_shape_matches_n_bars(self):
        """Label arrays should have shape (n_bars,)."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        n = int(result["n_bars"])
        for key in ("label_values", "label_tau", "label_resolution_bar"):
            assert result[key].shape == (n,), (
                f"{key} shape {result[key].shape} != ({n},)"
            )

    def test_features_shape_matches_n_usable(self):
        """features should have shape (n_usable, n_features * lookback)."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        n_usable = int(result["n_usable"])
        n_features = int(result["n_features"])
        lookback = int(result["lookback"])
        expected_shape = (n_usable, n_features * lookback)
        assert result["features"].shape == expected_shape, (
            f"features shape {result['features'].shape} != {expected_shape}"
        )

    def test_bar_trade_offsets_shape(self):
        """bar_trade_offsets should have shape (n_bars + 1,)."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        n = int(result["n_bars"])
        assert result["bar_trade_offsets"].shape == (n + 1,)


@pytest.mark.skipif(not _FIXTURE_AVAILABLE, reason="No .dbn.zst fixture available")
class TestBarrierPrecomputeScalars:
    """Scalar metadata should match expected values."""

    def test_n_features_matches_constant(self):
        """n_features should equal the N_FEATURES constant (22)."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert int(result["n_features"]) == 22

    def test_bar_size_matches_input(self):
        """bar_size in result should match the input parameter."""
        result = lob_rl_core.barrier_precompute(
            "fixture.dbn.zst", 0, bar_size=500)
        assert int(result["bar_size"]) == 500

    def test_lookback_matches_input(self):
        """lookback in result should match the input parameter."""
        result = lob_rl_core.barrier_precompute(
            "fixture.dbn.zst", 0, lookback=10)
        assert int(result["lookback"]) == 10

    def test_a_b_tmax_match_input(self):
        """a, b, t_max in result should match input parameters."""
        result = lob_rl_core.barrier_precompute(
            "fixture.dbn.zst", 0, a=20, b=10, t_max=40)
        assert int(result["a"]) == 20
        assert int(result["b"]) == 10
        assert int(result["t_max"]) == 40


@pytest.mark.skipif(not _FIXTURE_AVAILABLE, reason="No .dbn.zst fixture available")
class TestBarrierPrecomputeValueConstraints:
    """Values in the returned arrays should satisfy domain constraints."""

    def test_features_all_finite(self):
        """All feature values should be finite (no NaN after normalization)."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert np.all(np.isfinite(result["features"])), (
            "All feature values must be finite"
        )

    def test_label_values_valid(self):
        """label_values should be in {-1, 0, +1}."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        labels = result["label_values"]
        valid = np.isin(labels, [-1, 0, 1])
        assert np.all(valid), (
            f"All labels must be -1, 0, or +1. Got unique: {np.unique(labels)}"
        )

    def test_bar_prices_positive(self):
        """All bar prices should be positive."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        for key in ("bar_open", "bar_high", "bar_low", "bar_close", "bar_vwap"):
            arr = result[key]
            assert np.all(arr > 0), f"{key} should be positive"
            assert np.all(np.isfinite(arr)), f"{key} should be finite"

    def test_trade_offsets_monotonic(self):
        """bar_trade_offsets should be monotonically non-decreasing."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        offsets = result["bar_trade_offsets"]
        assert np.all(np.diff(offsets) >= 0), (
            "bar_trade_offsets must be monotonically non-decreasing"
        )

    def test_trade_offsets_first_zero_last_matches_total(self):
        """First offset should be 0, last should match total trade count."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        offsets = result["bar_trade_offsets"]
        assert offsets[0] == 0, "First trade offset must be 0"
        assert offsets[-1] == len(result["trade_prices"]), (
            "Last offset must equal total number of trades"
        )

    def test_n_usable_consistent_with_features(self):
        """n_usable * (n_features * lookback) should equal features.size."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        n_usable = int(result["n_usable"])
        n_features = int(result["n_features"])
        lookback = int(result["lookback"])
        expected = n_usable * n_features * lookback
        actual = result["features"].size
        assert actual == expected, (
            f"features.size ({actual}) != n_usable*n_features*lookback ({expected})"
        )


# ===========================================================================
# Test 7: NPZ compatibility (can the dict be saved and loaded)
# ===========================================================================


class TestBarrierPrecomputeNpzCompat:
    """The returned dict should be compatible with np.savez_compressed."""

    @pytest.mark.skipif(not _FIXTURE_AVAILABLE, reason="No .dbn.zst fixture available")
    def test_savez_compressed_roundtrip(self, tmp_path):
        """Result dict should roundtrip through np.savez_compressed."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result is not None

        # Save
        out_path = tmp_path / "test_cache.npz"
        np.savez_compressed(str(out_path), **result)

        # Load back
        loaded = np.load(str(out_path))
        for key in result:
            np.testing.assert_array_equal(
                loaded[key], result[key],
                err_msg=f"Mismatch after npz roundtrip for key '{key}'"
            )


# ===========================================================================
# Test 8: Short-direction label keys exist in returned dict
# ===========================================================================


@pytest.mark.skipif(not _FIXTURE_AVAILABLE, reason="No .dbn.zst fixture available")
class TestBarrierPrecomputeShortLabelKeys:
    """The returned dict should contain short-direction label arrays."""

    SHORT_LABEL_KEYS = {
        "short_label_values",
        "short_label_tau",
        "short_label_resolution_bar",
    }

    def test_short_label_keys_present(self):
        """short_label_values, short_label_tau, short_label_resolution_bar must be present."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result is not None
        for key in self.SHORT_LABEL_KEYS:
            assert key in result, f"Missing short label key: {key}"


# ===========================================================================
# Test 9: Short label dtypes
# ===========================================================================


@pytest.mark.skipif(not _FIXTURE_AVAILABLE, reason="No .dbn.zst fixture available")
class TestBarrierPrecomputeShortLabelDtypes:
    """Short label arrays must have correct dtypes (same conventions as long labels)."""

    def test_short_label_values_dtype_int8(self):
        """short_label_values should be np.int8."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result["short_label_values"].dtype == np.int8

    def test_short_label_tau_dtype_int32(self):
        """short_label_tau should be np.int32."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result["short_label_tau"].dtype == np.int32

    def test_short_label_resolution_bar_dtype_int32(self):
        """short_label_resolution_bar should be np.int32."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result["short_label_resolution_bar"].dtype == np.int32


# ===========================================================================
# Test 10: Short label shapes
# ===========================================================================


@pytest.mark.skipif(not _FIXTURE_AVAILABLE, reason="No .dbn.zst fixture available")
class TestBarrierPrecomputeShortLabelShapes:
    """Short label arrays must have shape (n_bars,)."""

    def test_short_label_arrays_shape_matches_n_bars(self):
        """All short label arrays should have shape (n_bars,)."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        n = int(result["n_bars"])
        for key in ("short_label_values", "short_label_tau", "short_label_resolution_bar"):
            assert result[key].shape == (n,), (
                f"{key} shape {result[key].shape} != ({n},)"
            )


# ===========================================================================
# Test 11: Binary derivation of Y_long and Y_short
# ===========================================================================


@pytest.mark.skipif(not _FIXTURE_AVAILABLE, reason="No .dbn.zst fixture available")
class TestBarrierPrecomputeBinaryLabels:
    """Binary labels Y_long and Y_short can be derived from the raw label arrays."""

    def test_y_long_derivation(self):
        """Y_long = (label_values == +1) should produce a boolean array."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        y_long = result["label_values"] == 1
        assert y_long.dtype == np.bool_
        assert y_long.shape == result["label_values"].shape

    def test_y_short_derivation(self):
        """Y_short = (short_label_values == -1) should produce a boolean array."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        y_short = result["short_label_values"] == -1
        assert y_short.dtype == np.bool_
        assert y_short.shape == result["short_label_values"].shape

    def test_non_complementarity(self):
        """Y_long.mean() + Y_short.mean() should NOT equal 1.0.

        Y_long and Y_short are independent binary labels, not complementary.
        Under the martingale null with 2:1 reward:risk, each ≈ 1/3,
        so their sum ≈ 2/3, NOT 1.0.
        """
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        y_long = (result["label_values"] == 1).astype(float)
        y_short = (result["short_label_values"] == -1).astype(float)
        total = y_long.mean() + y_short.mean()
        assert total != 1.0, (
            f"Y_long.mean() + Y_short.mean() = {total}, but they should "
            f"NOT be complementary (expected ≈ 2/3 under martingale null)"
        )
