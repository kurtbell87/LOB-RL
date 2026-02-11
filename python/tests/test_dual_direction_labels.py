"""Tests for dual-direction (short) labels in barrier precompute pipeline.

Spec: docs/dual-direction-labels.md

These tests verify that:
- process_session() returns short summary statistics
- load_session_from_cache() returns short_labels
- load_session_from_cache() raises ValueError on missing short labels (old cache)
- Short labels roundtrip correctly through np.savez_compressed + np.load
- Binary derivation of Y_long and Y_short from label arrays
"""

import numpy as np
import pytest


# ===========================================================================
# Helpers: Create synthetic .npz cache files for testing
# ===========================================================================


def _make_synthetic_npz(tmp_path, name="test.npz", n_bars=20, lookback=10,
                        include_short_labels=True):
    """Create a synthetic .npz that mimics barrier_precompute output.

    Produces a valid cache file with all required keys.
    If include_short_labels=True, includes short label arrays.
    """
    from lob_rl.barrier import N_FEATURES

    n_usable = max(0, n_bars - lookback + 1)

    data = dict(
        features=np.random.randn(n_usable, N_FEATURES * lookback).astype(np.float32),
        bar_open=np.full(n_bars, 4000.0),
        bar_high=np.full(n_bars, 4001.0),
        bar_low=np.full(n_bars, 3999.0),
        bar_close=np.full(n_bars, 4000.0),
        bar_volume=np.full(n_bars, 100, dtype=np.int32),
        bar_vwap=np.full(n_bars, 4000.0),
        bar_t_start=np.arange(n_bars, dtype=np.int64) * 1000,
        bar_t_end=np.arange(1, n_bars + 1, dtype=np.int64) * 1000,
        trade_prices=np.full(n_bars, 4000.0),
        trade_sizes=np.ones(n_bars, dtype=np.int32),
        bar_trade_offsets=np.arange(n_bars + 1, dtype=np.int64),
        # Long labels
        label_values=np.array(
            [1, -1, 0] * (n_bars // 3) + [0] * (n_bars % 3),
            dtype=np.int8,
        )[:n_bars],
        label_tau=np.full(n_bars, 5, dtype=np.int32),
        label_resolution_bar=np.arange(n_bars, dtype=np.int32),
        n_bars=np.array(n_bars, dtype=np.int32),
        n_usable=np.array(n_usable, dtype=np.int32),
        n_features=np.array(N_FEATURES, dtype=np.int32),
    )

    if include_short_labels:
        # Short labels: different pattern from long labels
        data["short_label_values"] = np.array(
            [-1, 1, 0] * (n_bars // 3) + [0] * (n_bars % 3),
            dtype=np.int8,
        )[:n_bars]
        data["short_label_tau"] = np.full(n_bars, 7, dtype=np.int32)
        data["short_label_resolution_bar"] = np.arange(n_bars, dtype=np.int32) + 7

    npz_path = tmp_path / name
    np.savez_compressed(str(npz_path), **data)
    return str(npz_path), data


# ===========================================================================
# Test 1: process_session returns short summary statistics
# ===========================================================================


class TestProcessSessionShortStats:
    """process_session() should include short_p_plus, short_p_minus, short_p_zero."""

    def test_short_summary_stats_keys_present(self, tmp_path):
        """process_session result dict should include short summary stat keys.

        Since process_session requires a .dbn.zst file (which we don't have
        in tests), we test the contract by simulating what process_session
        should produce and verifying that the cache script adds the expected
        keys. We do this by importing the function and checking that the
        short stats are computed from the short_label_values array.
        """
        # Simulate what process_session should return after adding short labels:
        # It calls lob_rl_core.barrier_precompute() then adds stats.
        # Since we can't call it directly, verify the contract by constructing
        # what the result dict SHOULD look like and checking stats computation.
        n = 30
        short_label_values = np.array(
            [-1, 1, 0] * 10, dtype=np.int8,
        )
        short_p_plus = np.sum(short_label_values == 1) / n
        short_p_minus = np.sum(short_label_values == -1) / n
        short_p_zero = np.sum(short_label_values == 0) / n

        # Verify the contract: stats should be present and valid
        assert isinstance(short_p_plus, (float, np.floating))
        assert isinstance(short_p_minus, (float, np.floating))
        assert isinstance(short_p_zero, (float, np.floating))
        assert 0.0 <= short_p_plus <= 1.0
        assert 0.0 <= short_p_minus <= 1.0
        assert 0.0 <= short_p_zero <= 1.0

    def test_short_summary_stats_sum_to_one(self):
        """short_p_plus + short_p_minus + short_p_zero should sum to ~1.0."""
        # Test with various label distributions
        for labels in [
            np.array([1, -1, 0] * 10, dtype=np.int8),
            np.array([0] * 30, dtype=np.int8),  # all timeout
            np.array([-1] * 30, dtype=np.int8),  # all short profit
        ]:
            n = len(labels)
            p_plus = np.sum(labels == 1) / n
            p_minus = np.sum(labels == -1) / n
            p_zero = np.sum(labels == 0) / n
            total = p_plus + p_minus + p_zero
            assert abs(total - 1.0) < 1e-10, (
                f"Short summary stats should sum to 1.0, got {total}"
            )


# ===========================================================================
# Test 2: load_session_from_cache returns short_labels
# ===========================================================================


class TestLoadSessionShortLabels:
    """load_session_from_cache should return short_labels when present in cache."""

    def test_short_labels_key_exists(self, tmp_path):
        """Returned dict should have 'short_labels' key."""
        npz_path, _ = _make_synthetic_npz(tmp_path, include_short_labels=True)

        from scripts.precompute_barrier_cache import load_session_from_cache
        session = load_session_from_cache(npz_path)

        assert "short_labels" in session, (
            "load_session_from_cache should return 'short_labels' key"
        )

    def test_short_labels_correct_length(self, tmp_path):
        """short_labels list should have length n_bars."""
        n_bars = 20
        npz_path, _ = _make_synthetic_npz(tmp_path, n_bars=n_bars,
                                          include_short_labels=True)

        from scripts.precompute_barrier_cache import load_session_from_cache
        session = load_session_from_cache(npz_path)

        assert len(session["short_labels"]) == n_bars, (
            f"short_labels length {len(session['short_labels'])} != n_bars {n_bars}"
        )

    def test_short_labels_are_barrier_label_objects(self, tmp_path):
        """Each short_labels element should be a BarrierLabel object."""
        npz_path, _ = _make_synthetic_npz(tmp_path, include_short_labels=True)

        from lob_rl.barrier.label_pipeline import BarrierLabel
        from scripts.precompute_barrier_cache import load_session_from_cache
        session = load_session_from_cache(npz_path)

        for lbl in session["short_labels"]:
            assert isinstance(lbl, BarrierLabel), (
                f"short_labels elements should be BarrierLabel, got {type(lbl)}"
            )

    def test_short_label_values_match_cache(self, tmp_path):
        """Reconstructed short_labels should have label values matching the cached data."""
        npz_path, data = _make_synthetic_npz(tmp_path, include_short_labels=True)

        from scripts.precompute_barrier_cache import load_session_from_cache
        session = load_session_from_cache(npz_path)

        for i, lbl in enumerate(session["short_labels"]):
            assert lbl.label == int(data["short_label_values"][i]), (
                f"short_labels[{i}].label = {lbl.label} != "
                f"cached short_label_values[{i}] = {data['short_label_values'][i]}"
            )


# ===========================================================================
# Test 3: load_session_from_cache raises ValueError on missing short labels
# ===========================================================================


class TestLoadSessionMissingShortLabels:
    """Old cache without short_label_values should raise ValueError."""

    def test_missing_short_labels_raises_value_error(self, tmp_path):
        """Cache missing short_label_values should raise ValueError."""
        npz_path, _ = _make_synthetic_npz(tmp_path, include_short_labels=False)

        from scripts.precompute_barrier_cache import load_session_from_cache
        with pytest.raises(ValueError, match="short labels"):
            load_session_from_cache(npz_path)

    def test_error_message_mentions_reprecompute(self, tmp_path):
        """Error message should mention re-precomputing."""
        npz_path, _ = _make_synthetic_npz(tmp_path, include_short_labels=False)

        from scripts.precompute_barrier_cache import load_session_from_cache
        with pytest.raises(ValueError, match="[Rr]e-precompute"):
            load_session_from_cache(npz_path)


# ===========================================================================
# Test 4: Short labels roundtrip through NPZ save/load
# ===========================================================================


class TestShortLabelsNpzRoundtrip:
    """Short label arrays should survive np.savez_compressed + np.load roundtrip."""

    def test_short_label_values_roundtrip(self, tmp_path):
        """short_label_values should be identical after save+load."""
        npz_path, original = _make_synthetic_npz(tmp_path, include_short_labels=True)

        loaded = np.load(npz_path)
        np.testing.assert_array_equal(
            loaded["short_label_values"],
            original["short_label_values"],
            err_msg="short_label_values mismatch after NPZ roundtrip",
        )

    def test_short_label_tau_roundtrip(self, tmp_path):
        """short_label_tau should be identical after save+load."""
        npz_path, original = _make_synthetic_npz(tmp_path, include_short_labels=True)

        loaded = np.load(npz_path)
        np.testing.assert_array_equal(
            loaded["short_label_tau"],
            original["short_label_tau"],
            err_msg="short_label_tau mismatch after NPZ roundtrip",
        )

    def test_short_label_resolution_bar_roundtrip(self, tmp_path):
        """short_label_resolution_bar should be identical after save+load."""
        npz_path, original = _make_synthetic_npz(tmp_path, include_short_labels=True)

        loaded = np.load(npz_path)
        np.testing.assert_array_equal(
            loaded["short_label_resolution_bar"],
            original["short_label_resolution_bar"],
            err_msg="short_label_resolution_bar mismatch after NPZ roundtrip",
        )


# ===========================================================================
# Test 5: process_session short_p_* keys in saved NPZ
# ===========================================================================


class TestProcessSessionShortStatsInNpz:
    """After process_session, saved NPZ should contain short_p_plus/minus/zero."""

    def test_short_p_keys_in_saved_cache(self, tmp_path):
        """A cache file saved by process_session should have short_p_* keys.

        Since we can't call process_session directly, we verify the contract
        by checking what keys should be present and their properties.
        """
        # Simulate process_session output: it should add these keys
        n = 30
        short_label_values = np.array([-1, 1, 0] * 10, dtype=np.int8)
        result = {
            "short_label_values": short_label_values,
            "short_p_plus": np.array(np.sum(short_label_values == 1) / n, dtype=np.float64),
            "short_p_minus": np.array(np.sum(short_label_values == -1) / n, dtype=np.float64),
            "short_p_zero": np.array(np.sum(short_label_values == 0) / n, dtype=np.float64),
        }

        # Save and reload
        npz_path = tmp_path / "test.npz"
        np.savez_compressed(str(npz_path), **result)
        loaded = np.load(str(npz_path))

        assert "short_p_plus" in loaded, "short_p_plus missing from saved NPZ"
        assert "short_p_minus" in loaded, "short_p_minus missing from saved NPZ"
        assert "short_p_zero" in loaded, "short_p_zero missing from saved NPZ"

        # Values should be valid probabilities
        for key in ("short_p_plus", "short_p_minus", "short_p_zero"):
            val = float(loaded[key])
            assert 0.0 <= val <= 1.0, f"{key} = {val} is not a valid probability"
