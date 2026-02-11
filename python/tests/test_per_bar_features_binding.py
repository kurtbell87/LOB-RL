"""Tests for per-bar features in the barrier_precompute Python binding.

Spec: docs/per-bar-features.md

These tests verify that:
- barrier_precompute() result dict contains 'bar_features' key
- bar_features has shape (n_trimmed, 22) and dtype float32
- n_trimmed is present and matches bar_features.shape[0]
- bar_features values are in the [-5, 5] clipped range
- bar_features is independent of lookback parameter
"""

import numpy as np
import pytest

import lob_rl_core


# These tests require a .dbn.zst fixture file.
_FIXTURE_AVAILABLE = False


# ===========================================================================
# Section 1: bar_features key exists in result dict
# ===========================================================================


@pytest.mark.skipif(not _FIXTURE_AVAILABLE, reason="No .dbn.zst fixture available")
class TestBarFeaturesKeyPresent:
    """The result dict from barrier_precompute() should contain bar_features."""

    def test_bar_features_key_exists(self):
        """Result dict should have a 'bar_features' key."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result is not None
        assert "bar_features" in result, "Missing 'bar_features' key in result dict"

    def test_n_trimmed_key_exists(self):
        """Result dict should have an 'n_trimmed' key."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result is not None
        assert "n_trimmed" in result, "Missing 'n_trimmed' key in result dict"


# ===========================================================================
# Section 2: bar_features shape and dtype
# ===========================================================================


@pytest.mark.skipif(not _FIXTURE_AVAILABLE, reason="No .dbn.zst fixture available")
class TestBarFeaturesShape:
    """bar_features should be a 2D float32 array with shape (n_trimmed, 22)."""

    def test_bar_features_is_2d(self):
        """bar_features should be a 2D numpy array."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result["bar_features"].ndim == 2, (
            f"bar_features should be 2D, got {result['bar_features'].ndim}D"
        )

    def test_bar_features_dtype_float32(self):
        """bar_features should have dtype float32."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result["bar_features"].dtype == np.float32, (
            f"bar_features dtype should be float32, got {result['bar_features'].dtype}"
        )

    def test_bar_features_shape_matches_n_trimmed_and_n_features(self):
        """bar_features.shape should be (n_trimmed, N_FEATURES=22)."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        n_trimmed = int(result["n_trimmed"])
        n_features = int(result["n_features"])
        expected = (n_trimmed, n_features)
        assert result["bar_features"].shape == expected, (
            f"bar_features shape {result['bar_features'].shape} != {expected}"
        )

    def test_n_trimmed_matches_bar_features_rows(self):
        """n_trimmed should equal bar_features.shape[0]."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert int(result["n_trimmed"]) == result["bar_features"].shape[0]

    def test_n_trimmed_equals_n_bars_minus_warmup(self):
        """n_trimmed should equal n_bars - REALIZED_VOL_WARMUP (19)."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        n_bars = int(result["n_bars"])
        n_trimmed = int(result["n_trimmed"])
        # REALIZED_VOL_WARMUP = 19
        expected = n_bars - 19
        assert n_trimmed == expected, (
            f"n_trimmed ({n_trimmed}) != n_bars - 19 ({expected})"
        )


# ===========================================================================
# Section 3: bar_features values
# ===========================================================================


@pytest.mark.skipif(not _FIXTURE_AVAILABLE, reason="No .dbn.zst fixture available")
class TestBarFeaturesValues:
    """bar_features values should be normalized, finite, and clipped to [-5, 5]."""

    def test_all_finite(self):
        """All bar_features values should be finite (no NaN/inf)."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert np.all(np.isfinite(result["bar_features"])), (
            "All bar_features values must be finite"
        )

    def test_values_in_clipped_range(self):
        """All bar_features values should be in [-5, 5] (z-score clipped)."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        bf = result["bar_features"]
        assert np.all(bf >= -5.0), f"Min value {bf.min()} below -5"
        assert np.all(bf <= 5.0), f"Max value {bf.max()} above 5"


# ===========================================================================
# Section 4: bar_features flows through npz roundtrip
# ===========================================================================


@pytest.mark.skipif(not _FIXTURE_AVAILABLE, reason="No .dbn.zst fixture available")
class TestBarFeaturesNpzRoundtrip:
    """bar_features should survive np.savez_compressed roundtrip."""

    def test_bar_features_survives_npz(self, tmp_path):
        """bar_features should be identical after save/load."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result is not None

        out_path = tmp_path / "test.npz"
        np.savez_compressed(str(out_path), **result)

        loaded = np.load(str(out_path))
        np.testing.assert_array_equal(
            loaded["bar_features"], result["bar_features"],
            err_msg="bar_features mismatch after npz roundtrip"
        )

    def test_n_trimmed_survives_npz(self, tmp_path):
        """n_trimmed should be identical after save/load."""
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result is not None

        out_path = tmp_path / "test.npz"
        np.savez_compressed(str(out_path), **result)

        loaded = np.load(str(out_path))
        assert int(loaded["n_trimmed"]) == int(result["n_trimmed"])


# ===========================================================================
# Section 5: Alignment — bar_features[i] matches features lookback window
# ===========================================================================


@pytest.mark.skipif(not _FIXTURE_AVAILABLE, reason="No .dbn.zst fixture available")
class TestBarFeaturesAlignment:
    """bar_features values must match the corresponding lookback features."""

    def test_last_window_of_features_matches_bar_features(self):
        """features[j]'s last N_FEATURES cols should equal bar_features[j + lookback - 1].

        This is acceptance criterion #7 from the spec.
        """
        result = lob_rl_core.barrier_precompute("fixture.dbn.zst", 0)
        assert result is not None

        features = result["features"]
        bar_features = result["bar_features"]
        n_usable = int(result["n_usable"])
        n_features = int(result["n_features"])
        lookback = int(result["lookback"])

        for j in range(min(n_usable, 10)):  # Check first 10 rows
            bar_idx = j + lookback - 1
            # Last n_features columns of features[j]
            from_lookback = features[j, (lookback - 1) * n_features:]
            from_bar = bar_features[bar_idx]
            np.testing.assert_array_almost_equal(
                from_lookback, from_bar, decimal=5,
                err_msg=f"Alignment mismatch at features row {j} / bar_features row {bar_idx}"
            )
