"""Tests for load_session_features() and temporal_split() for session dicts.

Spec: docs/per-bar-features.md

These tests verify:
- load_session_features() returns list of session dicts with correct keys
- Features shape is (n_bars, 22) per session
- Y_long and Y_short are boolean arrays aligned with features
- Sessions are sorted chronologically
- Sessions with missing 'bar_features' key are skipped
- Sessions with n_trimmed == 0 are skipped
- Label alignment: bar_features[0] corresponds to label_values[WARMUP]
- temporal_split() works with session dict lists
- temporal_split() preserves chronological order
"""

import os

import numpy as np
import pytest

# REALIZED_VOL_WARMUP from the C++ constants
WARMUP = 19
N_FEATURES = 22


# ===========================================================================
# Fixtures: Create synthetic .npz cache files
# ===========================================================================


def _make_session_npz(
    path,
    n_bars=50,
    n_features=N_FEATURES,
    include_bar_features=True,
    warmup=WARMUP,
    seed=42,
):
    """Create a synthetic barrier cache .npz file."""
    rng = np.random.default_rng(seed)

    n_trimmed = max(0, n_bars - warmup)
    lookback = 10
    n_usable = max(0, n_trimmed - lookback + 1)

    data = {
        "n_bars": np.array(n_bars),
        "n_usable": np.array(n_usable),
        "n_features": np.array(n_features),
        "n_trimmed": np.array(n_trimmed),
        "lookback": np.array(lookback),
        "bar_size": np.array(500),
        "a": np.array(20),
        "b": np.array(10),
        "t_max": np.array(40),
        # Features (lookback-assembled)
        "features": rng.standard_normal((n_usable, n_features * lookback)).astype(np.float32),
        # Labels (size n_bars)
        "label_values": rng.choice([-1, 0, 1], size=n_bars).astype(np.int8),
        "label_tau": rng.integers(1, 40, size=n_bars).astype(np.int32),
        "label_resolution_bar": np.arange(n_bars, dtype=np.int32),
        "short_label_values": rng.choice([-1, 0, 1], size=n_bars).astype(np.int8),
        "short_label_tau": rng.integers(1, 40, size=n_bars).astype(np.int32),
        "short_label_resolution_bar": np.arange(n_bars, dtype=np.int32),
        # Bar data (dummy)
        "bar_open": rng.uniform(4000, 4010, n_bars),
        "bar_high": rng.uniform(4005, 4015, n_bars),
        "bar_low": rng.uniform(3995, 4005, n_bars),
        "bar_close": rng.uniform(4000, 4010, n_bars),
        "bar_vwap": rng.uniform(4000, 4010, n_bars),
        "bar_volume": rng.integers(100, 1000, n_bars).astype(np.int32),
        "bar_t_start": np.arange(n_bars, dtype=np.int64) * 1_000_000_000,
        "bar_t_end": np.arange(1, n_bars + 1, dtype=np.int64) * 1_000_000_000,
        "trade_prices": rng.uniform(4000, 4010, n_bars * 5),
        "trade_sizes": rng.integers(1, 10, n_bars * 5).astype(np.int32),
        "bar_trade_offsets": np.arange(0, (n_bars + 1) * 5, 5, dtype=np.int64),
    }

    if include_bar_features and n_trimmed > 0:
        data["bar_features"] = rng.standard_normal((n_trimmed, n_features)).astype(np.float32)

    np.savez_compressed(path, **data)


@pytest.fixture
def cache_dir(tmp_path):
    """Create a cache directory with 5 synthetic session .npz files."""
    dates = ["20220103", "20220104", "20220105", "20220106", "20220107"]
    for i, date in enumerate(dates):
        fname = f"barrier-{date}.npz"
        _make_session_npz(tmp_path / fname, n_bars=50, seed=42 + i)
    return str(tmp_path)


@pytest.fixture
def cache_dir_with_gaps(tmp_path):
    """Cache with mixed sessions: some have bar_features, some don't."""
    # 3 good sessions
    for i, date in enumerate(["20220103", "20220104", "20220105"]):
        _make_session_npz(tmp_path / f"barrier-{date}.npz", n_bars=50, seed=42 + i)

    # 1 session WITHOUT bar_features (old cache format)
    _make_session_npz(
        tmp_path / "barrier-20220106.npz",
        n_bars=50,
        include_bar_features=False,
        seed=100,
    )

    # 1 session with n_trimmed == 0 (not enough bars)
    _make_session_npz(
        tmp_path / "barrier-20220107.npz",
        n_bars=15,  # < WARMUP (19), so n_trimmed = 0
        seed=200,
    )

    return str(tmp_path)


@pytest.fixture
def empty_cache_dir(tmp_path):
    """Cache directory with no .npz files."""
    return str(tmp_path)


# ===========================================================================
# Section 1: load_session_features() basic contract
# ===========================================================================


class TestLoadSessionFeaturesBasic:
    """load_session_features() should return a list of session dicts."""

    def test_returns_list(self, cache_dir):
        from lob_rl.barrier.first_passage_analysis import load_session_features

        sessions = load_session_features(cache_dir)
        assert isinstance(sessions, list)

    def test_returns_nonempty_for_valid_cache(self, cache_dir):
        from lob_rl.barrier.first_passage_analysis import load_session_features

        sessions = load_session_features(cache_dir)
        assert len(sessions) > 0

    def test_each_session_is_dict(self, cache_dir):
        from lob_rl.barrier.first_passage_analysis import load_session_features

        sessions = load_session_features(cache_dir)
        for s in sessions:
            assert isinstance(s, dict)

    def test_required_keys_present(self, cache_dir):
        from lob_rl.barrier.first_passage_analysis import load_session_features

        sessions = load_session_features(cache_dir)
        required_keys = {"features", "Y_long", "Y_short", "date"}
        for s in sessions:
            for key in required_keys:
                assert key in s, f"Missing key '{key}' in session dict"


# ===========================================================================
# Section 2: Feature shape and dtype
# ===========================================================================


class TestLoadSessionFeaturesShape:
    """Per-session features should be (n_bars, 22) float32."""

    def test_features_is_2d(self, cache_dir):
        from lob_rl.barrier.first_passage_analysis import load_session_features

        sessions = load_session_features(cache_dir)
        for s in sessions:
            assert s["features"].ndim == 2, (
                f"features should be 2D, got {s['features'].ndim}D"
            )

    def test_features_has_n_features_columns(self, cache_dir):
        from lob_rl.barrier.first_passage_analysis import load_session_features

        sessions = load_session_features(cache_dir)
        for s in sessions:
            assert s["features"].shape[1] == N_FEATURES, (
                f"features should have {N_FEATURES} columns, got {s['features'].shape[1]}"
            )

    def test_features_dtype_float32(self, cache_dir):
        from lob_rl.barrier.first_passage_analysis import load_session_features

        sessions = load_session_features(cache_dir)
        for s in sessions:
            assert s["features"].dtype == np.float32


# ===========================================================================
# Section 3: Label arrays aligned with features
# ===========================================================================


class TestLoadSessionFeaturesLabelAlignment:
    """Y_long and Y_short must be boolean arrays aligned with features."""

    def test_y_long_is_boolean(self, cache_dir):
        from lob_rl.barrier.first_passage_analysis import load_session_features

        sessions = load_session_features(cache_dir)
        for s in sessions:
            assert s["Y_long"].dtype == np.bool_

    def test_y_short_is_boolean(self, cache_dir):
        from lob_rl.barrier.first_passage_analysis import load_session_features

        sessions = load_session_features(cache_dir)
        for s in sessions:
            assert s["Y_short"].dtype == np.bool_

    def test_label_length_matches_features(self, cache_dir):
        from lob_rl.barrier.first_passage_analysis import load_session_features

        sessions = load_session_features(cache_dir)
        for s in sessions:
            n = s["features"].shape[0]
            assert len(s["Y_long"]) == n, (
                f"Y_long length {len(s['Y_long'])} != features rows {n}"
            )
            assert len(s["Y_short"]) == n, (
                f"Y_short length {len(s['Y_short'])} != features rows {n}"
            )

    def test_label_alignment_with_warmup_trim(self, tmp_path):
        """bar_features[0] should correspond to label_values[WARMUP].

        The loader must trim the first WARMUP label rows to align with
        bar_features, which starts after warmup.
        """
        from lob_rl.barrier.first_passage_analysis import load_session_features

        # Create a session where we can verify alignment
        n_bars = 50
        n_trimmed = n_bars - WARMUP  # 31

        # Set specific label pattern we can verify
        label_values = np.zeros(n_bars, dtype=np.int8)
        label_values[WARMUP] = 1  # First post-warmup bar has label +1
        label_values[WARMUP + 1] = -1  # Second post-warmup bar has label -1

        short_label_values = np.zeros(n_bars, dtype=np.int8)
        short_label_values[WARMUP] = -1  # First post-warmup bar has short label -1

        data = {
            "n_bars": np.array(n_bars),
            "n_trimmed": np.array(n_trimmed),
            "n_features": np.array(N_FEATURES),
            "n_usable": np.array(n_trimmed - 10 + 1),
            "lookback": np.array(10),
            "bar_size": np.array(500),
            "a": np.array(20),
            "b": np.array(10),
            "t_max": np.array(40),
            "features": np.zeros((n_trimmed - 10 + 1, N_FEATURES * 10), dtype=np.float32),
            "bar_features": np.zeros((n_trimmed, N_FEATURES), dtype=np.float32),
            "label_values": label_values,
            "label_tau": np.ones(n_bars, dtype=np.int32),
            "label_resolution_bar": np.arange(n_bars, dtype=np.int32),
            "short_label_values": short_label_values,
            "short_label_tau": np.ones(n_bars, dtype=np.int32),
            "short_label_resolution_bar": np.arange(n_bars, dtype=np.int32),
            "bar_open": np.ones(n_bars),
            "bar_high": np.ones(n_bars),
            "bar_low": np.ones(n_bars),
            "bar_close": np.ones(n_bars),
            "bar_vwap": np.ones(n_bars),
            "bar_volume": np.ones(n_bars, dtype=np.int32),
            "bar_t_start": np.arange(n_bars, dtype=np.int64),
            "bar_t_end": np.arange(n_bars, dtype=np.int64),
            "trade_prices": np.ones(10),
            "trade_sizes": np.ones(10, dtype=np.int32),
            "bar_trade_offsets": np.arange(0, n_bars + 1, dtype=np.int64),
        }

        np.savez_compressed(tmp_path / "barrier-20220103.npz", **data)

        sessions = load_session_features(str(tmp_path))
        assert len(sessions) == 1

        s = sessions[0]
        # After trimming first WARMUP labels:
        # Y_long[0] should correspond to label_values[WARMUP] == 1
        assert s["Y_long"][0] == True, (
            "Y_long[0] should be True (label_values[WARMUP] == 1)"
        )
        # Y_long[1] should be False (label_values[WARMUP+1] == -1, not +1)
        assert s["Y_long"][1] == False

        # Y_short[0] should be True (short_label_values[WARMUP] == -1)
        assert s["Y_short"][0] == True, (
            "Y_short[0] should be True (short_label_values[WARMUP] == -1)"
        )


# ===========================================================================
# Section 4: Chronological sorting
# ===========================================================================


class TestLoadSessionFeaturesSorting:
    """Sessions should be sorted chronologically by date."""

    def test_sessions_sorted_by_date(self, cache_dir):
        from lob_rl.barrier.first_passage_analysis import load_session_features

        sessions = load_session_features(cache_dir)
        dates = [s["date"] for s in sessions]
        assert dates == sorted(dates), (
            f"Sessions should be sorted chronologically: {dates}"
        )

    def test_date_format_is_string(self, cache_dir):
        from lob_rl.barrier.first_passage_analysis import load_session_features

        sessions = load_session_features(cache_dir)
        for s in sessions:
            assert isinstance(s["date"], str)
            assert len(s["date"]) == 8, (
                f"Date should be YYYYMMDD format, got '{s['date']}'"
            )


# ===========================================================================
# Section 5: Skipping sessions with missing or empty bar_features
# ===========================================================================


class TestLoadSessionFeaturesSkipping:
    """Sessions without bar_features or with n_trimmed==0 should be skipped."""

    def test_missing_bar_features_skipped(self, cache_dir_with_gaps):
        from lob_rl.barrier.first_passage_analysis import load_session_features

        sessions = load_session_features(cache_dir_with_gaps)
        # We created 5 files: 3 good, 1 missing bar_features, 1 with n_trimmed=0
        # Should return only 3 good sessions
        assert len(sessions) == 3, (
            f"Expected 3 sessions (skipping 2 bad ones), got {len(sessions)}"
        )

    def test_zero_trimmed_skipped(self, cache_dir_with_gaps):
        from lob_rl.barrier.first_passage_analysis import load_session_features

        sessions = load_session_features(cache_dir_with_gaps)
        dates = [s["date"] for s in sessions]
        assert "20220107" not in dates, (
            "Session with n_trimmed=0 should be skipped"
        )

    def test_old_format_without_bar_features_skipped(self, cache_dir_with_gaps):
        from lob_rl.barrier.first_passage_analysis import load_session_features

        sessions = load_session_features(cache_dir_with_gaps)
        dates = [s["date"] for s in sessions]
        assert "20220106" not in dates, (
            "Session without 'bar_features' key should be skipped"
        )

    def test_empty_cache_raises(self, empty_cache_dir):
        from lob_rl.barrier.first_passage_analysis import load_session_features

        with pytest.raises(ValueError, match="No .npz files|No valid sessions"):
            load_session_features(empty_cache_dir)


# ===========================================================================
# Section 6: temporal_split() with session dict lists
# ===========================================================================


class TestTemporalSplitWithSessions:
    """temporal_split() should work with session dict lists."""

    def test_basic_split_returns_three_lists(self, cache_dir):
        from lob_rl.barrier.first_passage_analysis import (
            load_session_features,
            temporal_split,
        )

        sessions = load_session_features(cache_dir)
        n = len(sessions)
        train_idx, val_idx, test_idx = temporal_split(n)

        assert isinstance(train_idx, np.ndarray)
        assert isinstance(val_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)

    def test_split_covers_all_sessions(self, cache_dir):
        from lob_rl.barrier.first_passage_analysis import (
            load_session_features,
            temporal_split,
        )

        sessions = load_session_features(cache_dir)
        n = len(sessions)
        train_idx, val_idx, test_idx = temporal_split(n)

        all_idx = np.concatenate([train_idx, val_idx, test_idx])
        assert len(np.unique(all_idx)) == n, (
            "Split should cover all sessions exactly once"
        )

    def test_split_preserves_chronological_order(self, cache_dir):
        from lob_rl.barrier.first_passage_analysis import (
            load_session_features,
            temporal_split,
        )

        sessions = load_session_features(cache_dir)
        n = len(sessions)
        train_idx, val_idx, test_idx = temporal_split(n)

        # Train indices should come before val, val before test
        if len(train_idx) > 0 and len(val_idx) > 0:
            assert train_idx.max() < val_idx.min(), (
                "Train indices should all be before val indices"
            )
        if len(val_idx) > 0 and len(test_idx) > 0:
            assert val_idx.max() < test_idx.min(), (
                "Val indices should all be before test indices"
            )

    def test_default_split_is_60_20_20(self):
        from lob_rl.barrier.first_passage_analysis import temporal_split

        # With 100 sessions, should split ~60/20/20
        train_idx, val_idx, test_idx = temporal_split(100)
        assert len(train_idx) == 60
        assert len(val_idx) == 20
        assert len(test_idx) == 20

    def test_custom_split_fractions(self):
        from lob_rl.barrier.first_passage_analysis import temporal_split

        train_idx, val_idx, test_idx = temporal_split(
            100, train_frac=0.8, val_frac=0.1
        )
        assert len(train_idx) == 80
        assert len(val_idx) == 10
        assert len(test_idx) == 10

    def test_single_session(self):
        from lob_rl.barrier.first_passage_analysis import temporal_split

        train_idx, val_idx, test_idx = temporal_split(1)
        assert len(train_idx) == 1
        assert len(val_idx) == 0
        assert len(test_idx) == 0

    def test_split_no_duplicates(self):
        from lob_rl.barrier.first_passage_analysis import temporal_split

        train_idx, val_idx, test_idx = temporal_split(50)
        all_idx = np.concatenate([train_idx, val_idx, test_idx])
        assert len(all_idx) == len(np.unique(all_idx)), (
            "No session should appear in multiple splits"
        )

    def test_indices_are_contiguous_ranges(self):
        """Each split should be a contiguous range of indices (temporal order)."""
        from lob_rl.barrier.first_passage_analysis import temporal_split

        train_idx, val_idx, test_idx = temporal_split(50)

        for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            if len(idx) > 1:
                diffs = np.diff(idx)
                assert np.all(diffs == 1), (
                    f"{name} indices should be contiguous: {idx}"
                )
