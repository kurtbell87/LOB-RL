"""Tests for scripts/precompute_cache.py — CLI tool to cache precomputed data.

Spec: docs/precompute-cache.md

These tests verify that:
- precompute_cache.py reads manifest.json from --data-dir
- Calls lob_rl_core.precompute() once per day
- Saves each day as {date}.npz with obs, mid, spread arrays
- Skips days that already exist in cache (unless --force)
- Prints summary (days cached, array shapes, total size)
- Default session config is default_rth()
- Holiday .bin files that produce 0 steps are not cached (skipped)
"""

import json
import os
import sys
import tempfile
import importlib.util

import numpy as np
import pytest

import lob_rl_core

# Fixture paths
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
DAY_FILES = [os.path.join(FIXTURE_DIR, f"day{i}.bin") for i in range(5)]

# Path to the CLI script under test
_SCRIPT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "precompute_cache.py")
)


def _run_precompute_cache(argv):
    """Load and run precompute_cache.py's main() with the given CLI argv list."""
    spec = importlib.util.spec_from_file_location("precompute_cache", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old_argv = sys.argv
    try:
        sys.argv = ["precompute_cache.py"] + argv
        mod.main()
    finally:
        sys.argv = old_argv


# ===========================================================================
# Helper: create a temp manifest.json pointing to fixture files
# ===========================================================================

def _make_manifest_dir(day_files, tmpdir, dates=None):
    """Create a data dir with symlinked .bin files and a manifest.json."""
    data_dir = os.path.join(tmpdir, "data")
    os.makedirs(data_dir, exist_ok=True)

    if dates is None:
        dates = [f"2025-01-{i + 10:02d}" for i in range(len(day_files))]

    manifest_entries = []
    for date, src in zip(dates, day_files):
        dst = os.path.join(data_dir, f"{date}.bin")
        os.symlink(os.path.abspath(src), dst)
        manifest_entries.append({"date": date, "record_count": 1000})

    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"files": manifest_entries}, f)

    return data_dir, dates


# ===========================================================================
# Test 1: precompute_cache.py script exists and is importable
# ===========================================================================


class TestScriptExists:
    """precompute_cache.py should exist as a runnable script."""

    def test_script_file_exists(self):
        """scripts/precompute_cache.py should exist on disk."""
        assert os.path.exists(_SCRIPT_PATH), (
            f"Script not found at {_SCRIPT_PATH}"
        )

    def test_script_is_readable(self):
        """scripts/precompute_cache.py should be readable."""
        with open(_SCRIPT_PATH) as f:
            source = f.read()
        assert len(source) > 0, "Script is empty"


# ===========================================================================
# Test 2: precompute_cache.py creates .npz files
# ===========================================================================


class TestCreatesNpzFiles:
    """precompute_cache.py should create one .npz file per valid trading day."""

    def test_creates_npz_per_day(self):
        """Running the script should create one .npz per day in --out dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir, dates = _make_manifest_dir(DAY_FILES[:3], tmpdir)
            out_dir = os.path.join(tmpdir, "cache")

            _run_precompute_cache(["--data-dir", data_dir, "--out", out_dir])

            # Check that .npz files were created
            npz_files = sorted([f for f in os.listdir(out_dir) if f.endswith(".npz")])
            assert len(npz_files) == 3, (
                f"Expected 3 .npz files, got {len(npz_files)}: {npz_files}"
            )

            # Each file should be named {date}.npz
            for date in dates:
                expected = f"{date}.npz"
                assert expected in npz_files, (
                    f"Expected {expected} in cache dir, got {npz_files}"
                )


# ===========================================================================
# Test 3: .npz file contains correct keys and dtypes
# ===========================================================================


class TestNpzFormat:
    """.npz files should contain obs, mid, spread with correct dtypes/shapes."""

    def _run_cache(self, tmpdir, day_files=None):
        """Helper: run precompute_cache and return (out_dir, dates)."""
        if day_files is None:
            day_files = DAY_FILES[:1]
        data_dir, dates = _make_manifest_dir(day_files, tmpdir)
        out_dir = os.path.join(tmpdir, "cache")

        _run_precompute_cache(["--data-dir", data_dir, "--out", out_dir])

        return out_dir, dates

    def test_npz_has_obs_key(self):
        """Each .npz should contain an 'obs' key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir, dates = self._run_cache(tmpdir)
            data = np.load(os.path.join(out_dir, f"{dates[0]}.npz"))
            assert "obs" in data, f"Missing 'obs' key, found: {list(data.keys())}"

    def test_npz_has_mid_key(self):
        """Each .npz should contain a 'mid' key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir, dates = self._run_cache(tmpdir)
            data = np.load(os.path.join(out_dir, f"{dates[0]}.npz"))
            assert "mid" in data, f"Missing 'mid' key, found: {list(data.keys())}"

    def test_npz_has_spread_key(self):
        """Each .npz should contain a 'spread' key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir, dates = self._run_cache(tmpdir)
            data = np.load(os.path.join(out_dir, f"{dates[0]}.npz"))
            assert "spread" in data, f"Missing 'spread' key, found: {list(data.keys())}"

    def test_obs_dtype_float32(self):
        """obs array should be float32."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir, dates = self._run_cache(tmpdir)
            data = np.load(os.path.join(out_dir, f"{dates[0]}.npz"))
            assert data["obs"].dtype == np.float32, f"Expected float32, got {data['obs'].dtype}"

    def test_obs_shape_n_by_43(self):
        """obs array should have shape (N, 43)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir, dates = self._run_cache(tmpdir)
            data = np.load(os.path.join(out_dir, f"{dates[0]}.npz"))
            assert data["obs"].ndim == 2, f"Expected 2D, got {data['obs'].ndim}D"
            assert data["obs"].shape[1] == 43, f"Expected 43 cols, got {data['obs'].shape[1]}"

    def test_mid_dtype_float64(self):
        """mid array should be float64."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir, dates = self._run_cache(tmpdir)
            data = np.load(os.path.join(out_dir, f"{dates[0]}.npz"))
            assert data["mid"].dtype == np.float64, f"Expected float64, got {data['mid'].dtype}"

    def test_mid_shape_1d(self):
        """mid array should be 1-dimensional with shape (N,)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir, dates = self._run_cache(tmpdir)
            data = np.load(os.path.join(out_dir, f"{dates[0]}.npz"))
            assert data["mid"].ndim == 1, f"Expected 1D, got {data['mid'].ndim}D"

    def test_spread_dtype_float64(self):
        """spread array should be float64."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir, dates = self._run_cache(tmpdir)
            data = np.load(os.path.join(out_dir, f"{dates[0]}.npz"))
            assert data["spread"].dtype == np.float64, f"Expected float64, got {data['spread'].dtype}"

    def test_spread_shape_1d(self):
        """spread array should be 1-dimensional with shape (N,)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir, dates = self._run_cache(tmpdir)
            data = np.load(os.path.join(out_dir, f"{dates[0]}.npz"))
            assert data["spread"].ndim == 1, f"Expected 1D, got {data['spread'].ndim}D"

    def test_array_lengths_match(self):
        """obs rows, mid length, and spread length should all match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir, dates = self._run_cache(tmpdir)
            data = np.load(os.path.join(out_dir, f"{dates[0]}.npz"))
            n_obs = data["obs"].shape[0]
            n_mid = data["mid"].shape[0]
            n_spread = data["spread"].shape[0]
            assert n_obs == n_mid == n_spread, (
                f"Array lengths differ: obs={n_obs}, mid={n_mid}, spread={n_spread}"
            )

    def test_arrays_match_direct_precompute(self):
        """Cached arrays should match direct lob_rl_core.precompute() output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir, dates = self._run_cache(tmpdir)

            # Direct precompute
            cfg = lob_rl_core.SessionConfig.default_rth()
            obs_direct, mid_direct, spread_direct, _ = lob_rl_core.precompute(
                DAY_FILES[0], cfg
            )

            # Cached
            data = np.load(os.path.join(out_dir, f"{dates[0]}.npz"))
            np.testing.assert_array_equal(data["obs"], obs_direct,
                                          err_msg="Cached obs differs from direct precompute")
            np.testing.assert_array_equal(data["mid"], mid_direct,
                                          err_msg="Cached mid differs from direct precompute")
            np.testing.assert_array_equal(data["spread"], spread_direct,
                                          err_msg="Cached spread differs from direct precompute")


# ===========================================================================
# Test 4: --force flag re-caches existing files
# ===========================================================================


class TestForceFlag:
    """--force should re-cache even if .npz already exists."""

    def test_skips_existing_without_force(self):
        """Without --force, existing .npz files should not be re-created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir, dates = _make_manifest_dir(DAY_FILES[:1], tmpdir)
            out_dir = os.path.join(tmpdir, "cache")
            os.makedirs(out_dir, exist_ok=True)

            # Pre-create an npz file with a sentinel
            sentinel = os.path.join(out_dir, f"{dates[0]}.npz")
            np.savez(sentinel, obs=np.array([[1.0]]), mid=np.array([1.0]),
                     spread=np.array([1.0]))

            _run_precompute_cache(["--data-dir", data_dir, "--out", out_dir])

            # File should NOT have been overwritten (mtime unchanged or content matches sentinel)
            data = np.load(sentinel)
            assert data["obs"].shape == (1, 1), (
                "Existing .npz was overwritten without --force"
            )

    def test_rewrites_with_force(self):
        """With --force, existing .npz files should be overwritten."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir, dates = _make_manifest_dir(DAY_FILES[:1], tmpdir)
            out_dir = os.path.join(tmpdir, "cache")
            os.makedirs(out_dir, exist_ok=True)

            # Pre-create an npz file with a sentinel
            sentinel = os.path.join(out_dir, f"{dates[0]}.npz")
            np.savez(sentinel, obs=np.array([[1.0]]), mid=np.array([1.0]),
                     spread=np.array([1.0]))

            _run_precompute_cache(["--data-dir", data_dir, "--out", out_dir, "--force"])

            # File should have been overwritten (shape != (1, 1))
            data = np.load(sentinel)
            assert data["obs"].shape[1] == 43, (
                f"Expected 43-col obs after --force, got shape {data['obs'].shape}"
            )
            assert data["obs"].shape[0] > 1, (
                f"Expected > 1 rows after --force, got {data['obs'].shape[0]}"
            )


# ===========================================================================
# Test 5: Skips holiday/empty .bin files that produce 0 steps
# ===========================================================================


class TestSkipsEmptyDays:
    """Holiday .bin files producing 0 steps should be skipped, not cached."""

    def test_holiday_file_not_cached(self):
        """A .bin file producing < 2 BBO snapshots should not produce an .npz file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # episode_200records.bin has epoch-era timestamps -> 0 RTH steps
            bad_file = os.path.join(FIXTURE_DIR, "episode_200records.bin")
            data_dir, dates = _make_manifest_dir([bad_file], tmpdir, dates=["2025-12-25"])
            out_dir = os.path.join(tmpdir, "cache")

            _run_precompute_cache(["--data-dir", data_dir, "--out", out_dir])

            # No .npz files should have been created
            if os.path.exists(out_dir):
                npz_files = [f for f in os.listdir(out_dir) if f.endswith(".npz")]
                assert len(npz_files) == 0, (
                    f"Holiday file should not produce .npz, but found {npz_files}"
                )


# ===========================================================================
# Test 6: Temporal features and step_interval NOT cached
# ===========================================================================


class TestNoCachedTemporalOrInterval:
    """Cached .npz should contain raw 43-col obs, not temporal features or subsampled data."""

    def test_obs_has_43_columns_not_54(self):
        """Cached obs should have 43 columns (no temporal features, no position)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir, dates = _make_manifest_dir(DAY_FILES[:1], tmpdir)
            out_dir = os.path.join(tmpdir, "cache")

            _run_precompute_cache(["--data-dir", data_dir, "--out", out_dir])

            data = np.load(os.path.join(out_dir, f"{dates[0]}.npz"))
            assert data["obs"].shape[1] == 43, (
                f"Cached obs should have 43 cols (raw), got {data['obs'].shape[1]}"
            )

    def test_obs_not_subsampled(self):
        """Cached obs should contain ALL rows, not subsampled by step_interval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir, dates = _make_manifest_dir(DAY_FILES[:1], tmpdir)
            out_dir = os.path.join(tmpdir, "cache")

            _run_precompute_cache(["--data-dir", data_dir, "--out", out_dir])

            # Compare with direct precompute
            cfg = lob_rl_core.SessionConfig.default_rth()
            _, _, _, num_steps = lob_rl_core.precompute(DAY_FILES[0], cfg)

            data = np.load(os.path.join(out_dir, f"{dates[0]}.npz"))
            assert data["obs"].shape[0] == num_steps, (
                f"Cached obs should have {num_steps} rows (not subsampled), "
                f"got {data['obs'].shape[0]}"
            )
