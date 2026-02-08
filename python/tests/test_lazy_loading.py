"""Tests for Lazy Loading for MultiDayEnv.

Spec: docs/lazy-loading.md

These tests verify:
1. MultiDayEnv(cache_dir=...) lazy loading: stores file paths only, no arrays in memory
2. reset() loads the day's .npz arrays on demand, releases previous day
3. Internal state uses _npz_paths (list of str) instead of _precomputed_days
4. Bar-size filtering at init: temporary load, check bar count, discard arrays
5. file_paths mode unchanged (eager load, arrays in memory)
6. New cache_files parameter: explicit list of .npz paths (lazy load)
7. Mutual exclusivity: exactly one of file_paths / cache_dir / cache_files
8. train.py train-split filtering: uses cache_files for train split
9. Backward compatibility: all existing behaviors preserved
10. Edge cases: bad .npz at init vs reset, empty cache_files, both cache_files + cache_dir

Key acceptance criteria:
- A MultiDayEnv(cache_dir=...) with 100 .npz files holds NO numpy arrays after __init__()
- reset() returns same observations as eager-load for same day/seed
- Shuffle ordering preserved (file ordering, RNG seeding, epoch wraparound)
- contract_ids property returns instrument_ids extracted at init time
- Days with < 2 bars filtered at init (temporary load + discard)
- train.py --cache-dir only passes train-split .npz paths to workers
"""

import ast
import os
import sys
import tempfile
import warnings

import numpy as np
import pytest
import gymnasium as gym

from lob_rl.multi_day_env import MultiDayEnv

from conftest import (
    make_realistic_obs,
    create_synthetic_cache_dir,
    run_episode,
    save_cache_with_instrument_id as _save_cache_with_instrument_id,
    save_cache_without_instrument_id as _save_cache_without_instrument_id,
)


def _make_npz_files(tmpdir, n_days=5, n_rows=50, with_instrument_id=False):
    """Create multiple .npz files in tmpdir, return sorted list of paths."""
    paths = []
    for i in range(n_days):
        obs, mid, spread = make_realistic_obs(n_rows, mid_start=100.0 + i * 10)
        filename = f"2025-01-{i + 10:02d}.npz"
        if with_instrument_id:
            # Alternate instrument IDs to test contract tracking
            inst_id = 11355 if i < n_days // 2 else 13615
            path = _save_cache_with_instrument_id(tmpdir, filename, obs, mid, spread, inst_id)
        else:
            path = _save_cache_without_instrument_id(tmpdir, filename, obs, mid, spread)
        paths.append(path)
    return sorted(paths)


def _has_numpy_arrays(obj, attr_name):
    """Check if an object attribute contains numpy arrays (non-trivially).

    Returns True if the attribute holds any numpy arrays with shape > ().
    Scalar instrument_id arrays (uint32, shape=(1,)) are excluded.
    """
    val = getattr(obj, attr_name, None)
    if val is None:
        return False
    if isinstance(val, list):
        for item in val:
            if isinstance(item, np.ndarray) and item.nbytes > 8:
                return True
            if isinstance(item, tuple):
                for elem in item:
                    if isinstance(elem, np.ndarray) and elem.nbytes > 8:
                        return True
    elif isinstance(val, np.ndarray) and val.nbytes > 8:
        return True
    return False


# ===========================================================================
# 1. Lazy Loading — cache_dir stores paths only, no arrays
# ===========================================================================


class TestLazyLoadingCacheDirNoArrays:
    """After __init__(cache_dir=...), no numpy arrays should be held in memory."""

    def test_no_precomputed_days_attribute(self):
        """cache_dir mode should NOT populate _precomputed_days with arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=5)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            # _precomputed_days should either not exist or be empty
            if hasattr(env, "_precomputed_days"):
                assert len(env._precomputed_days) == 0, (
                    f"_precomputed_days has {len(env._precomputed_days)} items; "
                    "lazy loading should not store arrays at init"
                )

    def test_has_npz_paths_attribute(self):
        """cache_dir mode should store file paths in _npz_paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=3)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            assert hasattr(env, "_npz_paths"), (
                "MultiDayEnv should have _npz_paths attribute for lazy loading"
            )
            assert isinstance(env._npz_paths, list)
            assert len(env._npz_paths) == 3

    def test_npz_paths_are_strings(self):
        """_npz_paths should contain string file paths, not arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=3)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            for p in env._npz_paths:
                assert isinstance(p, str), f"Expected str path, got {type(p)}"
                assert p.endswith(".npz"), f"Expected .npz path, got {p}"

    def test_npz_paths_are_valid_files(self):
        """Each path in _npz_paths should point to an existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=3)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            for p in env._npz_paths:
                assert os.path.isfile(p), f"Path does not exist: {p}"

    def test_no_large_arrays_in_memory_after_init(self):
        """After init with cache_dir, env should not hold any large numpy arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=10, n_rows=100)
            cache_dir = tmpdir
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            # Check that _precomputed_days doesn't hold arrays
            assert not _has_numpy_arrays(env, "_precomputed_days"), (
                "_precomputed_days should not hold numpy arrays in lazy mode"
            )

    def test_contract_ids_extracted_at_init(self):
        """instrument_id should be extracted at init time (stored as int, not array)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=5, with_instrument_id=True)
            cache_dir = tmpdir
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            ids = env.contract_ids
            assert len(ids) == 5
            # Each id should be an int or None, not a numpy array
            for cid in ids:
                assert cid is None or isinstance(cid, int), (
                    f"contract_id should be int or None, got {type(cid)}"
                )

    def test_many_files_no_memory_blowup(self):
        """With many .npz files, init should not hold arrays — only paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            n_days = 50
            paths = _make_npz_files(tmpdir, n_days=n_days, n_rows=100)
            cache_dir = tmpdir
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            # Should have paths but no arrays
            assert hasattr(env, "_npz_paths")
            assert len(env._npz_paths) == n_days
            assert not _has_numpy_arrays(env, "_precomputed_days")


# ===========================================================================
# 2. Load on reset() — arrays loaded on demand
# ===========================================================================


class TestLoadOnReset:
    """reset() should load the selected day's .npz arrays on demand."""

    def test_reset_returns_valid_obs(self):
        """reset() in lazy mode should return valid observation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=3)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            obs, info = env.reset()
            assert obs.shape == (54,)
            assert obs.dtype == np.float32
            assert not np.any(np.isnan(obs))

    def test_step_works_after_lazy_reset(self):
        """step() should work correctly after a lazy-loading reset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=3)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            env.reset()
            obs, reward, terminated, truncated, info = env.step(2)
            assert obs.shape == (54,)
            assert isinstance(reward, (float, np.floating))
            assert isinstance(terminated, (bool, np.bool_))

    def test_full_episode_after_lazy_reset(self):
        """A full episode should complete after lazy-loading reset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=3)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            env.reset()
            steps = run_episode(env)
            assert steps > 0

    def test_multiple_resets_cycle_through_days(self):
        """Multiple resets should cycle through different days."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=3)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            observations = []
            for _ in range(3):
                obs, _ = env.reset()
                observations.append(obs.copy())
                run_episode(env)

            # At least some observations should differ (different days)
            all_same = all(
                np.allclose(observations[0], observations[i])
                for i in range(1, len(observations))
            )
            assert not all_same, (
                "All resets returned identical observations — "
                "lazy loading may not be cycling through files"
            )

    def test_previous_day_arrays_released(self):
        """After loading a new day, the previous day's arrays should not be held."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=3)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            # Reset twice - should only hold current day's env
            env.reset()
            run_episode(env)
            env.reset()

            # After second reset, _precomputed_days should not hold arrays
            # (only the inner env holds current day data)
            assert not _has_numpy_arrays(env, "_precomputed_days"), (
                "Previous day's arrays should be released after loading new day"
            )

    def test_day_index_in_info(self):
        """reset() info should still contain day_index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=3)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            _, info = env.reset()
            assert "day_index" in info


# ===========================================================================
# 3. Correctness — lazy load produces same results as eager load
# ===========================================================================


class TestLazyLoadCorrectnessVsCacheDir:
    """Lazy-loaded cache_dir should produce identical results to the old eager load."""

    def test_same_reset_obs_first_day(self):
        """First reset obs should be deterministic for the same cache_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=3)

            env1 = MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            env2 = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            obs1, _ = env1.reset()
            obs2, _ = env2.reset()

            np.testing.assert_array_almost_equal(
                obs1, obs2,
                err_msg="Two identical cache_dir envs produce different first obs"
            )

    def test_same_rewards_for_same_actions(self):
        """Rewards should be identical for the same action sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=1)

            env1 = MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            env2 = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            env1.reset()
            env2.reset()

            for action in [2, 0, 1, 2, 0]:
                _, r1, _, _, _ = env1.step(action)
                _, r2, _, _, _ = env2.step(action)
                assert r1 == pytest.approx(r2), (
                    f"Reward mismatch: {r1} vs {r2}"
                )

    def test_same_obs_at_each_step(self):
        """Observations should be identical at each step for two identical envs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=1)

            env1 = MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            env2 = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            env1.reset()
            env2.reset()

            for step in range(10):
                obs1, _, t1, _, _ = env1.step(step % 3)
                obs2, _, t2, _, _ = env2.step(step % 3)
                np.testing.assert_array_almost_equal(
                    obs1, obs2,
                    err_msg=f"Obs differs at step {step}"
                )
                if t1:
                    break

    def test_sequential_day_ordering_preserved(self):
        """Sequential (shuffle=False) should visit days in sorted filename order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=5)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            day_indices = []
            for _ in range(5):
                _, info = env.reset()
                day_indices.append(info["day_index"])
                run_episode(env)

            assert day_indices == [0, 1, 2, 3, 4], (
                f"Expected sequential [0,1,2,3,4], got {day_indices}"
            )


# ===========================================================================
# 4. Shuffle ordering preserved
# ===========================================================================


class TestLazyLoadShuffle:
    """Shuffle ordering should work identically with lazy loading."""

    def test_shuffle_with_seed_deterministic(self):
        """Same seed should produce same day ordering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=5, n_rows=30)

            def collect_indices(seed):
                env = MultiDayEnv(cache_dir=cache_dir, shuffle=True, seed=seed)
                indices = []
                for _ in range(5):
                    _, info = env.reset()
                    indices.append(info["day_index"])
                    run_episode(env)
                return indices

            a = collect_indices(42)
            b = collect_indices(42)
            assert a == b, f"Same seed should give same order: {a} vs {b}"

    def test_different_seeds_different_order(self):
        """Different seeds should produce different day orderings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=5, n_rows=30)

            def collect_indices(seed):
                env = MultiDayEnv(cache_dir=cache_dir, shuffle=True, seed=seed)
                indices = []
                for _ in range(5):
                    _, info = env.reset()
                    indices.append(info["day_index"])
                    run_episode(env)
                return indices

            a = collect_indices(42)
            b = collect_indices(9999)
            assert a != b, "Different seeds should produce different orderings"

    def test_epoch_wraparound_reshuffles(self):
        """At epoch boundary, day order should be re-shuffled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=5, n_rows=30)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=True, seed=99)

            # Epoch 1
            epoch1 = []
            for _ in range(5):
                _, info = env.reset()
                epoch1.append(info["day_index"])
                run_episode(env)

            # Epoch 2
            epoch2 = []
            for _ in range(5):
                _, info = env.reset()
                epoch2.append(info["day_index"])
                run_episode(env)

            # With 5 days, probability of same shuffle = 1/120
            assert epoch1 != epoch2, (
                "Epoch 1 and Epoch 2 had identical day orders — "
                "re-shuffle at epoch boundary may not be working"
            )

    def test_all_days_visited_in_shuffle_epoch(self):
        """In shuffle mode, all days should still be visited within an epoch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=5, n_rows=30)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=True, seed=42)

            indices = set()
            for _ in range(5):
                _, info = env.reset()
                indices.add(info["day_index"])
                run_episode(env)

            assert indices == {0, 1, 2, 3, 4}, (
                f"Not all days visited in epoch: {indices}"
            )


# ===========================================================================
# 5. Contract tracking — contract_ids from lazy init
# ===========================================================================


class TestLazyLoadContractIds:
    """contract_ids property should return instrument_ids extracted at init time."""

    def test_contract_ids_extracted_at_init_not_reset(self):
        """contract_ids should be available before any reset() is called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=3, with_instrument_id=True)
            env = MultiDayEnv(cache_dir=tmpdir, shuffle=False)

            # Should work without calling reset()
            ids = env.contract_ids
            assert len(ids) == 3
            assert all(isinstance(cid, (int, type(None))) for cid in ids)

    def test_contract_ids_values_correct(self):
        """contract_ids should match instrument_ids in the .npz files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            expected_ids = [11355, 11355, 13615, 13615, 10039]
            for i, inst_id in enumerate(expected_ids):
                obs, mid, spread = make_realistic_obs(50, mid_start=100.0 + i * 10)
                _save_cache_with_instrument_id(
                    tmpdir, f"day{i:02d}.npz", obs, mid, spread, inst_id
                )

            env = MultiDayEnv(cache_dir=tmpdir, shuffle=False)
            assert env.contract_ids == expected_ids

    def test_contract_ids_none_for_legacy_files(self):
        """Legacy .npz without instrument_id should give None in contract_ids."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=3, with_instrument_id=False)
            env = MultiDayEnv(cache_dir=tmpdir, shuffle=False)

            ids = env.contract_ids
            assert all(cid is None for cid in ids)

    def test_contract_roll_detection_with_lazy_load(self):
        """Contract roll detection should work with lazy loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs, mid, spread = make_realistic_obs(50, mid_start=100.0)
            _save_cache_with_instrument_id(
                tmpdir, "day00.npz", obs, mid, spread, 11355
            )
            obs, mid, spread = make_realistic_obs(50, mid_start=110.0)
            _save_cache_with_instrument_id(
                tmpdir, "day01.npz", obs, mid, spread, 13615
            )

            env = MultiDayEnv(cache_dir=tmpdir, shuffle=False)

            # Day 0
            _, info0 = env.reset()
            assert info0["contract_roll"] is False
            assert info0["instrument_id"] == 11355
            run_episode(env)

            # Day 1 — contract roll
            _, info1 = env.reset()
            assert info1["contract_roll"] is True
            assert info1["instrument_id"] == 13615


# ===========================================================================
# 6. Bar-size filtering at init — temporary load, check, discard
# ===========================================================================


class TestBarSizeFilteringLazy:
    """Bar-size filtering should load arrays temporarily and discard them."""

    def test_bar_size_filtering_works_in_lazy_mode(self):
        """Days with too few bars should be filtered out at init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Day with enough ticks for bars
            obs, mid, spread = make_realistic_obs(200, mid_start=100.0)
            _save_cache_without_instrument_id(tmpdir, "day00.npz", obs, mid, spread)

            # Day with too few ticks for even 2 bars (e.g., bar_size=1000, only 10 ticks)
            obs_tiny, mid_tiny, spread_tiny = make_realistic_obs(10, mid_start=200.0)
            _save_cache_without_instrument_id(tmpdir, "day01.npz", obs_tiny, mid_tiny, spread_tiny)

            env = MultiDayEnv(cache_dir=tmpdir, shuffle=False, bar_size=100)

            # Only day00 should survive filtering
            assert len(env._npz_paths) == 1

    def test_bar_size_filtering_no_arrays_after_init(self):
        """After bar-size filtering, arrays should NOT be held in memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                obs, mid, spread = make_realistic_obs(200, mid_start=100.0 + i * 10)
                _save_cache_without_instrument_id(
                    tmpdir, f"day{i:02d}.npz", obs, mid, spread
                )

            env = MultiDayEnv(cache_dir=tmpdir, shuffle=False, bar_size=10)

            # After filtering, no arrays should be stored
            assert not _has_numpy_arrays(env, "_precomputed_days"), (
                "Arrays should be discarded after bar-size filtering"
            )

    def test_bar_size_filtering_preserves_contract_ids(self):
        """Bar-size filtering should keep contract_ids aligned with surviving days."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Day 0: enough ticks, instrument_id=11355
            obs, mid, spread = make_realistic_obs(200, mid_start=100.0)
            _save_cache_with_instrument_id(
                tmpdir, "day00.npz", obs, mid, spread, 11355
            )

            # Day 1: too few ticks, instrument_id=13615
            obs_tiny, mid_tiny, spread_tiny = make_realistic_obs(5, mid_start=200.0)
            _save_cache_with_instrument_id(
                tmpdir, "day01.npz", obs_tiny, mid_tiny, spread_tiny, 13615
            )

            env = MultiDayEnv(cache_dir=tmpdir, shuffle=False, bar_size=100)

            # Only day 0 should survive
            assert len(env.contract_ids) == 1
            assert env.contract_ids[0] == 11355


# ===========================================================================
# 7. cache_files parameter — explicit list of .npz paths
# ===========================================================================


class TestCacheFilesParameter:
    """New cache_files parameter: accepts an explicit list of .npz file paths."""

    def test_cache_files_accepted(self):
        """MultiDayEnv(cache_files=[...]) should be accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=3)
            env = MultiDayEnv(cache_files=paths, shuffle=False)
            obs, info = env.reset()
            assert obs.shape == (54,)

    def test_cache_files_uses_lazy_loading(self):
        """cache_files mode should also use lazy loading (no arrays after init)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=5, n_rows=100)
            env = MultiDayEnv(cache_files=paths, shuffle=False)

            assert hasattr(env, "_npz_paths")
            assert len(env._npz_paths) == 5
            assert not _has_numpy_arrays(env, "_precomputed_days")

    def test_cache_files_uses_exact_files(self):
        """cache_files should use exactly the provided files, not glob."""
        with tempfile.TemporaryDirectory() as tmpdir:
            all_paths = _make_npz_files(tmpdir, n_days=5)
            # Only pass first 3
            subset = all_paths[:3]
            env = MultiDayEnv(cache_files=subset, shuffle=False)

            assert len(env._npz_paths) == 3

    def test_cache_files_order_preserved(self):
        """cache_files should use files in the order given."""
        with tempfile.TemporaryDirectory() as tmpdir:
            all_paths = _make_npz_files(tmpdir, n_days=5)
            # Reverse order
            reversed_paths = list(reversed(all_paths))
            env = MultiDayEnv(cache_files=reversed_paths, shuffle=False)

            # _npz_paths should match the provided order
            assert env._npz_paths == reversed_paths

    def test_cache_files_reset_works(self):
        """reset() should work with cache_files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=3)
            env = MultiDayEnv(cache_files=paths, shuffle=False)

            obs, info = env.reset()
            assert obs.shape == (54,)
            assert "day_index" in info

    def test_cache_files_step_works(self):
        """step() should work after cache_files reset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=3)
            env = MultiDayEnv(cache_files=paths, shuffle=False)

            env.reset()
            obs, reward, terminated, truncated, info = env.step(2)
            assert obs.shape == (54,)

    def test_cache_files_full_episode(self):
        """A full episode should complete with cache_files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=3)
            env = MultiDayEnv(cache_files=paths, shuffle=False)

            env.reset()
            steps = run_episode(env)
            assert steps > 0

    def test_cache_files_cycles_through_days(self):
        """Multiple resets should cycle through the provided files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=3)
            env = MultiDayEnv(cache_files=paths, shuffle=False)

            day_indices = []
            for _ in range(3):
                _, info = env.reset()
                day_indices.append(info["day_index"])
                run_episode(env)

            assert day_indices == [0, 1, 2]

    def test_cache_files_with_shuffle(self):
        """cache_files with shuffle=True should randomize order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=5, n_rows=30)

            def collect_indices(seed):
                env = MultiDayEnv(cache_files=paths, shuffle=True, seed=seed)
                indices = []
                for _ in range(5):
                    _, info = env.reset()
                    indices.append(info["day_index"])
                    run_episode(env)
                return indices

            a = collect_indices(42)
            b = collect_indices(42)
            assert a == b, f"Same seed should give same order: {a} vs {b}"

    def test_cache_files_with_instrument_id(self):
        """cache_files should extract instrument_id from .npz files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=4, with_instrument_id=True)
            env = MultiDayEnv(cache_files=paths, shuffle=False)

            ids = env.contract_ids
            assert len(ids) == 4
            assert all(isinstance(cid, int) for cid in ids)

    def test_cache_files_with_bar_size(self):
        """cache_files should work with bar_size parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=3, n_rows=200)
            env = MultiDayEnv(cache_files=paths, shuffle=False, bar_size=10)

            obs, info = env.reset()
            assert obs.shape == (21,)  # bar-level obs

    def test_cache_files_with_execution_cost(self):
        """cache_files should forward execution_cost to inner env."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=1)
            env = MultiDayEnv(cache_files=paths, shuffle=False, execution_cost=True)

            env.reset()
            _, reward, _, _, _ = env.step(2)
            assert isinstance(reward, (float, np.floating))

    def test_cache_files_check_env(self):
        """gymnasium check_env should pass for cache_files mode."""
        from gymnasium.utils.env_checker import check_env
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=3, n_rows=30)
            env = MultiDayEnv(cache_files=paths, shuffle=False)
            check_env(env, skip_render_check=True)


# ===========================================================================
# 8. Mutual exclusivity — exactly one of file_paths / cache_dir / cache_files
# ===========================================================================


class TestMutualExclusivityThreeWay:
    """Exactly one of file_paths, cache_dir, or cache_files must be provided."""

    def test_cache_files_and_cache_dir_raises(self):
        """Providing both cache_files and cache_dir should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=2)
            with pytest.raises(ValueError):
                MultiDayEnv(cache_files=paths, cache_dir=tmpdir)

    def test_cache_files_and_file_paths_raises(self):
        """Providing both cache_files and file_paths should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=2)
            with pytest.raises(ValueError):
                MultiDayEnv(cache_files=paths, file_paths=["/some/file.bin"])

    def test_all_three_raises(self):
        """Providing all three should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=2)
            with pytest.raises(ValueError):
                MultiDayEnv(
                    file_paths=["/some/file.bin"],
                    cache_dir=tmpdir,
                    cache_files=paths,
                )

    def test_none_provided_raises(self):
        """Providing none of the three should raise ValueError."""
        with pytest.raises((ValueError, TypeError)):
            MultiDayEnv(file_paths=None, cache_dir=None, cache_files=None)

    def test_cache_files_only_works(self):
        """Providing only cache_files should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=2)
            env = MultiDayEnv(cache_files=paths, shuffle=False)
            obs, _ = env.reset()
            assert obs.shape == (54,)

    def test_cache_dir_only_works(self):
        """Providing only cache_dir should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            obs, _ = env.reset()
            assert obs.shape == (54,)

    def test_empty_cache_files_raises(self):
        """cache_files=[] should raise ValueError."""
        with pytest.raises(ValueError):
            MultiDayEnv(cache_files=[])


# ===========================================================================
# 9. file_paths mode unchanged — eager load, arrays in memory
# ===========================================================================


class TestFilePathsModeUnchanged:
    """file_paths mode should remain unchanged (eager load)."""

    def test_file_paths_still_works(self):
        """MultiDayEnv(file_paths=...) should still work as before."""
        from conftest import DAY_FILES
        env = MultiDayEnv(file_paths=DAY_FILES[:2], shuffle=False)
        obs, info = env.reset()
        assert obs.shape == (54,)

    def test_file_paths_holds_arrays_in_memory(self):
        """file_paths mode should eagerly hold arrays in _precomputed_days."""
        from conftest import DAY_FILES
        env = MultiDayEnv(file_paths=DAY_FILES[:2], shuffle=False)

        # file_paths mode should still hold arrays
        assert hasattr(env, "_precomputed_days")
        assert len(env._precomputed_days) > 0, (
            "file_paths mode should eagerly load arrays"
        )

    def test_file_paths_contract_ids_are_none(self):
        """file_paths mode should give None contract_ids."""
        from conftest import DAY_FILES
        env = MultiDayEnv(file_paths=DAY_FILES[:2], shuffle=False)
        ids = env.contract_ids
        assert all(cid is None for cid in ids)


# ===========================================================================
# 10. Backward compatibility — cache_dir still works as before
# ===========================================================================


class TestBackwardCompatCacheDir:
    """MultiDayEnv(cache_dir=...) should still work — just without holding arrays."""

    def test_cache_dir_glob_npz_sorted(self):
        """cache_dir should glob *.npz files sorted by name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with specific dates
            for date in ["2025-01-15", "2025-01-10", "2025-01-12"]:
                obs, mid, spread = make_realistic_obs(50, mid_start=100.0)
                _save_cache_without_instrument_id(tmpdir, f"{date}.npz", obs, mid, spread)

            env = MultiDayEnv(cache_dir=tmpdir, shuffle=False)

            # Paths should be sorted
            filenames = [os.path.basename(p) for p in env._npz_paths]
            assert filenames == sorted(filenames), (
                f"Expected sorted filenames, got {filenames}"
            )

    def test_cache_dir_skips_invalid_npz(self):
        """cache_dir should skip invalid .npz files (missing keys)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # One valid
            obs, mid, spread = make_realistic_obs(50)
            _save_cache_without_instrument_id(tmpdir, "2025-01-10.npz", obs, mid, spread)

            # One invalid (wrong keys)
            np.savez(os.path.join(tmpdir, "2025-01-11.npz"), wrong_key=np.array([1.0]))

            env = MultiDayEnv(cache_dir=tmpdir, shuffle=False)
            assert len(env._npz_paths) == 1

    def test_cache_dir_empty_raises(self):
        """Empty cache_dir (no .npz files) should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises((ValueError, FileNotFoundError)):
                MultiDayEnv(cache_dir=tmpdir, shuffle=False)

    def test_cache_dir_all_invalid_raises(self):
        """cache_dir with only invalid .npz files should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez(os.path.join(tmpdir, "bad1.npz"), wrong=np.array([1.0]))
            np.savez(os.path.join(tmpdir, "bad2.npz"), wrong=np.array([2.0]))

            with pytest.raises(ValueError):
                MultiDayEnv(cache_dir=tmpdir, shuffle=False)

    def test_cache_dir_observation_space(self):
        """cache_dir mode should have correct observation_space."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            assert env.observation_space.shape == (54,)

    def test_cache_dir_action_space(self):
        """cache_dir mode should have Discrete(3) action space."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            assert env.action_space.n == 3

    def test_cache_dir_is_gymnasium_env(self):
        """cache_dir-based MultiDayEnv should be a gymnasium.Env."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            assert isinstance(env, gym.Env)


# ===========================================================================
# 11. Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases from the spec."""

    def test_npz_fails_at_init_skipped_with_warning(self):
        """A .npz that fails to load at init (during instrument_id extraction)
        should be warned and skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid .npz
            obs, mid, spread = make_realistic_obs(50)
            _save_cache_without_instrument_id(tmpdir, "day00.npz", obs, mid, spread)

            # Create a corrupt file
            with open(os.path.join(tmpdir, "day01.npz"), "wb") as f:
                f.write(b"this is not a valid npz file")

            # Should warn but not crash
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                env = MultiDayEnv(cache_dir=tmpdir, shuffle=False)
                # Only the valid file should be loaded
                assert len(env._npz_paths) == 1

    def test_npz_fails_at_reset_raises_error(self):
        """A .npz that fails to load at reset() time should raise an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid .npz files
            paths = _make_npz_files(tmpdir, n_days=2)
            env = MultiDayEnv(cache_files=paths, shuffle=False)

            # Delete the first file after init (simulates corruption at reset time)
            os.remove(paths[0])

            with pytest.raises(Exception):
                env.reset()

    def test_cache_files_empty_list_raises(self):
        """cache_files=[] should raise ValueError."""
        with pytest.raises(ValueError):
            MultiDayEnv(cache_files=[])

    def test_single_file_cache_files(self):
        """cache_files with a single file should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=1)
            env = MultiDayEnv(cache_files=paths, shuffle=False)
            obs, info = env.reset()
            assert obs.shape == (54,)

    def test_single_file_cache_dir(self):
        """cache_dir with a single .npz file should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_npz_files(tmpdir, n_days=1)
            env = MultiDayEnv(cache_dir=tmpdir, shuffle=False)
            obs, info = env.reset()
            assert obs.shape == (54,)


# ===========================================================================
# 12. DummyVecEnv compatibility with lazy loading
# ===========================================================================


class TestDummyVecEnvLazyLoading:
    """Lazy-loaded MultiDayEnv should work inside SB3's DummyVecEnv."""

    def test_wrappable_in_dummy_vec_env_cache_dir(self):
        """DummyVecEnv should wrap lazy-loaded cache_dir MultiDayEnv."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_rows=30)
            vec_env = DummyVecEnv([
                lambda: MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            ])
            obs = vec_env.reset()
            assert obs.shape == (1, 54)

    def test_wrappable_in_dummy_vec_env_cache_files(self):
        """DummyVecEnv should wrap cache_files MultiDayEnv."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=3, n_rows=30)
            vec_env = DummyVecEnv([
                lambda: MultiDayEnv(cache_files=paths, shuffle=False)
            ])
            obs = vec_env.reset()
            assert obs.shape == (1, 54)

    def test_dummy_vec_env_step_lazy(self):
        """step() through DummyVecEnv should work with lazy loading."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_rows=30)
            vec_env = DummyVecEnv([
                lambda: MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            ])
            vec_env.reset()
            obs, rewards, dones, infos = vec_env.step([1])
            assert obs.shape == (1, 54)
            assert rewards.shape == (1,)


# ===========================================================================
# 13. train.py — train-split filtering with cache_files
# ===========================================================================


from conftest import TRAIN_SCRIPT


class TestTrainScriptTrainSplitFiltering:
    """train.py should pass train-split .npz paths to workers, not the full dir."""

    def test_make_train_env_accepts_cache_files(self):
        """make_train_env should accept cache_files parameter."""
        # Import train.py's make_train_env
        sys.path.insert(0, os.path.dirname(TRAIN_SCRIPT))
        try:
            import importlib
            spec_mod = importlib.util.spec_from_file_location("train", TRAIN_SCRIPT)
            train_module = importlib.util.module_from_spec(spec_mod)
            spec_mod.loader.exec_module(train_module)

            make_train_env = train_module.make_train_env

            # Check it accepts cache_files parameter
            import inspect
            sig = inspect.signature(make_train_env)
            assert "cache_files" in sig.parameters, (
                "make_train_env should accept cache_files parameter"
            )
        finally:
            sys.path.pop(0)

    def test_make_train_env_cache_files_creates_env(self):
        """make_train_env(cache_files=...) should return a callable that creates an env."""
        sys.path.insert(0, os.path.dirname(TRAIN_SCRIPT))
        try:
            import importlib
            spec_mod = importlib.util.spec_from_file_location("train", TRAIN_SCRIPT)
            train_module = importlib.util.module_from_spec(spec_mod)
            spec_mod.loader.exec_module(train_module)

            with tempfile.TemporaryDirectory() as tmpdir:
                paths = _make_npz_files(tmpdir, n_days=3, n_rows=30)
                factory = train_module.make_train_env(cache_files=paths)
                env = factory()
                assert isinstance(env, MultiDayEnv)
                obs, _ = env.reset()
                assert obs.shape == (54,)
        finally:
            sys.path.pop(0)

    def test_train_script_source_uses_cache_files_for_train_split(self):
        """train.py source should pass cache_files (not cache_dir) for train split."""
        with open(TRAIN_SCRIPT) as f:
            source = f.read()

        # The train.py --cache-dir branch should:
        # 1. Split npz files into train/val/test
        # 2. Pass train-split files via cache_files= (not cache_dir=) to make_train_env
        # This ensures each worker only gets train-split files
        assert "cache_files" in source, (
            "train.py should use cache_files parameter to pass train-split files"
        )

    def test_train_script_does_not_pass_full_cache_dir_to_workers(self):
        """train.py should NOT pass the full cache_dir to make_train_env for SubprocVecEnv.

        The current code does: make_train_env(cache_dir=train_cache_dir, ...)
        which passes ALL 249 files to every worker. After the fix, it should pass
        only the train-split paths via cache_files.
        """
        with open(TRAIN_SCRIPT) as f:
            source = f.read()

        # Parse the AST to find calls to make_train_env
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Look for calls to make_train_env
                func = node.func
                func_name = None
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    func_name = func.attr

                if func_name == "make_train_env":
                    # Check that it doesn't use cache_dir= in the --cache-dir branch
                    # It should use cache_files= instead
                    keywords = {kw.arg: kw for kw in node.keywords}
                    # If cache_dir is still used, the fix hasn't been applied
                    # We check: if cache_files is present, that's correct
                    if "cache_files" in keywords:
                        # Good: using cache_files
                        pass
                    elif "cache_dir" in keywords:
                        pytest.fail(
                            "train.py still passes cache_dir to make_train_env. "
                            "It should pass cache_files with train-split paths."
                        )


# ===========================================================================
# 14. Forwarding parameters — lazy load mode
# ===========================================================================


class TestLazyLoadForwardsParams:
    """Lazy-loaded envs should forward all parameters to the inner env."""

    def test_forwards_reward_mode(self):
        """reward_mode should be forwarded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=1)
            env = MultiDayEnv(
                cache_files=paths, shuffle=False,
                reward_mode="pnl_delta_penalized", lambda_=0.5,
            )
            env.reset()
            _, reward, _, _, _ = env.step(2)
            assert isinstance(reward, (float, np.floating))

    def test_forwards_execution_cost(self):
        """execution_cost should be forwarded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=1)
            env = MultiDayEnv(
                cache_files=paths, shuffle=False, execution_cost=True,
            )
            env.reset()
            _, reward, _, _, _ = env.step(2)
            assert isinstance(reward, (float, np.floating))

    def test_forwards_step_interval(self):
        """step_interval should be forwarded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=1, n_rows=100)

            env_1 = MultiDayEnv(cache_files=paths, shuffle=False, step_interval=1)
            env_10 = MultiDayEnv(cache_files=paths, shuffle=False, step_interval=10)

            env_1.reset()
            env_10.reset()

            steps_1 = run_episode(env_1)
            steps_10 = run_episode(env_10)

            assert steps_10 < steps_1, (
                f"step_interval=10 ({steps_10}) should have fewer steps "
                f"than step_interval=1 ({steps_1})"
            )

    def test_forwards_participation_bonus(self):
        """participation_bonus should be forwarded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=1)
            env = MultiDayEnv(
                cache_files=paths, shuffle=False, participation_bonus=0.01,
            )
            env.reset()
            _, reward, _, _, _ = env.step(2)
            assert isinstance(reward, (float, np.floating))

    def test_forwards_bar_size(self):
        """bar_size should be forwarded and result in 21-dim obs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=1, n_rows=200)
            env = MultiDayEnv(cache_files=paths, shuffle=False, bar_size=10)
            obs, _ = env.reset()
            assert obs.shape == (21,)


# ===========================================================================
# 15. Forced flatten still works with lazy loading
# ===========================================================================


class TestForcedFlattenWithLazyLoading:
    """Forced flatten on terminal step should work with lazy loading."""

    def test_terminal_forced_flatten_cache_dir(self):
        """Terminal step should force flatten with lazy-loaded cache_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs, mid, spread = make_realistic_obs(50, mid_start=100.0, mid_step=0.0)
            _save_cache_without_instrument_id(tmpdir, "day00.npz", obs, mid, spread)

            env = MultiDayEnv(cache_dir=tmpdir, shuffle=False)
            env.reset()

            terminated = False
            last_info = {}
            while not terminated:
                obs_out, _, terminated, _, last_info = env.step(2)

            assert terminated
            assert last_info.get("forced_flatten") is True
            assert obs_out[53] == pytest.approx(0.0)

    def test_terminal_forced_flatten_cache_files(self):
        """Terminal step should force flatten with cache_files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _make_npz_files(tmpdir, n_days=1)
            env = MultiDayEnv(cache_files=paths, shuffle=False)
            env.reset()

            terminated = False
            last_info = {}
            while not terminated:
                obs_out, _, terminated, _, last_info = env.step(2)

            assert terminated
            assert last_info.get("forced_flatten") is True
            assert obs_out[53] == pytest.approx(0.0)

    def test_position_zero_at_every_reset_lazy(self):
        """Position should be 0 at every episode start with lazy loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=3)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            for _ in range(6):  # 2 full epochs
                obs, _ = env.reset()
                assert obs[53] == pytest.approx(0.0)
                # Go long then finish episode
                env.step(2)
                run_episode(env)
