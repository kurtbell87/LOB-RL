"""Vectorized barrier environment helpers.

Helper functions for creating SB3-compatible vectorized environments:
make_barrier_env_fn() and make_barrier_vec_env().
"""

import lob_rl.barrier._sb3_compat  # noqa: F401 — patch legacy net_arch format

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from lob_rl.barrier.multi_session_env import MultiSessionBarrierEnv


def make_barrier_env_fn(sessions, config=None, shuffle=False, seed=None):
    """Return a callable that creates a MultiSessionBarrierEnv.

    Parameters
    ----------
    sessions : list[dict]
        Session data dicts with keys: bars, labels, features.
    config : RewardConfig, optional
    shuffle : bool
    seed : int, optional

    Returns
    -------
    callable
        A zero-argument function that returns a MultiSessionBarrierEnv.
    """
    def _make():
        return MultiSessionBarrierEnv(
            sessions, config=config, shuffle=shuffle, seed=seed,
        )
    return _make


def make_barrier_vec_env(sessions, n_envs=1, use_subprocess=False,
                         config=None, shuffle=False, seed=None):
    """Create a vectorized barrier environment.

    Parameters
    ----------
    sessions : list[dict]
        Session data dicts.
    n_envs : int
        Number of parallel environments.
    use_subprocess : bool
        If True, use SubprocVecEnv; else DummyVecEnv.
    config : RewardConfig, optional
    shuffle : bool
    seed : int, optional

    Returns
    -------
    VecEnv
        SB3 vectorized environment.
    """
    env_fns = [
        make_barrier_env_fn(
            sessions, config=config, shuffle=shuffle,
            seed=(seed + i if seed is not None else None),
        )
        for i in range(n_envs)
    ]
    if use_subprocess:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)
