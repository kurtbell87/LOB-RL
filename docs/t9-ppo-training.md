# T9: PPO Training Infrastructure

## What to Build

Training infrastructure for barrier-hit PPO agent: a multi-session environment wrapper, MaskablePPO training script, and training diagnostic callbacks. This provides the code needed to train on real data; actual GPU training is a follow-on research experiment.

**Spec reference:** Section 5 (PPO Configuration), orchestrator T9.

## Dependencies

- `barrier_env.py` — `BarrierEnv`, `BarrierEnv.from_bars()`
- `bar_pipeline.py` — `TradeBar`, `build_bars_from_trades()`, `build_session_bars()`
- `label_pipeline.py` — `compute_labels()`
- `feature_pipeline.py` — `build_feature_matrix()`
- `reward_accounting.py` — `RewardConfig`, `ACTION_LONG`, `ACTION_SHORT`, `ACTION_FLAT`, `ACTION_HOLD`
- `gamblers_ruin.py` — `generate_random_walk()` (for synthetic test data)
- `__init__.py` — `TICK_SIZE`, `build_synthetic_trades()`
- `sb3_contrib` — `MaskablePPO`
- `stable_baselines3` — `SubprocVecEnv`, `DummyVecEnv`, `VecNormalize`, `VecMonitor`
- `gymnasium`
- `numpy`
- `torch.nn` (for activation_fn)

## Module 1: MultiSessionBarrierEnv

### File: `python/lob_rl/barrier/multi_session_env.py`

A Gymnasium wrapper that cycles through multiple trading sessions. Each `reset()` loads the next session's pre-built data and creates a fresh `BarrierEnv`. One episode = one session.

```python
class MultiSessionBarrierEnv(gymnasium.Env):
    def __init__(self, session_data_list, config=None, shuffle=False, seed=None):
        """
        Parameters
        ----------
        session_data_list : list[dict]
            Each dict has keys: 'bars', 'labels', 'features'.
            - bars: list[TradeBar]
            - labels: list[BarrierLabel]
            - features: np.ndarray of shape (n_usable, 13*h)
        config : RewardConfig, optional
        shuffle : bool
            If True, randomize session order each cycle.
        seed : int, optional
            Random seed for shuffling.
        """

    def reset(self, seed=None, options=None):
        """Load next session, create inner BarrierEnv, return first obs."""

    def step(self, action):
        """Delegate to inner BarrierEnv."""

    def action_masks(self):
        """Delegate to inner BarrierEnv."""

    @property
    def current_session_index(self):
        """Index of current session in the session_data_list."""

    @classmethod
    def from_bar_lists(cls, bar_lists, h=10, config=None, shuffle=False, seed=None):
        """Convenience factory: compute labels and features for each bar list."""
```

**Behavior:**
- Cycles through sessions round-robin. After all sessions consumed, wraps around to session 0 (or reshuffles if `shuffle=True`).
- `observation_space` and `action_space` are set from the first session's BarrierEnv and remain fixed.
- If a session has too few bars (< h), skip it silently.
- `action_masks()` delegates to the inner env.

### File: `python/lob_rl/barrier/barrier_vec_env.py`

Helper functions for creating vectorized environments compatible with SB3.

```python
def make_barrier_env_fn(session_data_list, config=None, shuffle=False, seed=None):
    """Return a callable that creates a MultiSessionBarrierEnv.

    For use with SubprocVecEnv / DummyVecEnv.
    """

def make_barrier_vec_env(session_data_list, n_envs=4, config=None,
                          shuffle=True, seed=42, use_subprocess=True):
    """Create a vectorized barrier environment.

    Parameters
    ----------
    session_data_list : list[dict]
        Session data as for MultiSessionBarrierEnv.
    n_envs : int
        Number of parallel environments.
    config : RewardConfig, optional
    shuffle : bool
    seed : int
    use_subprocess : bool
        If True, use SubprocVecEnv (multiprocess). Else DummyVecEnv.

    Returns
    -------
    VecEnv
        Ready for SB3 training.
    """
```

## Module 2: Training Script

### File: `scripts/train_barrier.py`

Standalone training script for barrier-hit PPO. Separate from `train.py` (which handles the legacy tick-level/bar-level pipeline).

**CLI arguments:**
```
--data-dir PATH        Directory with .dbn.zst files (or precomputed .npz)
--output-dir PATH      Where to save model, logs, metrics
--bar-size INT         Trade bar size (default: 500)
--lookback INT         Feature lookback h (default: 10)
--n-envs INT           Parallel environments (default: 4)
--total-timesteps INT  Total training steps (default: 2_000_000)
--eval-freq INT        Steps between eval runs (default: 10_000)
--checkpoint-freq INT  Steps between checkpoints (default: 50_000)
--train-frac FLOAT     Fraction of sessions for training (default: 0.8)
--seed INT             Random seed (default: 42)
--resume PATH          Resume from checkpoint
```

**Section 5.2 hyperparameters (hardcoded or with overrides):**
```python
policy_kwargs = dict(
    net_arch=[256, 256, dict(pi=[64], vf=[64])],
    activation_fn=nn.ReLU,
)
model = MaskablePPO(
    'MlpPolicy',
    env,
    learning_rate=linear_schedule(1e-4),  # linear decay to 0
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    seed=seed,
    verbose=1,
    tensorboard_log=output_dir / "tb",
)
```

**Data loading:**
1. Discover all `.dbn.zst` files in `data-dir` (or precomputed `.npz` bar caches).
2. Split into train/val/test by session (80/10/10 default, shuffled with seed).
3. For each session: load bars → compute labels → build features → store as dict.
4. Create `MultiSessionBarrierEnv` instances wrapped in `SubprocVecEnv`.
5. Optionally wrap with `VecNormalize` (features are already z-scored, but position/PnL may benefit).

**Training loop:**
1. Create model with Section 5.2 hyperparameters.
2. Train for `total_timesteps` with callbacks: eval, checkpoint, diagnostics.
3. Save final model + VecNormalize stats.

## Module 3: Training Diagnostics

### File: `python/lob_rl/barrier/training_diagnostics.py`

Custom SB3 callback that monitors Section 5.3 metrics.

```python
class BarrierDiagnosticCallback(BaseCallback):
    """Monitor training health per spec Section 5.3.

    Parameters
    ----------
    check_freq : int
        How often to check diagnostics (in updates, not steps).
    output_dir : Path, optional
        Where to write diagnostic CSV.
    verbose : int
    """

    def __init__(self, check_freq=1, output_dir=None, verbose=0):
        ...

    def _on_step(self) -> bool:
        """Called after each step. Accumulates per-step metrics."""

    def _on_rollout_end(self) -> None:
        """Called after each rollout collection. Computes and logs diagnostics."""

    @property
    def diagnostics(self) -> list[dict]:
        """Return all diagnostic snapshots (one per rollout)."""
```

**Metrics tracked per rollout (Section 5.3):**
- `entropy_flat`: Mean entropy on flat-state steps (healthy: 0.5–1.1)
- `value_loss`: Value function loss (healthy: decreasing trend)
- `policy_loss`: Policy gradient loss (healthy: small magnitude, stable)
- `episode_reward_mean`: Mean episode reward (healthy: > -0.20)
- `flat_action_rate`: Fraction of flat-state steps where agent chose flat (healthy: 10–90%)
- `trade_win_rate`: Fraction of completed trades with positive reward (healthy: > 33%)
- `n_trades`: Total trades completed in this rollout
- `has_nan`: Whether any NaN appeared in losses/gradients

**Red flag detection:**
```python
def check_red_flags(self) -> list[str]:
    """Return list of red flag descriptions, empty if healthy."""
```

Red flags (from Section 5.3):
- Entropy collapse: flat-state entropy < 0.3 in first 100 updates
- Value loss not decreasing: moving avg over 50 updates is increasing
- Episode reward below random baseline (-0.20) after 500 updates
- Flat action rate outside [10%, 90%]
- NaN in any loss or gradient

## Module 4: Linear Schedule Utility

### In `training_diagnostics.py` or `train_barrier.py`:

```python
def linear_schedule(initial_value: float):
    """Return a function that computes linear decay from initial_value to 0."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func
```

## Test Cases

### MultiSessionBarrierEnv (12 tests)

1. **`test_multi_session_reset_returns_obs_info`**: `reset()` returns `(obs, info)` with correct shape.

2. **`test_multi_session_obs_shape`**: Observation shape is `(132,)` (13×10 + 2).

3. **`test_multi_session_action_space`**: `action_space` is `Discrete(4)`.

4. **`test_multi_session_cycles_through_sessions`**: After completing episodes for N sessions, wraps back to session 0.

5. **`test_multi_session_episode_per_session`**: Each episode corresponds to one session's bars.

6. **`test_multi_session_shuffle`**: With `shuffle=True`, session order differs across cycles.

7. **`test_multi_session_deterministic_seed`**: Same seed produces same session order.

8. **`test_multi_session_skip_short_sessions`**: Sessions with fewer bars than lookback `h` are silently skipped.

9. **`test_multi_session_action_masks`**: `action_masks()` returns correct masks at each state.

10. **`test_multi_session_single_session`**: Works correctly with only 1 session.

11. **`test_multi_session_from_bar_lists`**: `from_bar_lists()` correctly computes labels and features.

12. **`test_multi_session_random_agent_10_episodes`**: Random agent completes 10 episodes across multiple sessions without crashing.

### Vectorized Environment (6 tests)

13. **`test_make_barrier_env_fn_callable`**: `make_barrier_env_fn()` returns a callable that creates a valid env.

14. **`test_make_barrier_vec_env_dummy`**: `make_barrier_vec_env(use_subprocess=False)` creates a working DummyVecEnv.

15. **`test_make_barrier_vec_env_n_envs`**: Created VecEnv has correct `num_envs`.

16. **`test_vec_env_step_returns_correct_shapes`**: Step returns arrays with correct batch dimensions.

17. **`test_vec_env_reset_returns_obs`**: Reset returns observation array of shape `(n_envs, 132)`.

18. **`test_vec_env_random_agent_completes`**: Random agent runs 5 episodes on vec env without error.

### Training Diagnostics (10 tests)

19. **`test_diagnostic_callback_init`**: Callback can be created with default parameters.

20. **`test_diagnostic_tracks_episode_reward`**: After a rollout, `episode_reward_mean` is populated.

21. **`test_diagnostic_tracks_flat_action_rate`**: After a rollout, `flat_action_rate` is in [0, 1].

22. **`test_diagnostic_tracks_trade_win_rate`**: After a rollout with trades, `trade_win_rate` is in [0, 1].

23. **`test_diagnostic_detects_nan`**: If NaN appears in losses, `has_nan` is True and red flag is raised.

24. **`test_diagnostic_no_red_flags_initially`**: Before training, `check_red_flags()` returns empty list.

25. **`test_diagnostic_entropy_collapse_flag`**: If flat-state entropy drops below 0.3 in first 100 updates, red flag raised.

26. **`test_diagnostic_writes_csv`**: When `output_dir` is set, diagnostics are written to a CSV file.

27. **`test_diagnostic_snapshots_accumulate`**: Each rollout adds one entry to `diagnostics` list.

28. **`test_diagnostic_flat_rate_red_flag`**: Flat action rate outside [10%, 90%] raises red flag.

### Training Script Smoke Tests (6 tests)

29. **`test_linear_schedule`**: `linear_schedule(1e-4)` returns 1e-4 at progress=1.0, 0 at progress=0.0, 5e-5 at progress=0.5.

30. **`test_train_barrier_creates_model`**: Training script can create a MaskablePPO model with correct hyperparameters on synthetic data.

31. **`test_train_barrier_model_predicts_with_masks`**: Model can predict actions given observations and action masks.

32. **`test_train_barrier_short_training_run`**: Training for 2048 steps (1 update) completes without error on synthetic data.

33. **`test_train_barrier_checkpoint_saved`**: After training with `checkpoint_freq`, checkpoint files exist in output dir.

34. **`test_train_barrier_resume_training`**: Training can be resumed from a saved checkpoint.

### Integration Tests (4 tests)

35. **`test_end_to_end_synthetic_training`**: Full pipeline: generate synthetic bars → build multi-session env → train MaskablePPO for 4096 steps → evaluate on held-out sessions → no errors.

36. **`test_action_masking_respected`**: During training, MaskablePPO never selects masked-out actions (verify via custom wrapper that logs actions).

37. **`test_eval_callback_runs`**: Eval callback evaluates on held-out sessions and logs mean reward.

38. **`test_policy_kwargs_architecture`**: Model's policy network has the architecture from Section 5.1: shared [256,256], policy head [64], value head [64].

## Implementation Notes

- **MaskablePPO from sb3-contrib** handles action masking natively. The env must implement `action_masks()` returning a numpy array. `MultiSessionBarrierEnv` delegates this to the inner `BarrierEnv`.
- **Session data is precomputed** before training starts. Each session's bars, labels, and features are computed once and stored in memory (or lazy-loaded from cache).
- **SubprocVecEnv** requires that env creation be done inside the subprocess. Pass `session_data_list` to the factory function; each subprocess creates its own `MultiSessionBarrierEnv`.
- **VecNormalize** may not be needed since features are already z-scored. However, the position and unrealized_pnl fields are not normalized. Consider normalizing only the last 2 dims, or skipping VecNormalize entirely.
- **Linear schedule** for learning rate: `progress_remaining` goes from 1.0 (start) to 0.0 (end). `lr = initial * progress_remaining`.
- **For testing**, use synthetic data from `generate_random_walk()` to create sessions of ~50-100 bars each. 5-10 sessions is enough for unit tests.
- **The training script** is primarily tested via its constituent modules (MultiSessionBarrierEnv, diagnostics callback, vec env helpers). The script itself gets a lightweight smoke test.
- **Serialization**: MaskablePPO uses the same `.save()` / `.load()` API as standard PPO.
- **train_barrier.py** is a NEW script, not an extension of `train.py`. The data loading, env creation, and model setup are fundamentally different from the legacy pipeline.
