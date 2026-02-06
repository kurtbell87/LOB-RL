# Training Pipeline v2 — Fix PPO Hyperparameters & Normalization

## Problem

The baseline training run (500k steps) produced Sortino -1.05 val / -14.4 test with entropy collapse. Root causes identified:

1. `ent_coef=0.0` (SB3 default) — no entropy regularization, causes premature convergence
2. No observation normalization — raw LOB features on wildly different scales
3. No reward normalization — reward magnitudes vary across market conditions
4. Single environment — only 1 env in `DummyVecEnv`, low sample throughput
5. Evaluation bug — `evaluate_sortino()` doesn't pass `execution_cost` to eval envs

## What to Build

All changes are in `scripts/train.py`. No C++ or environment code changes.

### 1. Entropy coefficient

Set `ent_coef=0.01` in the PPO constructor. Add CLI flag `--ent-coef` (default 0.01) so it's tunable without code changes.

### 2. VecNormalize wrapper

Wrap the vectorized env with `stable_baselines3.common.vec_env.VecNormalize`:

```python
from stable_baselines3.common.vec_env import VecNormalize

env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
```

After training, save the normalization statistics alongside the model:

```python
env.save(os.path.join(args.output_dir, 'vec_normalize.pkl'))
```

During evaluation, load the stats and wrap eval envs with VecNormalize in eval mode (`norm_reward=False`, `training=False`):

```python
eval_env = VecNormalize.load(stats_path, eval_env)
eval_env.training = False
eval_env.norm_reward = False
```

Add CLI flag `--no-norm` to disable normalization (for A/B testing against old behavior).

### 3. Parallel environments

Replace `DummyVecEnv` with `SubprocVecEnv` using `n_envs` parallel environments.

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_train_env(file_paths, session_config, reward_mode, lambda_, execution_cost):
    def _init():
        return MultiDayEnv(
            file_paths=file_paths,
            session_config=session_config,
            steps_per_episode=0,
            reward_mode=reward_mode,
            lambda_=lambda_,
            shuffle=True,
            execution_cost=execution_cost,
        )
    return _init

env = SubprocVecEnv([
    make_train_env(train_paths, session_config, args.reward_mode, args.lambda_, args.execution_cost)
    for _ in range(args.n_envs)
])
```

Add CLI flag `--n-envs` (default 8).

**Important:** The `make_train_env` factory must use a closure that captures parameters by value (not a lambda with late binding), otherwise all subprocesses will share the same variable references.

### 4. Fix evaluation to use execution_cost

`evaluate_sortino()` currently calls `make_env()` which doesn't forward `execution_cost`. Fix:

- Pass `execution_cost` through to `make_env()` and into `LOBGymEnv`.
- Also wrap eval env with VecNormalize (loaded from training stats, `training=False`).

The evaluation function should accept `execution_cost` as a parameter:

```python
def evaluate_sortino(model, eval_files, n_eval_episodes=10, execution_cost=False, vec_normalize_path=None):
```

For each eval episode:
- Create a `DummyVecEnv` wrapping the single eval env (needed for VecNormalize)
- If `vec_normalize_path` is set, load and apply VecNormalize in eval mode
- Run the episode through the wrapped env
- Collect returns

### 5. Updated PPO hyperparameters

```python
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,       # was 64 — larger for stability with more envs
    n_epochs=5,           # was 10 — less overfitting per rollout
    gamma=0.99,
    ent_coef=args.ent_coef,  # was 0.0 (!)
    vf_coef=0.5,
    max_grad_norm=0.5,
    clip_range=0.2,
    verbose=1,
    tensorboard_log=...,
)
```

Add CLI flags for key hyperparameters:
- `--ent-coef` (default 0.01)
- `--n-envs` (default 8)
- `--batch-size` (default 256)
- `--n-epochs` (default 5)
- `--learning-rate` (default 3e-4)

### 6. New CLI flags summary

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--ent-coef` | float | 0.01 | Entropy coefficient for PPO |
| `--n-envs` | int | 8 | Number of parallel training environments |
| `--batch-size` | int | 256 | PPO minibatch size |
| `--n-epochs` | int | 5 | PPO epochs per rollout |
| `--learning-rate` | float | 3e-4 | Learning rate |
| `--no-norm` | flag | False | Disable VecNormalize |

Existing flags are unchanged: `--data-dir`, `--train-days`, `--total-timesteps`, `--reward-mode`, `--lambda`, `--execution-cost`, `--output-dir`.

## Edge Cases

- **`--n-envs 1`**: Should still work, using `SubprocVecEnv` with 1 process. Alternatively fall back to `DummyVecEnv` for n_envs=1 to avoid subprocess overhead — either approach is acceptable.
- **`--no-norm` with evaluation**: When normalization is disabled, eval should skip loading VecNormalize stats. The `vec_normalize.pkl` file won't exist.
- **Eval with fewer files than `n_eval_episodes`**: Already handled (slices `eval_files[:n_eval_episodes]`).
- **MultiDayEnv pickling for SubprocVecEnv**: `SubprocVecEnv` uses `multiprocessing` which pickles the env factory. The factory function (not lambda) pattern above avoids pickling issues. The `lob_rl_core` C++ module is imported inside `PrecomputedEnv.from_file()`, so each subprocess imports it independently.

## Acceptance Criteria

1. `train.py --help` shows all new CLI flags with correct defaults.
2. Training with `--n-envs 1 --no-norm` produces equivalent behavior to the old code (minus hyperparameter changes).
3. Training with default flags uses SubprocVecEnv with 8 envs, VecNormalize, and `ent_coef=0.01`.
4. After training, `vec_normalize.pkl` is saved alongside the model.
5. Evaluation uses `execution_cost` when `--execution-cost` is passed.
6. Evaluation applies VecNormalize stats in eval mode (no updating running stats).
7. All existing tests continue to pass (no changes to env code).

## Files to Modify

- `scripts/train.py` — all changes are here.

## Files NOT to Modify

- `python/lob_rl/precomputed_env.py` — no changes.
- `python/lob_rl/multi_day_env.py` — no changes.
- `python/lob_rl/gym_env.py` — no changes.
- C++ code — no changes.
