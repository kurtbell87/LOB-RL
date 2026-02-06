# Last Touch — Cold-Start Briefing

## What to do next

**Step 4c is DONE.** First training run completed. Results below. Next steps are hyperparameter tuning and improving the agent's performance.

### Baseline results (500k timesteps, vanilla PPO)

| Metric | Validation (5 days) | Test (2 days) |
|---|---|---|
| Mean return | -39.8 | -76.7 |
| Sortino ratio | -1.05 | -14.4 |
| Profitable episodes | 0/5 | 0/2 |

- **Throughput:** 3,350 steps/sec sustained, 2.5 min total training time
- **Model saved:** `runs/ppo_lob.zip`
- **TensorBoard:** `runs/tb_logs/`
- **Key observation:** Entropy collapsed to near-zero early (degenerate policy), then recovered at ~450k steps. Suggests the reward signal is sparse and the agent needs more timesteps, better reward shaping, or both.

### Recommended next steps (pick one)

1. **More timesteps:** Re-run with `--total-timesteps 5000000` (5M). ~25 min at current throughput.
2. **Inventory penalty:** Try `--reward-mode pnl_delta_penalized --lambda 0.01` to discourage holding.
3. **Observation normalization:** Wrap env with `VecNormalize` for stable learning.
4. **Network architecture:** Try larger MLP (256x256) or LSTM policy for temporal patterns.

### To re-run training

```bash
cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py --data-dir ../data/mes --total-timesteps 5000000
```

## Key files for current task

| File | Role |
|---|---|
| `scripts/train.py` | Training entry point. PPO with SB3, Sortino evaluation. Uses `MultiDayEnv`. |
| `runs/ppo_lob.zip` | Saved baseline model (500k steps, negative Sortino). |
| `runs/tb_logs/` | TensorBoard training logs. |
| `python/lob_rl/multi_day_env.py` | Gym env cycling through day files. Precomputes all days at construction. |
| `data/mes/manifest.json` | 27 days of /MES MBO data. 20 train / 5 val / 2 test split. |

## Don't waste time on

- **Build verification** — `build-release/` is current, 404 C++ tests pass.
- **Dependency checks** — SB3, gymnasium, numpy, tensorboard all installed.
- **Reading PRD.md** — everything relevant is in this file.
- **Codebase exploration** — read directory `README.md` files instead (every key dir has one).

## Architecture overview

```
data/mes/*.bin  →  BinaryFileSource (C++)  →  Book (C++)  →  LOBEnv (C++)
                                                                   ↓
                                                           precompute() (C++)
                                                                   ↓
                                                         numpy arrays (Python)
                                                                   ↓
                                                    MultiDayEnv (Python/Gym)
                                                                   ↓
                                                      SB3 PPO (scripts/train.py)
```

## Test coverage

- **404 C++ tests** — `cd build-release && ./lob_tests`
- **463 Python tests** — `cd build-release && PYTHONPATH=.:../python uv run pytest ../python/tests/`
- **867 total**, all passing.

## Remaining work

| Item | Priority | Notes |
|---|---|---|
| Hyperparameter tuning / longer training | High | Baseline Sortino is negative; agent needs more signal |
| `binary_file_source.cpp:62` int64→double precision loss | Medium | Low-impact for real financial data |
| Test efficiency refactoring | Optional | Consolidate empty book tests, extract shared helpers |
| DST handling for session boundaries | Low | Current dataset is entirely non-DST |

---

For project history, see `git log`.
