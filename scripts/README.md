# scripts — Utility Scripts

## Files

| File | Role |
|---|---|
| `train.py` | **Main training entry point.** PPO/RecurrentPPO via SB3 on `MultiDayEnv`. CLI flags: `--cache-dir`, `--bar-size`, `--step-interval`, `--shuffle-split`, `--seed`, `--frame-stack`, `--recurrent`, `--train-days`, `--total-timesteps`, `--policy-arch`, `--activation`, `--ent-coef`, `--learning-rate`, `--execution-cost`, `--checkpoint-freq`, `--resume`. Sortino evaluation. Run with: `cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py --cache-dir ../cache/mes/` |
| `precompute_cache.py` | **Cache builder.** Reads `.dbn.zst` files, calls C++ `precompute()` per day, saves `{date}.npz`. Supports `--roll-calendar` and `--workers N` for parallel processing (`ProcessPoolExecutor`). Run with: `--data-dir ../data/mes --out ../cache/mes/ --roll-calendar ../data/mes/roll_calendar.json --workers 8` |
| `supervised_diagnostic.py` | **MLP capacity diagnostic.** Tests if MLP can predict optimal actions in supervised setting. Supports `--bar-size N` for bar-level features. |
| `watch_download.sh` | Data download progress monitor. |
| `tdd-watch.py` | Live-tail TDD phase output. Used by `./tdd.sh watch`. |
