# scripts — Utility Scripts

## Files

| File | Role |
|---|---|
| `train.py` | **Main training entry point.** PPO via SB3 on `MultiDayEnv`. 16 CLI flags incl. `--cache-dir`, `--step-interval`. Sortino evaluation. Run with: `cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py --cache-dir ../cache/mes/` |
| `precompute_cache.py` | **Cache builder.** Reads manifest, calls C++ `precompute()` once per day, saves `{date}.npz`. Run with: `cd build-release && PYTHONPATH=.:../python uv run python ../scripts/precompute_cache.py --data-dir ../data/mes --out ../cache/mes/` |
| `supervised_diagnostic.py` | **MLP capacity diagnostic.** Tests if MLP can predict optimal actions in supervised setting. Supports `--bar-size N` for bar-level features. |
| `tdd-watch.py` | Live-tail TDD phase output. Used by `./tdd.sh watch`. |
