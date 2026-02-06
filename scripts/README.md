# scripts — Utility Scripts

## Files

| File | Role |
|---|---|
| `train.py` | **Main training entry point.** PPO via SB3 on `MultiDayEnv`. 20 train days / 7 test days from `data/mes/manifest.json`. Sortino evaluation. Run with: `cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py --data-dir ../data/mes` |
| `tdd-watch.py` | Live-tail TDD phase output. Used by `./tdd.sh watch`. |
