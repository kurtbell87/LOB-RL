# scripts — Utility Scripts

## Files

| File | Role |
|---|---|
| `train.py` | **Main training entry point.** PPO/RecurrentPPO via SB3 on `MultiDayEnv`. CLI flags: `--cache-dir`, `--bar-size`, `--step-interval`, `--shuffle-split`, `--seed`, `--frame-stack`, `--recurrent`, `--train-days`, `--total-timesteps`, `--policy-arch`, `--activation`, `--ent-coef`, `--learning-rate`, `--execution-cost`, `--checkpoint-freq`, `--resume`, `--output-dir`, `--config`, `--val-days`, `--n-eval-episodes`. Writes `run_manifest.yaml` and schema-validated `metrics.json`. Sortino evaluation. Run with: `cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py --cache-dir ../cache/mes/` |
| `start.sh` | **RunPod container entrypoint.** Starts sshd (picks up `PUBLIC_KEY` env var), constructs unique output dir `/workspace/runs/{EXP_NAME}_{RUNPOD_POD_ID}/`, runs `train.py --output-dir ... "$@"` in background, keeps container alive via `sleep infinity`. With no args, idles for manual SSH use. Used as Dockerfile `ENTRYPOINT`. |
| `precompute_cache.py` | **Cache builder.** Reads `.dbn.zst` files, calls C++ `precompute()` per day, saves `{date}.npz`. Supports `--roll-calendar` and `--workers N` for parallel processing (`ProcessPoolExecutor`). Run with: `--data-dir ../data/mes --out ../cache/mes/ --roll-calendar ../data/mes/roll_calendar.json --workers 8` |
| `precompute_barrier_cache.py` | **Barrier cache builder.** Reads `.dbn.zst` files through barrier pipeline (bars → labels → features with MBO book reconstruction). Saves per-session `.npz` with N_FEATURES version check. Supports `--workers N`. Key functions: `parse_date_str()`, `process_session()`, `load_session_from_cache()`. Run with: `--data-dir ../data/mes/ --output-dir ../cache/barrier/ --roll-calendar ../data/mes/roll_calendar.json --workers 8` |
| `supervised_diagnostic.py` | **MLP capacity diagnostic.** Tests if MLP can predict optimal actions in supervised setting. Supports `--bar-size N` for bar-level features. |
| `watch_download.sh` | Data download progress monitor. |
| `tdd-watch.py` | Live-tail TDD phase output. Used by `./tdd.sh watch`. |
