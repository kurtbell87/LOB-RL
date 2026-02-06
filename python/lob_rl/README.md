# python/lob_rl — Python Package

Gymnasium wrappers and utilities on top of the C++ `lob_rl_core` module.

## Files

| File | Role |
|---|---|
| `__init__.py` | Package init. |
| `_config.py` | Shared `SessionConfig` conversion helper. |
| `gym_env.py` | `LOBGymEnv` — single-day `gymnasium.Env` wrapping `lob_rl_core.LOBEnv`. 44-float obs, Discrete(3) actions. |
| `precomputed_env.py` | `PrecomputedEnv` — single-day env backed by precomputed numpy arrays. Zero C++ at runtime. |
| `multi_day_env.py` | `MultiDayEnv` — cycles through multiple day files. Precomputes all at construction. Sequential or shuffle mode. **This is what `train.py` uses.** |
| `convert_dbn.py` | Converts Databento `.dbn.zst` → flat binary `.bin`. CLI: `uv run python -m lob_rl.convert_dbn --input-dir ... --output-dir ... --instrument-id ...` |

## Typical usage

```python
from lob_rl.multi_day_env import MultiDayEnv

env = MultiDayEnv(data_dir="data/mes", shuffle=True, seed=42)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(1)  # Buy
```
