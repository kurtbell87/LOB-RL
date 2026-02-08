# data — Market Data

## Directories

| Directory | Contents |
|---|---|
| `mes/` | 312 days of /MES (Micro E-mini S&P 500) MBO data as `.mbo.dbn.zst` files. Jan–Dec 2022. 57GB total. |

## data/mes/

- `roll_calendar.json` — maps each date → front-month `instrument_id`. 4 roll periods (MESH2, MESM2, MESU2, MESZ2). Used by `precompute_cache.py --roll-calendar`.
- `manifest.json` — legacy file listing (from old .bin era). Not used by current pipeline.
- `*.mbo.dbn.zst` — Databento native compressed MBO files, one per trading day. Read by `DbnFileSource` (C++) via `precompute_cache.py`.

## Precomputed cache

`../cache/mes/` contains 249 `.npz` files (one per trading day, weekends/holidays excluded). Built with:
```bash
cd build-release && PYTHONPATH=.:../python uv run python ../scripts/precompute_cache.py \
  --data-dir ../data/mes/ --out ../cache/mes/ --roll-calendar ../data/mes/roll_calendar.json --workers 8
```
