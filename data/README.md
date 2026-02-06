# data — Market Data

## Directories

| Directory | Contents |
|---|---|
| `mes/` | 27 days of /MES (Micro E-mini S&P 500) MBO data as flat `.bin` files. Dec 2024 – Jan 2025. |

## data/mes/

- `manifest.json` — lists all files with train/test split (20 train / 7 test)
- `*.bin` — flat binary files (16-byte header + 36-byte records), one per trading day
- Created by: `uv run python -m lob_rl.convert_dbn --input-dir <dbn_dir> --output-dir data/mes --instrument-id <id>`
