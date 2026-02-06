# B2: Make convert_dbn.py Read Real .dbn.zst Files

## Problem

`convert_dbn.py` only handles `MockRecord` objects and `.mock.json` files. There is no `import databento`, no code that opens `.dbn.zst` files, and no CLI entry point. Real Databento data cannot be converted.

## What to Build

Extend `convert_dbn.py` so `convert_directory()` reads real `.dbn.zst` files (via `databento.DBNStore.from_file()`) in addition to the existing `.mock.json` test path. Add a `__main__` CLI entry point.

## Interface

### `convert_directory()` changes

- Accept `.dbn.zst` files in `input_dir` (glob pattern: `*.mbo.dbn.zst`)
- For each `.dbn.zst` file, open with `databento.DBNStore.from_file(path)`, iterate records
- Pass records directly to `convert_file()` — `MBOMsg` objects have the same field names as `MockRecord` (ts_event, order_id, price, size, action, side, flags, instrument_id)
- Extract date from filename pattern `glbx-mdp3-YYYYMMDD.mbo.dbn.zst`
- If both `.mock.json` and `.dbn.zst` files exist, process both (mock files first for test compatibility)
- `symbol` parameter used for manifest metadata only (not for filtering — use `instrument_id`)

### CLI entry point (`__main__`)

When run as `uv run python -m lob_rl.convert_dbn`:
- `--input-dir`: Path to directory with `.dbn.zst` files (required)
- `--output-dir`: Path to write `.bin` files and `manifest.json` (required)
- `--symbol`: Symbol name for manifest metadata (default: "MESH5")
- `--instrument-id`: Integer instrument ID to filter (required)

### Action mapping note

Databento MBO data includes action 'F' (fill), which should be treated the same as 'T' (trade). Update `convert_file()` to map 'F' -> 'T' before writing, so BinaryFileSource only needs to handle A/C/M/T.

## Requirements

1. `convert_directory()` finds and processes `*.mbo.dbn.zst` files using `databento.DBNStore.from_file()`
2. Date extracted from filename via regex `glbx-mdp3-(\d{8})\.mbo\.dbn\.zst`
3. Action 'F' mapped to 'T' in `convert_file()` before writing the action byte
4. `__main__.py` or `if __name__ == "__main__"` block provides CLI with argparse
5. Existing `.mock.json` path continues to work (backward compatible)
6. Manifest includes all converted files with correct record counts and timestamps

## Edge Cases

- Skip `.dbn.zst` files that produce zero records for the given instrument_id (don't write empty .bin files, or write them with record_count=0)
- Handle the case where `databento` is not installed gracefully (import error with helpful message)

## Acceptance Criteria

- `convert_directory()` successfully processes a directory of `.dbn.zst` files
- Output `.bin` files are readable by `BinaryFileSource` (valid header, correct record format)
- `manifest.json` written with correct metadata
- CLI entry point works: `uv run python -m lob_rl.convert_dbn --input-dir ... --output-dir ... --instrument-id 42005347`
- Existing mock-based tests still pass
- New test verifying 'F' action is mapped to 'T' (ord('T')) in output
