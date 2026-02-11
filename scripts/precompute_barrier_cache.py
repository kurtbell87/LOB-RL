#!/usr/bin/env python3
"""Precompute barrier pipeline data from .dbn.zst files.

Uses C++ backend (lob_rl_core.barrier_precompute) for single-pass
MBO stream processing: bar construction + feature computation +
normalization + lookback assembly + label computation.

Usage:
  cd build-release
  PYTHONPATH=.:../python uv run python ../scripts/precompute_barrier_cache.py \
    --data-dir ../data/mes/ --output-dir ../cache/barrier/ \
    --roll-calendar ../data/mes/roll_calendar.json \
    --bar-size 500 --lookback 10 --workers 8
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def parse_date_str(filepath):
    """Extract YYYYMMDD date string from a .dbn.zst filename."""
    return filepath.name.split("-")[-1].split(".")[0]


def get_instrument_id(date_str, roll_calendar):
    """Look up front-month instrument_id for a YYYY-MM-DD date."""
    for roll in roll_calendar["rolls"]:
        if roll["start"] <= date_str <= roll["end"]:
            return roll["instrument_id"]
    return None


def process_session(filepath, instrument_id, bar_size, lookback, a, b, t_max):
    """Process a single .dbn.zst file into bars, labels, features via C++.

    Returns dict with numpy arrays, or None if session has insufficient data.
    """
    import lob_rl_core

    result = lob_rl_core.barrier_precompute(
        str(filepath), instrument_id,
        bar_size=bar_size, lookback=lookback, a=a, b=b, t_max=t_max)
    if result is None:
        return None

    # Add summary stats computed from long labels
    label_values = result["label_values"]
    n = len(label_values)
    result["p_plus"] = np.array(np.sum(label_values == 1) / n, dtype=np.float64)
    result["p_minus"] = np.array(np.sum(label_values == -1) / n, dtype=np.float64)
    result["p_zero"] = np.array(np.sum(label_values == 0) / n, dtype=np.float64)
    result["tiebreak_freq"] = np.array(0.0, dtype=np.float64)

    # Add summary stats computed from short labels
    short_label_values = result["short_label_values"]
    result["short_p_plus"] = np.array(np.sum(short_label_values == 1) / n, dtype=np.float64)
    result["short_p_minus"] = np.array(np.sum(short_label_values == -1) / n, dtype=np.float64)
    result["short_p_zero"] = np.array(np.sum(short_label_values == 0) / n, dtype=np.float64)

    return result


def process_file(args_tuple):
    """Worker function for ProcessPoolExecutor."""
    filepath, instrument_id, bar_size, lookback, a, b, t_max, output_path = args_tuple

    date_str = parse_date_str(filepath)

    try:
        t0 = time.time()
        result = process_session(filepath, instrument_id, bar_size, lookback, a, b, t_max)
        elapsed = time.time() - t0

        if result is None:
            return date_str, "skipped", 0, 0, elapsed

        np.savez_compressed(output_path, **result)
        n_bars = int(result["n_bars"])
        n_usable = int(result["n_usable"])
        return date_str, "ok", n_bars, n_usable, elapsed

    except Exception as e:
        return date_str, f"error: {e}", 0, 0, 0.0


def load_session_from_cache(npz_path):
    """Load a cached session .npz and reconstruct TradeBar/BarrierLabel objects.

    Returns dict with keys: bars, labels, features (compatible with MultiSessionBarrierEnv).

    Raises ValueError if the cache has an n_features key that doesn't match
    the current N_FEATURES constant.
    """
    from lob_rl.barrier import N_FEATURES
    from lob_rl.barrier.bar_pipeline import TradeBar
    from lob_rl.barrier.label_pipeline import BarrierLabel

    data = np.load(npz_path, allow_pickle=False)

    # Version check: if n_features is stored, it must match current N_FEATURES
    if "n_features" in data:
        stored_n_features = int(data["n_features"])
        if stored_n_features != N_FEATURES:
            raise ValueError(
                f"Cache n_features mismatch: stored {stored_n_features}, "
                f"expected {N_FEATURES}. Re-precompute with "
                f"precompute_barrier_cache.py."
            )

    # Check for short labels (required for modern caches with n_features)
    has_n_features = "n_features" in data
    if has_n_features and "short_label_values" not in data:
        raise ValueError(
            "Cache missing short labels. Re-precompute with "
            "precompute_barrier_cache.py to add dual-direction labels."
        )

    n_bars = int(data["n_bars"])
    features = data["features"]
    offsets = data["bar_trade_offsets"]
    trade_prices_flat = data["trade_prices"]
    trade_sizes_flat = data["trade_sizes"]

    # Reconstruct TradeBar objects
    bars = []
    for i in range(n_bars):
        start_off = offsets[i]
        end_off = offsets[i + 1]
        tp = trade_prices_flat[start_off:end_off]
        ts = trade_sizes_flat[start_off:end_off]

        bar = TradeBar(
            bar_index=i,
            open=float(data["bar_open"][i]),
            high=float(data["bar_high"][i]),
            low=float(data["bar_low"][i]),
            close=float(data["bar_close"][i]),
            volume=int(data["bar_volume"][i]),
            vwap=float(data["bar_vwap"][i]),
            t_start=int(data["bar_t_start"][i]),
            t_end=int(data["bar_t_end"][i]),
            session_date="",
            trade_prices=tp,
            trade_sizes=ts,
        )
        bars.append(bar)

    # Reconstruct BarrierLabel objects
    label_values = data["label_values"]
    label_tau = data["label_tau"]
    label_resolution_bar = data["label_resolution_bar"]
    labels = []
    for i in range(n_bars):
        lbl = BarrierLabel(
            bar_index=i,
            label=int(label_values[i]),
            tau=int(label_tau[i]),
            resolution_type="precomputed",
            entry_price=float(data["bar_close"][i]),
            resolution_bar=int(label_resolution_bar[i]),
        )
        labels.append(lbl)

    result = {"bars": bars, "labels": labels, "features": features}

    # Reconstruct short-direction BarrierLabel objects (if present)
    if "short_label_values" in data:
        short_label_values = data["short_label_values"]
        short_label_tau = data["short_label_tau"]
        short_label_resolution_bar = data["short_label_resolution_bar"]
        short_labels = []
        for i in range(n_bars):
            lbl = BarrierLabel(
                bar_index=i,
                label=int(short_label_values[i]),
                tau=int(short_label_tau[i]),
                resolution_type="precomputed",
                entry_price=float(data["bar_close"][i]),
                resolution_bar=int(short_label_resolution_bar[i]),
            )
            short_labels.append(lbl)
        result["short_labels"] = short_labels

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Precompute barrier pipeline data from .dbn.zst files.",
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with .mbo.dbn.zst files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for .npz cache files")
    parser.add_argument("--roll-calendar", type=str, required=True,
                        help="Path to roll_calendar.json")
    parser.add_argument("--bar-size", type=int, default=500,
                        help="Trades per bar (default: 500)")
    parser.add_argument("--lookback", type=int, default=10,
                        help="Feature lookback h (default: 10)")
    parser.add_argument("--a", type=int, default=20,
                        help="Profit barrier in ticks (default: 20)")
    parser.add_argument("--b", type=int, default=10,
                        help="Stop barrier in ticks (default: 10)")
    parser.add_argument("--t-max", type=int, default=40,
                        help="Max holding period in bars (default: 40)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default: 1)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.roll_calendar) as f:
        roll_calendar = json.load(f)

    # Discover .dbn.zst files
    files = sorted(data_dir.glob("*.mbo.dbn.zst"))
    if not files:
        print(f"ERROR: No .mbo.dbn.zst files in {data_dir}")
        sys.exit(1)

    print(f"=== Barrier Pipeline Precompute ===")
    print(f"Data dir:    {data_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Files:       {len(files)}")
    print(f"Bar size:    {args.bar_size}")
    print(f"Lookback:    {args.lookback}")
    print(f"Barriers:    a={args.a}, b={args.b}, T_max={args.t_max}")
    print(f"Workers:     {args.workers}")
    print()

    # Build work items
    work_items = []
    for filepath in files:
        date_str = parse_date_str(filepath)
        iso_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        instrument_id = get_instrument_id(iso_date, roll_calendar)

        if instrument_id is None:
            print(f"  SKIP {filepath.name}: no roll calendar entry for {iso_date}")
            continue

        output_path = output_dir / f"{date_str}.npz"
        if output_path.exists():
            print(f"  SKIP {filepath.name}: already cached")
            continue

        work_items.append((
            filepath, instrument_id, args.bar_size, args.lookback,
            args.a, args.b, args.t_max, output_path,
        ))

    if not work_items:
        print("Nothing to process — all files already cached.")
        return

    print(f"Processing {len(work_items)} files...")
    print()

    t0_total = time.time()
    ok_count = 0
    skip_count = 0
    error_count = 0
    total_bars = 0
    total_usable = 0

    if args.workers <= 1:
        # Sequential
        for item in work_items:
            date_str, status, n_bars, n_usable, elapsed = process_file(item)
            if status == "ok":
                ok_count += 1
                total_bars += n_bars
                total_usable += n_usable
                print(f"  {date_str}: {n_bars} bars, {n_usable} usable, {elapsed:.1f}s")
            elif status == "skipped":
                skip_count += 1
                print(f"  {date_str}: skipped (insufficient data)")
            else:
                error_count += 1
                print(f"  {date_str}: {status}")
    else:
        # Parallel
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_file, item): item for item in work_items}
            for future in as_completed(futures):
                date_str, status, n_bars, n_usable, elapsed = future.result()
                if status == "ok":
                    ok_count += 1
                    total_bars += n_bars
                    total_usable += n_usable
                    print(f"  {date_str}: {n_bars} bars, {n_usable} usable, {elapsed:.1f}s")
                elif status == "skipped":
                    skip_count += 1
                    print(f"  {date_str}: skipped (insufficient data)")
                else:
                    error_count += 1
                    print(f"  {date_str}: {status}")

    elapsed_total = time.time() - t0_total

    print()
    print(f"=== Precompute Complete ===")
    print(f"OK:      {ok_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors:  {error_count}")
    print(f"Total bars:   {total_bars}")
    print(f"Total usable: {total_usable}")
    print(f"Wall time:    {elapsed_total:.1f}s")
    print(f"Output:       {output_dir}/")

    # Summary of cache
    cache_files = sorted(output_dir.glob("*.npz"))
    total_size = sum(f.stat().st_size for f in cache_files)
    print(f"Cache files:  {len(cache_files)}")
    print(f"Cache size:   {total_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
