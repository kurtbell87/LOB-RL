"""Precompute LOB data and cache as .npz files for fast training startup."""

import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import numpy as np
import lob_rl_core


def _extract_date_from_filename(filename):
    """Extract YYYYMMDD or YYYY-MM-DD date from filename."""
    # Try YYYY-MM-DD pattern first
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return match.group(1)
    # Try YYYYMMDD pattern (used by dbn files like glbx-mdp3-20240115.mbo.dbn.zst)
    match = re.search(r'(\d{8})', filename)
    if match:
        return match.group(1)
    return None


def _normalize_date(date_str):
    """Normalize date to YYYYMMDD format for comparison."""
    return date_str.replace('-', '')


def _load_roll_calendar(path):
    """Load roll calendar JSON → dict mapping YYYYMMDD → instrument_id.

    Returns None if no calendar provided. Otherwise returns a dict
    where each key is YYYYMMDD and value is the instrument_id for
    that date's front-month contract.
    """
    with open(path) as f:
        cal = json.load(f)

    date_to_id = {}
    for roll in cal['rolls']:
        inst_id = roll['instrument_id']
        start = _normalize_date(roll['start'])
        end = _normalize_date(roll['end'])
        # Fill every date in range (we only need dates that have data files,
        # but pre-filling is cheap and avoids date arithmetic)
        from datetime import date, timedelta
        d = date(int(start[:4]), int(start[4:6]), int(start[6:8]))
        d_end = date(int(end[:4]), int(end[4:6]), int(end[6:8]))
        while d <= d_end:
            date_to_id[d.strftime('%Y%m%d')] = inst_id
            d += timedelta(days=1)

    return date_to_id


def _precompute_one(data_path, npz_path, date_norm, instrument_id):
    """Precompute a single day. Returns a status string or None if skipped."""
    cfg = lob_rl_core.SessionConfig.default_rth()
    obs, mid, spread, num_steps = lob_rl_core.precompute(
        data_path, cfg, instrument_id
    )
    if num_steps < 2:
        return None
    np.savez(npz_path, obs=obs, mid=mid, spread=spread)
    return (f"  Cached {date_norm} (inst={instrument_id}): "
            f"obs={obs.shape}, mid={mid.shape}, spread={spread.shape}")


def main():
    parser = argparse.ArgumentParser(description='Precompute and cache LOB data as .npz files')
    parser.add_argument('--data-dir', required=True, help='Directory with .dbn.zst files')
    parser.add_argument('--out', required=True, help='Output directory for cached .npz files')

    id_group = parser.add_mutually_exclusive_group(required=True)
    id_group.add_argument('--instrument-id', type=int,
                          help='Fixed instrument ID for all files (uint32)')
    id_group.add_argument('--roll-calendar', type=str,
                          help='Path to roll_calendar.json (maps dates to instrument IDs)')

    parser.add_argument('--force', action='store_true', default=False,
                        help='Re-cache even if .npz already exists')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel worker processes (default: 1)')
    args = parser.parse_args()

    # Load roll calendar if provided
    roll_map = None
    if args.roll_calendar:
        roll_map = _load_roll_calendar(args.roll_calendar)
        print(f"Loaded roll calendar: {len(roll_map)} dates mapped")

    # Glob for .dbn.zst files
    data_files = sorted(glob.glob(os.path.join(args.data_dir, '*.mbo.dbn.zst')))

    os.makedirs(args.out, exist_ok=True)

    # Build work items (filter before dispatching to workers)
    work_items = []
    skipped_exist = 0
    skipped_no_roll = 0

    for data_path in data_files:
        date = _extract_date_from_filename(os.path.basename(data_path))
        if date is None:
            print(f"  Skipping {data_path}: cannot extract date from filename")
            continue

        date_norm = _normalize_date(date)

        if roll_map is not None:
            if date_norm not in roll_map:
                skipped_no_roll += 1
                continue
            instrument_id = roll_map[date_norm]
        else:
            instrument_id = args.instrument_id

        npz_path = os.path.join(args.out, f"{date_norm}.npz")

        if not args.force and os.path.exists(npz_path):
            skipped_exist += 1
            print(f"  Skipping {date_norm}: already cached")
            continue

        work_items.append((data_path, npz_path, date_norm, instrument_id))

    print(f"Dispatching {len(work_items)} files to {args.workers} workers "
          f"({skipped_exist} already cached, {skipped_no_roll} not in roll calendar)")

    cached = 0
    skipped_empty = 0

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_precompute_one, data_path, npz_path, date_norm, inst_id): date_norm
                for data_path, npz_path, date_norm, inst_id in work_items
            }
            for future in as_completed(futures):
                date_norm = futures[future]
                result = future.result()
                if result is None:
                    skipped_empty += 1
                else:
                    cached += 1
                    print(result, flush=True)
    else:
        for data_path, npz_path, date_norm, inst_id in work_items:
            result = _precompute_one(data_path, npz_path, date_norm, inst_id)
            if result is None:
                skipped_empty += 1
            else:
                cached += 1
                print(result, flush=True)

    total_size = sum(
        os.path.getsize(os.path.join(args.out, f))
        for f in os.listdir(args.out)
        if f.endswith('.npz')
    ) if os.path.exists(args.out) else 0

    print(f"\nSummary: {cached} days cached, {skipped_exist} already existed, "
          f"{skipped_empty} empty/holiday, {skipped_no_roll} not in roll calendar. "
          f"Total size: {total_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
