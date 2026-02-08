"""Precompute LOB data and cache as .npz files for fast training startup."""

import argparse
import glob
import os
import re
import sys

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


def main():
    parser = argparse.ArgumentParser(description='Precompute and cache LOB data as .npz files')
    parser.add_argument('--data-dir', required=True, help='Directory with data files')
    parser.add_argument('--out', required=True, help='Output directory for cached .npz files')
    parser.add_argument('--instrument-id', type=int, required=True,
                        help='Instrument ID to filter records (uint32)')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Re-cache even if .npz already exists')
    args = parser.parse_args()

    # Glob for .dbn.zst files
    data_files = sorted(glob.glob(os.path.join(args.data_dir, '*.mbo.dbn.zst')))

    os.makedirs(args.out, exist_ok=True)

    cfg = lob_rl_core.SessionConfig.default_rth()
    cached = 0
    skipped_exist = 0
    skipped_empty = 0

    for data_path in data_files:
        date = _extract_date_from_filename(os.path.basename(data_path))
        if date is None:
            print(f"  Skipping {data_path}: cannot extract date from filename")
            continue

        npz_path = os.path.join(args.out, f"{date}.npz")

        if not args.force and os.path.exists(npz_path):
            skipped_exist += 1
            print(f"  Skipping {date}: already cached")
            continue

        obs, mid, spread, num_steps = lob_rl_core.precompute(
            data_path, cfg, args.instrument_id
        )

        if num_steps < 2:
            skipped_empty += 1
            print(f"  Skipping {date}: only {num_steps} BBO snapshots (need >= 2)")
            continue

        np.savez(npz_path, obs=obs, mid=mid, spread=spread)
        cached += 1
        print(f"  Cached {date}: obs={obs.shape}, mid={mid.shape}, spread={spread.shape}")

    total_size = sum(
        os.path.getsize(os.path.join(args.out, f))
        for f in os.listdir(args.out)
        if f.endswith('.npz')
    ) if os.path.exists(args.out) else 0

    print(f"\nSummary: {cached} days cached, {skipped_exist} already existed, "
          f"{skipped_empty} empty/holiday. Total size: {total_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
