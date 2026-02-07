"""Precompute LOB data and cache as .npz files for fast training startup."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import numpy as np
import lob_rl_core


def main():
    parser = argparse.ArgumentParser(description='Precompute and cache LOB data as .npz files')
    parser.add_argument('--data-dir', required=True, help='Directory with .bin files and manifest.json')
    parser.add_argument('--out', required=True, help='Output directory for cached .npz files')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Re-cache even if .npz already exists')
    args = parser.parse_args()

    # Load manifest
    manifest_path = os.path.join(args.data_dir, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)

    os.makedirs(args.out, exist_ok=True)

    cfg = lob_rl_core.SessionConfig.default_rth()
    cached = 0
    skipped_exist = 0
    skipped_empty = 0

    for entry in manifest['files']:
        date = entry['date']
        bin_path = os.path.join(args.data_dir, f"{date}.bin")
        npz_path = os.path.join(args.out, f"{date}.npz")

        if not args.force and os.path.exists(npz_path):
            skipped_exist += 1
            print(f"  Skipping {date}: already cached")
            continue

        if not os.path.exists(bin_path):
            print(f"  Skipping {date}: .bin file not found")
            continue

        obs, mid, spread, num_steps = lob_rl_core.precompute(bin_path, cfg)

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
