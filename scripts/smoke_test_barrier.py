#!/usr/bin/env python3
"""Smoke test: full barrier-hit pipeline on real .dbn.zst data.

Tests the complete chain:
  .dbn.zst → build_session_bars → compute_labels → build_feature_matrix
  → MultiSessionBarrierEnv → MaskablePPO → train 2048 steps

This is NOT a unit test — it's an integration test on real data to catch
pipeline bugs before committing GPU hours.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np


def get_instrument_id(date_str, roll_calendar):
    """Look up front-month instrument_id for a given date."""
    for roll in roll_calendar["rolls"]:
        if roll["start"] <= date_str <= roll["end"]:
            return roll["instrument_id"]
    return None


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "mes"
    roll_cal_path = data_dir / "roll_calendar.json"

    with open(roll_cal_path) as f:
        roll_calendar = json.load(f)

    # Pick 5 files spread across the year
    test_dates = ["20220103", "20220401", "20220701", "20221003", "20221201"]
    files = []
    for d in test_dates:
        p = data_dir / f"glbx-mdp3-{d}.mbo.dbn.zst"
        if p.exists():
            files.append((p, d))

    if not files:
        print("ERROR: No .dbn.zst files found for test dates")
        sys.exit(1)

    print(f"=== Barrier Pipeline Smoke Test ===")
    print(f"Files: {len(files)}")
    print()

    # --- Phase 1: Bar construction ---
    print("--- Phase 1: Bar Construction ---")
    from lob_rl.barrier.bar_pipeline import build_session_bars

    sessions_bars = []
    for filepath, date_str in files:
        iid = get_instrument_id(f"2022-{date_str[4:6]}-{date_str[6:8]}", roll_calendar)
        print(f"  {filepath.name} (instrument_id={iid})...", end=" ", flush=True)
        t0 = time.time()
        bars = build_session_bars(str(filepath), n=500, instrument_id=iid)
        elapsed = time.time() - t0
        print(f"{len(bars)} bars in {elapsed:.1f}s")
        if len(bars) > 0:
            sessions_bars.append((bars, date_str))

    if not sessions_bars:
        print("ERROR: No sessions produced any bars")
        sys.exit(1)

    print(f"\nTotal sessions with bars: {len(sessions_bars)}")
    print()

    # --- Phase 2: Label construction ---
    print("--- Phase 2: Label Construction ---")
    from lob_rl.barrier.label_pipeline import (
        compute_labels,
        compute_label_distribution,
        compute_tiebreak_frequency,
    )

    sessions_labeled = []
    for bars, date_str in sessions_bars:
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        dist = compute_label_distribution(labels)
        tiebreak = compute_tiebreak_frequency(labels)
        print(f"  {date_str}: {len(labels)} labels, "
              f"p+={dist['p_plus']:.3f} p-={dist['p_minus']:.3f} p0={dist['p_zero']:.3f}, "
              f"tiebreak={tiebreak:.3f}")
        sessions_labeled.append((bars, labels, date_str))

    print()

    # --- Phase 3: Feature extraction ---
    print("--- Phase 3: Feature Extraction ---")
    from lob_rl.barrier.feature_pipeline import build_feature_matrix

    session_data = []
    for bars, labels, date_str in sessions_labeled:
        features = build_feature_matrix(bars, h=10)
        has_nan = np.any(np.isnan(features))
        has_inf = np.any(np.isinf(features))
        print(f"  {date_str}: features shape={features.shape}, "
              f"NaN={has_nan}, Inf={has_inf}, "
              f"mean={features.mean():.3f}, std={features.std():.3f}")
        if has_nan or has_inf:
            print(f"    WARNING: Bad values detected!")
        session_data.append({
            "bars": bars,
            "labels": labels,
            "features": features,
        })

    print()

    # --- Phase 4: Environment creation ---
    print("--- Phase 4: Environment Creation ---")
    from lob_rl.barrier.multi_session_env import MultiSessionBarrierEnv
    from lob_rl.barrier.reward_accounting import RewardConfig

    config = RewardConfig()

    # Filter sessions with enough bars for features (need h=10, so at least 10 bars)
    valid_sessions = [s for s in session_data if s["features"].shape[0] > 0]
    print(f"  Valid sessions (features.shape[0] > 0): {len(valid_sessions)}")

    if not valid_sessions:
        print("ERROR: No valid sessions")
        sys.exit(1)

    env = MultiSessionBarrierEnv(valid_sessions, config=config, shuffle=True, seed=42)
    obs, info = env.reset()
    print(f"  Obs shape: {obs.shape}")
    print(f"  Obs space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Action mask: {env.action_masks()}")

    # Quick random agent rollout
    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        mask = env.action_masks()
        valid_actions = np.where(mask)[0]
        action = np.random.choice(valid_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    print(f"  Random agent: {steps} steps, total reward={total_reward:.3f}, "
          f"trades={info.get('n_trades', 'N/A')}")
    print()

    # --- Phase 5: VecEnv + MaskablePPO ---
    print("--- Phase 5: VecEnv + MaskablePPO Training ---")
    from lob_rl.barrier.barrier_vec_env import make_barrier_vec_env
    from lob_rl.barrier.training_diagnostics import BarrierDiagnosticCallback, linear_schedule
    import lob_rl.barrier._sb3_compat  # noqa: F401 — auto-patches on import
    import torch.nn as nn
    from sb3_contrib import MaskablePPO

    # Use DummyVecEnv (not subprocess) for smoke test
    vec_env = make_barrier_vec_env(
        valid_sessions, n_envs=1, use_subprocess=False,
        config=config, shuffle=True, seed=42,
    )
    print(f"  VecEnv created: {vec_env.num_envs} env(s)")

    policy_kwargs = dict(
        net_arch=[256, 256, dict(pi=[64], vf=[64])],
        activation_fn=nn.ReLU,
    )

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=linear_schedule(1e-4),
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        seed=42,
        verbose=1,
    )
    print(f"  MaskablePPO created")
    print(f"  Policy: {model.policy.__class__.__name__}")
    print(f"  Obs space: {model.observation_space}")
    print(f"  Action space: {model.action_space}")

    callback = BarrierDiagnosticCallback(check_freq=1, verbose=0)

    print(f"\n  Training for 2048 steps...")
    t0 = time.time()
    model.learn(total_timesteps=2048, callback=callback)
    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed:.1f}s")

    # Check diagnostics
    red_flags = callback.check_red_flags()
    if red_flags:
        print(f"\n  RED FLAGS:")
        for flag in red_flags:
            print(f"    - {flag}")
    else:
        print(f"\n  No red flags detected.")

    if callback.diagnostics:
        last = callback.diagnostics[-1]
        print(f"  Last diagnostic snapshot:")
        for k, v in sorted(last.items()):
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    vec_env.close()

    print()
    print("=== SMOKE TEST PASSED ===")
    print("The full pipeline works end-to-end on real data:")
    print("  .dbn.zst → bars → labels → features → env → MaskablePPO → train")


if __name__ == "__main__":
    main()
