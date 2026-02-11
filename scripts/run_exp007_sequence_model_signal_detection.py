#!/usr/bin/env python
"""exp-007: Sequence Model Signal Detection runner.

Trains LSTM and Transformer on per-bar barrier features,
evaluates using Brier score framework (BSS, bootstrap CI).
Writes all metrics to results/exp-007-sequence-model-signal-detection/metrics.json.
"""

import json
import os
import sys
import time
import traceback

import numpy as np
import torch

from lob_rl.barrier.first_passage_analysis import (
    load_session_features,
    temporal_split,
)
from lob_rl.barrier.sequence_models import (
    BarrierLSTM,
    BarrierTransformer,
    LinearBarEmbedding,
    evaluate_sequence_model,
    train_sequence_model,
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache", "barrier")
RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "exp-007-sequence-model-signal-detection"
)
SEED = 42

# Spec parameters
EPOCHS = 50
BATCH_SIZE = 4
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
DROPOUT = 0.1
N_BOOTSTRAP = 1000

# Abort criteria
PER_MODEL_ABORT_SECONDS = 45 * 60  # 45 min
TOTAL_ABORT_SECONDS = 2 * 3600     # 2 hours


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    notes = []

    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # ---------------------------------------------------------------
    # Step 1: Load data
    # ---------------------------------------------------------------
    print("=== Step 1: Loading session data ===")
    sessions = load_session_features(CACHE_DIR)
    n_sessions = len(sessions)
    n_bars_total = sum(len(s["features"]) for s in sessions)
    n_features = sessions[0]["features"].shape[1]
    print(f"  N_sessions = {n_sessions}")
    print(f"  N_bars_total = {n_bars_total}")
    print(f"  N_features = {n_features}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ---------------------------------------------------------------
    # Step 2: Temporal split (60/20/20)
    # ---------------------------------------------------------------
    print("\n=== Step 2: Temporal split ===")
    train_idx, val_idx, test_idx = temporal_split(n_sessions)
    train_sessions = [sessions[i] for i in train_idx]
    val_sessions = [sessions[i] for i in val_idx]
    test_sessions = [sessions[i] for i in test_idx]
    print(f"  Train: {len(train_sessions)} sessions, "
          f"{sum(len(s['features']) for s in train_sessions)} bars")
    print(f"  Val: {len(val_sessions)} sessions, "
          f"{sum(len(s['features']) for s in val_sessions)} bars")
    print(f"  Test: {len(test_sessions)} sessions, "
          f"{sum(len(s['features']) for s in test_sessions)} bars")

    # ---------------------------------------------------------------
    # Step 3: Sanity checks
    # ---------------------------------------------------------------
    print("\n=== Step 3: Sanity checks ===")
    train_y_long = np.concatenate([s["Y_long"] for s in train_sessions])
    train_y_short = np.concatenate([s["Y_short"] for s in train_sessions])
    ybar_train_long = float(train_y_long.mean())
    ybar_train_short = float(train_y_short.mean())
    print(f"  ybar_train_long = {ybar_train_long:.6f}")
    print(f"  ybar_train_short = {ybar_train_short:.6f}")

    sanity_ok = True
    if n_sessions < 200:
        notes.append(f"SANITY FAIL: N_sessions = {n_sessions} < 200")
        sanity_ok = False
    if not (0.28 <= ybar_train_long <= 0.38):
        notes.append(f"SANITY FAIL: ybar_train_long = {ybar_train_long}")
        sanity_ok = False
    if not (0.28 <= ybar_train_short <= 0.38):
        notes.append(f"SANITY FAIL: ybar_train_short = {ybar_train_short}")
        sanity_ok = False

    # Check for NaN in features
    nan_sessions = sum(
        1 for s in sessions if np.isnan(s["features"]).any()
    )
    if nan_sessions > 0:
        notes.append(f"WARNING: {nan_sessions} sessions with NaN features")
        print(f"  WARNING: {nan_sessions} sessions with NaN features")

    # ---------------------------------------------------------------
    # Step 4: Train and evaluate models
    # ---------------------------------------------------------------
    # Compute max_len from data (some sessions exceed default 2048)
    max_session_len = max(len(s["features"]) for s in sessions)
    max_len = max_session_len + 100  # small buffer
    print(f"  Max session length: {max_session_len}, positional encoding max_len: {max_len}")

    model_configs = {
        "lstm": lambda: BarrierLSTM(
            n_features=n_features,
            d_model=D_MODEL,
            n_layers=N_LAYERS,
            dropout=DROPOUT,
            embedding=LinearBarEmbedding(
                n_features=n_features, d_model=D_MODEL,
                max_len=max_len, dropout=DROPOUT,
            ),
        ),
        "transformer": lambda: BarrierTransformer(
            n_features=n_features,
            d_model=D_MODEL,
            n_layers=N_LAYERS,
            n_heads=N_HEADS,
            dropout=DROPOUT,
            embedding=LinearBarEmbedding(
                n_features=n_features, d_model=D_MODEL,
                max_len=max_len, dropout=DROPOUT,
            ),
        ),
    }

    model_results = {}
    total_runs = 0
    abort_triggered = False
    abort_reason = None

    for model_name, model_factory in model_configs.items():
        print(f"\n=== Step 4.{total_runs + 1}: Training {model_name.upper()} ===")
        model_t0 = time.time()

        # Check total wall clock abort
        elapsed = time.time() - t0
        if elapsed > TOTAL_ABORT_SECONDS:
            abort_triggered = True
            abort_reason = f"Total wall clock {elapsed:.0f}s > {TOTAL_ABORT_SECONDS}s"
            notes.append(f"ABORT: {abort_reason}")
            print(f"  ABORT: {abort_reason}")
            break

        try:
            model = model_factory()
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {n_params:,}")

            # Train
            history = train_sequence_model(
                model=model,
                train_sessions=train_sessions,
                val_sessions=val_sessions,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                patience=PATIENCE,
                seed=SEED,
                device=device,
            )

            train_time = time.time() - model_t0
            print(f"  Training time: {train_time:.1f}s")
            print(f"  Best epoch: {history['best_epoch']}")
            print(f"  Best val Brier: {history['best_val_brier']:.6f}")
            print(f"  Stopped early: {history['stopped_early']}")

            # Check per-model abort
            if train_time > PER_MODEL_ABORT_SECONDS:
                notes.append(
                    f"WARNING: {model_name} took {train_time:.0f}s "
                    f"(> {PER_MODEL_ABORT_SECONDS}s abort threshold)"
                )

            # Check for NaN loss
            if any(np.isnan(l) for l in history["train_loss_history"]):
                notes.append(f"WARNING: {model_name} produced NaN loss during training")
                model_results[model_name] = {
                    "error": "NaN loss during training",
                    "train_time_seconds": train_time,
                    "n_params": n_params,
                }
                total_runs += 1
                continue

            # Evaluate on val
            print(f"  Evaluating on val set...")
            val_metrics = evaluate_sequence_model(
                model=model,
                sessions=val_sessions,
                batch_size=BATCH_SIZE * 2,
                device=device,
                n_bootstrap=N_BOOTSTRAP,
                seed=SEED,
            )

            # Evaluate on test
            print(f"  Evaluating on test set...")
            test_metrics = evaluate_sequence_model(
                model=model,
                sessions=test_sessions,
                batch_size=BATCH_SIZE * 2,
                device=device,
                n_bootstrap=N_BOOTSTRAP,
                seed=SEED,
            )

            model_results[model_name] = {
                "n_params": n_params,
                "train_time_seconds": round(train_time, 1),
                "best_epoch": history["best_epoch"],
                "best_val_brier": history["best_val_brier"],
                "stopped_early": history["stopped_early"],
                "epochs_trained": len(history["train_loss_history"]),
                "final_train_loss": history["train_loss_history"][-1],
                "train_loss_history": history["train_loss_history"],
                "val_brier_history": history["val_brier_history"],
                "val": val_metrics,
                "test": test_metrics,
            }

            # Print key val results
            for label in ["long", "short"]:
                bss = val_metrics[f"bss_{label}"]
                boot = val_metrics[f"bootstrap_{label}"]
                print(f"  Val {label}: BSS={bss:.6f}, "
                      f"delta={boot['delta']:.6f}, "
                      f"p={boot['p_value']:.4f}, "
                      f"CI=[{boot['ci_lower']:.6f}, {boot['ci_upper']:.6f}]")

            total_runs += 1

        except Exception as e:
            train_time = time.time() - model_t0
            error_msg = f"{model_name} failed: {e}"
            notes.append(error_msg)
            print(f"  ERROR: {error_msg}")
            traceback.print_exc()
            model_results[model_name] = {
                "error": str(e),
                "train_time_seconds": round(train_time, 1),
            }
            total_runs += 1

    # Check if all models produced NaN
    all_nan = all(
        "error" in model_results.get(m, {}) and "NaN" in str(model_results.get(m, {}).get("error", ""))
        for m in model_configs
    )
    if all_nan and len(model_results) == len(model_configs):
        abort_triggered = True
        abort_reason = "All models produced NaN loss"
        notes.append(f"ABORT: {abort_reason}")

    # ---------------------------------------------------------------
    # Step 5: Evaluate success criteria
    # ---------------------------------------------------------------
    print("\n=== Step 5: Evaluating success criteria ===")

    # C1: At least one (model, label) pair on val has BSS > 0 with bootstrap p < 0.05
    c1_pass = False
    best_bss = -999.0
    best_pair = None

    for model_name in model_configs:
        if model_name not in model_results or "val" not in model_results[model_name]:
            continue
        val = model_results[model_name]["val"]
        for label in ["long", "short"]:
            bss = val[f"bss_{label}"]
            boot = val[f"bootstrap_{label}"]
            if bss > 0 and boot["p_value"] < 0.05:
                c1_pass = True
            if bss > best_bss:
                best_bss = bss
                best_pair = f"{model_name}/{label}"

    # C2: Best BSS on val >= 0.005
    c2_pass = best_bss >= 0.005

    # C3: Best model's val Brier < constant Brier AND < exp-006's best flat model Brier
    c3_pass = False
    exp006_best_brier = None
    if best_pair:
        best_model, best_label = best_pair.split("/")
        if best_model in model_results and "val" in model_results[best_model]:
            val = model_results[best_model]["val"]
            brier_model = val[f"brier_model_{best_label}"]
            brier_constant = val[f"brier_constant_{best_label}"]
            # exp-006 best flat model Brier (logistic/short had best BSS)
            # From metrics: logistic_short brier=0.2185, logistic_long brier=0.2168
            exp006_best_brier_long = 0.2167793347339828
            exp006_best_brier_short = 0.2184907901990017
            exp006_best_brier = (
                exp006_best_brier_long if best_label == "long"
                else exp006_best_brier_short
            )
            c3_pass = (brier_model < brier_constant) and (brier_model < exp006_best_brier)

    print(f"  C1 (BSS > 0, p < 0.05): {c1_pass}")
    print(f"  C2 (BSS >= 0.005): {c2_pass} (best BSS = {best_bss:.6f}, pair = {best_pair})")
    print(f"  C3 (Brier < constant AND < exp-006 best): {c3_pass}")

    # ---------------------------------------------------------------
    # Step 6: Assemble and write metrics.json
    # ---------------------------------------------------------------
    print("\n=== Step 6: Writing metrics.json ===")

    # Build primary metrics for each (model, label) pair on val
    primary = {}
    for model_name in model_configs:
        if model_name not in model_results or "val" not in model_results[model_name]:
            continue
        val = model_results[model_name]["val"]
        for label in ["long", "short"]:
            key = f"{model_name}_{label}"
            boot = val[f"bootstrap_{label}"]
            primary[key] = {
                "brier_model": val[f"brier_model_{label}"],
                "brier_constant": val[f"brier_constant_{label}"],
                "bss": val[f"bss_{label}"],
                "delta": boot["delta"],
                "delta_ci_lower": boot["ci_lower"],
                "delta_ci_upper": boot["ci_upper"],
                "p_value": boot["p_value"],
                "p_hat_mean": val[f"p_hat_mean_{label}"],
                "p_hat_std": val[f"p_hat_std_{label}"],
            }

    # Build test metrics
    test_primary = {}
    for model_name in model_configs:
        if model_name not in model_results or "test" not in model_results[model_name]:
            continue
        test = model_results[model_name]["test"]
        for label in ["long", "short"]:
            key = f"{model_name}_{label}"
            boot = test[f"bootstrap_{label}"]
            test_primary[key] = {
                "brier_model": test[f"brier_model_{label}"],
                "brier_constant": test[f"brier_constant_{label}"],
                "bss": test[f"bss_{label}"],
                "delta": boot["delta"],
                "delta_ci_lower": boot["ci_lower"],
                "delta_ci_upper": boot["ci_upper"],
                "p_value": boot["p_value"],
                "p_hat_mean": test[f"p_hat_mean_{label}"],
                "p_hat_std": test[f"p_hat_std_{label}"],
            }

    # Training details per model
    training_details = {}
    for model_name in model_configs:
        if model_name not in model_results:
            continue
        r = model_results[model_name]
        detail = {
            "n_params": r.get("n_params"),
            "train_time_seconds": r.get("train_time_seconds"),
            "best_epoch": r.get("best_epoch"),
            "epochs_trained": r.get("epochs_trained"),
            "stopped_early": r.get("stopped_early"),
            "final_train_loss": r.get("final_train_loss"),
        }
        if "error" in r:
            detail["error"] = r["error"]
        training_details[model_name] = detail

    wall_clock = round(time.time() - t0, 1)

    metrics = {
        "experiment": "exp-007-sequence-model-signal-detection",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sanity_checks": {
            "N_sessions": n_sessions,
            "N_bars_total": n_bars_total,
            "N_features": n_features,
            "ybar_train_long": ybar_train_long,
            "ybar_train_short": ybar_train_short,
            "n_train_sessions": len(train_sessions),
            "n_val_sessions": len(val_sessions),
            "n_test_sessions": len(test_sessions),
            "n_train_bars": sum(len(s["features"]) for s in train_sessions),
            "n_val_bars": sum(len(s["features"]) for s in val_sessions),
            "n_test_bars": sum(len(s["features"]) for s in test_sessions),
            "device": device,
            "sanity_ok": sanity_ok,
        },
        "primary_val": primary,
        "primary_test": test_primary,
        "training_details": training_details,
        "exp006_comparison": {
            "exp006_best_bss": -0.0002898695757111991,
            "exp006_best_pair": "logistic/short",
            "exp006_best_brier_long": 0.2167793347339828,
            "exp006_best_brier_short": 0.2184907901990017,
        },
        "success_criteria": {
            "C1_bss_positive_significant": c1_pass,
            "C2_bss_gte_0005": c2_pass,
            "C2_best_bss": best_bss,
            "C2_best_pair": best_pair,
            "C3_brier_lt_constant_and_exp006": c3_pass,
            "C3_exp006_best_brier": exp006_best_brier,
            "sanity_ok": sanity_ok,
        },
        "resource_usage": {
            "gpu_hours": 0,
            "wall_clock_seconds": wall_clock,
            "total_training_steps": 0,
            "total_runs": total_runs,
        },
        "abort_triggered": abort_triggered,
        "abort_reason": abort_reason,
        "notes": "; ".join(notes) if notes else None,
    }

    # Write config.json
    config = {
        "experiment": "exp-007-sequence-model-signal-detection",
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "dropout": DROPOUT,
        "n_bootstrap": N_BOOTSTRAP,
        "device": device,
        "models": {
            "lstm": {
                "class": "BarrierLSTM",
                "n_features": n_features,
                "d_model": D_MODEL,
                "n_layers": N_LAYERS,
                "dropout": DROPOUT,
            },
            "transformer": {
                "class": "BarrierTransformer",
                "n_features": n_features,
                "d_model": D_MODEL,
                "n_layers": N_LAYERS,
                "n_heads": N_HEADS,
                "dropout": DROPOUT,
            },
        },
        "data": {
            "cache_dir": "cache/barrier/",
            "split": "temporal 60/20/20",
        },
    }

    config_path = os.path.join(RESULTS_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config written to {config_path}")

    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics written to {metrics_path}")
    print(f"\nTotal wall clock: {wall_clock:.1f}s")


if __name__ == "__main__":
    main()
