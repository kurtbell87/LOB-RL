"""Tests for config loading and run artifact contracts."""

import json

from lob_rl.config import load_json_config
from lob_rl.orchestration import build_run_manifest
from lob_rl.reporting import METRICS_SCHEMA_VERSION, validate_metrics_payload


def test_load_json_config_round_trip(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"train_days": 42, "seed": 7}), encoding="utf-8")

    loaded = load_json_config(str(config_path))

    assert loaded["train_days"] == 42
    assert loaded["seed"] == 7


def test_build_run_manifest_contains_hashes():
    manifest = build_run_manifest(
        args={"seed": 42, "train_days": 20},
        train_files=[("2022-01-03", "/cache/2022-01-03.npz", 0)],
        val_files=[("2022-01-04", "/cache/2022-01-04.npz", 0)],
        test_files=[("2022-01-05", "/cache/2022-01-05.npz", 0)],
        artifact_schema_version=METRICS_SCHEMA_VERSION,
    )

    assert manifest["schema_version"] == "run_manifest.v1"
    assert manifest["data"]["train_count"] == 1
    assert len(manifest["data"]["train_hash"]) == 64
    assert manifest["artifact_schema_version"] == METRICS_SCHEMA_VERSION


def test_validate_metrics_payload_success_and_failure():
    valid_payload = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "evaluation": {"n_eval_episodes": 10},
        "metrics": {
            "validation": {
                "mean_return": 1.0,
                "std_return": 2.0,
                "sortino": 0.5,
                "downside_std": 1.2,
                "n_episodes": 10,
            }
        },
    }
    assert validate_metrics_payload(valid_payload) == []

    invalid_payload = {
        "schema_version": "bad",
        "evaluation": {"n_eval_episodes": "10"},
        "metrics": {"validation": {"mean_return": 1.0}},
    }
    errors = validate_metrics_payload(invalid_payload)
    assert any("schema_version" in error for error in errors)
    assert any("n_eval_episodes" in error for error in errors)
    assert any("std_return" in error for error in errors)
