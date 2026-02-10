"""Run manifest generation and serialization."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _hash_json(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _hash_paths(paths: list[str]) -> str:
    canonical = "\n".join(sorted(paths)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_run_manifest(
    *,
    args: dict,
    train_files: list[tuple[str, str, int]],
    val_files: list[tuple[str, str, int]],
    test_files: list[tuple[str, str, int]],
    artifact_schema_version: str = "1.0",
) -> dict:
    """Build a run manifest payload."""
    train_paths = [path for _, path, _ in train_files]
    val_paths = [path for _, path, _ in val_files]
    test_paths = [path for _, path, _ in test_files]

    return {
        "schema_version": "run_manifest.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "artifact_schema_version": artifact_schema_version,
        "config_hash": _hash_json(args),
        "data": {
            "train_count": len(train_files),
            "val_count": len(val_files),
            "test_count": len(test_files),
            "train_hash": _hash_paths(train_paths),
            "val_hash": _hash_paths(val_paths),
            "test_hash": _hash_paths(test_paths),
            "train_dates": [date for date, _, _ in train_files],
            "val_dates": [date for date, _, _ in val_files],
            "test_dates": [date for date, _, _ in test_files],
        },
        "args": args,
    }


def write_run_manifest(manifest: dict, output_dir: str | Path) -> Path:
    """Write manifest to output_dir/run_manifest.yaml."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_manifest.yaml"

    # YAML-compatible plaintext without third-party dependency.
    with path.open("w", encoding="utf-8") as f:
        f.write("# run_manifest.v1\n")
        f.write(json.dumps(manifest, indent=2, sort_keys=True))
        f.write("\n")

    return path
