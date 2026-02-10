"""Simple metrics schema validation for training outputs."""

from __future__ import annotations

METRICS_SCHEMA_VERSION = "metrics.v1"


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_split_metrics(name: str, metrics: dict, errors: list[str]) -> None:
    required_numeric_fields = [
        "mean_return",
        "std_return",
        "sortino",
        "downside_std",
    ]
    for field in required_numeric_fields:
        if field not in metrics:
            errors.append(f"{name}.{field} missing")
            continue
        if not _is_number(metrics[field]):
            errors.append(f"{name}.{field} must be numeric")

    if "n_episodes" not in metrics:
        errors.append(f"{name}.n_episodes missing")
    elif not _is_int(metrics["n_episodes"]):
        errors.append(f"{name}.n_episodes must be int")


def validate_metrics_payload(payload: dict) -> list[str]:
    """Validate metrics payload and return list of schema errors."""
    errors: list[str] = []

    if payload.get("schema_version") != METRICS_SCHEMA_VERSION:
        errors.append("schema_version must equal metrics.v1")

    eval_cfg = payload.get("evaluation")
    if not isinstance(eval_cfg, dict):
        errors.append("evaluation must be an object")
    else:
        if not _is_int(eval_cfg.get("n_eval_episodes")):
            errors.append("evaluation.n_eval_episodes must be int")

    split_metrics = payload.get("metrics")
    if not isinstance(split_metrics, dict):
        errors.append("metrics must be an object")
        return errors

    for split_name in ("validation", "test"):
        if split_name not in split_metrics:
            continue
        split_payload = split_metrics[split_name]
        if not isinstance(split_payload, dict):
            errors.append(f"metrics.{split_name} must be an object")
            continue
        _validate_split_metrics(f"metrics.{split_name}", split_payload, errors)

    return errors
