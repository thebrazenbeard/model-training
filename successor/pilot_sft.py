from __future__ import annotations

import hashlib
import math
from pathlib import Path


_REQUIRED = {
    "run_id",
    "base_repo_id",
    "base_revision",
    "parent_adapter_path",
    "train_path",
    "validation_path",
    "output_dir",
    "seed",
    "learning_rate",
    "gradient_accumulation",
    "assistant_only_loss",
    "neutral_system_override",
}


def validate_run_spec(spec: dict, base_manifest: dict) -> dict:
    missing = sorted(_REQUIRED - set(spec))
    if missing:
        raise ValueError(f"missing run spec fields: {', '.join(missing)}")
    if base_manifest.get("mutable_revision_allowed") is not False:
        raise ValueError("base manifest must forbid mutable revisions")
    if spec["base_repo_id"] != base_manifest.get("repo_id"):
        raise ValueError("base repository does not match manifest")
    if spec["base_revision"] != base_manifest.get("revision"):
        raise ValueError("base revision does not match manifest")
    if spec["assistant_only_loss"] is not True:
        raise ValueError("assistant_only_loss must be true")
    if spec["neutral_system_override"] is not True:
        raise ValueError("neutral_system_override must be true")
    if not isinstance(spec["seed"], int):
        raise ValueError("seed must be an integer")
    if float(spec["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")
    if int(spec["gradient_accumulation"]) < 1:
        raise ValueError("gradient_accumulation must be positive")
    return spec


def build_run_manifest(
    spec: dict,
    *,
    code_commit: str,
    train_sha256: str,
    validation_sha256: str,
    parent_adapter_sha256: str,
    output_adapter_sha256: str,
    gpu_type: str,
    package_versions: dict,
) -> dict:
    return {
        "run_id": spec["run_id"],
        "code_commit": code_commit,
        "base": {
            "repo_id": spec["base_repo_id"],
            "revision": spec["base_revision"],
        },
        "seed": spec["seed"],
        "hyperparameters": {
            "learning_rate": float(spec["learning_rate"]),
            "gradient_accumulation": int(spec["gradient_accumulation"]),
        },
        "assistant_only_loss": True,
        "neutral_system_override": True,
        "dataset_digests": {
            "train_sha256": train_sha256,
            "validation_sha256": validation_sha256,
        },
        "parent_adapter_sha256": parent_adapter_sha256,
        "output_adapter_sha256": output_adapter_sha256,
        "gpu_type": gpu_type,
        "package_versions": dict(package_versions),
    }


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cosine_floor_scale(
    step: int,
    total_steps: int,
    *,
    warmup_steps: int,
    floor: float = 0.2,
) -> float:
    if total_steps < 1 or warmup_steps < 1 or warmup_steps > total_steps:
        raise ValueError("invalid scheduler steps")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must be between 0 and 1")
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 1:
        return floor
    progress = (step - warmup_steps) / (decay_steps - 1)
    progress = min(1.0, max(0.0, progress))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
