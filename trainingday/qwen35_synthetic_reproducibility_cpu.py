"""Verify repeated synthetic Qwen3.5 checkpoints are reproducible on one CPU.

This is a source-only, randomly initialized test. It neither authenticates
official pretrained weights nor grants training authority.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys


SCHEMA = "TRAININGDAY_QWEN35_SYNTHETIC_REPRODUCIBILITY_V1"
PACKAGES = ("torch", "transformers", "peft", "safetensors", "tokenizers",
            "numpy", "huggingface_hub")


def hash_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_tensor_digest(path: Path) -> tuple[str, int]:
    """Hash typed tensor values independent of safetensors header ordering."""
    import torch
    from safetensors import safe_open

    digest = hashlib.sha256()
    count = 0
    with safe_open(str(path), framework="pt", device="cpu") as source:
        for name in sorted(source.keys()):
            tensor = source.get_tensor(name).detach().contiguous()
            header = json.dumps({"name": name, "shape": list(tensor.shape),
                                 "dtype": str(tensor.dtype)},
                                sort_keys=True, separators=(",", ":")).encode()
            payload = tensor.view(torch.uint8).numpy().tobytes()
            digest.update(len(header).to_bytes(8, "big"))
            digest.update(header)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
            count += 1
    return digest.hexdigest(), count


def fingerprint(root: Path) -> dict:
    model_file = root / "merged" / "model.safetensors"
    if not model_file.is_file():
        raise AssertionError("missing merged safetensors model")
    tensor_digest, count = canonical_tensor_digest(model_file)
    return {"file_sha256": hash_file(model_file),
            "tensor_values_sha256": tensor_digest,
            "tensor_count": count, "bytes": model_file.stat().st_size}


def assert_reproduced(first: dict, second: dict) -> None:
    if first["tensor_values_sha256"] != second["tensor_values_sha256"]:
        raise AssertionError("synthetic merged tensor values drifted with fixed seed")
    if first["file_sha256"] != second["file_sha256"]:
        raise AssertionError("safetensors serialization bytes drifted with fixed seed")
    if first["tensor_count"] != second["tensor_count"]:
        raise AssertionError("synthetic merged tensor inventory changed")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m trainingday.qwen35_synthetic_reproducibility_cpu OUTDIR")
    from trainingday.qwen35_tiny_gguf_e2e_cpu import prepare

    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    observations = []
    for label in ("a", "b"):
        import numpy as np
        random.seed(35032)
        np.random.seed(35032)
        destination = root / label
        prior = prepare(destination)
        observations.append({"label": label, "loss": prior["loss"],
                             **fingerprint(destination)})
    first, second = observations
    if first["loss"] != second["loss"]:
        raise AssertionError("synthetic CPU training loss drifted with fixed seed")
    assert_reproduced(first, second)

    receipt = {"schema": SCHEMA, "status": "SAME_PROCESS_CPU_TWO_PREPARES_PASS",
               "official_pretrained_weights_exercised": False,
               "gpu_fit": "UNKNOWN", "observations": observations,
               "python": sys.version.split()[0],
               "pythonhashseed": os.environ.get("PYTHONHASHSEED", "UNSET"),
               "packages": {p: importlib.metadata.version(p) for p in PACKAGES}}
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
