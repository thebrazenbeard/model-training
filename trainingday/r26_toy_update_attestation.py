"""R26 PUBLIC toy tensor-update falsifier; deliberately NOT a Qwen-4B receipt.

The fixed two-parameter SGD task is fully reproducible with the Python stdlib.
It can detect a claimed optimizer step with no matching byte update, and can
reload the toy tensor in a fresh interpreter. It cannot attest an actual
transformer optimizer execution, data rights, external signing, or learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
from pathlib import Path


class AttestationError(ValueError):
    """A public-toy synthetic receipt failed controlled verification."""


SCHEMA = "TRAININGDAY_R26_PUBLIC_TOY_UPDATE_V1"
FIXTURE = "R26_PUBLIC_ORTHOGONAL_TWO_PARAMETER_V1"
MAGIC = b"VERA_R26_PUBLIC_TOY_F64_V1\n"
STRUCT = struct.Struct("<dd")
NAMES = ("base.bin", "tokenizer.txt", "adapter_before.bin", "adapter_after.bin")
SHA = re.compile(r"[0-9a-f]{64}\Z")


def _hex(value: object) -> bool:
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bounded_real(value: object, lower: float, upper: float) -> bool:
    """Do not float-convert unbounded ints: math.isfinite(int) can overflow."""
    if type(value) is int:
        return lower <= value <= upper
    if type(value) is float:
        return math.isfinite(value) and lower <= value <= upper
    return False


def _no_nan(name: str):
    raise AttestationError("noncanonical JSON number: " + name)


def _no_duplicates(items):
    result = {}
    for key, value in items:
        if key in result:
            raise AttestationError("duplicate JSON key")
        result[key] = value
    return result


def _read_bounded(path: Path, limit: int) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise AttestationError("source file absent or symlinked")
    if path.stat().st_size > limit:
        raise AttestationError("source file exceeds public toy cap")
    try:
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
    except OSError as exc:
        raise AttestationError("source file unreadable") from exc
    if not data or len(data) > limit:
        raise AttestationError("source file empty or unbounded")
    return data


def encode_toy_adapter(weights: tuple[float, float]) -> bytes:
    if not isinstance(weights, tuple) or len(weights) != 2:
        raise AttestationError("public toy tensor dimensions malformed")
    if any(not _bounded_real(w, -100.0, 100.0) for w in weights):
        raise AttestationError("public toy tensor value malformed")
    return MAGIC + STRUCT.pack(*weights)


def _decode_toy_adapter(data: bytes) -> tuple[float, float]:
    if len(data) != len(MAGIC) + STRUCT.size or not data.startswith(MAGIC):
        raise AttestationError("public toy tensor bytes malformed")
    weights = STRUCT.unpack(data[len(MAGIC):])
    if any(not math.isfinite(x) or abs(x) > 100 for x in weights):
        raise AttestationError("public toy tensor value malformed")
    return weights


def replay_public_toy(initial: tuple[float, float], mode: str,
                      steps: int, learning_rate: float) -> list[tuple[float, float]]:
    """Deterministic orthogonal two-label SGD (not pretrained-model training)."""
    if mode not in ("zero", "correct", "corrupt"):
        raise AttestationError("public toy control mode malformed")
    if type(steps) is not int or steps < 0 or steps > 16:
        raise AttestationError("public toy optimizer step count malformed")
    if mode == "zero" and steps != 0 or mode != "zero" and steps == 0:
        raise AttestationError("public toy control steps inconsistent")
    if not (_bounded_real(learning_rate, 0.0, 0.5)
            and learning_rate > 0.0):
        raise AttestationError("public toy optimizer learning rate malformed")
    weights = _decode_toy_adapter(encode_toy_adapter(initial))
    targets = (1.0, -1.0) if mode != "corrupt" else (-1.0, 1.0)
    states = []
    for _ in range(steps):
        # One synthetic optimizer step processes both orthogonal public cases.
        weights = tuple(w - learning_rate * (w - target)
                        for w, target in zip(weights, targets))
        states.append(weights)
    return states


def _load_manifest(root: Path) -> dict:
    if not root.is_dir() or root.is_symlink():
        raise AttestationError("public toy run root absent or symlinked")
    raw = _read_bounded(root / "manifest.json", 16 * 1024)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates,
                           parse_constant=_no_nan)
    except (ValueError, UnicodeError, TypeError) as exc:
        raise AttestationError("public toy manifest malformed") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema", "fixture_id", "mode", "claimed_optimizer_steps",
        "learning_rate", "sha256", "claimed_step_sha256",
    }:
        raise AttestationError("public toy manifest schema keys malformed")
    if value["schema"] != SCHEMA or value["fixture_id"] != FIXTURE:
        raise AttestationError("public toy fixture mismatch")
    if not isinstance(value["sha256"], dict) or set(value["sha256"]) != set(NAMES):
        raise AttestationError("public toy file digest map malformed")
    if any(not _hex(v) for v in value["sha256"].values()):
        raise AttestationError("public toy file digest malformed")
    if (not isinstance(value["claimed_step_sha256"], list)
            or len(value["claimed_step_sha256"]) > 16
            or any(not _hex(v) for v in value["claimed_step_sha256"])):
        raise AttestationError("public toy step digest list malformed")
    return value


def _toy_public_loss(weights: tuple[float, float]) -> float:
    return sum((w - target) ** 2 for w, target in zip(weights, (1.0, -1.0))) / 2


def _fresh_process_reload(path: Path) -> dict:
    # Caller cannot provide an executable, module, shell fragment or script.
    # The fixed module merely parses and hashes the public two-float tensor.
    cmd = [sys.executable, "-m", "trainingday.r26_toy_update_attestation",
           "--reload", str(path)]
    env = os.environ.copy()
    env.pop("PYTHONINSPECT", None)
    try:
        run = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1],
                             capture_output=True, text=True, env=env,
                             timeout=10, check=False)
        if run.returncode != 0 or len(run.stdout) > 2048:
            raise AttestationError("fresh-process public toy reload failed")
        value = json.loads(run.stdout, parse_constant=_no_nan,
                           object_pairs_hook=_no_duplicates)
    except (OSError, subprocess.TimeoutExpired, UnicodeError, ValueError) as exc:
        raise AttestationError("fresh-process public toy reload unavailable") from exc
    if not isinstance(value, dict) or set(value) != {"sha256", "weights", "count"}:
        raise AttestationError("fresh-process public toy response malformed")
    return value


def verify_public_toy_run(root: Path, *, expected_base_sha256: str,
                          expected_tokenizer_sha256: str,
                          expected_initial_adapter_sha256: str) -> dict:
    """Read actual toy tensor bytes and test declared step/reload consistency.

    Expected pins are EXTERNAL PARAMETERS, not independently authenticated.
    An attacker controlling both these pins and files can fabricate a coherent
    toy witness; the result therefore NEVER qualifies real neural learning.
    """
    pinned = {
        "base.bin": expected_base_sha256,
        "tokenizer.txt": expected_tokenizer_sha256,
        "adapter_before.bin": expected_initial_adapter_sha256,
    }
    if any(not _hex(value) for value in pinned.values()):
        raise AttestationError("external pin malformed")
    root = Path(root)
    manifest = _load_manifest(root)
    mode, steps, learning_rate = (manifest["mode"],
                                   manifest["claimed_optimizer_steps"],
                                   manifest["learning_rate"])
    # Validate bounded count before reading all content or opening subprocess.
    replay_public_toy((0.0, 0.0), mode, steps, learning_rate)
    files = {name: _read_bounded(root / name, 64 * 1024) for name in NAMES}
    observed = {name: _sha(data) for name, data in files.items()}
    if observed != manifest["sha256"]:
        raise AttestationError("public toy file digest mismatch")
    if any(observed[name] != expected for name, expected in pinned.items()):
        raise AttestationError("external pin mismatch")
    initial = _decode_toy_adapter(files["adapter_before.bin"])
    final = _decode_toy_adapter(files["adapter_after.bin"])
    states = replay_public_toy(initial, mode, steps, learning_rate)
    expected_step_hashes = [_sha(encode_toy_adapter(state)) for state in states]
    expected_final = encode_toy_adapter(states[-1] if states else initial)
    if (manifest["claimed_step_sha256"] != expected_step_hashes
            or expected_final != files["adapter_after.bin"]):
        raise AttestationError("optimizer replay differs from claimed steps/weights")
    changed = files["adapter_before.bin"] != files["adapter_after.bin"]
    if changed is not (mode != "zero"):
        raise AttestationError("optimizer replay update/control mismatch")
    fresh = _fresh_process_reload(root / "adapter_after.bin")
    if (fresh["sha256"] != observed["adapter_after.bin"]
            or fresh["count"] != 2 or fresh["weights"] != list(final)):
        raise AttestationError("fresh-process toy reload differs from saved bytes")
    return {
        "status": "PUBLIC_TOY_UPDATE_REPLAY_ONLY",
        "mode": mode,
        "source_file_sha256": observed,
        "claimed_optimizer_steps": steps,
        "learning_rate": learning_rate,
        "replayed_step_sha256": expected_step_hashes,
        "physical_toy_adapter_bytes_changed": changed,
        "declared_step_replay_matches": True,
        "fresh_process_toy_reload_matches": True,
        "correct_label_toy_loss_before": _toy_public_loss(initial),
        "correct_label_toy_loss_after": _toy_public_loss(final),
        "qwen35_neural_update_verified": False,
        "behavioral_generalization_verified": False,
        "host_lease_authenticated": False,
        "authorizes_gpu_start": False,
        "authorizes_promotion": False,
        "pin_authority": "CALLER_DECLARED_NOT_EXTERNAL_ATTESTED",
    }


def verify_public_toy_control_triplet(paths: dict, **pins) -> dict:
    if not isinstance(paths, dict) or set(paths) != {"zero", "correct", "corrupt"}:
        raise AttestationError("public toy control triplet incomplete")
    reports = {mode: verify_public_toy_run(root, **pins) for mode, root in paths.items()}
    if any(report["mode"] != mode for mode, report in reports.items()):
        raise AttestationError("public toy control label mismatch")
    zero, correct, corrupt = reports["zero"], reports["correct"], reports["corrupt"]
    if (correct["claimed_optimizer_steps"] != corrupt["claimed_optimizer_steps"]
            or len({report["learning_rate"] for report in reports.values()}) != 1):
        raise AttestationError("unmatched public toy control schedule")
    if (zero["physical_toy_adapter_bytes_changed"]
            or not correct["physical_toy_adapter_bytes_changed"]
            or not corrupt["physical_toy_adapter_bytes_changed"]
            or not correct["correct_label_toy_loss_after"] < correct["correct_label_toy_loss_before"]
            or not corrupt["correct_label_toy_loss_after"] > corrupt["correct_label_toy_loss_before"]
            or len({r["source_file_sha256"]["adapter_after.bin"] for r in reports.values()}) != 3):
        raise AttestationError("public toy control behavior inconsistent")
    return {
        "status": "PUBLIC_TOY_CONTROL_REPLAY_ONLY",
        "control_report": reports,
        "qualifies_neural_training": False,
        "model_behavioral_improvement_verified": False,
        "host_lease_authenticated": False,
        "authorizes_promotion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reload", type=Path, required=True)
    args = parser.parse_args()
    data = _read_bounded(args.reload, len(MAGIC) + STRUCT.size)
    weights = _decode_toy_adapter(data)
    print(json.dumps({"sha256": _sha(data), "weights": list(weights), "count": 2},
                     sort_keys=True))


if __name__ == "__main__":
    main()
