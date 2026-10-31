"""R28 public two-parameter CPU trajectory and restart probe.

This is a nonauthorizing synthetic learning test. It does not load Qwen,
authenticate cross-host execution, evaluate private OOD data, or grant a GPU
lease. A self-consistent SHA-256 is not an independently signed receipt.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import struct
import sys


class PublicTrajectoryError(ValueError):
    """Public-toy state, checkpoint, or trajectory is invalid."""


MAX_STEPS = 256
MAX_CHECKPOINT_BYTES = 4096
UINT32_MAX = 2**32 - 1
MAX_ABS_FLOAT = 1_000_000.0
MAX_ABS_TOLERANCE = 0.001
FORMAT = "R28_PUBLIC_TWO_PARAMETER_CPU_TRAJECTORY_V1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")

# Six invented, non-personal, public-toy linear-regression cases. Never corpus
# input from the actual Vera training workbench or any protected evaluation.
PUBLIC_EXAMPLES = (
    ((1.0, 0.0), 0.75),
    ((0.0, 1.0), -0.5),
    ((1.0, 1.0), 0.2),
    ((-0.5, 0.25), -0.3),
    ((0.5, -1.0), 0.625),
    ((0.25, 0.75), 0.45),
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


PUBLIC_DATASET_SHA256 = hashlib.sha256(_canonical_json([
    [[a.hex(), b.hex()], target.hex()]
    for (a, b), target in PUBLIC_EXAMPLES
])).hexdigest()


def _number(value: object) -> float:
    if type(value) is not float or not math.isfinite(value) or abs(value) > MAX_ABS_FLOAT:
        raise PublicTrajectoryError("expected finite bounded binary64 public value")
    return value


def _floats(value: object, name: str) -> tuple[float, float]:
    if type(value) not in (list, tuple) or len(value) != 2:
        raise PublicTrajectoryError(f"{name} must contain exactly two binary64 values")
    return _number(value[0]), _number(value[1])


def _uint32(value: object, *, nonzero: bool = False) -> int:
    if type(value) is not int or not (1 if nonzero else 0) <= value <= UINT32_MAX:
        raise PublicTrajectoryError("invalid 32-bit deterministic PRNG state/seed")
    return value


@dataclass(frozen=True)
class ToyState:
    seed: int
    step: int
    rng: int
    weights: tuple[float, float]
    momentum: tuple[float, float]
    public_dataset_sha256: str


def _state(state: ToyState) -> ToyState:
    if type(state) is not ToyState:
        raise PublicTrajectoryError("expected exact immutable toy state type")
    _uint32(state.seed, nonzero=True)
    _uint32(state.rng, nonzero=True)
    if type(state.step) is not int or not 0 <= state.step <= MAX_STEPS:
        raise PublicTrajectoryError("invalid toy step counter")
    _floats(state.weights, "weights")
    _floats(state.momentum, "momentum")
    if state.public_dataset_sha256 != PUBLIC_DATASET_SHA256:
        raise PublicTrajectoryError("checkpoint dataset commitment differs")
    return state


def new_trajectory(seed: int) -> ToyState:
    """Freeze a nonzero 32-bit Xorshift seed and zero-initialized toy state."""
    _uint32(seed, nonzero=True)
    return ToyState(seed, 0, seed, (0.0, 0.0), (0.0, 0.0),
                    PUBLIC_DATASET_SHA256)


def _xor32(value: int) -> int:
    value ^= (value << 13) & UINT32_MAX
    value ^= value >> 17
    value ^= (value << 5) & UINT32_MAX
    return value & UINT32_MAX


RECEIPT_KEYS = frozenset((
    "step", "seed", "sample_index", "rng", "public_dataset_sha256",
    "weights", "optimizer_momentum", "pre_loss", "post_loss", "state_sha256",
))


def _validated_receipt(row: object, *, check_digest: bool) -> dict:
    if type(row) is not dict or set(row) != RECEIPT_KEYS:
        raise PublicTrajectoryError("untrusted step receipt shape")
    if (type(row["step"]) is not int or not 1 <= row["step"] <= MAX_STEPS
            or type(row["sample_index"]) is not int
            or not 0 <= row["sample_index"] < len(PUBLIC_EXAMPLES)):
        raise PublicTrajectoryError("step/sample index invalid")
    _uint32(row["seed"], nonzero=True)
    _uint32(row["rng"], nonzero=True)
    if row["public_dataset_sha256"] != PUBLIC_DATASET_SHA256:
        raise PublicTrajectoryError("receipt references different public dataset")
    _floats(row["weights"], "weights")
    _floats(row["optimizer_momentum"], "optimizer momentum")
    _number(row["pre_loss"])
    _number(row["post_loss"])
    if type(row["state_sha256"]) is not str or not SHA256.fullmatch(row["state_sha256"]):
        raise PublicTrajectoryError("invalid trajectory step digest")
    if check_digest and row["state_sha256"] != receipt_digest(row):
        raise PublicTrajectoryError("trajectory step bytes/digest mismatch")
    return row


def receipt_digest(row: dict) -> str:
    """Recompute internal byte-consistency only; never a signed attestation."""
    _validated_receipt(row, check_digest=False)
    raw = struct.pack(
        "<QQQI6d", row["step"], row["seed"], row["rng"],
        row["sample_index"], *row["weights"], *row["optimizer_momentum"],
        row["pre_loss"], row["post_loss"],
    ) + bytes.fromhex(row["public_dataset_sha256"])
    return hashlib.sha256(raw).hexdigest()


def advance(initial: ToyState, steps: int) -> tuple[ToyState, tuple[dict, ...]]:
    """Apply real small SGD-with-momentum CPU updates and immutable receipts."""
    current = _state(initial)
    if type(steps) is not int or not 0 <= steps <= MAX_STEPS - current.step:
        raise PublicTrajectoryError("toy step budget exceeded")
    receipts = []
    for _ in range(steps):
        rng = _xor32(current.rng)
        index = rng % len(PUBLIC_EXAMPLES)
        (x0, x1), truth = PUBLIC_EXAMPLES[index]
        w0, w1 = current.weights
        m0, m1 = current.momentum
        error = ((w0 * x0) + (w1 * x1)) - truth
        pre_loss = error * error
        # Fixed binary-fraction coefficients avoid hidden optimizer schedules.
        next_m0 = (m0 * 0.125) + (error * (0.025 * x0))
        next_m1 = (m1 * 0.125) + (error * (0.025 * x1))
        next_w0, next_w1 = w0 - next_m0, w1 - next_m1
        post_error = ((next_w0 * x0) + (next_w1 * x1)) - truth
        post_loss = post_error * post_error
        current = _state(ToyState(
            seed=current.seed, step=current.step + 1, rng=rng,
            weights=(next_w0, next_w1), momentum=(next_m0, next_m1),
            public_dataset_sha256=PUBLIC_DATASET_SHA256,
        ))
        row = {
            "step": current.step, "seed": current.seed,
            "sample_index": index, "rng": current.rng,
            "public_dataset_sha256": PUBLIC_DATASET_SHA256,
            "weights": list(current.weights),
            "optimizer_momentum": list(current.momentum),
            "pre_loss": pre_loss, "post_loss": post_loss,
            "state_sha256": "0" * 64,
        }
        row["state_sha256"] = receipt_digest(row)
        receipts.append(row)
    return current, tuple(receipts)


def save_checkpoint(state: ToyState) -> bytes:
    """Serialize exact binary64/PRNG/optimizer values as canonical hex + hash."""
    state = _state(state)
    payload = {
        "seed": state.seed, "step": state.step, "rng": state.rng,
        "weights_hex": [value.hex() for value in state.weights],
        "momentum_hex": [value.hex() for value in state.momentum],
        "public_dataset_sha256": PUBLIC_DATASET_SHA256,
    }
    envelope = {
        "format": FORMAT, "payload": payload,
        "payload_sha256": hashlib.sha256(_canonical_json(payload)).hexdigest(),
    }
    encoded = _canonical_json(envelope)
    if len(encoded) > MAX_CHECKPOINT_BYTES:
        raise PublicTrajectoryError("toy checkpoint exceeds byte cap")
    return encoded


def _unique_json_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise PublicTrajectoryError("duplicate toy checkpoint key")
        result[key] = value
    return result


def _hex_float(value: object) -> float:
    if type(value) is not str or len(value) > 32 or not value.startswith(("0x", "-0x")):
        raise PublicTrajectoryError("noncanonical public binary64 hex value")
    try:
        number = float.fromhex(value)
    except (ValueError, OverflowError) as exc:
        raise PublicTrajectoryError("bad toy binary64 value") from exc
    _number(number)
    if number.hex() != value:
        raise PublicTrajectoryError("noncanonical toy binary64 encoding")
    return number


def load_checkpoint(encoded: bytes) -> ToyState:
    """Check canonical source-only checkpoint; no trust-anchor is asserted."""
    if type(encoded) is not bytes or not 20 <= len(encoded) <= MAX_CHECKPOINT_BYTES:
        raise PublicTrajectoryError("public checkpoint byte size invalid")
    try:
        envelope = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_json_keys,
                              parse_constant=lambda _: (_ for _ in ()).throw(
                                  PublicTrajectoryError("nonfinite JSON")))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise PublicTrajectoryError("invalid public checkpoint JSON") from exc
    if encoded != _canonical_json(envelope):
        raise PublicTrajectoryError("noncanonical public checkpoint bytes")
    if (type(envelope) is not dict or set(envelope) != {"format", "payload", "payload_sha256"}
            or envelope["format"] != FORMAT or type(envelope["payload"]) is not dict):
        raise PublicTrajectoryError("invalid public checkpoint schema")
    payload = envelope["payload"]
    if set(payload) != {"seed", "step", "rng", "weights_hex", "momentum_hex", "public_dataset_sha256"}:
        raise PublicTrajectoryError("invalid optimizer/RNG checkpoint schema")
    digest = envelope["payload_sha256"]
    if (type(digest) is not str or not SHA256.fullmatch(digest)
            or digest != hashlib.sha256(_canonical_json(payload)).hexdigest()):
        raise PublicTrajectoryError("toy checkpoint checksum mismatch")
    for key in ("weights_hex", "momentum_hex"):
        if type(payload[key]) is not list or len(payload[key]) != 2:
            raise PublicTrajectoryError("optimizer weight shape invalid")
    return _state(ToyState(
        seed=payload["seed"], step=payload["step"], rng=payload["rng"],
        weights=tuple(_hex_float(v) for v in payload["weights_hex"]),
        momentum=tuple(_hex_float(v) for v in payload["momentum_hex"]),
        public_dataset_sha256=payload["public_dataset_sha256"],
    ))


def _trajectory(rows: object) -> tuple[dict, ...]:
    if type(rows) not in (list, tuple) or not 1 <= len(rows) <= MAX_STEPS:
        raise PublicTrajectoryError("public trajectory must contain 1..256 receipts")
    result = tuple(_validated_receipt(row, check_digest=True) for row in rows)
    first = result[0]
    for offset, row in enumerate(result):
        if row["step"] != first["step"] + offset or row["seed"] != first["seed"]:
            raise PublicTrajectoryError("non-contiguous or mismatched seeded trajectory")
    return result


def compare_trajectories(left: object, right: object, *, tolerance: float = 0.0) -> dict:
    """Compare two untrusted public-toy receipts with explicit finite absolute tol."""
    if type(tolerance) not in (int, float) or type(tolerance) is bool or not math.isfinite(tolerance) or not 0 <= tolerance <= MAX_ABS_TOLERANCE:
        raise PublicTrajectoryError("absolute comparison tolerance invalid")
    a, b = _trajectory(left), _trajectory(right)
    if len(a) != len(b):
        raise PublicTrajectoryError("unequal or truncated replay length")
    max_abs_delta = 0.0
    first_divergence = None
    exact = True
    per_step = []
    for x, y in zip(a, b):
        control_match = (
            x["step"] == y["step"] and x["seed"] == y["seed"]
            and x["rng"] == y["rng"] and x["sample_index"] == y["sample_index"]
            and x["public_dataset_sha256"] == y["public_dataset_sha256"]
        )
        sha_match = x["state_sha256"] == y["state_sha256"]
        if not sha_match:
            exact = False
        comparisons = (
            *zip(x["weights"], y["weights"]),
            *zip(x["optimizer_momentum"], y["optimizer_momentum"]),
            (x["pre_loss"], y["pre_loss"]),
            (x["post_loss"], y["post_loss"]),
        )
        delta = max(abs(v1 - v2) for v1, v2 in comparisons)
        max_abs_delta = max(max_abs_delta, delta)
        step_ok = control_match and delta <= tolerance and (tolerance > 0 or sha_match)
        if not step_ok and first_divergence is None:
            first_divergence = x["step"]
        per_step.append({
            "step": x["step"], "control_state_identical": control_match,
            "sha256_identical": sha_match,
            "max_absolute_numeric_difference": delta,
            "within_declared_tolerance": step_ok,
        })
    status = ("TRAJECTORY_DIVERGENCE" if first_divergence is not None
              else "EXACT_MATCH" if exact else "WITHIN_DECLARED_TOLERANCE")
    return {
        "status": status, "first_divergence_step": first_divergence,
        "max_absolute_numeric_difference": max_abs_delta,
        "per_step_differences": per_step,
        "absolute_tolerance": float(tolerance), "steps_compared": len(a),
        "public_dataset_sha256": PUBLIC_DATASET_SHA256,
        "independent_receipts_authenticated": False,
        "cross_os_replay_physically_verified": False,
        "qwen4b_qualified": False,
        "gpu_training_authorized": False,
    }


def verify_public_reference_trajectory(rows: object, *, tolerance: float = 0.0) -> dict:
    """Recompute all public-toy updates from step one, not from caller verdicts."""
    observed = _trajectory(rows)
    if observed[0]["step"] != 1:
        raise PublicTrajectoryError("reference verification requires trajectory from step one")
    _, expected = advance(new_trajectory(observed[0]["seed"]), len(observed))
    comparison = compare_trajectories(expected, observed, tolerance=tolerance)
    return {
        **comparison,
        "reference_recomputed_from_pinned_public_toy": True,
        "reference_source_fingerprint_sha256": environment_fingerprint()[
            "source_fingerprint_sha256"],
        "physical_cross_host_attested": False,
    }


def environment_fingerprint() -> dict:
    """Nonidentifying observations; OS family varies without invalidating replay."""
    os_name = platform.system()
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": ".".join(str(i) for i in sys.version_info[:3]),
        "byteorder": sys.byteorder,
        "os_family": os_name if os_name in {"Windows", "Linux", "Darwin"} else "Other",
        "binary64_size": struct.calcsize("d"),
        "source_fingerprint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "public_dataset_sha256": PUBLIC_DATASET_SHA256,
        "physical_cross_host_attested": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="R28 public CPU toy trajectory, nonauthorizing")
    parser.add_argument("--seed", type=int, default=721)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--resume-checkpoint-hex", default=None)
    args = parser.parse_args()
    try:
        start = (new_trajectory(args.seed) if args.resume_checkpoint_hex is None
                 else load_checkpoint(bytes.fromhex(args.resume_checkpoint_hex)))
        final, receipts = advance(start, args.steps)
    except (ValueError, PublicTrajectoryError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "scope": "PUBLIC_CPU_TOY_ONLY", "environment": environment_fingerprint(),
        "steps": receipts, "checkpoint_hex": save_checkpoint(final).hex(),
        "qwen4b_qualified": False, "gpu_training_authorized": False,
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
