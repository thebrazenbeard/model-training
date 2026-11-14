"""Bounded read-only R13 adapter forensic delta; NOT a training receipt.

The original R13 controller reported FAILED_OR_INTERRUPTED_NOT_VERIFIED.
This tool does not rewrite that observation or confer model/runtime authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

_MAX_TENSOR_FILE = 128 * 1024 * 1024
_MAX_HEADER = 2 * 1024 * 1024
_MAX_JSON = 128 * 1024
_MAX_STDOUT = 1024 * 1024


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _read_json(path: Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_JSON:
        raise ValueError("missing or oversized JSON source")
    data = path.read_bytes()
    obj = json.loads(data.decode("utf-8-sig"), object_pairs_hook=_unique_pairs)
    if type(obj) is not dict:
        raise ValueError("JSON source must be an object")
    return obj, hashlib.sha256(data).hexdigest()


def _sha256(path: Path, max_size: int):
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or symlinked source")
    if path.stat().st_size > max_size:
        raise ValueError("oversized source")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _read_tensor_file(path: Path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or symlinked adapter")
    size = path.stat().st_size
    if not 9 <= size <= _MAX_TENSOR_FILE:
        raise ValueError("adapter size outside bound")
    blob = path.read_bytes()
    length = struct.unpack_from("<Q", blob)[0]
    if length < 2 or length > _MAX_HEADER or 8 + length >= len(blob):
        raise ValueError("invalid safetensors header bound")
    header = json.loads(blob[8:8+length].decode("utf8"),
                        object_pairs_hook=_unique_pairs)
    if type(header) is not dict:
        raise ValueError("invalid safetensors header")
    binary = memoryview(blob)[8+length:]
    entries = {}
    intervals = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if type(name) is not str or type(meta) is not dict:
            raise ValueError("invalid safetensors entry")
        if meta.get("dtype") != "F32":
            raise ValueError("non-FP32 adapter tensor")
        shape, offsets = meta.get("shape"), meta.get("data_offsets")
        if not isinstance(shape, list) or not shape or not all(
            type(d) is int and d > 0 for d in shape
        ):
            raise ValueError("invalid tensor shape")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(
            type(offset) is int for offset in offsets
        ):
            raise ValueError("invalid tensor offsets")
        begin, end = offsets
        if begin < 0 or end <= begin or end > len(binary) \
                or 4 * math.prod(shape) != end - begin:
            raise ValueError("invalid tensor byte extent")
        intervals.append((begin, end))
        entries[name] = (tuple(shape), binary[begin:end])
    if not entries:
        raise ValueError("missing LoRA tensors")
    position = 0
    for begin, end in sorted(intervals):
        if begin != position:
            raise ValueError("noncontiguous or overlapping tensor data")
        position = end
    if position != len(binary):
        raise ValueError("trailing tensor bytes")
    return blob, entries


def _canonical_receipt_self_sha(receipt: dict[str, Any]) -> str:
    body = dict(receipt)
    observed = body.pop("receipt_sha256", None)
    calculated = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf8")).hexdigest()
    if observed != calculated:
        raise ValueError("R12 receipt self-hash mismatch")
    return calculated


def audit_paths(r12_adapter: Path, r13_adapter: Path,
                r12_receipt_path: Path, r13_spec_path: Path,
                controller_path: Path, trainer_stdout: Path, *,
                expected_tensors: int = 496) -> dict[str, Any]:
    if type(expected_tensors) is not int or expected_tensors < 1:
        raise ValueError("invalid expected tensor count")
    if (r13_adapter.parent.parent/"DEV_TRAINING_COMPLETE.json").exists():
        raise ValueError("R13 is not unreceipted; use strict receipt auditor")
    r12, _ = _read_json(r12_receipt_path)
    spec, spec_sha = _read_json(r13_spec_path)
    controller, _ = _read_json(controller_path)
    r12_sha = _sha256(r12_adapter, _MAX_TENSOR_FILE)
    r13_sha = _sha256(r13_adapter, _MAX_TENSOR_FILE)
    self_sha = _canonical_receipt_self_sha(r12)
    if r12.get("cumulative_optimizer_steps") != 52 or \
            r12.get("adapter_artifacts", {}).get("files", {}).get(
                "adapter_model.safetensors") != r12_sha:
        raise ValueError("R12 receipt/adapter source binding mismatch")
    resume = spec.get("resume_adapter", {})
    if (resume.get("adapter_model_sha256") != r12_sha or
        resume.get("source_receipt_sha256") != self_sha or
        resume.get("source_cumulative_optimizer_steps") != 52):
        raise ValueError("R13 spec/R12 source binding mismatch")
    if spec.get("trainer", {}).get("max_optimizer_steps") != 32:
        raise ValueError("R13 spec must request 32 optimizer steps")
    if (controller.get("status") != "R13_FAILED_OR_INTERRUPTED_NOT_VERIFIED"
        or controller.get("reason") !=
            "Adapter/receipt missing despite trainer exit 0"
        or controller.get("trainer_exit_code") != 0
        or controller.get("target_cumulative_optimizer_steps") != 84
        or controller.get("source_spec_sha256") != spec_sha
        or controller.get("parent_r12_sha256") != r12_sha
        or controller.get("source_corpus_sha256") !=
            spec.get("source_subject", {}).get("train_sha256")):
        raise ValueError("R13 failed controller/spec binding mismatch")
    if trainer_stdout.is_symlink() or not trainer_stdout.is_file() \
            or trainer_stdout.stat().st_size > _MAX_STDOUT:
        raise ValueError("missing or oversized trainer stdout")
    lines = [line.strip() for line in
             trainer_stdout.read_text(encoding="utf8").splitlines()
             if line.strip()]
    rows = [line for line in lines if line.startswith("{'loss':")
            or line.startswith('{"loss":')]
    if len(rows) != 32:
        raise ValueError("R13 stdout must contain 32 loss rows")
    try:
        terminal = json.loads(lines[-1])
    except (ValueError, IndexError) as exc:
        raise ValueError("R13 terminal HOLD record missing") from exc
    if not isinstance(terminal, dict) or \
            terminal.get("status") != "HOLD" or \
            terminal.get("reason") != "effect label step count is unsupported" or \
            terminal.get("schema") != "V10R2_QWEN35_DEV_TRAINING_RECEIPT_V1":
        raise ValueError("R13 terminal failure mismatch")
    r12_blob, old = _read_tensor_file(r12_adapter)
    r13_blob, new = _read_tensor_file(r13_adapter)
    if len(old) != expected_tensors or old.keys() != new.keys():
        raise ValueError("R13 tensor topology/count mismatch")
    changed_tensors = changed_elements = total_elements = 0
    max_delta = 0.0
    for key in sorted(old):
        old_shape, old_data = old[key]
        new_shape, new_data = new[key]
        if old_shape != new_shape:
            raise ValueError("R13 tensor shape mismatch")
        changed_here = 0
        for (a,), (b,) in zip(struct.iter_unpack("<f", old_data),
                              struct.iter_unpack("<f", new_data)):
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("nonfinite FP32 adapter tensor")
            total_elements += 1
            if a != b:
                changed_here += 1
                max_delta = max(max_delta, abs(b - a))
        changed_elements += changed_here
        changed_tensors += int(changed_here > 0)
    if changed_tensors == 0:
        raise ValueError("no observed saved FP32 tensor delta")
    return {
        "schema": "R13_UNRECEIPTED_SAVED_TENSOR_FORENSICS_V1",
        "status": "R13_FP32_DELTA_OBSERVED_COMPLETION_UNVERIFIED",
        "evidence_class": "LOCAL_FILE_BYTES_AND_HISTORIC_LOGS",
        "r12_adapter_sha256": r12_sha,
        "r13_adapter_sha256": r13_sha,
        "r12_receipt_self_sha256": self_sha,
        "r13_spec_sha256": spec_sha,
        "controller_status": controller["status"],
        "terminal_hold_reason": terminal["reason"],
        "tensor_count": len(old),
        "changed_tensors": changed_tensors,
        "changed_elements": changed_elements,
        "total_elements": total_elements,
        "max_abs_saved_delta": max_delta,
        "reported_loss_steps": len(rows),
        "saved_weight_delta_observed": True,
        "training_completion_verified": False,
        "causal_optimizer_update_proven": False,
        "r13_training_base_shards_independently_bound": False,
        "behavior_qualified": False,
        "runtime_qualified": False,
        "authority_effect": "NONE",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("r12-adapter", "r13-adapter", "r12-receipt",
                 "r13-spec", "controller", "trainer-stdout"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = audit_paths(args.r12_adapter, args.r13_adapter,
                             args.r12_receipt, args.r13_spec,
                             args.controller, args.trainer_stdout)
    except (OSError, ValueError, UnicodeError) as exc:
        print(json.dumps({"schema": "R13_UNRECEIPTED_SAVED_TENSOR_FORENSICS_V1",
                          "status": "REFUSED", "reason": str(exc),
                          "training_completion_verified": False,
                          "authority_effect": "NONE"}, sort_keys=True))
        return 3
    print(json.dumps(report, sort_keys=True))
    return 2  # Always HOLD: an unreceipted adapter cannot qualify.


if __name__ == "__main__":
    sys.exit(main())
