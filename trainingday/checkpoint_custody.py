"""Fail-closed bytes and step custody for local Trainer checkpoints.

Inventory is not evidence of successful optimizer/RNG restoration. The
checkpoint receipt SHA must be pinned outside its own training output before
any serialization can be loaded by a resume run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from trainingday.admission import AdmissionError, _hash_file

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT = re.compile(r"checkpoint-([1-9][0-9]*)\Z")
_REQUIRED = {
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "adapter_model.safetensors",
    "adapter_config.json",
}


def canonical_sha(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checkpoint_inventory(directory: Path) -> tuple[int, dict[str, str]]:
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink() or any(p.is_symlink() for p in directory.parents):
        raise AdmissionError("checkpoint directory absent/symlinked")
    match = _CHECKPOINT.fullmatch(directory.name)
    if match is None:
        raise AdmissionError("checkpoint name must bind optimizer step")
    step = int(match.group(1))
    files: dict[str, str] = {}
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise AdmissionError("symlinked checkpoint member")
        if path.is_dir():
            continue
        if not path.is_file() or len(files) >= 128:
            raise AdmissionError("unexpected checkpoint file count/type")
        files[path.relative_to(directory).as_posix()] = _hash_file(path)
    if not _REQUIRED.issubset(files):
        raise AdmissionError("checkpoint missing optimizer/scheduler/adapter/state")
    if not ("rng_state.pth" in files or any(re.fullmatch(r"rng_state_\d+\.pth", f) for f in files)):
        raise AdmissionError("checkpoint missing RNG state")
    try:
        state = json.loads((directory / "trainer_state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AdmissionError("checkpoint Trainer state invalid") from exc
    if not isinstance(state, dict) or state.get("global_step") != step:
        raise AdmissionError("checkpoint directory/step mismatch")
    return step, dict(sorted(files.items()))


def create_custody(directory: Path, *, contract_sha256: str) -> dict:
    if not isinstance(contract_sha256, str) or not _SHA.fullmatch(contract_sha256):
        raise AdmissionError("training contract hash missing")
    step, files = checkpoint_inventory(directory)
    return {"schema": "VERA_TRAINER_STEP_CUSTODY_R16_V1", "run_contract_sha256": contract_sha256,
            "checkpoint_name": directory.name, "step": step, "files": files,
            "state": "CHECKPOINT_BYTES_PINNED_NOT_RESUME_ATTESTED"}


def validate_custody(directory: Path, custody: dict, *, contract_sha256: str) -> int:
    if (not isinstance(custody, dict)
            or custody.get("schema") != "VERA_TRAINER_STEP_CUSTODY_R16_V1"
            or custody.get("state") != "CHECKPOINT_BYTES_PINNED_NOT_RESUME_ATTESTED"
            or custody.get("run_contract_sha256") != contract_sha256):
        raise AdmissionError("checkpoint custody contract/state mismatch")
    step, files = checkpoint_inventory(directory)
    if (step != custody.get("step") or directory.name != custody.get("checkpoint_name")
            or files != custody.get("files")):
        raise AdmissionError("checkpoint bytes/step mismatch")
    return step


def write_external_custody(checkpoint: Path, output_receipt: Path) -> dict:
    """Create an immutable external receipt for a completed Trainer checkpoint.

    Use only after a checkpoint write is known to have completed. Do not use
    this helper to create fake runtime evidence from arbitrary local bytes.
    """
    checkpoint, output_receipt = Path(checkpoint), Path(output_receipt)
    run_output = checkpoint.parent.parent
    contract_file = run_output / "r16-run-contract.json"
    if (not checkpoint.is_absolute() or not output_receipt.is_absolute()
            or not contract_file.is_file() or contract_file.is_symlink()
            or output_receipt.exists() or output_receipt.is_symlink()
            or not output_receipt.parent.is_dir() or output_receipt.parent.is_symlink()
            or output_receipt.is_relative_to(run_output)):
        raise AdmissionError("external checkpoint custody output unsafe")
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    if contract.get("schema") != "VERA_TRAININGDAY_R16_RUN_CONTRACT_V1":
        raise AdmissionError("run contract schema invalid")
    custody = create_custody(checkpoint, contract_sha256=canonical_sha(contract))
    # Exclusive creation: refuse an existing file rather than overwriting a pin.
    with output_receipt.open("x", encoding="utf-8") as out:
        out.write(json.dumps(custody, sort_keys=True, indent=2) + "\n")
    return {"step": custody["step"], "receipt_sha256": _hash_file(output_receipt),
            "state": custody["state"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory R16 checkpoint for external SHA pin")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_external_custody(args.checkpoint, args.output_receipt), sort_keys=True))


if __name__ == "__main__":
    main()
