"""Fail-closed bytes and step custody for local Trainer checkpoints.

Inventory is not evidence of successful optimizer/RNG restoration. The
checkpoint receipt SHA must be pinned outside its own training output before
any serialization can be loaded by a resume run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import uuid
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


def _sync_directory(directory: Path) -> None:
    """Require a directory flush; lack of host support is *not* durable proof."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(directory, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_external_receipt(output_receipt: Path, payload: bytes) -> str:
    """No-overwrite publication of complete bytes; never authorize resume.

    Hard-link publication is exclusive and uses a staging file in the same
    directory. The stage is intentionally retained: unlinking a pathname
    after checking its inode could delete an adversary's last-moment
    substitution. Its later cleanup requires separate trusted ownership.
    A missing directory-fsync capability results in a controlled unknown
    outcome, never an assertion of physical crash durability.
    """
    stage = output_receipt.with_name(
        f".{output_receipt.name}.r16-stage-{uuid.uuid4().hex}"
    )
    file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    published = False
    expected_hash = hashlib.sha256(payload).hexdigest()
    try:
        # These path checks attest a bounded source-level snapshot, not
        # immutable custody against future concurrent filesystem mutation.
        parent = output_receipt.parent
        admitted_ancestors = tuple((ancestor, ancestor.lstat())
                                   for ancestor in output_receipt.parents)
        if any(not stat.S_ISDIR(pinned.st_mode) for _, pinned in admitted_ancestors):
            raise AdmissionError("external custody output ancestor is not a real directory")

        def require_original_parent() -> None:
            # A stable immediate-parent inode is insufficient: the parent can
            # become reachable through a replaced symlinked outer ancestor.
            for ancestor, pinned in admitted_ancestors:
                current = ancestor.lstat()
                if (not stat.S_ISDIR(current.st_mode)
                        or (current.st_dev, current.st_ino)
                        != (pinned.st_dev, pinned.st_ino)):
                    raise AdmissionError("external custody output ancestor identity changed")

        fd = os.open(stage, file_flags, 0o600)
        with os.fdopen(fd, "wb") as out:
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
            staged_stat = os.fstat(out.fileno())
        if (not stat.S_ISREG(staged_stat.st_mode)
                or staged_stat.st_size != len(payload) or staged_stat.st_nlink != 1):
            raise AdmissionError("external custody staging file identity invalid")

        # Both names must be in the same directory, so link() is an atomic,
        # no-overwrite publication boundary. The final target is verified from
        # a no-follow file descriptor before this function returns success.
        try:
            require_original_parent()
            os.link(stage, output_receipt, follow_symlinks=False)
        except FileExistsError:
            # A late FileExistsError may be a lost acknowledgement after our
            # hardlink was actually published. Distinguish that case by inode
            # without mistaking an unrelated competing destination for ours.
            try:
                candidate = output_receipt.lstat()
                if (stat.S_ISREG(candidate.st_mode)
                        and (candidate.st_dev, candidate.st_ino)
                        == (staged_stat.st_dev, staged_stat.st_ino)):
                    published = True
            except OSError:
                pass
            raise
        except Exception:
            # A failed acknowledgement need not mean the filesystem did not
            # publish the hard link. Preserve an unknown outcome for custody.
            if output_receipt.exists() or output_receipt.is_symlink():
                published = True
            raise
        published = True
        read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        committed_fd = os.open(output_receipt, read_flags)
        with os.fdopen(committed_fd, "rb") as committed:
            actual_stat = os.fstat(committed.fileno())
            committed_bytes = committed.read()
        if (not stat.S_ISREG(actual_stat.st_mode)
                or (actual_stat.st_dev, actual_stat.st_ino)
                != (staged_stat.st_dev, staged_stat.st_ino)
                or committed_bytes != payload):
            raise AdmissionError("external custody published bytes/identity mismatch")
        # Confirm the stage has not already been replaced. Do not unlink it:
        # an inode check followed by pathname deletion is not atomic, so an
        # attacker could substitute another file between those operations.
        current_stage = stage.lstat()
        if (not stat.S_ISREG(current_stage.st_mode)
                or (current_stage.st_dev, current_stage.st_ino)
                != (staged_stat.st_dev, staged_stat.st_ino)):
            raise AdmissionError("external custody staging identity changed after publication")
        _sync_directory(output_receipt.parent)
        # Directory synchronization is not the last custody observation:
        # a substituted parent or target after it must never yield success.
        require_original_parent()
        verified_fd = os.open(output_receipt, read_flags)
        with os.fdopen(verified_fd, "rb") as verified:
            verified_stat = os.fstat(verified.fileno())
            verified_bytes = verified.read()
        if (not stat.S_ISREG(verified_stat.st_mode)
                or (verified_stat.st_dev, verified_stat.st_ino)
                != (staged_stat.st_dev, staged_stat.st_ino)
                or verified_bytes != payload):
            raise AdmissionError("external custody final published identity/bytes mismatch")
        # A verified descriptor alone is insufficient if its directory entry
        # was replaced while those bytes were being read.
        final_entry = output_receipt.lstat()
        if (not stat.S_ISREG(final_entry.st_mode)
                or (final_entry.st_dev, final_entry.st_ino)
                != (staged_stat.st_dev, staged_stat.st_ino)):
            raise AdmissionError("external custody final directory entry changed")
        require_original_parent()
        return expected_hash
    except FileExistsError as exc:
        if published:
            raise AdmissionError(
                f"external custody POST_PUBLISH_DURABILITY_UNKNOWN target={output_receipt}"
            ) from exc
        raise AdmissionError("external checkpoint custody output unsafe/existing") from exc
    except Exception as exc:
        if published:
            raise AdmissionError(
                f"external custody POST_PUBLISH_DURABILITY_UNKNOWN target={output_receipt}"
            ) from exc
        raise AdmissionError("external custody staging failed") from exc


def write_external_custody(checkpoint: Path, output_receipt: Path) -> dict:
    """Create a no-overwrite source-level receipt for a completed Trainer checkpoint.

    Use only after a checkpoint write is known to have completed. Do not use
    this helper to create fake runtime evidence from arbitrary local bytes.
    A separate, hidden staging hard link is intentionally retained for
    trusted host-owned cleanup, because deleting an untrusted pathname is
    unsafe under concurrent substitution. The returned hash is a verified
    source snapshot, not an immutable future filesystem or resume attestation.
    """
    checkpoint, output_receipt = Path(checkpoint), Path(output_receipt)
    run_output = checkpoint.parent.parent
    contract_file = run_output / "r16-run-contract.json"
    if (not checkpoint.is_absolute() or not output_receipt.is_absolute()
            or not contract_file.is_file() or contract_file.is_symlink()
            or output_receipt.exists() or output_receipt.is_symlink()
            or not output_receipt.parent.is_dir()
            or any(parent.is_symlink() for parent in output_receipt.parents)
            or ".." in output_receipt.parts
            or output_receipt.resolve().is_relative_to(run_output.resolve())):
        raise AdmissionError("external checkpoint custody output unsafe")
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    if contract.get("schema") != "VERA_TRAININGDAY_R16_RUN_CONTRACT_V1":
        raise AdmissionError("run contract schema invalid")
    custody = create_custody(checkpoint, contract_sha256=canonical_sha(contract))
    encoded = (json.dumps(custody, sort_keys=True, indent=2) + "\n").encode("utf-8")
    receipt_sha = _publish_external_receipt(output_receipt, encoded)
    return {"step": custody["step"], "receipt_sha256": receipt_sha,
            "state": custody["state"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory R16 checkpoint for external SHA pin")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_external_custody(args.checkpoint, args.output_receipt), sort_keys=True))


if __name__ == "__main__":
    main()
