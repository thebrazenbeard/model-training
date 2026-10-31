"""Source-only R18 checkpoint publication with an explicit commit boundary.

This module never launches training and never authorizes resume. A directory
rename is not proof of power-loss durability, optimizer/RNG restoration,
authenticated host custody, or tensor reload. Linux uses RENAME_NOREPLACE to
prevent replacement races; Windows uses its non-replacing directory rename.
Other platforms fail closed. Directory flush support differs by filesystem
and host and must be qualified independently before any runtime use.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Callable


SCHEMA = "R18_CHECKPOINT_COMMIT_V1"
WEIGHTS = "lora-trainables.safetensors"
STATE = "state.json"
COMMIT = "commit.json"
MAX_WEIGHTS_BYTES = 512 * 1024 * 1024
MAX_STATE_BYTES = 1024 * 1024
SHA256 = re.compile(r"[a-f0-9]{64}\Z")
# pathlib permits subclasses to disagree about __fspath__ (filesystem I/O)
# and __str__ (receipt provenance). Accept only the native concrete type.
_NATIVE_PATH_TYPE = type(Path())


class CheckpointPublicationError(RuntimeError):
    """Fail-closed source custody error; remnants remain quarantined."""

    def __init__(self, reason: str, *, stage_path: Path | None = None,
                 target_path: Path | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.stage_path = stage_path
        self.target_path = target_path


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict:
    value: dict = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _plain_json(value: object) -> bool:
    stack = [(value, 0)]
    count = 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if count > 100_000 or depth > 64:
            return False
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                return False
            stack.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            stack.extend((child, depth + 1) for child in item)
        elif type(item) is float:
            if not math.isfinite(item):
                return False
        elif type(item) not in (str, int, bool, type(None)):
            return False
    return True


def _read_strict_json_and_hash(
    path: Path, *, cap: int, expected_sha256: str | None = None,
) -> tuple[str, dict]:
    """Hash and decode exactly the bytes read through one checked descriptor."""
    metadata = _regular_owned_file(path, cap=cap)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CheckpointPublicationError("MEMBER_OPEN_FAILED") from exc
    try:
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(cap + 1)
        actual = os.fstat(fd)
        if (not stat.S_ISREG(actual.st_mode)
                or (actual.st_dev, actual.st_ino, actual.st_size) != (
                    metadata.st_dev, metadata.st_ino, metadata.st_size)
                or not raw or len(raw) != metadata.st_size):
            raise CheckpointPublicationError("MEMBER_CHANGED_DURING_READ")
        os.fsync(fd)
    except OSError as exc:
        raise CheckpointPublicationError("MEMBER_READ_OR_FSYNC_FAILED") from exc
    finally:
        os.close(fd)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise CheckpointPublicationError("COMMIT_HASH_MISMATCH")
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_strict_pairs,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON number: {constant}")
            ),
        )
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise CheckpointPublicationError("STATE_NOT_STRICT_JSON") from exc
    if type(value) is not dict or not _plain_json(value):
        raise CheckpointPublicationError("STATE_NOT_STRICT_JSON")
    return digest, value


def _regular_owned_file(path: Path, *, cap: int) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CheckpointPublicationError("MEMBER_MISSING") from exc
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            or metadata.st_size <= 0 or metadata.st_size > cap):
        raise CheckpointPublicationError("MEMBER_TYPE_OR_SIZE_INVALID")
    return metadata


def _hash_file(path: Path, *, cap: int) -> str:
    expected = _regular_owned_file(path, cap=cap)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CheckpointPublicationError("MEMBER_OPEN_FAILED") from exc
    try:
        actual = os.fstat(fd)
        if not stat.S_ISREG(actual.st_mode) or (
                actual.st_dev, actual.st_ino, actual.st_size) != (
                expected.st_dev, expected.st_ino, expected.st_size):
            raise CheckpointPublicationError("MEMBER_IDENTITY_CHANGED")
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(fd, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk)
                if size > cap:
                    raise CheckpointPublicationError("MEMBER_EXCEEDS_SIZE_CAP")
                digest.update(chunk)
        if size != expected.st_size:
            raise CheckpointPublicationError("MEMBER_CHANGED_DURING_HASH")
        os.fsync(fd)
        return digest.hexdigest()
    except OSError as exc:
        raise CheckpointPublicationError("MEMBER_READ_OR_FSYNC_FAILED") from exc
    finally:
        os.close(fd)


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        # Do not assert Windows directory flush/power-loss semantics from Python.
        return
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise CheckpointPublicationError("DIRECTORY_FSYNC_FAILED") from exc


def _exclusive_rename(stage: Path, target: Path) -> None:
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise CheckpointPublicationError("EXCLUSIVE_RENAME_UNAVAILABLE")
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p,
                           ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        # AT_FDCWD = -100, RENAME_NOREPLACE = 1
        result = rename(-100, os.fsencode(stage), -100, os.fsencode(target), 1)
        if result != 0:
            err = ctypes.get_errno()
            if err == errno.EEXIST:
                raise CheckpointPublicationError("DESTINATION_ALREADY_EXISTS")
            raise CheckpointPublicationError(
                f"EXCLUSIVE_RENAME_FAILED_{errno.errorcode.get(err, 'UNKNOWN')}"
            )
    elif os.name == "nt":
        try:
            os.rename(stage, target)  # Windows refuses existing destinations.
        except FileExistsError as exc:
            raise CheckpointPublicationError("DESTINATION_ALREADY_EXISTS") from exc
        except OSError as exc:
            raise CheckpointPublicationError("EXCLUSIVE_RENAME_FAILED") from exc
    else:
        raise CheckpointPublicationError("EXCLUSIVE_RENAME_UNAVAILABLE")


def _validate_stage(stage: Path, *, step: int) -> dict[str, str]:
    if stage.is_symlink() or not stage.is_dir():
        raise CheckpointPublicationError("STAGING_DIRECTORY_INVALID")
    try:
        actual = {child.name for child in stage.iterdir()}
    except OSError as exc:
        raise CheckpointPublicationError("STAGING_LIST_FAILED") from exc
    if actual != {WEIGHTS, STATE}:
        raise CheckpointPublicationError("STAGING_MEMBERS_NOT_EXACT")
    return _validate_payload(stage, step=step)


def _directory_identity(directory: Path) -> tuple[int, int]:
    """Observe an actual directory, never a symlink to a directory."""
    try:
        metadata = directory.lstat()
    except OSError as exc:
        raise CheckpointPublicationError("STAGING_DIRECTORY_CHANGED") from exc
    if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_ino <= 0):
        raise CheckpointPublicationError("STAGING_DIRECTORY_CHANGED")
    return metadata.st_dev, metadata.st_ino


def _assert_directory_identity(directory: Path, original: tuple[int, int]) -> None:
    if _directory_identity(directory) != original:
        raise CheckpointPublicationError("STAGING_DIRECTORY_CHANGED")


def _pin_directory_chain(directory: Path) -> tuple[tuple[Path, tuple[int, int]], ...]:
    """Bind every pathname component, not just its final directory inode.

    A parent-directory rename followed by a symlink alias preserves the
    child's inode. The caller must therefore check every ancestor with lstat
    before attesting source-byte custody. This is a bounded snapshot check,
    not a guarantee that hostile filesystem mutation cannot occur afterward.
    """
    pinned: list[tuple[Path, tuple[int, int]]] = []
    current = directory
    while True:
        pinned.append((current, _directory_identity(current)))
        if current.parent == current:
            break
        current = current.parent
    return tuple(pinned)


def _assert_directory_chain(
    pinned: tuple[tuple[Path, tuple[int, int]], ...],
) -> None:
    for path, identity in pinned:
        _assert_directory_identity(path, identity)


def _target_now_present(target: Path) -> bool:
    """An ambiguous post-rename target means never blindly retry."""
    try:
        target.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        # Inability to determine target state must also hold replays.
        return True
    return True


def _target_is_original_stage(target: Path, original: tuple[int, int]) -> bool:
    try:
        return _directory_identity(target) == original
    except CheckpointPublicationError:
        return False


def _validate_payload(directory: Path, *, step: int) -> dict[str, str]:
    weights_hash = _hash_file(directory / WEIGHTS, cap=MAX_WEIGHTS_BYTES)
    state_hash, state_value = _read_strict_json_and_hash(
        directory / STATE, cap=MAX_STATE_BYTES
    )
    elapsed = state_value.get("optimizer_window_seconds")
    if (state_value.get("schema") != "R18_MEZO_STATE_V1"
            or state_value.get("status") != "CHECKPOINT_BYTES_PINNED_NOT_RELOADED"
            or type(state_value.get("completed_steps")) is not int
            or state_value["completed_steps"] != step
            or state_value.get("weights_sha256") != weights_hash
            or type(state_value.get("run_contract_sha256")) is not str
            or SHA256.fullmatch(state_value["run_contract_sha256"]) is None
            or type(elapsed) not in (int, float)
            or not math.isfinite(elapsed) or elapsed < 0
            or type(state_value.get("counters")) is not dict):
        raise CheckpointPublicationError("STATE_CONTRACT_MISMATCH")
    return {WEIGHTS: weights_hash, STATE: state_hash}


def _member_identity(path: Path, *, cap: int) -> tuple[int, int, int]:
    """Pin the regular-file inode and size; no symlinks or hardlink aliases."""
    metadata = _regular_owned_file(path, cap=cap)
    return metadata.st_dev, metadata.st_ino, metadata.st_size


def _assert_pinned_member_bytes(
    directory: Path, pins: dict[str, tuple[int, int, int]],
    expected_hashes: dict[str, str],
) -> None:
    """Re-open after payload validation to detect a changed member pathname.

    Earlier hash/parse validity cannot attest a file replaced after parsing.
    These are source-snapshot checks, not a promise of future immutable custody.
    """
    caps = {WEIGHTS: MAX_WEIGHTS_BYTES, STATE: MAX_STATE_BYTES, COMMIT: 8 * 1024}
    for name, cap in caps.items():
        path = directory / name
        if _member_identity(path, cap=cap) != pins[name]:
            raise CheckpointPublicationError("READBACK_MEMBER_CHANGED")
        if _hash_file(path, cap=cap) != expected_hashes[name]:
            raise CheckpointPublicationError("READBACK_MEMBER_CHANGED")
        if _member_identity(path, cap=cap) != pins[name]:
            raise CheckpointPublicationError("READBACK_MEMBER_CHANGED")


def publish_checkpoint_atomic(
    output: Path, *, step: int, write_stage: Callable[[Path], None],
) -> dict:
    """Stage, hash, mark complete, then publish one checkpoint-N directory.

    Only a local source-byte snapshot at the final verification boundary is
    reported. No pathname check guarantees future custody against concurrent
    hostile filesystem mutation. On failure, staged remnants are non-adoptable;
    a post-rename ambiguity is UNKNOWN and must never be blindly retried.
    """
    if (type(output) is not _NATIVE_PATH_TYPE
            or type(step) is not int or step < 1
            or step > 2**31 - 1 or not callable(write_stage)
            or output.is_symlink() or not output.is_dir()):
        raise CheckpointPublicationError("PUBLICATION_REQUEST_INVALID")
    try:
        output_chain = _pin_directory_chain(output)
    except CheckpointPublicationError as exc:
        raise CheckpointPublicationError("PUBLICATION_REQUEST_INVALID") from exc
    target = output / f"checkpoint-{step}"
    if target.exists() or target.is_symlink():
        raise CheckpointPublicationError("DESTINATION_ALREADY_EXISTS")
    try:
        stage = Path(tempfile.mkdtemp(prefix=f".checkpoint-{step}-staging-", dir=output))
    except OSError as exc:
        raise CheckpointPublicationError("STAGING_CREATE_FAILED") from exc
    renamed = False
    stage_identity = _directory_identity(stage)
    try:
        write_stage(stage)
        hashes = _validate_stage(stage, step=step)
        _assert_directory_identity(stage, stage_identity)
        declaration = {
            "schema": SCHEMA,
            "status": "SOURCE_ONLY_BYTES_COMMITTED_NOT_RELOADED",
            "completed_steps": step,
            "members": hashes,
            "authorizes_resume": False,
            "authorizes_training": False,
        }
        manifest = (json.dumps(
            declaration, sort_keys=True, separators=(",", ":"), allow_nan=False
        ) + "\n").encode("utf-8")
        with (stage / COMMIT).open("xb") as stream:
            stream.write(manifest)
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(stage)
        _assert_directory_identity(stage, stage_identity)
        _assert_directory_chain(output_chain)
        _exclusive_rename(stage, target)
        renamed = True
        # The exclusive rename can be tricked by an exchanged stage pathname.
        # A successful syscall alone is not provenance of the validated inode.
        _assert_directory_chain(output_chain)
        _assert_directory_identity(target, stage_identity)
        pinned_commit_sha256 = hashlib.sha256(manifest).hexdigest()
        verified = verify_published_checkpoint(
            target, expected_step=step,
            expected_commit_sha256=pinned_commit_sha256,
        )
        if (verified["checkpoint_state_sha256"] != hashes[STATE]
                or verified["checkpoint_weights_sha256"] != hashes[WEIGHTS]):
            raise CheckpointPublicationError("PUBLISHED_MEMBERS_CHANGED")
        _sync_directory(output)
        # The parent can be rebound while the sync call is in progress, or the
        # published child can be swapped after the earlier hash verification.
        # Re-verify both directory identities and the exact pinned bytes just
        # before emitting the source-only snapshot receipt.
        _assert_directory_chain(output_chain)
        _assert_directory_identity(target, stage_identity)
        final = verify_published_checkpoint(
            target, expected_step=step,
            expected_commit_sha256=pinned_commit_sha256,
        )
        if (final["checkpoint_state_sha256"] != hashes[STATE]
                or final["checkpoint_weights_sha256"] != hashes[WEIGHTS]):
            raise CheckpointPublicationError("PUBLISHED_MEMBERS_CHANGED")
        # The final readback can race with a pathname exchange. Bind its
        # return to the original stage inode and output-root identity too.
        _assert_directory_identity(target, stage_identity)
        _assert_directory_chain(output_chain)
        return {
            "status": "SOURCE_ONLY_BYTES_COMMITTED_NOT_RELOADED",
            "checkpoint_path": os.fspath(target),
            "checkpoint_state_sha256": hashes[STATE],
            "checkpoint_weights_sha256": hashes[WEIGHTS],
            "checkpoint_commit_sha256": pinned_commit_sha256,
            "authorizes_resume": False,
            "authorizes_training": False,
            "physical_durability_verified": False,
        }
    except CheckpointPublicationError as exc:
        ambiguous = renamed or _target_is_original_stage(target, stage_identity) or (
            exc.reason not in {"DESTINATION_ALREADY_EXISTS"}
            and _target_now_present(target)
        )
        raise CheckpointPublicationError(
            "POST_RENAME_DURABILITY_UNKNOWN" if ambiguous else exc.reason,
            stage_path=None if ambiguous else stage,
            target_path=target if ambiguous else None,
        ) from exc
    except Exception as exc:
        ambiguous = renamed or _target_now_present(target)
        raise CheckpointPublicationError(
            "POST_RENAME_DURABILITY_UNKNOWN" if ambiguous
            else "STAGING_WRITER_FAILED",
            stage_path=None if ambiguous else stage,
            target_path=target if ambiguous else None,
        ) from exc


def verify_published_checkpoint(
    directory: Path, *, expected_step: int, expected_commit_sha256: str,
) -> dict:
    """Verify externally pinned *source bytes*; never authorize resume.

    The SHA caller supplies is not itself authenticated by this module.
    Verification is non-adopting and does not deserialize safetensors.
    """
    if (type(directory) is not _NATIVE_PATH_TYPE or directory.is_symlink()
            or not directory.is_dir() or type(expected_step) is not int
            or not 1 <= expected_step <= 2**31 - 1
            or directory.name != f"checkpoint-{expected_step}"
            or type(expected_commit_sha256) is not str
            or SHA256.fullmatch(expected_commit_sha256) is None):
        raise CheckpointPublicationError("READBACK_REQUEST_INVALID")
    # Pin every ancestor as well as the checkpoint directory. A symlink
    # rebound in a grandparent can preserve the immediate parent's inode.
    # Hashing member bytes through such an alias must not certify custody.
    try:
        directory_chain = _pin_directory_chain(directory)
    except CheckpointPublicationError as exc:
        raise CheckpointPublicationError("READBACK_DIRECTORY_CHANGED") from exc
    try:
        present = {item.name for item in directory.iterdir()}
    except OSError as exc:
        raise CheckpointPublicationError("READBACK_DIRECTORY_INVALID") from exc
    if present != {WEIGHTS, STATE, COMMIT}:
        raise CheckpointPublicationError("READBACK_MEMBERS_NOT_EXACT")
    caps = {WEIGHTS: MAX_WEIGHTS_BYTES, STATE: MAX_STATE_BYTES, COMMIT: 8 * 1024}
    member_pins = {
        name: _member_identity(directory / name, cap=cap)
        for name, cap in caps.items()
    }
    commit_hash, marker = _read_strict_json_and_hash(
        directory / COMMIT, cap=8 * 1024,
        expected_sha256=expected_commit_sha256,
    )
    hashes = _validate_payload(directory, step=expected_step)
    if (set(marker) != {
            "schema", "status", "completed_steps", "members",
            "authorizes_resume", "authorizes_training",
    } or marker.get("schema") != SCHEMA
            or marker.get("status") != "SOURCE_ONLY_BYTES_COMMITTED_NOT_RELOADED"
            or type(marker.get("completed_steps")) is not int
            or marker["completed_steps"] != expected_step
            or type(marker.get("members")) is not dict
            or marker["members"] != hashes
            or marker.get("authorizes_resume") is not False
            or marker.get("authorizes_training") is not False):
        raise CheckpointPublicationError("COMMIT_DECLARATION_INVALID")
    try:
        # The payload parser (or concurrent filesystem actor) may change a
        # member pathname after its bytes were hashed. Verify all three names
        # still refer to the pinned inodes and the same bytes before success.
        _assert_pinned_member_bytes(
            directory, member_pins,
            {COMMIT: commit_hash, WEIGHTS: hashes[WEIGHTS], STATE: hashes[STATE]},
        )
        _assert_directory_chain(directory_chain)
    except CheckpointPublicationError as exc:
        raise CheckpointPublicationError("READBACK_DIRECTORY_CHANGED") from exc
    return {
        "status": "SOURCE_ONLY_BYTES_VERIFIED_NOT_RELOADED",
        "source_bytes_verified": True,
        "checkpoint_path": os.fspath(directory),
        "checkpoint_state_sha256": hashes[STATE],
        "checkpoint_weights_sha256": hashes[WEIGHTS],
        "checkpoint_commit_sha256": commit_hash,
        "authorizes_resume": False,
        "authorizes_training": False,
        "physical_reload_verified": False,
        "physical_durability_verified": False,
    }
