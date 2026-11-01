"""Linux-only, descriptor-pinned *source* checkpoint publication prototype.

The caller supplies an already-open directory descriptor. This module does not
authenticate who owns that descriptor and cannot prevent a hostile process
from modifying files with independent write access. An external host-owned
authorization service, filesystem ownership/isolation and crash qualification
remain mandatory before runtime use. Neither fsync nor a successful rename is
physical power-loss evidence, and no result here authorizes training/resume.

Only pre-staged R18 MeZO checkpoint bytes are admitted. This intentionally
does not replace ``r18_checkpoint_commit`` or implement tensor reload.
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
from pathlib import Path

from .r18_checkpoint_commit import _finite_nonnegative_elapsed


_WEIGHTS = "lora-trainables.safetensors"
_STATE = "state.json"
_COMMIT = "commit.json"
_MEMBER_CAPS = {_WEIGHTS: 512 * 1024 * 1024, _STATE: 1024 * 1024}
_COMMIT_CAP = 8192
_RENAME_NOREPLACE = 1
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
# The step segment is bounded before int() conversion. A 5000-digit ASCII
# number exceeds CPython's configured decimal conversion ceiling.
_STAGE = re.compile(r"\.checkpoint-([1-9][0-9]{0,9})-staging-[A-Za-z0-9_-]{1,64}\Z")
_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


class LinuxCustodyError(RuntimeError):
    """An unsafe or ambiguous source-publication state. Never auto-retry."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _linux_only() -> None:
    if not sys.platform.startswith("linux") or not hasattr(os, "O_NOFOLLOW"):
        raise LinuxCustodyError("LINUX_FD_ENFORCEMENT_UNAVAILABLE")


def open_linux_directory_no_symlinks(path: Path) -> int:
    """Open every absolute path component with O_NOFOLLOW; caller closes fd.

    This is an optional test/operator helper. In production the trusted host
    broker must establish and deliver the custody directory capability.
    """
    _linux_only()
    if type(path) is not type(Path()) or not path.is_absolute() or path == Path("/"):
        raise LinuxCustodyError("OUTPUT_PATH_INVALID")
    fd = os.open("/", _DIR_FLAGS)
    try:
        for component in path.parts[1:]:
            if component in ("", ".", ".."):
                raise LinuxCustodyError("OUTPUT_PATH_INVALID")
            try:
                next_fd = os.open(component, _DIR_FLAGS, dir_fd=fd)
            except OSError as exc:
                raise LinuxCustodyError("OUTPUT_COMPONENT_UNSAFE") from exc
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _require_dirfd(fd: int) -> os.stat_result:
    if type(fd) is not int or fd < 0:
        raise LinuxCustodyError("OUTPUT_DESCRIPTOR_INVALID")
    try:
        info = os.fstat(fd)
    except OSError as exc:
        raise LinuxCustodyError("OUTPUT_DESCRIPTOR_INVALID") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise LinuxCustodyError("OUTPUT_DESCRIPTOR_NOT_DIRECTORY")
    return info


def _open_member(directory_fd: int, name: str, cap: int) -> tuple[int, bytes]:
    try:
        fd = os.open(name, _FILE_FLAGS, dir_fd=directory_fd)
    except OSError as exc:
        raise LinuxCustodyError("MEMBER_OPEN_UNSAFE") from exc
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size <= 0 or before.st_size > cap):
            raise LinuxCustodyError("MEMBER_TYPE_OR_SIZE_INVALID")
        with os.fdopen(os.dup(fd), "rb") as stream:
            data = stream.read(cap + 1)
        after = os.fstat(fd)
        if len(data) != before.st_size or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns
        ):
            raise LinuxCustodyError("MEMBER_CHANGED_DURING_READ")
        os.fsync(fd)
        return fd, data
    except BaseException:
        os.close(fd)
        raise


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict:
    obj: dict = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def _plain_json_bounded(value: object) -> bool:
    """Mirror canonical R18 finite/typed JSON plus depth and node ceilings."""
    pending = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > 100_000 or depth > 64:
            return False
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                return False
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is float:
            if not math.isfinite(item):
                return False
        elif type(item) not in (str, int, bool, type(None)):
            return False
    return True


def _decode_json(data: bytes) -> dict:
    try:
        result = json.loads(
            data.decode("utf-8"), object_pairs_hook=_strict_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite number")),
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise LinuxCustodyError("STATE_JSON_INVALID") from exc
    if type(result) is not dict or not _plain_json_bounded(result):
        raise LinuxCustodyError("STATE_JSON_INVALID")
    return result


def _validate_state(state_data: bytes, *, step: int, weights_sha256: str) -> None:
    state = _decode_json(state_data)
    elapsed = state.get("optimizer_window_seconds")
    # Use the R18 checkpoint publisher's exact numeric admission rule. In
    # particular, a finite JSON integer above 2**53 - 1 is not an admissible
    # elapsed duration even though math.isfinite(int) may return True.
    valid_elapsed = _finite_nonnegative_elapsed(elapsed)
    if (state.get("schema") != "R18_MEZO_STATE_V1"
            or state.get("status") != "CHECKPOINT_BYTES_PINNED_NOT_RELOADED"
            or type(state.get("completed_steps")) is not int
            or state["completed_steps"] != step
            or state.get("weights_sha256") != weights_sha256
            or type(state.get("run_contract_sha256")) is not str
            or _SHA256.fullmatch(state["run_contract_sha256"]) is None
            or type(state.get("counters")) is not dict
            or not valid_elapsed):
        raise LinuxCustodyError("STATE_CONTRACT_MISMATCH")


def _rename_noreplace(directory_fd: int, stage_name: str, target_name: str) -> None:
    """Both pathnames are single components interpreted under ONE pinned fd."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise LinuxCustodyError("RENAME_NOREPLACE_UNAVAILABLE")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    if renameat2(directory_fd, os.fsencode(stage_name), directory_fd,
                 os.fsencode(target_name), _RENAME_NOREPLACE) != 0:
        err = ctypes.get_errno()
        if err == errno.EEXIST:
            raise LinuxCustodyError("DESTINATION_ALREADY_EXISTS")
        raise LinuxCustodyError(f"RENAME_NOREPLACE_FAILED_{errno.errorcode.get(err, 'UNKNOWN')}")


def _same_dir(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _readback(
    directory_fd: int,
    expected_sha: dict[str, str],
    original_members: dict[str, tuple[int, int]],
) -> None:
    if set(os.listdir(directory_fd)) != set(expected_sha):
        raise LinuxCustodyError("READBACK_MEMBERS_NOT_EXACT")
    caps = {**_MEMBER_CAPS, _COMMIT: _COMMIT_CAP}
    for name, expected in expected_sha.items():
        fd, raw = _open_member(directory_fd, name, caps[name])
        try:
            info = os.fstat(fd)
            if (info.st_dev, info.st_ino) != original_members[name]:
                raise LinuxCustodyError("READBACK_MEMBER_REBOUND")
            if hashlib.sha256(raw).hexdigest() != expected:
                raise LinuxCustodyError("READBACK_MEMBER_CHANGED")
        finally:
            os.close(fd)


def publish_prepared_linux(output_fd: int, *, stage_name: str, step: int) -> dict:
    """Bind source bytes, write a marker, rename without overwrite, and read back.

    ``stage_name`` refers to an existing directory that the host owns, with
    exactly the two R18 checkpoint members. No arbitrary stage-writing callback
    is executed. A post-rename failure is permanently ambiguous to this caller.
    The returned status remains SOURCE_ONLY and never grants resume.
    """
    _linux_only()
    original_parent = _require_dirfd(output_fd)
    stage_match = _STAGE.fullmatch(stage_name) if type(stage_name) is str else None
    if (type(step) is not int or not 1 <= step <= 2**31 - 1
            or stage_match is None or int(stage_match.group(1)) != step):
        raise LinuxCustodyError("PUBLICATION_REQUEST_INVALID")
    target_name = f"checkpoint-{step}"
    try:
        stage_fd = os.open(stage_name, _DIR_FLAGS, dir_fd=output_fd)
    except OSError as exc:
        raise LinuxCustodyError("STAGE_DESCRIPTOR_UNSAFE") from exc
    renamed = False
    rename_attempted = False
    try:
        stage_identity = os.fstat(stage_fd)
        if set(os.listdir(stage_fd)) != {_WEIGHTS, _STATE}:
            raise LinuxCustodyError("STAGING_MEMBERS_NOT_EXACT")
        weights_fd, weights = _open_member(stage_fd, _WEIGHTS, _MEMBER_CAPS[_WEIGHTS])
        try:
            state_fd, state = _open_member(stage_fd, _STATE, _MEMBER_CAPS[_STATE])
            try:
                weight_sha = hashlib.sha256(weights).hexdigest()
                state_sha = hashlib.sha256(state).hexdigest()
                _validate_state(state, step=step, weights_sha256=weight_sha)
                manifest = {
                    "schema": "R18_CHECKPOINT_COMMIT_V1",
                    "status": "SOURCE_ONLY_BYTES_COMMITTED_NOT_RELOADED",
                    "completed_steps": step,
                    "members": {_WEIGHTS: weight_sha, _STATE: state_sha},
                    "authorizes_resume": False,
                    "authorizes_training": False,
                }
                encoded = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
                try:
                    marker_fd = os.open(_COMMIT, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                                        getattr(os, "O_CLOEXEC", 0) |
                                        getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=stage_fd)
                except OSError as exc:
                    raise LinuxCustodyError("COMMIT_CREATE_FAILED") from exc
                try:
                    count = os.write(marker_fd, encoded)
                    if count != len(encoded):
                        raise LinuxCustodyError("COMMIT_SHORT_WRITE")
                    os.fsync(marker_fd)
                    marker_info = os.fstat(marker_fd)
                finally:
                    os.close(marker_fd)
                os.fsync(stage_fd)
                expected = {_WEIGHTS: weight_sha, _STATE: state_sha, _COMMIT: hashlib.sha256(encoded).hexdigest()}
                member_ids = {}
                for name, fd in ((_WEIGHTS, weights_fd), (_STATE, state_fd)):
                    info = os.fstat(fd)
                    member_ids[name] = (info.st_dev, info.st_ino)
                member_ids[_COMMIT] = (marker_info.st_dev, marker_info.st_ino)
                _readback(stage_fd, expected, member_ids)
                if not _same_dir(original_parent, os.fstat(output_fd)):
                    raise LinuxCustodyError("OUTPUT_DESCRIPTOR_CHANGED")
                # Even a syscall wrapper reporting failure after attempting a
                # rename cannot establish that no bytes were published. Hold
                # all such outcomes as UNKNOWN; EEXIST is the sole reliable
                # non-replacing, no-write rejection.
                rename_attempted = True
                _rename_noreplace(output_fd, stage_name, target_name)
                renamed = True
                try:
                    target_fd = os.open(target_name, _DIR_FLAGS, dir_fd=output_fd)
                except OSError as exc:
                    raise LinuxCustodyError("PUBLISHED_DIRECTORY_REBOUND") from exc
                try:
                    if not _same_dir(stage_identity, os.fstat(target_fd)):
                        raise LinuxCustodyError("PUBLISHED_DIRECTORY_REBOUND")
                    _readback(target_fd, expected, member_ids)
                    os.fsync(output_fd)
                    if not _same_dir(stage_identity, os.fstat(target_fd)) or not _same_dir(original_parent, os.fstat(output_fd)):
                        raise LinuxCustodyError("PUBLISHED_DIRECTORY_REBOUND")
                    _readback(target_fd, expected, member_ids)
                finally:
                    os.close(target_fd)
                return {
                    "status": "SOURCE_ONLY_FD_PINNED_NOT_RELOADED",
                    "checkpoint_name": target_name,
                    "checkpoint_weights_sha256": weight_sha,
                    "checkpoint_state_sha256": state_sha,
                    "checkpoint_commit_sha256": expected[_COMMIT],
                    "authorizes_training": False,
                    "authorizes_resume": False,
                    "host_authority_verified": False,
                    "physical_durability_verified": False,
                    "physical_reload_verified": False,
                }
            finally:
                os.close(state_fd)
        finally:
            os.close(weights_fd)
    except BaseException as exc:
        if renamed or (rename_attempted and not (
            isinstance(exc, LinuxCustodyError)
            and exc.reason == "DESTINATION_ALREADY_EXISTS"
        )):
            raise LinuxCustodyError("POST_RENAME_CUSTODY_UNKNOWN") from exc
        raise
    finally:
        os.close(stage_fd)
