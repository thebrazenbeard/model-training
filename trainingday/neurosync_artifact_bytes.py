"""Bounded local-byte readback for TRAININGDAY/Neuro-Sync artifact roles.

This is *not* an independent host witness: a caller can supply invented
expected digests or counterfeit files. The module never authenticates
artifact origin, model training, rights, checkpoints, or launch authority.

On POSIX, each path is opened component-by-component relative to held
directory descriptors with O_NOFOLLOW; on platforms lacking that primitive
(notably ordinary Windows Python), the operation fails closed instead of
silently relying on a symlink-prone resolved-path check.

No deserialization, model loading, file mutation, network, or GPU use occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from typing import Any, Mapping


SCHEMA = "trainingday.neurosync.local-artifact-bytes.v1"
ROLES = (
    "donor_trained_checkpoint",
    "recipient_base_checkpoint",
    "adapter_weights",
    "adapter_config",
    "tokenizer",
)
PEFT_ONLY_ROLES = ("adapter_weights", "adapter_config", "tokenizer")
PROFILES = {
    "FULL_FIVE_ROLE": ROLES,
    "PEFT_ADAPTER_ONLY_PARTIAL_HOLD": PEFT_ONLY_ROLES,
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+@-]{0,254}\Z")
_CHUNK = 1024 * 1024
_MAX_SINGLE = 32 * 1024**3
_MAX_TOTAL = 64 * 1024**3
_MANIFEST_LIMIT = 32768
_ELAPSED_CEILING_SECONDS = 600
_LOCAL_RECEIPT_CONSTRUCTOR = object()


class ArtifactHold(ValueError):
    """No trustworthy bounded local-byte observation could be produced."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ArtifactByteObservation:
    role: str
    relative_path: str
    sha256: str
    size_bytes: int
    device_id: int
    inode_id: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True, init=False)
class LocalByteEvidence:
    observations: tuple[ArtifactByteObservation, ...]
    profile: str

    def __init__(self, observations: tuple[ArtifactByteObservation, ...],
                 profile: str, *, _internal: object = None) -> None:
        # Guard accidental public construction and especially empty-success
        # receipts. This is NOT cryptographic authentication: Python module
        # internals can be inspected/forged by same-process adversaries.
        if _internal is not _LOCAL_RECEIPT_CONSTRUCTOR:
            raise ArtifactHold("BYTE_EVIDENCE_REQUIRES_LOCAL_SCANNER")
        expected_roles = PROFILES.get(profile)
        if (expected_roles is None or type(observations) is not tuple
                or len(observations) != len(expected_roles)
                or any(not isinstance(obs, ArtifactByteObservation)
                       or obs.role != role
                       for obs, role in zip(observations, expected_roles))):
            raise ArtifactHold("INCOMPLETE_LOCAL_BYTE_OBSERVATIONS")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "profile", profile)

    @property
    def status(self) -> str:
        if self.profile == "PEFT_ADAPTER_ONLY_PARTIAL_HOLD":
            return "PEFT_ADAPTER_ONLY_PARTIAL_HOLD__ORIGIN_UNVERIFIED"
        return "LOCAL_FILE_BYTES_MATCH_DECLARATIONS__ORIGIN_UNVERIFIED_HOLD"

    @property
    def physical_launch_authorized(self) -> bool:
        return False

    @property
    def state_commit_authorized(self) -> bool:
        return False

    @property
    def rights_verified(self) -> bool:
        return False

    @property
    def independent_host_witness(self) -> bool:
        return False

    @property
    def trained_model_verified(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "profile": self.profile,
            "observations": [
                {
                    "role": x.role, "relative_path": x.relative_path,
                    "sha256": x.sha256, "size_bytes": x.size_bytes,
                    "device_id": x.device_id, "inode_id": x.inode_id,
                    "mtime_ns": x.mtime_ns, "ctime_ns": x.ctime_ns,
                }
                for x in self.observations
            ],
            "physical_launch_authorized": False,
            "state_commit_authorized": False,
            "rights_verified": False,
            "independent_host_witness": False,
            "trained_model_verified": False,
            "evidence_class": "LOCAL_READBACK_UNATTESTED",
        }


def _require_secure_open() -> None:
    if (os.name != "posix"
            or not hasattr(os, "O_NOFOLLOW")
            or not hasattr(os, "O_DIRECTORY")
            or os.open not in os.supports_dir_fd
            or os.stat not in os.supports_dir_fd):
        raise ArtifactHold("PLATFORM_NO_ATOMIC_PATH_CONTAINMENT")


def _segments(relative_path: object) -> tuple[str, ...]:
    if not isinstance(relative_path, str) or not relative_path or len(relative_path) > 4096:
        raise ArtifactHold("INVALID_RELATIVE_PATH")
    if ("\\" in relative_path or ":" in relative_path
            or relative_path.startswith("/") or "\x00" in relative_path):
        raise ArtifactHold("INVALID_RELATIVE_PATH")
    parts = relative_path.split("/")
    if any(x in ("", ".", "..") or not _SAFE_SEGMENT.fullmatch(x) for x in parts):
        raise ArtifactHold("INVALID_RELATIVE_PATH")
    return tuple(parts)


def _manifest_rows(manifest: Mapping[str, Any]) -> tuple[str, tuple[tuple[str, tuple[str, ...], str, int], ...]]:
    if not isinstance(manifest, Mapping) or set(manifest) != {"schema", "profile", "artifacts"}:
        raise ArtifactHold("INVALID_MANIFEST_FIELDS")
    if manifest["schema"] != SCHEMA:
        raise ArtifactHold("INVALID_MANIFEST_SCHEMA")
    profile = manifest["profile"]
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ArtifactHold("UNSUPPORTED_ARTIFACT_PROFILE")
    roles = PROFILES[profile]
    try:
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError):
        raise ArtifactHold("INVALID_MANIFEST_JSON") from None
    if len(encoded) > _MANIFEST_LIMIT:
        raise ArtifactHold("MANIFEST_TOO_LARGE")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(roles):
        raise ArtifactHold("ARTIFACT_ROLES_REQUIRED")
    rows = []
    paths = set()
    digests = set()
    total = 0
    for role in roles:
        entry = artifacts[role]
        if not isinstance(entry, Mapping) or set(entry) != {"relative_path", "sha256", "size_bytes"}:
            raise ArtifactHold("INVALID_ARTIFACT_FIELDS")
        parts = _segments(entry["relative_path"])
        digest = entry["sha256"]
        size = entry["size_bytes"]
        if not isinstance(digest, str) or not _SHA.fullmatch(digest):
            raise ArtifactHold("INVALID_SHA256")
        if type(size) is not int or size <= 0 or size > _MAX_SINGLE:
            raise ArtifactHold("INVALID_SIZE_CEILING")
        if parts in paths:
            raise ArtifactHold("DUPLICATE_ARTIFACT_PATH")
        if digest in digests:
            raise ArtifactHold("ALIASED_ARTIFACT_DIGEST")
        paths.add(parts)
        digests.add(digest)
        total += size
        if total > _MAX_TOTAL:
            raise ArtifactHold("TOTAL_SIZE_CEILING")
        rows.append((role, parts, digest, size))
    return profile, tuple(rows)


def _open_directory_components(absolute_root: Path) -> int:
    """Return held root fd, rejecting symlink ancestors (not just last path)."""
    if not absolute_root.is_absolute() or ".." in absolute_root.parts:
        raise ArtifactHold("ROOT_MUST_BE_ABSOLUTE_CLEAN_PATH")
    parts = absolute_root.parts
    if len(parts) < 3:
        raise ArtifactHold("ALLOWROOT_TOO_BROAD")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    fd = None
    try:
        fd = os.open(parts[0], flags)
        for segment in parts[1:]:
            next_fd = os.open(segment, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except OSError:
        if fd is not None:
            os.close(fd)
        raise ArtifactHold("ROOT_DIRECTORY_UNAVAILABLE_OR_LINKED") from None


def _open_bounded_file(root_fd: int, parts: tuple[str, ...]) -> tuple[int, int]:
    """Return held parent and file fds, both caller-owned."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    parent_fd = os.dup(root_fd)
    try:
        for segment in parts[:-1]:
            next_fd = os.open(segment, flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        file_flags = (os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
                      | getattr(os, "O_NONBLOCK", 0))
        file_fd = os.open(parts[-1], file_flags, dir_fd=parent_fd)
        return parent_fd, file_fd
    except OSError:
        os.close(parent_fd)
        raise ArtifactHold("ARTIFACT_UNAVAILABLE_OR_LINKED") from None


def _scan_file(root_fd: int, role: str, parts: tuple[str, ...],
               expected_digest: str, expected_size: int,
               deadline: float) -> ArtifactByteObservation:
    parent_fd, file_fd = _open_bounded_file(root_fd, parts)
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ArtifactHold("ARTIFACT_NOT_EXCLUSIVE_REGULAR_FILE")
        if before.st_size != expected_size:
            raise ArtifactHold("ARTIFACT_SIZE_MISMATCH")
        hasher = hashlib.sha256()
        remaining = expected_size
        while remaining:
            if time.monotonic() > deadline:
                raise ArtifactHold("ARTIFACT_TIME_BUDGET_EXCEEDED")
            block = os.read(file_fd, min(_CHUNK, remaining))
            if not block:
                raise ArtifactHold("ARTIFACT_TRUNCATED_DURING_READ")
            hasher.update(block)
            remaining -= len(block)
        if time.monotonic() > deadline:
            raise ArtifactHold("ARTIFACT_TIME_BUDGET_EXCEEDED")
        if os.read(file_fd, 1):
            raise ArtifactHold("ARTIFACT_GREW_DURING_READ")
        after = os.fstat(file_fd)
        try:
            current = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
        except OSError:
            raise ArtifactHold("ARTIFACT_PATH_CHANGED_DURING_READ") from None
        identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (any(getattr(before, a) != getattr(after, a) for a in identity)
                or any(getattr(after, a) != getattr(current, a) for a in identity)
                or not stat.S_ISREG(current.st_mode)):
            raise ArtifactHold("ARTIFACT_CHANGED_DURING_READ")
        actual_digest = hasher.hexdigest()
        if actual_digest != expected_digest:
            raise ArtifactHold("ARTIFACT_SHA256_MISMATCH")
        return ArtifactByteObservation(
            role=role, relative_path="/".join(parts), sha256=actual_digest,
            size_bytes=expected_size, device_id=after.st_dev, inode_id=after.st_ino,
            mtime_ns=after.st_mtime_ns, ctime_ns=after.st_ctime_ns,
        )
    except OSError:
        raise ArtifactHold("ARTIFACT_READ_FAILED") from None
    finally:
        os.close(file_fd)
        os.close(parent_fd)


def verify_local_artifact_bytes(
    allowroot: str | Path, manifest: Mapping[str, Any],
) -> LocalByteEvidence:
    """Read *existing* explicit files without writing or asserting custody.

    Successful completion means only that the selected profile's held file
    descriptors matched
    untrusted expected digests and sizes during this local read. An untrusted
    caller cannot transform this into an independently signed host receipt.
    In particular, Neuro-Sync's `target_map_sha256` is a canonical *structural*
    digest, NOT the SHA-256 of a target-map file, and is outside this module.
    """
    profile, rows = _manifest_rows(manifest)
    _require_secure_open()
    if not isinstance(allowroot, (str, Path)):
        raise ArtifactHold("INVALID_ALLOWROOT")
    root = Path(allowroot)
    root_fd = _open_directory_components(root)
    deadline = time.monotonic() + _ELAPSED_CEILING_SECONDS
    try:
        original_root = os.fstat(root_fd)
        seen_inodes: set[tuple[int, int]] = set()
        observations = []
        for role, parts, digest, size in rows:
            observed = _scan_file(root_fd, role, parts, digest, size, deadline)
            inode = (observed.device_id, observed.inode_id)
            if inode in seen_inodes:
                raise ArtifactHold("ARTIFACT_HARDLINK_ALIAS")
            seen_inodes.add(inode)
            observations.append(observed)

        # Held dirfds intentionally survive a rename, which is good for
        # descriptor safety but can make a path-named receipt misleading.
        # Reopen the requested absolute root through no-follow ancestors to
        # catch ordinary rename/replacement during this bounded scan.
        try:
            current_fd = _open_directory_components(root)
        except ArtifactHold:
            raise ArtifactHold("ALLOWROOT_PATH_CHANGED_DURING_READ") from None
        try:
            current_root = os.fstat(current_fd)
            held_root = os.fstat(root_fd)
            identity = ("st_dev", "st_ino", "st_ctime_ns")
            if (any(getattr(original_root, a) != getattr(held_root, a)
                    for a in identity)
                    or any(getattr(held_root, a) != getattr(current_root, a)
                           for a in identity)):
                raise ArtifactHold("ALLOWROOT_PATH_CHANGED_DURING_READ")
        finally:
            os.close(current_fd)

        return LocalByteEvidence(tuple(observations), profile,
                                 _internal=_LOCAL_RECEIPT_CONSTRUCTOR)
    finally:
        os.close(root_fd)


__all__ = [
    "SCHEMA", "ROLES", "PEFT_ONLY_ROLES", "PROFILES", "ArtifactHold", "ArtifactByteObservation",
    "LocalByteEvidence", "verify_local_artifact_bytes",
]
