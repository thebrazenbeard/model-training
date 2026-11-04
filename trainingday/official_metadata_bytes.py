"""Source-only byte identities for the *official* Qwen3.5-4B metadata.

The hashes describe upstream files at a pinned revision. Matching local bytes
does not authenticate model weight shards, license a dataset, create a trusted
signature, grant GPU custody, or admit a training run. In particular this
module does NOT populate pinned_base.TRUSTED_OFFICIAL_METADATA_PIN_ITEMS.

tokenizer.json must be the actual 12,807,982-byte LFS object from Hugging
Face's /resolve/ endpoint, never the 133-byte /raw/ LFS pointer.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from trainingday.official_public_base import (
    METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS, OFFICIAL_ID, OFFICIAL_REVISION,
)


SCHEMA = "TRAININGDAY_OFFICIAL_QWEN35_4B_METADATA_SOURCE_V1"
EVIDENCE_CLASS = "PINNED_UPSTREAM_METADATA_BYTES_NOT_WEIGHT_OR_HOST_ADMISSION"
MAX_METADATA_BYTES = 32 * 1024 * 1024
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]{1,120}\Z")
_SECURE_DIRFD_SUPPORTED = (os.name == "posix"
                           and hasattr(os, "O_DIRECTORY")
                           and hasattr(os, "O_NOFOLLOW")
                           and os.open in os.supports_dir_fd)

# (filename, actual upstream object byte length, streamed SHA-256).
# Derived at official revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.
# Six Git blobs were read through /raw; tokenizer.json through /resolve
# because /raw returns only the Git LFS pointer, not the tokenizer object.
OFFICIAL_METADATA_BYTE_PINS: tuple[tuple[str, int, str], ...] = (
    ("config.json", 3161,
     "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"),
    ("tokenizer_config.json", 16710,
     "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"),
    ("tokenizer.json", 12807982,
     "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
    ("model.safetensors.index.json", 76196,
     "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d"),
    ("chat_template.jinja", 7756,
     "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"),
    ("merges.txt", 3353259,
     "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"),
    ("vocab.json", 6722759,
     "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"),
)


class OfficialMetadataBytesError(ValueError):
    """The source-only byte check cannot establish exact metadata identity."""


def _validate_pins(pins: tuple[tuple[str, int, str], ...]) -> None:
    if type(pins) is not tuple or not pins:
        raise OfficialMetadataBytesError("invalid pin set")
    names: set[str] = set()
    for entry in pins:
        if type(entry) is not tuple or len(entry) != 3:
            raise OfficialMetadataBytesError("invalid pin entry")
        name, size, digest = entry
        if (type(name) is not str or _SAFE_NAME.fullmatch(name) is None
                or name in {".", ".."} or name in names
                or type(size) is not int or not (0 < size <= MAX_METADATA_BYTES)
                or type(digest) is not str or _HEX64.fullmatch(digest) is None):
            raise OfficialMetadataBytesError("invalid or duplicated pin")
        names.add(name)


def _open_snapshot_directory_nofollow(snapshot: Path) -> int:
    """Anchor each directory component to a preceding FD, never a path race.

    This deliberately fails closed on platforms without POSIX openat and
    O_DIRECTORY/O_NOFOLLOW. Windows requires a separately reviewed handle-
    based implementation rather than unsafe pre-check/then-open emulation.
    """
    if not _SECURE_DIRFD_SUPPORTED:
        raise OfficialMetadataBytesError("secure directory-FD traversal unavailable")
    if not snapshot.is_absolute() or ".." in snapshot.parts:
        raise OfficialMetadataBytesError("snapshot must be an absolute untraversed path")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current: int | None = None
    try:
        current = os.open(snapshot.anchor, flags)
        for component in snapshot.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = next_fd
        result = current
        current = None  # caller now owns the returned descriptor
        return result
    except (OSError, NotImplementedError, TypeError, ValueError) as exc:
        raise OfficialMetadataBytesError("snapshot path not securely openable") from exc
    finally:
        if current is not None:
            try:
                os.close(current)
            except (OSError, NotImplementedError, TypeError) as exc:
                raise OfficialMetadataBytesError(
                    "cannot close snapshot directory descriptor") from exc


def _fd_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode,
            metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)


@contextmanager
def _filesystem_failures_as_hold():
    """Cover directory fstat and ExitStack close errors as well as file reads.

    Narrow inner handlers preserve filename-specific rejection messages.
    This outer guard also covers syscalls during context cleanup, so the CLI
    cannot accidentally emit an uncaught OSError instead of structured HOLD.
    """
    try:
        yield
    except (OSError, NotImplementedError, TypeError) as exc:
        raise OfficialMetadataBytesError(
            "metadata snapshot directory or file-descriptor operation failed") from exc


def _verify_bytes_against_pins(
    snapshot_dir: str | Path,
    pins: tuple[tuple[str, int, str], ...],
) -> tuple[str, ...]:
    """Low-level injectable test seam; test pins carry NO official authority."""
    _validate_pins(pins)
    if not isinstance(snapshot_dir, (str, Path)):
        raise OfficialMetadataBytesError("invalid snapshot path")
    snapshot = Path(snapshot_dir)
    if not snapshot.is_absolute() or ".." in snapshot.parts:
        raise OfficialMetadataBytesError("snapshot must be a real absolute directory")
    with _filesystem_failures_as_hold(), ExitStack() as stack:
        directory_fd = _open_snapshot_directory_nofollow(snapshot)
        stack.callback(os.close, directory_fd)
        initial_directory = _fd_signature(os.fstat(directory_fd))
        held_files = []
        for name, expected_size, expected_digest in pins:
            flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
            try:
                fd = os.open(name, flags, dir_fd=directory_fd)
                stack.callback(os.close, fd)
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
                    raise OfficialMetadataBytesError(f"wrong size or type: {name}")
                actual_digest = hashlib.sha256()
                actual_size = 0
                while True:
                    block = os.read(fd, 1024 * 1024)
                    if not block:
                        break
                    actual_size += len(block)
                    if actual_size > expected_size:
                        raise OfficialMetadataBytesError(f"oversized metadata: {name}")
                    actual_digest.update(block)
                after = os.fstat(fd)
                if (actual_size != expected_size or after.st_size != expected_size
                        or _fd_signature(before) != _fd_signature(after)
                        or actual_digest.hexdigest() != expected_digest):
                    raise OfficialMetadataBytesError(f"incorrect metadata bytes: {name}")
                held_files.append((name, fd, _fd_signature(after)))
            except (OSError, NotImplementedError, TypeError) as exc:
                raise OfficialMetadataBytesError(
                    f"cannot securely open regular metadata: {name}") from exc

        # Keep every verified file descriptor open until all names have been
        # checked. Revisit both the FD and the directory entry so a deleted,
        # renamed or replaced earlier file does not silently pass in normal
        # concurrent-update cases. This is NOT an atomic multi-file snapshot.
        for name, fd, original_signature in held_files:
            try:
                current_fd = os.fstat(fd)
                current_name = os.stat(name, dir_fd=directory_fd,
                                       follow_symlinks=False)
            except (OSError, NotImplementedError, TypeError) as exc:
                raise OfficialMetadataBytesError(
                    f"metadata entry disappeared or changed: {name}") from exc
            if (not stat.S_ISREG(current_name.st_mode)
                    or _fd_signature(current_fd) != original_signature
                    or _fd_signature(current_name) != original_signature):
                raise OfficialMetadataBytesError(
                    f"metadata entry disappeared or changed: {name}")

        # Rewalk the caller's lexical absolute path to detect persistent
        # ancestor swaps. The opened directory remains the authority for
        # relative file reads; neither pass is atomic against active writers.
        fresh_directory_fd = _open_snapshot_directory_nofollow(snapshot)
        stack.callback(os.close, fresh_directory_fd)
        if _fd_signature(os.fstat(fresh_directory_fd)) != initial_directory:
            raise OfficialMetadataBytesError("snapshot root changed during check")
    return tuple(name for name, _, _ in pins)


def inspect_official_metadata_bytes_source_only(snapshot_dir: str | Path) -> dict:
    """Check seven upstream metadata objects without touching any shard.

    This is a SOURCE-ONLY observation, explicitly *not* the protected
    production checkpoint admission gate or a custody/signature attestation.
    """
    _validate_pins(OFFICIAL_METADATA_BYTE_PINS)
    if set(name for name, _, _ in OFFICIAL_METADATA_BYTE_PINS) != set(
            METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS):
        raise OfficialMetadataBytesError("official metadata pin inventory incomplete")
    matched = _verify_bytes_against_pins(snapshot_dir, OFFICIAL_METADATA_BYTE_PINS)
    return {
        "schema": SCHEMA,
        "model_id": OFFICIAL_ID,
        "revision": OFFICIAL_REVISION,
        "evidence_class": EVIDENCE_CLASS,
        "metadata_byte_matches": list(matched),
        "atomic_immutable_snapshot_verified": False,
        "host_path_custody_verified": False,
        "shard_bytes_verified": False,
        "owner_signature_verified": False,
        "training_admission_granted": False,
        "physical_launch_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True,
                        help="absolute path to a LOCAL metadata snapshot (no downloads)")
    args = parser.parse_args(argv)
    try:
        result = inspect_official_metadata_bytes_source_only(args.snapshot)
    except OfficialMetadataBytesError as exc:
        print(json.dumps({"schema": SCHEMA, "status": "HOLD", "error": str(exc),
                          "physical_launch_allowed": False}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
