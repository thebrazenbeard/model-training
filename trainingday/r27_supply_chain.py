"""R27 static public-toy supply-chain witness; never fetches or runs artifacts.

No human or service signs these declarations. A matching SHA-256 binds supplied
bytes to a *caller-provided* manifest, not to an authenticated upstream
publisher. This code cannot authorize executable code, training, or release.
"""

from __future__ import annotations

import hashlib
import gzip
import io
import re
import stat
import tarfile
import unicodedata
import zipfile
from dataclasses import dataclass


_HASH = re.compile(r"[a-f0-9]{64}\Z")
_REVISION = re.compile(r"[a-f0-9]{40}\Z")
_NAME = re.compile(r"[a-z][a-z0-9_-]{2,63}\Z")
_LOCATOR = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/+=-]{2,127}\Z")
_LICENSE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{1,63}\Z")
_REQUIREMENT = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([0-9][A-Za-z0-9._+-]*)\Z")
_ENTRY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,255}\Z")
_ROLES = frozenset({"training-package", "quantizer-binary", "model-weights", "adapter", "runtime-config", "archive"})
_COMPONENT_KEYS = frozenset({"name", "role", "locator", "source_revision", "artifact_sha256", "declared_bytes", "license_spdx"})
_MANIFEST_KEYS = frozenset({"schema", "requirements_sha256", "components"})
_SCHEMA = "TRAININGDAY_R27_STATIC_SUPPLY_CHAIN_V1"
_MAX_REQUIREMENTS_BYTES = 4096
_MAX_REQUIREMENTS_LINES = 64
_MAX_COMPONENTS = 32
_MAX_DECLARED_BYTES = 16 * 1024**3
_MAX_OBSERVED_BYTES = 4 * 1024**2
_MAX_ARCHIVE_ENTRIES = 64
_MAX_ARCHIVE_TOTAL_UNCOMPRESSED = 32 * 1024**2
_MAX_ARCHIVE_ENTRY_UNCOMPRESSED = 16 * 1024**2
_MAX_COMPRESSION_RATIO = 20
_WIN_DEVICE_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


@dataclass(frozen=True)
class SupplyChainAssessment:
    reason: str
    static_receipt_consistent: bool = False
    artifact_bytes_matched: bool = False
    direct_requirement_versions_pinned: bool = False
    archive_metadata_safe: bool = False
    dependency_wheel_hashes_verified: bool = False
    reproducible_execution_verified: bool = False
    licenses_authenticated: bool = False
    binary_execution_authorized: bool = False
    model_release_authorized: bool = False


def _reject(reason: str) -> SupplyChainAssessment:
    return SupplyChainAssessment(reason)


def _requirements_valid(raw: object) -> bool:
    if type(raw) is not str or len(raw) > _MAX_REQUIREMENTS_BYTES:
        return False
    lines = raw.splitlines()
    if len(lines) > _MAX_REQUIREMENTS_LINES or not lines:
        return False
    names: set[str] = set()
    found = False
    for line in lines:
        if not line or line.startswith("#"):
            continue
        match = _REQUIREMENT.fullmatch(line)
        if match is None:
            return False
        package = match.group(1).lower().replace("_", "-").replace(".", "-")
        if package in names:
            return False
        names.add(package)
        found = True
    return found


def _component_valid(c: object) -> bool:
    if type(c) is not dict or c.keys() != _COMPONENT_KEYS:
        return False
    if any(type(c[k]) is not str for k in _COMPONENT_KEYS if k != "declared_bytes"):
        return False
    if type(c["declared_bytes"]) is not int or not (1 <= c["declared_bytes"] <= _MAX_DECLARED_BYTES):
        return False
    locator = c["locator"]
    if (_NAME.fullmatch(c["name"]) is None or c["role"] not in _ROLES
            or _LOCATOR.fullmatch(locator) is None or ".." in locator or "//" in locator
            or _REVISION.fullmatch(c["source_revision"]) is None
            or _HASH.fullmatch(c["artifact_sha256"]) is None
            or _LICENSE.fullmatch(c["license_spdx"]) is None):
        return False
    return True


def _safe_member_name(name: object) -> bool:
    """Portable Windows/POSIX relative member name, with no filesystem I/O.

    Windows device aliases apply even to file extensions, nested path segments
    and the superscript ¹²³ variants of COM/LPT. NFKC is only a rejection
    check; it never rewrites an archive path into something allegedly safe.
    """
    if type(name) is not str or _ENTRY.fullmatch(name) is None:
        return False
    for segment in name.split("/"):
        if segment in ("", ".", "..") or segment.endswith((".", " ")):
            return False
        stem = unicodedata.normalize("NFKC", segment).split(".", 1)[0].upper()
        if stem in _WIN_DEVICE_STEMS:
            return False
    return True


def _safe_zip_metadata(payload: bytes) -> bool:
    """Inspect ZIP central directory without extracting/opening members."""
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if not (1 <= len(entries) <= _MAX_ARCHIVE_ENTRIES):
                return False
            names: set[str] = set()
            cumulative = 0
            for info in entries:
                name = info.filename
                if not _safe_member_name(name):
                    return False
                if name.casefold() in names:
                    return False
                names.add(name.casefold())
                if name.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz")):
                    return False
                file_type = stat.S_IFMT(info.external_attr >> 16)
                # Directory entries are not needed in this bounded fixture;
                # a directory declaration bearing data must never look safe.
                if info.is_dir() or file_type not in (0, stat.S_IFREG):
                    return False
                if info.flag_bits & 1 or info.file_size < 0 or info.compress_size < 0:
                    return False
                if info.file_size > _MAX_ARCHIVE_ENTRY_UNCOMPRESSED:
                    return False
                cumulative += info.file_size
                if cumulative > _MAX_ARCHIVE_TOTAL_UNCOMPRESSED:
                    return False
                if (info.file_size > 65536
                        and info.file_size > _MAX_COMPRESSION_RATIO * max(1, info.compress_size)):
                    return False
            return True
    except (zipfile.BadZipFile, OSError, ValueError, EOFError, OverflowError, UnicodeError):
        return False


def _safe_tar_gz_metadata(payload: bytes) -> bool:
    """Bounded gzip header/member metadata parse; never extract tar payloads."""
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as source:
            expanded = source.read(_MAX_ARCHIVE_TOTAL_UNCOMPRESSED + 1)
        if len(expanded) > _MAX_ARCHIVE_TOTAL_UNCOMPRESSED:
            return False
        if (len(expanded) > 65536
                and len(expanded) > _MAX_COMPRESSION_RATIO * max(1, len(payload))):
            return False
        with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:") as archive:
            # PAX extended/global headers can override member paths, link
            # targets, ownership and OS xattrs. This tiny public fixture needs
            # none; refuse them instead of pretending we vet their semantics.
            if archive.pax_headers:
                return False
            entries = archive.getmembers()
            if not (1 <= len(entries) <= _MAX_ARCHIVE_ENTRIES):
                return False
            names: set[str] = set()
            cumulative = 0
            for info in entries:
                if info.pax_headers:
                    return False
                name = info.name.rstrip("/") if info.isdir() else info.name
                if not _safe_member_name(name):
                    return False
                if name.casefold() in names:
                    return False
                names.add(name.casefold())
                # Disallow all links, devices, FIFO, PAX global special files.
                if not (info.isfile() or info.isdir()):
                    return False
                if info.size < 0 or info.size > _MAX_ARCHIVE_ENTRY_UNCOMPRESSED:
                    return False
                cumulative += info.size
                if cumulative > _MAX_ARCHIVE_TOTAL_UNCOMPRESSED:
                    return False
            return True
    except (tarfile.TarError, OSError, EOFError, ValueError, OverflowError, UnicodeError):
        return False


def _safe_archive_metadata(payload: bytes) -> bool:
    """Inspect bounded ZIP/tar.gz manifests only; never extract/run members."""
    if payload.startswith(b"PK\x03\x04"):
        return _safe_zip_metadata(payload)
    if payload.startswith(b"\x1f\x8b"):
        return _safe_tar_gz_metadata(payload)
    return False


def assess_supply_chain(
    manifest: object, *, requirements_text: str,
    observed_artifacts: object,
) -> SupplyChainAssessment:
    """Assess bounded caller-supplied *public* bytes without external effects.

    Declared version/revision/licensing is not authenticated provenance. Only
    test-size bytes (<=4 MiB each) can be directly witnessed here. Incomplete
    observations stay HOLD, including large real 4B weight files by design.
    """
    if type(manifest) is not dict or manifest.keys() != _MANIFEST_KEYS:
        return _reject("MALFORMED_MANIFEST")
    if (type(manifest["schema"]) is not str or manifest["schema"] != _SCHEMA
            or type(manifest["requirements_sha256"]) is not str
            or _HASH.fullmatch(manifest["requirements_sha256"]) is None
            or type(manifest["components"]) is not list
            or not (1 <= len(manifest["components"]) <= _MAX_COMPONENTS)):
        return _reject("MALFORMED_MANIFEST")
    if not _requirements_valid(requirements_text):
        return _reject("UNPINNED_REQUIREMENT")
    try:
        requirements_digest = hashlib.sha256(requirements_text.encode("utf-8")).hexdigest()
    except (UnicodeEncodeError, AttributeError):
        return _reject("UNPINNED_REQUIREMENT")
    if requirements_digest != manifest["requirements_sha256"]:
        return _reject("REQUIREMENTS_DIGEST_MISMATCH")

    components = manifest["components"]
    names: set[str] = set()
    for component in components:
        if not _component_valid(component):
            return _reject("MALFORMED_COMPONENT")
        name = component["name"]
        if name in names:
            return _reject("DUPLICATE_COMPONENT")
        names.add(name)

    if type(observed_artifacts) is not dict:
        return _reject("MALFORMED_OBSERVATION")
    if len(observed_artifacts) > _MAX_COMPONENTS:
        return _reject("UNDECLARED_OBSERVED_ARTIFACT")
    for name, payload in observed_artifacts.items():
        if type(name) is not str or type(payload) is not bytes:
            return _reject("MALFORMED_OBSERVATION")
        if name not in names:
            return _reject("UNDECLARED_OBSERVED_ARTIFACT")
        if len(payload) > _MAX_OBSERVED_BYTES:
            return _reject("OVERSIZED_OBSERVATION")
    if len(observed_artifacts) != len(components):
        return _reject("MISSING_OBSERVED_ARTIFACT")

    saw_archive = False
    for c in components:
        payload = observed_artifacts[c["name"]]
        if len(payload) != c["declared_bytes"] or hashlib.sha256(payload).hexdigest() != c["artifact_sha256"]:
            return _reject("ARTIFACT_HASH_MISMATCH")
        if c["role"] == "archive":
            saw_archive = True
            if not _safe_archive_metadata(payload):
                return _reject("ARCHIVE_METADATA_REJECTED")
    return SupplyChainAssessment(
        reason="STATIC_BYTES_MATCH_ONLY",
        static_receipt_consistent=True,
        artifact_bytes_matched=True,
        direct_requirement_versions_pinned=True,
        archive_metadata_safe=True if saw_archive else False,
        dependency_wheel_hashes_verified=False,
        reproducible_execution_verified=False,
        licenses_authenticated=False,
        binary_execution_authorized=False,
        model_release_authorized=False,
    )
