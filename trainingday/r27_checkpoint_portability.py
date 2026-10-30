"""Bounded public-toy safetensors portability probes (TrainingDay R27-01).

No torch, pickle, model weights, private corpus, host credentials, or optimizer
resumption. The standard-library reader handles a strict F32 subset of the
safetensors layout for *synthetic* test fixtures. A matching declaration from
three machines is not a host-authenticated readback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import unicodedata


class PortabilityError(ValueError):
    """Public portability input is unsafe or malformed."""


MAX_FILE_BYTES = 1024 * 1024
MAX_HEADER_BYTES = 128 * 1024
MAX_TENSORS = 32
MAX_ELEMENTS = 65536
MAX_PATHS = 32
EXPECTED_HOSTS = frozenset(("Lappy", "VERA-NODE-1", "VN2"))
SHA = re.compile(r"[0-9a-f]{64}\Z")
TENSOR_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,100}\Z")
WINDOWS_DEVICES = frozenset(("CON", "PRN", "AUX", "NUL")) | frozenset(
    f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)
) | frozenset(
    # Win32 also reserves COM/LPT followed by superscript one, two or three.
    f"{prefix}{digit}" for prefix in ("COM", "LPT") for digit in "¹²³"
)


def _bounded_shape(shape: object) -> tuple[int, ...]:
    if type(shape) not in (list, tuple) or not 1 <= len(shape) <= 4:
        raise PortabilityError("tensor shape must be a 1..4 dimensional sequence")
    result = []
    elements = 1
    for dimension in shape:
        if type(dimension) is not int or not 1 <= dimension <= MAX_ELEMENTS:
            raise PortabilityError("tensor shape component invalid or excessive")
        elements *= dimension
        if elements > MAX_ELEMENTS:
            raise PortabilityError("tensor exceeds public element cap")
        result.append(dimension)
    return tuple(result)


def _name(name: object) -> str:
    if type(name) is not str or not TENSOR_NAME.fullmatch(name):
        raise PortabilityError("invalid public tensor name")
    return name


def _f32(value: object) -> bytes:
    if type(value) not in (float, int) or not -3.4028234e38 <= value <= 3.4028234e38:
        raise PortabilityError("nonfinite or out-of-bounds public float")
    try:
        encoded = struct.pack("<f", float(value))
        if not math.isfinite(struct.unpack("<f", encoded)[0]):
            raise PortabilityError("nonfinite encoded public float")
        return encoded
    except (OverflowError, ValueError, struct.error) as exc:
        raise PortabilityError("malformed public float") from exc


def build_public_f32_safetensors(tensors: dict[str, tuple[object, object]]) -> bytes:
    """Create deterministic little-endian .safetensors bytes from public F32s."""
    if type(tensors) is not dict or not 1 <= len(tensors) <= MAX_TENSORS:
        raise PortabilityError("unbounded or empty public tensor map")
    header = {}
    chunks = []
    offset = 0
    for name in sorted(tensors):
        _name(name)
        row = tensors[name]
        if type(row) not in (list, tuple) or len(row) != 2:
            raise PortabilityError("tensor declaration requires shape and values")
        shape = _bounded_shape(row[0])
        values = row[1]
        if type(values) not in (list, tuple) or math.prod(shape) != len(values):
            raise PortabilityError("tensor shape/value count mismatch")
        encoded = b"".join(_f32(value) for value in values)
        if len(encoded) != math.prod(shape) * 4:
            raise PortabilityError("tensor F32 encoded byte count differs from shape")
        header[name] = {"dtype": "F32", "shape": list(shape),
                        "data_offsets": [offset, offset + len(encoded)]}
        chunks.append(encoded)
        offset += len(encoded)
        if offset > MAX_FILE_BYTES:
            raise PortabilityError("unbounded public tensor bytes")
    head = json.dumps(header, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(head) > MAX_HEADER_BYTES or 8 + len(head) + offset > MAX_FILE_BYTES:
        raise PortabilityError("public safetensors file exceeds cap")
    return struct.pack("<Q", len(head)) + head + b"".join(chunks)


def _json_no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    record = {}
    for key, value in pairs:
        if key in record:
            raise PortabilityError("duplicate safetensors header key")
        record[key] = value
    return record


def inspect_public_f32_safetensors(
        payload: bytes, *, expected_sha256: str | None = None) -> dict:
    """Validate the strict public F32 safetensors subset without model loading."""
    if type(payload) is not bytes or not 10 <= len(payload) <= MAX_FILE_BYTES:
        raise PortabilityError("public safetensors input length invalid")
    sha = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and (type(expected_sha256) is not str
            or not SHA.fullmatch(expected_sha256) or expected_sha256 != sha):
        raise PortabilityError("public checkpoint SHA mismatch")
    header_length = struct.unpack("<Q", payload[:8])[0]
    if not 2 <= header_length <= MAX_HEADER_BYTES or 8 + header_length > len(payload):
        raise PortabilityError("invalid little-endian safetensors header length")
    try:
        header = json.loads(payload[8:8 + header_length].decode("utf-8"),
                            object_pairs_hook=_json_no_duplicate_keys,
                            parse_constant=lambda _: (_ for _ in ()).throw(
                                PortabilityError("nonfinite safetensors header")))
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise PortabilityError("invalid safetensors header JSON") from exc
    if type(header) is not dict or not 1 <= len(header) <= MAX_TENSORS:
        raise PortabilityError("safetensors public header is empty or too large")
    body = payload[8 + header_length:]
    rows = []
    offset = 0
    for name, row in sorted(header.items()):
        _name(name)
        if type(row) is not dict or set(row) != {"dtype", "shape", "data_offsets"}:
            raise PortabilityError("invalid public F32 tensor schema")
        if row["dtype"] != "F32":
            raise PortabilityError("unsupported dtype; only public F32 supported")
        shape = _bounded_shape(row["shape"])
        offsets = row["data_offsets"]
        if (type(offsets) is not list or len(offsets) != 2
                or any(type(item) is not int or item < 0 for item in offsets)
                or offsets[1] - offsets[0] != math.prod(shape) * 4):
            raise PortabilityError("tensor offsets do not match F32 shape")
        rows.append((offsets[0], offsets[1], name, shape))
    for start, end, name, shape in sorted(rows):
        if start != offset or end > len(body):
            raise PortabilityError("overlap, gap, or truncated public tensor")
        for (value,) in struct.iter_unpack("<f", body[start:end]):
            if not math.isfinite(value):
                raise PortabilityError("nonfinite public F32 tensor value")
        offset = end
    if offset != len(body):
        raise PortabilityError("trailing unclaimed safetensors bytes")
    return {"format": "SAFETENSORS_F32_PUBLIC_SUBSET", "tensor_count": len(rows),
            "tensor_shapes": {name: list(shape) for _, _, name, shape in rows},
            "bytes_sha256": sha, "byte_count": len(payload),
            "actual_cross_host_verified": False, "optimizer_resume_authorized": False,
            "qwen4b_release_authorized": False}


def normalize_artifact_inventory(paths: object) -> tuple[str, ...]:
    """Reject aliases/traversal that resolve differently on Linux vs Windows."""
    if type(paths) not in (list, tuple) or not 1 <= len(paths) <= MAX_PATHS:
        raise PortabilityError("unbounded or empty virtual path inventory")
    canonical = []
    folded = set()
    for path in paths:
        if type(path) is not str or not path or len(path) > 240:
            raise PortabilityError("invalid virtual relative path")
        if path.startswith(("/", "\\")) or ":" in path:
            raise PortabilityError("absolute, UNC, or Windows ADS path forbidden")
        parts = path.replace("\\", "/").split("/")
        if any(not part or part in (".", "..") or part.rstrip(". ") != part
               or any(ord(ch) < 32 or ord(ch) == 127 for ch in part)
               or any(ch in '<>"|?*' for ch in part)
               or part.split(".", 1)[0].upper() in WINDOWS_DEVICES
               or unicodedata.normalize("NFC", part) != part
               or part.casefold() in {"optimizer.pt", "scheduler.pt", "rng_state.pth", "random_states.pkl"}
               for part in parts):
            raise PortabilityError("cross-OS ambiguous or escaping path")
        name = "/".join(parts)
        try:
            name.encode("utf-8", "strict")
        except UnicodeEncodeError as exc:
            raise PortabilityError("invalid Unicode path") from exc
        if name.casefold() in folded:
            raise PortabilityError("casefold-colliding virtual paths")
        folded.add(name.casefold())
        canonical.append(name)
    return tuple(sorted(canonical))


def assess_declared_cross_host_readbacks(payload: bytes, receipts: object) -> dict:
    """Compare *declarations*, never turn them into authenticated host proof."""
    info = inspect_public_f32_safetensors(payload)
    if type(receipts) not in (list, tuple) or len(receipts) != len(EXPECTED_HOSTS):
        raise PortabilityError("exactly three bounded declarations required")
    observed = set()
    consistent = True
    for item in receipts:
        if (type(item) is not dict or set(item) != {"host", "sha256"}
                or type(item["host"]) is not str or item["host"] not in EXPECTED_HOSTS
                or item["host"] in observed or type(item["sha256"]) is not str
                or not SHA.fullmatch(item["sha256"])):
            raise PortabilityError("invalid or duplicate host declaration")
        observed.add(item["host"])
        consistent &= item["sha256"] == info["bytes_sha256"]
    if observed != EXPECTED_HOSTS:
        raise PortabilityError("missing declared host")
    return {"source_status": "DECLARED_DIGEST_MATCH_ONLY" if consistent
                             else "DECLARED_DIGEST_MISMATCH",
            "declared_digest_consistent": bool(consistent),
            "bytes_sha256": info["bytes_sha256"],
            "actual_cross_host_verified": False,
            "optimizer_resume_authorized": False,
            "qwen4b_release_authorized": False,
            "unsupported_optimizer_and_rng_resume": True}


def read_bounded_local_file(path: Path) -> bytes:
    """Read only one local synthetic regular file, declining symlinks."""
    if type(path) is not type(Path(".")):
        raise PortabilityError("native local Path required")
    try:
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode) or path.is_symlink() or path.stat().st_size > MAX_FILE_BYTES:
            raise PortabilityError("not a bounded regular public file")
        with path.open("rb") as handle:
            data = handle.read(MAX_FILE_BYTES + 1)
    except OSError as exc:
        raise PortabilityError("public file unavailable") from exc
    if len(data) > MAX_FILE_BYTES:
        raise PortabilityError("public file too large")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only public synthetic F32 safetensors probe")
    parser.add_argument("--read-public-file", type=Path, required=True)
    args = parser.parse_args(argv)
    info = inspect_public_f32_safetensors(read_bounded_local_file(args.read_public_file))
    print(json.dumps(info, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
