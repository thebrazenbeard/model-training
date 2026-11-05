"""Bounded, read-only GGUF metadata triage; NEVER a weight-lineage verifier.

Derived from ggml's public GGUF v3 serialization specification. No third-party
dependencies, tensor reads, model load, download, or modification. Metadata is
UNTRUSTED and never proves original upstream tensors or adapter compatibility.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import struct
from pathlib import Path


MAX_SCAN_BYTES = 64 * 1024 * 1024
MAX_READ_BYTES = 16 * 1024 * 1024
MAX_ITEMS = 1_000_000
MAX_KV_PAIRS = 20_000
MAX_KEY_BYTES = 1024
MAX_PRINT_STRING = 512
MAX_ARRAY_DEPTH = 3
VALUE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
               7: 1, 10: 8, 11: 8, 12: 8}
# Return *only* these fields, not arbitrary potentially sensitive metadata.
SELECTED_KEYS = frozenset({
    "general.name", "general.basename", "general.finetune",
    "general.architecture", "general.file_type", "general.type",
    "general.source.url", "general.source.huggingface.repository",
    "general.source.huggingface.revision", "general.source.huggingface.commit",
    "general.url", "tokenizer.ggml.model",
})


class _Bounded:
    def __init__(self, f, size: int):
        self.f = f
        self.size = size
        self.bytes_read = 0
        self.items = 0

    def _span(self, n: int):
        if not isinstance(n, int) or n < 0 or self.f.tell() + n > self.size:
            raise ValueError("GGUF metadata runs past file bounds")
        if self.f.tell() + n > MAX_SCAN_BYTES:
            raise ValueError("GGUF metadata exceeds bounded scan window")

    def read(self, n: int) -> bytes:
        self._span(n)
        if n > MAX_READ_BYTES - self.bytes_read:
            raise ValueError("GGUF header exceeds bounded byte-read budget")
        result = self.f.read(n)
        if len(result) != n:
            raise ValueError("GGUF metadata truncated")
        self.bytes_read += n
        return result

    def skip(self, n: int) -> None:
        self._span(n)
        self.f.seek(n, os.SEEK_CUR)

    def unpack(self, kind: str):
        size = struct.calcsize(kind)
        return struct.unpack("<" + kind, self.read(size))[0]

    def string(self, *, keep: bool, max_kept: int = MAX_PRINT_STRING) -> str | None:
        n = self.unpack("Q")
        if keep:
            if n > max_kept:
                raise ValueError("GGUF selected metadata string too long")
            return self.read(n).decode("utf-8", errors="strict")
        self.skip(n)
        return None

    def value(self, typ: int, *, keep: bool, depth: int = 0):
        if typ == 8:  # GGUF_TYPE_STRING
            return self.string(keep=keep)
        if typ == 9:  # GGUF_TYPE_ARRAY
            if depth >= MAX_ARRAY_DEPTH:
                raise ValueError("GGUF metadata has excessive array nesting")
            subtyp = self.unpack("I")
            count = self.unpack("Q")
            if count > MAX_ITEMS - self.items:
                raise ValueError("GGUF metadata array element budget exceeded")
            self.items += count
            if subtyp in VALUE_SIZES:
                self.skip(count * VALUE_SIZES[subtyp])
            elif subtyp in (8, 9):
                for _ in range(count):
                    self.value(subtyp, keep=False, depth=depth + 1)
            else:
                raise ValueError("GGUF metadata array has unknown element type")
            return "<array omitted>" if keep else None
        if typ not in VALUE_SIZES:
            raise ValueError("GGUF metadata has unknown scalar type")
        size = VALUE_SIZES[typ]
        if not keep:
            self.skip(size)
            return None
        raw = self.read(size)
        fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
               7: "B", 10: "Q", 11: "q", 12: "d"}[typ]
        result = struct.unpack("<" + fmt, raw)[0]
        if typ == 7:
            if result not in (0, 1):
                raise ValueError("GGUF metadata boolean is not 0 or 1")
            return bool(result)
        return result


def inspect(path: Path) -> dict:
    """Read only GGUF v3 KV metadata within hard read/scan budgets.

    The GGUF may have been renamed or repackaged. All fields are self-reported.
    A matching title, origin URL or architecture NEVER certifies base tensor
    lineage, tokenizer identity, file integrity or R13 adapter compatibility.
    """
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        with os.fdopen(fd, "rb") as f:
            fd = -1
            st = os.fstat(f.fileno())
            if not stat.S_ISREG(st.st_mode):
                raise ValueError("GGUF must be a regular file")
            reader = _Bounded(f, st.st_size)
            if reader.read(4) != b"GGUF":
                raise ValueError("not GGUF magic")
            version = reader.unpack("I")
            if version != 3:
                raise ValueError("only GGUF version 3 metadata is supported")
            tensor_count = reader.unpack("Q")
            kv_count = reader.unpack("Q")
            if kv_count > MAX_KV_PAIRS:
                raise ValueError("too many GGUF metadata KV entries")
            if tensor_count > 10_000_000:
                raise ValueError("implausible GGUF tensor count")
            selected = {}
            seen_keys = set()
            for _ in range(kv_count):
                key = reader.string(keep=True, max_kept=MAX_KEY_BYTES)
                if key in seen_keys:
                    raise ValueError("duplicate GGUF metadata key")
                seen_keys.add(key)
                typ = reader.unpack("I")
                value = reader.value(typ, keep=key in SELECTED_KEYS)
                if key in SELECTED_KEYS:
                    selected[key] = value
            return {
                "evidence_class": "UNTRUSTED_GGUF_HEADER_METADATA_ONLY",
                "gguf_version": version,
                "file_bytes": st.st_size,
                "tensor_count_declared": tensor_count,
                "metadata_pair_count": kv_count,
                "metadata_scan_bytes": f.tell(),
                "actual_header_read_bytes": reader.bytes_read,
                "selected_metadata": selected,
                "upstream_tensor_ancestry": "UNKNOWN_NOT_MEASURED",
                "adapter_compatibility": "NOT_ESTABLISHED",
            }
    finally:
        if fd != -1:
            os.close(fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path, help="existing local GGUF; read-only")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(inspect(args.gguf), indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, UnicodeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
