"""Read-only pinned Git-tree inventory: alternative to copying donor executables.

Reports metadata about EXACT repository tree objects. Never checks out, opens,
imports, executes, packages, or publishes file contents. An OID is content
identity, not provenance/authorship certification or scan approval.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json
import re
import subprocess

_SHA=re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")
_DIR=re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MODES={"100644":"REGULAR_NO_COPY","100755":"EXECUTABLE_NO_COPY",
        "120000":"SYMLINK_NO_COPY","160000":"SUBMODULE_NO_COPY"}
_MAX_ITEMS=30000


class SourceTreeError(ValueError):
    pass


def _run(repo: Path, *args: str) -> bytes:
    p=subprocess.run(["git","-C",str(repo),*args],capture_output=True,
                     timeout=12,shell=False,check=False)
    if p.returncode!=0 or len(p.stdout)>8_000_000:
        raise SourceTreeError("Unable to verify exact local Git tree")
    return p.stdout


def _decode_tree(output: bytes):
    if type(output) is not bytes or len(output)>8_000_000:
        raise SourceTreeError("Invalid tree byte stream")
    if output and not output.endswith(b"\0"):
        raise SourceTreeError("Truncated Git -z tree record")
    rows=[]
    for line in output.split(b"\0"):
        if not line:
            continue
        if len(rows)>=_MAX_ITEMS:
            raise SourceTreeError("Inventory entry limit exceeded")
        if b"\t" not in line:
            raise SourceTreeError("Malformed Git tree entry")
        header,path_bytes=line.split(b"\t",1)
        bits=header.split(b" ")
        if len(bits)!=3:
            raise SourceTreeError("Malformed Git tree header")
        try:
            mode,kind,oid=(x.decode("ascii") for x in bits)
            path=path_bytes.decode("utf8","strict")
        except UnicodeError as exc:
            raise SourceTreeError("Git tree contains unsupported path encoding") from exc
        if mode not in _MODES or kind not in ("blob","commit"):
            raise SourceTreeError("Unsupported tree file mode")
        if not _SHA.fullmatch(oid):
            raise SourceTreeError("Malformed Git OID")
        parts=path.split("/")
        if any(not p or p in (".","..") or "\\" in p or
               not p.isprintable() for p in parts):
            raise SourceTreeError("Unsafe repository inventory path")
        rows.append({"path":path,"oid":oid,"mode":mode,"kind":kind,
                     "handling":_MODES[mode]})
    return rows


def source_tree_manifest(repo: str | Path, *, commit: str,
                         expected_folders: tuple[str,...]) -> dict:
    if type(commit) is not str or not _SHA.fullmatch(commit):
        raise SourceTreeError("Exact full immutable commit ID required")
    if (type(expected_folders) is not tuple or not expected_folders
            or len(expected_folders)>128 or
            len(set(expected_folders))!=len(expected_folders) or
            any(type(x) is not str or not _DIR.fullmatch(x) for x in expected_folders)):
        raise SourceTreeError("Bounded unique folder inventory required")
    repo=Path(repo)
    if not repo.is_dir():
        raise SourceTreeError("Local Git repo directory required")
    readback=_run(repo,"rev-parse","--verify",commit+"^{commit}").decode("ascii").strip()
    if readback!=commit:
        raise SourceTreeError("Pinned source commit mismatch")
    tree=_decode_tree(_run(repo,"ls-tree","-r","--full-tree","-z",commit))
    wanted=set(expected_folders)
    selected=[row for row in tree if row["path"].split("/",1)[0] in wanted]
    present={row["path"].split("/",1)[0] for row in selected}
    report={
        "schema":"VERA_PINNED_SOURCE_TREE_METADATA_V1",
        "source_commit":commit,
        "expected_folders":list(expected_folders),
        "present_folders":sorted(present),
        "missing_folders":sorted(wanted-present),
        "entries":selected,
        "copies_performed":0,
        "content_bytes_published":0,
        "source_files_executed":False,
        "compatibility_qualified":False,
        "source_identity_authenticated":False,
        "warning":"Metadata-only Git tree inventory; no content, installation, publication rights or scan clearance.",
    }
    raw=json.dumps(report,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf8")
    report["manifest_sha256"]=sha256(raw).hexdigest()
    return report
