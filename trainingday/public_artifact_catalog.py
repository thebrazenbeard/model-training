"""Read-only OCD verification catalog for the two pinned public HF inventories.

No download, deletion, model loading, or training authority is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/artifacts/qwen35-4b-public-base-20261102.json"
CACHE = ROOT / "docs/artifacts/public-hf-cache-models-20261102.json"
CATALOG = ROOT / "docs/artifacts/ocd-verify-public-hf-20261102.json"
_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_REV = re.compile(r"[0-9a-f]{40}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*\Z")
_LOCAL_DIR = re.compile(r"--local-dir ABSOLUTE_WORKER_DIRECTORY/([A-Za-z0-9_.-]+)\Z")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate manifest key")
        result[key] = value
    return result


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)


def _validate(repo, revision, filename, size, digest, destination):
    if not (isinstance(repo, str) and _REPO.fullmatch(repo)
            and isinstance(revision, str) and _REV.fullmatch(revision)
            and isinstance(filename, str) and _NAME.fullmatch(filename)
            and filename not in {".", ".."} and not filename.endswith(".")
            and type(size) is int and size > 0
            and isinstance(digest, str) and _SHA.fullmatch(digest)
            and isinstance(destination, str)
            and _NAME.fullmatch(destination) and not destination.endswith(".")):
        raise ValueError("invalid source pin or destination")
    return {
        "url": f"https://huggingface.co/{repo}/resolve/{revision}/{filename}",
        "sha256": digest,
        "size": size,
        "destination_relpath": f"{destination}/{filename}",
    }


def build_catalog(base_path=BASE, cache_path=CACHE):
    """Consolidate two existing source inventories into OCD's verifier schema."""
    base = _read(Path(base_path))
    cache = _read(Path(cache_path))
    if (base.get("schema_version") != 1
            or base.get("kind") != "public_upstream_base_model"
            or cache.get("schema_version") != 1
            or cache.get("kind") != "verified_public_huggingface_weight_cache"):
        raise ValueError("unsupported source manifest schema")
    source = base["source"]
    entries = [
        _validate(source["repository"], source["revision"], item["path"],
                  item["size_bytes"], item["sha256"], "qwen35-base")
        for item in base["files"]
    ]
    cache_entries = []
    for item in cache["models"]:
        match = _LOCAL_DIR.search(item["download_command"])
        if match is None:
            raise ValueError("public cache restore directory is not pinned")
        cache_entries.append(
            _validate(item["model_id"], item["revision"], item["path"],
                      item["size"], item["sha256"], match.group(1))
        )
    if sum(entry["size"] for entry in cache_entries) != cache["total_weight_bytes"]:
        raise ValueError("public cache total does not match entries")
    entries.extend(cache_entries)
    paths = [entry["destination_relpath"].casefold() for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate case-insensitive destination")
    return {"artifacts": entries}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="confirm tracked OCD manifest matches source pins")
    args = parser.parse_args(argv)
    try:
        generated = build_catalog()
        if args.check:
            if generated != _read(CATALOG):
                raise ValueError("tracked OCD verification catalog is stale")
            print(f"PASS: {len(generated['artifacts'])} pinned public weight files")
        else:
            print(json.dumps(generated, indent=2))
    except (KeyError, TypeError, ValueError, OSError, UnicodeError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
