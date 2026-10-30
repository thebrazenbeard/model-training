"""Bind an already downloaded official Qwen3.5 snapshot to exact Hub LFS bytes.

This does not download or change any model files. The output is a new,
content-addressed custody manifest for the neural runner, not model training.
"""

import argparse
import json
import re
from pathlib import Path

from trainingday.admission import AdmissionError, _hash_file, _require
from trainingday.train_neural import BASE_ID


def freeze(snapshot: Path, revision: str, destination: Path, *, api) -> dict:
    snapshot, destination = Path(snapshot), Path(destination)
    _require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "base revision unpinned")
    _require(snapshot.is_absolute() and snapshot.is_dir() and not snapshot.is_symlink(),
             "snapshot must be an existing isolated directory")
    _require(not destination.exists(), "base binding output already exists")
    _require(destination.parent.is_dir() and not destination.parent.is_symlink(),
             "base binding parent absent/symlinked")
    info = api.model_info(BASE_ID, revision=revision, files_metadata=True)
    _require(info.sha == revision, "Hub resolved unexpected model revision")
    hub_tensors = {item.rfilename: item for item in info.siblings
                   if item.rfilename.endswith(".safetensors")}
    local_tensors = {file.name for file in snapshot.glob("*.safetensors")}
    _require(bool(hub_tensors) and set(hub_tensors) == local_tensors,
             "missing/extra downloaded model tensor shards")
    files = {}
    for file in sorted(snapshot.iterdir()):
        if file.name.startswith("."):
            continue  # Hugging Face local download metadata, not model assets.
        _require(file.is_file() and not file.is_symlink(),
                 "unexpected nested/symlinked snapshot asset")
        digest = _hash_file(file)
        if file.name in hub_tensors:
            target = hub_tensors[file.name]
            _require(target.lfs is not None and target.lfs.sha256 == digest
                     and target.size == file.stat().st_size,
                     "downloaded tensor differs from published Hub LFS SHA/size")
        files[file.name] = digest
    _require("config.json" in files and "tokenizer_config.json" in files,
             "required tokenizer/config metadata absent")
    record = {"schema": "TRAININGDAY_PINNED_BASE_V1", "model_id": BASE_ID,
              "revision": revision, "loader_class": "Qwen3_5ForCausalLM",
              "quantization": "NF4_DOUBLE_BF16", "snapshot_path": str(snapshot),
              "files": files, "hub_lfs_verified": True}
    with destination.open("x", encoding="utf-8") as writer:
        json.dump(record, writer, sort_keys=True, indent=2)
        writer.write("\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import HfApi
    result = freeze(args.snapshot, args.revision, args.destination, api=HfApi())
    print(json.dumps({"status": "PINNED_BASE_BINDING_CREATED_NOT_TRAINED",
                      "revision": result["revision"],
                      "sha256": _hash_file(args.destination),
                      "files": len(result["files"])}))


if __name__ == "__main__":
    main()
