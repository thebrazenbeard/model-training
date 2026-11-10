"""Read-only local tokenizer preflight; emits aggregate counts, never prompts.

No model weights, network, installs, or training. Explicit file paths only.
The caller must supply the exact already-installed snapshot and public-case JSON.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


BUDGETS = (64, 96, 128, 256)
EXPECTED_CORPUS_SHA256 = "4fbe54d97504628aa88ac7be37d994ed5f39be8d2a73db99f4cf54684989f31e"
EXPECTED_SOURCE_RECEIPT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPECTED_TOKENIZER_FILES = {
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bins(values: list[int]) -> dict:
    v = sorted(values)
    return {
        "count": len(v), "min": v[0], "median": statistics.median(v),
        "max": v[-1],
        "within_budget": {str(b): sum(n <= b for n in v) for b in BUDGETS},
        "over_budget": {str(b): sum(n > b for n in v) for b in BUDGETS},
    }


def _token_count(encoded: object) -> int:
    """Transformers versions may return BatchEncoding, not a token ID list."""
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if not isinstance(ids, list) or not ids:
        raise ValueError("unexpected tokenization response")
    if isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("unexpected batched tokenizer response")
        ids = ids[0]
    if not all(type(t) is int for t in ids):
        raise ValueError("token IDs are not integers")
    return len(ids)


def _verify_public_candidate(rows: list[dict]) -> None:
    if len(rows) != 80 or Counter(row["split"] for row in rows) != Counter(
            {"train": 64, "dev": 16}):
        raise ValueError("not the expected 64/16 source-candidate set")
    if any(row.get("training_authorized") is not False or
           row.get("heldout") is not False for row in rows):
        raise ValueError("not a held rights-only TRAIN/DEV candidate")
    encoded = (json.dumps(rows, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")) + "\n").encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != EXPECTED_CORPUS_SHA256:
        raise ValueError("candidate content hash differs from frozen W51 source")


def _verify_tokenizer_snapshot(path: Path) -> dict[str, str]:
    if not path.is_dir() or any(not (path / name).is_file()
                                for name in EXPECTED_TOKENIZER_FILES):
        raise ValueError("explicit tokenizer snapshot path incomplete")
    hashes = {name: _digest(path / name) for name in EXPECTED_TOKENIZER_FILES}
    if hashes != EXPECTED_TOKENIZER_FILES:
        raise ValueError("tokenizer file bytes differ from pinned source receipt")
    receipt_path = path / "TRAININGDAY_PUBLIC_SOURCE_RECEIPT.json"
    if not receipt_path.is_file():
        raise ValueError("source receipt missing from this snapshot")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("repo") != "Qwen/Qwen3.5-4B" or
            receipt.get("revision") != EXPECTED_SOURCE_RECEIPT_REVISION):
        raise ValueError("source receipt model/revision mismatch")
    recorded = {r.get("file"): r.get("sha256") for r in receipt.get("files", [])}
    if any(recorded.get(name) != digest for name, digest in hashes.items()):
        raise ValueError("source receipt tokenizer digest mismatch")
    return hashes


def probe(rows: list[dict], snapshot: str, system_text: str | None = None,
          *, thinking: str = "default") -> dict:
    _verify_public_candidate(rows)
    path = Path(snapshot)
    file_hashes = _verify_tokenizer_snapshot(path)
    import tokenizers
    import transformers
    from transformers import AutoTokenizer

    if thinking not in ("default", "off"):
        raise ValueError("unsupported chat-template mode")

    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True,
                                        trust_remote_code=False)
    lengths = {s: {"prompt": [], "full_sequence": []} for s in ("train", "dev")}
    keyword = {} if thinking == "default" else {"enable_thinking": False}
    for row in rows:
        messages = list(row["messages"])
        if system_text is not None:
            messages.insert(0, {"role": "system", "content": system_text})
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("last message must be assistant completion")
        prompt = tok.apply_chat_template(messages[:-1], tokenize=True,
                                         add_generation_prompt=True, **keyword)
        full = tok.apply_chat_template(messages, tokenize=True,
                                       add_generation_prompt=False, **keyword)
        lengths[row["split"]]["prompt"].append(_token_count(prompt))
        lengths[row["split"]]["full_sequence"].append(_token_count(full))

    return {
        "status": "MEASURED_PINNED_PUBLIC_CASES_ONLY" if system_text is None else
                  "MEASURED_CALLER_SUPPLIED_SYSTEM_TEXT",
        "caller_system_text_included": system_text is not None,
        "full_private_runtime_prefix_version_verified": False,
        "scope": "only caller-supplied messages and optional system text",
        "snapshot_name": path.name,
        "receipt_claimed_revision": EXPECTED_SOURCE_RECEIPT_REVISION,
        "candidate_sha256": EXPECTED_CORPUS_SHA256,
        "tokenizer_class": type(tok).__name__,
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "thinking_mode": thinking,
        "file_sha256": file_hashes,
        "splits": {split: {kind: _bins(lengths[split][kind]) for kind in
                            ("prompt", "full_sequence")} for split in
                   ("train", "dev")},
        "no_weights_loaded": True,
        "no_training_or_rights_admission": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="exact existing local tokenizer snapshot")
    parser.add_argument("--system-file", help="optional private rendered system prefix file; never printed")
    parser.add_argument("--thinking", choices=("default", "off"), default="default")
    args = parser.parse_args()
    from trainingday.dialogue_curriculum import candidate_rows
    rows = candidate_rows()
    try:
        system = (Path(args.system_file).read_text(encoding="utf-8")
                  if args.system_file else None)
        print(json.dumps(probe(rows, args.snapshot, system, thinking=args.thinking),
                         indent=2, sort_keys=True))
    except Exception as exc:
        # Never print raw prompts, private system text or tokenizer tracebacks.
        print(json.dumps({"status": "NOT_MEASURED", "error_type":
                          type(exc).__name__}))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
