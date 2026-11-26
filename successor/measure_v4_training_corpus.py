from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

from successor import build_v4_training_corpus as corpus
from successor.build_v4_hf_rehearsal import TOKENIZER_REPO, TOKENIZER_REVISION, SYSTEM


def render_lengths(tokenizer, row: dict) -> tuple[int, int]:
    prompt_text = tokenizer.apply_chat_template(
        [SYSTEM, {"role": "user", "content": row["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        [SYSTEM, {"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["response"]}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    full_len = len(tokenizer(full_text, add_special_tokens=False)["input_ids"])
    if prompt_len >= full_len:
        raise RuntimeError(f"no assistant supervision: {row['record_id']}")
    return full_len, full_len - prompt_len


def summary(values: list[int]) -> dict:
    ordered = sorted(values)
    def pct(q: float) -> int:
        return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))]
    return {
        "count": len(values),
        "total": sum(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": max(values),
    }


def measure(path: Path, tokenizer) -> dict:
    total_lengths = []
    assistant_lengths = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        total, assistant = render_lengths(tokenizer, row)
        total_lengths.append(total)
        assistant_lengths.append(assistant)
    return {
        "rendered_tokens": summary(total_lengths),
        "assistant_supervised_tokens": summary(assistant_lengths),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO, revision=TOKENIZER_REVISION)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        train = td / "train.jsonl"
        validation = td / "validation.jsonl"
        manifest_path = td / "manifest.json"
        manifest = corpus.build(train, validation, manifest_path)
        result = {
            "schema": "VERA_V4_TOKEN_VOLUME_RECEIPT_V1",
            "corpus_manifest_sha256": manifest["manifest_sha256"],
            "train_sha256": manifest["train_sha256"],
            "validation_sha256": manifest["validation_sha256"],
            "tokenizer_repo": TOKENIZER_REPO,
            "tokenizer_revision": TOKENIZER_REVISION,
            "train": measure(train, tokenizer),
            "validation": measure(validation, tokenizer),
        }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
