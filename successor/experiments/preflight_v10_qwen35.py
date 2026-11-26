from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import median


BASE_REPO = "rodrigomt/Qwen3.5-4B-Uncensored-Aggressive"
BASE_REVISION = "d61dd146c8fd44c9a49cdb7f59f34e17b61902d8"
BASE_SHARDS = {
    "model.safetensors-00001-of-00002.safetensors": "36dd0f7de62c546e7d163c2a4461208d731fad91ebea9bc31226b54c1c86511f",
    "model.safetensors-00002-of-00002.safetensors": "e2fb50b735a8e10210eece763097f906955ca9db6ba557215f549a46b622157f",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_sha256(path: Path, expected: str) -> str:
    actual = _sha256_bytes(path.read_bytes())
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {path}: {actual} != {expected}")
    return actual


def _generation_prefix(tokenizer, prompt: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )


def _token_count(tokenizer, prompt: str, response: str) -> int:
    text = _generation_prefix(tokenizer, prompt) + response + (tokenizer.eos_token or "")
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _percentile(sorted_values: list[int], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def token_budget_report(tokenizer, rows: list[dict], *, max_length: int) -> dict:
    lengths = []
    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        response = row.get("response")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompt missing at row {index}")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"response missing at row {index}")
        lengths.append(_token_count(tokenizer, prompt, response))
    ordered = sorted(lengths)
    return {
        "rows": len(lengths),
        "max_length": max_length,
        "min_tokens": ordered[0] if ordered else 0,
        "median_tokens": median(ordered) if ordered else 0,
        "p95_tokens": _percentile(ordered, 0.95),
        "p99_tokens": _percentile(ordered, 0.99),
        "max_tokens": ordered[-1] if ordered else 0,
        "over_budget": sum(length > max_length for length in ordered),
    }


def token_budget(tokenizer, rows: list[dict], *, max_length: int) -> dict:
    report = token_budget_report(tokenizer, rows, max_length=max_length)
    if report["over_budget"]:
        raise RuntimeError("token budget exceeded: " + json.dumps(report, sort_keys=True))
    return report


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_base_dir(base_dir: Path) -> dict:
    for filename, expected in BASE_SHARDS.items():
        verify_sha256(base_dir / filename, expected)
    tokenizer_path = base_dir / "tokenizer.json"
    return {
        "repo": BASE_REPO,
        "revision": BASE_REVISION,
        "shards": dict(BASE_SHARDS),
        "tokenizer_sha256": _sha256_bytes(tokenizer_path.read_bytes()),
    }


def runtime_versions() -> dict:
    names = ("torch", "transformers", "trl", "peft")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        import torch
        versions["cuda_available"] = bool(torch.cuda.is_available())
        versions["cuda_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        versions["cuda_runtime"] = torch.version.cuda
    except Exception:
        versions["cuda_available"] = False
        versions["cuda_name"] = None
        versions["cuda_runtime"] = None
    return versions


def run_preflight(
    *,
    train_path: Path,
    validation_path: Path,
    base_dir: Path,
    expected_train_sha256: str,
    expected_validation_sha256: str,
    max_length: int = 512,
) -> dict:
    from transformers import AutoTokenizer

    train_sha = verify_sha256(train_path, expected_train_sha256)
    validation_sha = verify_sha256(validation_path, expected_validation_sha256)
    base = verify_base_dir(base_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train = load_jsonl(train_path)
    validation = load_jsonl(validation_path)
    train_report = token_budget_report(tokenizer, train, max_length=max_length)
    validation_report = token_budget_report(tokenizer, validation, max_length=max_length)
    overflow = train_report["over_budget"] + validation_report["over_budget"]
    return {
        "schema": "V10_QWEN35_TOKEN_RUNTIME_PREFLIGHT_V1",
        "status": "PASS" if overflow == 0 else "HOLD_OVER_BUDGET",
        "effect": "READ_ONLY_NO_WEIGHT_CHANGE",
        "base": base,
        "train_sha256": train_sha,
        "validation_sha256": validation_sha,
        "train_token_budget": train_report,
        "validation_token_budget": validation_report,
        "runtime": runtime_versions(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-validation-sha256", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = run_preflight(
            train_path=args.train,
            validation_path=args.validation,
            base_dir=args.base_dir,
            expected_train_sha256=args.expected_train_sha256,
            expected_validation_sha256=args.expected_validation_sha256,
            max_length=args.max_length,
        )
        code = 0 if receipt["status"] == "PASS" else 2
    except Exception as exc:
        receipt = {
            "schema": "V10_QWEN35_TOKEN_RUNTIME_PREFLIGHT_V1",
            "status": "HOLD",
            "effect": "READ_ONLY_NO_WEIGHT_CHANGE",
            "reason": str(exc),
        }
        code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
