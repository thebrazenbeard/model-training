from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path


def normalize_prompt(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be nonempty text")
    normalized = unicodedata.normalize("NFKC", prompt).casefold()
    return " ".join(normalized.split())


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fingerprint_jsonl(path: Path, *, suite_id: str) -> dict:
    raw = path.read_bytes()
    rows = []
    hashes = []
    exact_hashes = []
    for lineno, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompt missing at line {lineno}")
        rows.append(row)
        exact_hashes.append(_sha256_text(prompt.strip()))
        hashes.append(_sha256_text(normalize_prompt(prompt)))
    return {
        "schema": "V10_QWEN35_PROMPT_FINGERPRINT_SUITE_V1",
        "suite_id": suite_id,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "rows": len(rows),
        "unique_exact_prompts": len(set(exact_hashes)),
        "unique_normalized_prompts": len(set(hashes)),
        "exact_prompt_sha256": sorted(set(exact_hashes)),
        "normalized_prompt_sha256": sorted(set(hashes)),
    }


def reject_excluded_prompts(candidate_prompts: list[str], exclusion_receipts: list[dict]) -> dict:
    excluded = set()
    for receipt in exclusion_receipts:
        excluded.update(receipt.get("normalized_prompt_sha256", []))

    overlap_indices = []
    for index, prompt in enumerate(candidate_prompts):
        digest = _sha256_text(normalize_prompt(prompt))
        if digest in excluded:
            overlap_indices.append(index)

    return {
        "schema": "V10_QWEN35_EXCLUSION_CHECK_V1",
        "status": "REJECT" if overlap_indices else "PASS",
        "candidate_count": len(candidate_prompts),
        "overlap_count": len(overlap_indices),
        "overlap_indices": overlap_indices,
    }
