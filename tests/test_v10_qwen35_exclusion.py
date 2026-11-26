from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from successor.experiments.v10_exclusion import (
    fingerprint_jsonl,
    normalize_prompt,
    reject_excluded_prompts,
)


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_normalize_prompt_is_nfkc_casefolded_whitespace_collapsed() -> None:
    assert normalize_prompt("  FOO\n\tＢａｒ  ") == "foo bar"


def test_fingerprint_jsonl_emits_hashes_without_plaintext(tmp_path: Path) -> None:
    path = tmp_path / "suite.jsonl"
    _jsonl(path, [{"prompt": "Alpha  Beta"}, {"prompt": "Gamma"}])
    receipt = fingerprint_jsonl(path, suite_id="suite-a")
    assert receipt["rows"] == 2
    assert receipt["unique_normalized_prompts"] == 2
    assert receipt["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    serialized = json.dumps(receipt)
    assert "Alpha" not in serialized
    assert "Gamma" not in serialized
    assert all(len(x) == 64 for x in receipt["normalized_prompt_sha256"])


def test_fingerprint_jsonl_rejects_missing_prompt(tmp_path: Path) -> None:
    path = tmp_path / "suite.jsonl"
    _jsonl(path, [{"record_id": "x"}])
    with pytest.raises(ValueError, match="prompt"):
        fingerprint_jsonl(path, suite_id="suite-a")


def test_reject_excluded_prompts_reports_normalized_overlap(tmp_path: Path) -> None:
    path = tmp_path / "suite.jsonl"
    _jsonl(path, [{"prompt": "Do the current check."}])
    receipt = fingerprint_jsonl(path, suite_id="suite-a")
    result = reject_excluded_prompts(
        ["  DO the current\ncheck.  ", "Entirely new prompt"],
        [receipt],
    )
    assert result["status"] == "REJECT"
    assert result["overlap_count"] == 1
    assert result["overlap_indices"] == [0]
    assert "Do the current check." not in json.dumps(result)


def test_reject_excluded_prompts_passes_disjoint_input(tmp_path: Path) -> None:
    path = tmp_path / "suite.jsonl"
    _jsonl(path, [{"prompt": "Old final prompt"}])
    receipt = fingerprint_jsonl(path, suite_id="suite-a")
    result = reject_excluded_prompts(["Fresh prompt"], [receipt])
    assert result == {
        "schema": "V10_QWEN35_EXCLUSION_CHECK_V1",
        "status": "PASS",
        "candidate_count": 1,
        "overlap_count": 0,
        "overlap_indices": [],
    }
