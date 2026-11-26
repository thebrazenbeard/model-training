from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from successor.experiments.v10_exclusion import fingerprint_jsonl


REPAIRED_HASHES = {
    "V10_QWEN512_TRAIN": "a232732a7db1f22c7edabb984fb16a3fe5350022e0a152576d80877eb9430300",
    "V10_QWEN512_VALIDATION": "ccc57ad20e8dfbc826ce49f064e692e0eab3fa396ed602dfba54ad9e252bd6d7",
}
HISTORICAL = {
    "V3_BEHAVIORAL": "final_holdout_v3.jsonl",
    "V3_RETENTION": "final_retention_v3.jsonl",
    "V3_ADVERSARIAL": "final_adversarial_proxy_v3.jsonl",
    "HISTORY_BEHAVIOR_HOLDOUT_V1": "history_behavior_holdout_v1.jsonl",
    "H07_FINAL_HOLDOUT_V1": "h07_final_holdout_v1.jsonl",
    "V4_BEHAVIORAL": "final_holdout_v4.jsonl",
    "V4_RETENTION": "final_retention_v4.jsonl",
    "V4_ADVERSARIAL": "final_adversarial_proxy_v4.jsonl",
}


def _set_digest(values: list[str]) -> str:
    payload = ("\n".join(sorted(values)) + "\n").encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-train", type=Path, required=True)
    parser.add_argument("--original-validation", type=Path, required=True)
    parser.add_argument("--repaired-train", type=Path, required=True)
    parser.add_argument("--repaired-validation", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--donor-qualification-root", type=Path, required=True)
    parser.add_argument("--exclusion-registry", type=Path, required=True)
    parser.add_argument("--output-hashes", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    registry = json.loads(args.exclusion_registry.read_text(encoding="utf-8"))
    expected = {
        item["id"]: item.get("sha256")
        for group in ("current_v10", "historical_consumed")
        for item in registry.get(group, [])
    }
    expected.update(REPAIRED_HASHES)

    sources = [
        ("V10_TRAIN", args.original_train),
        ("V10_VALIDATION", args.original_validation),
        ("V10_QWEN512_TRAIN", args.repaired_train),
        ("V10_QWEN512_VALIDATION", args.repaired_validation),
        ("V10_REVIEW_SAMPLE", args.review),
    ]
    sources.extend(
        (suite_id, args.donor_qualification_root / filename)
        for suite_id, filename in HISTORICAL.items()
    )

    combined: set[str] = set()
    summaries = []
    for suite_id, path in sources:
        receipt = fingerprint_jsonl(path, suite_id=suite_id)
        expected_sha = expected.get(suite_id)
        if expected_sha and receipt["source_sha256"] != expected_sha:
            raise RuntimeError(
                f"source hash mismatch for {suite_id}: "
                f"{receipt['source_sha256']} != {expected_sha}"
            )
        hashes = receipt["normalized_prompt_sha256"]
        combined.update(hashes)
        summaries.append(
            {
                "suite_id": suite_id,
                "source_sha256": receipt["source_sha256"],
                "rows": receipt["rows"],
                "unique_exact_prompts": receipt["unique_exact_prompts"],
                "unique_normalized_prompts": receipt["unique_normalized_prompts"],
                "normalized_prompt_set_sha256": _set_digest(hashes),
            }
        )

    ordered = sorted(combined)
    payload = ("\n".join(ordered) + "\n").encode("ascii")
    args.output_hashes.parent.mkdir(parents=True, exist_ok=True)
    args.output_hashes.write_bytes(payload)
    manifest = {
        "schema": "V10_QWEN35_PROMPT_FINGERPRINT_REGISTRY_V2",
        "status": "FROZEN_EXCLUSION_FINGERPRINTS",
        "normalization": "UNICODE_NFKC_CASEFOLD_COLLAPSE_WHITESPACE_V1",
        "plaintext_prompts_included": False,
        "source_count": len(summaries),
        "source_rows": sum(item["rows"] for item in summaries),
        "unique_normalized_prompts": len(ordered),
        "hash_file_sha256": hashlib.sha256(payload).hexdigest(),
        "sources": summaries,
        "scope": "ORIGINAL_V10_DEVELOPMENT_PLUS_QWEN512_REPAIRED_MIXTURE_PLUS_CONSUMED_FINALS",
        "claim_ceiling": "NORMALIZED_HASH_SCREEN_INPUT / NOT_SEMANTIC_CONTAMINATION_PROOF",
    }
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
