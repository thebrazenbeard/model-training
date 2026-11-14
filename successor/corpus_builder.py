from __future__ import annotations

import hashlib
import json
from pathlib import Path


def fingerprint(prompt: str, response: str) -> str:
    payload = f"{prompt.strip()}\0{response.strip()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _eligible(record: dict) -> bool:
    uses = set(record.get("training_use") or [])
    return (
        record.get("promotion_status") == "reviewed_candidate"
        and bool({"sft_candidate", "patrick_specific_sft_candidate"} & uses)
        and isinstance(record.get("trigger"), str)
        and isinstance(record.get("accepted_target"), str)
    )


def build_examples(corpus_paths, anchors, holdout_hashes):
    rows = []
    seen = set()
    for path in corpus_paths:
        path = Path(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not _eligible(record):
                continue
            prompt, response = record["trigger"].strip(), record["accepted_target"].strip()
            fp = fingerprint(prompt, response)
            if fp in holdout_hashes or fp in seen:
                continue
            seen.add(fp)
            rows.append({"prompt": prompt, "response": response, "source": str(path), "record_id": record.get("record_id"), "weight": 1})
    for anchor in anchors:
        prompt, response = anchor["prompt"].strip(), anchor["response"].strip()
        fp = fingerprint(prompt, response)
        if fp in holdout_hashes or fp in seen:
            continue
        seen.add(fp)
        rows.append({"prompt": prompt, "response": response, "source": "CURRENT_SELF_ANCHOR", "record_id": anchor.get("record_id"), "weight": int(anchor.get("weight", 1))})
    return rows
