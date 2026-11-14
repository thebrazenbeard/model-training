from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import Request, urlopen


MODEL = "ministral-3:14b"
MODEL_BLOB_SHA256 = (
    "bfb40fc6bb9c3b2ed529b480e04f824c005ea8f86733d4ebbf0c204de484891e"
)
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
SEED = 20261001
REQUIRED_REVIEW_KEYS = {
    "verdict",
    "prompt_well_posed",
    "grader_matches_prompt",
    "source_evidence_sufficient",
    "issue_code",
    "reason",
    "confidence",
}
def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_model_blob_path(modelfile_text: str) -> Path:
    for line in modelfile_text.splitlines():
        if line.startswith("FROM "):
            return Path(line[5:].strip().strip('"'))
    raise RuntimeError("Ollama Modelfile has no FROM blob")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reviewer_identity() -> dict:
    shown = subprocess.run(
        ["ollama", "show", MODEL, "--modelfile"],
        capture_output=True,
        text=True,
        check=True,
    )
    blob_path = parse_model_blob_path(shown.stdout)
    actual = sha256_file(blob_path)
    if actual != MODEL_BLOB_SHA256:
        raise RuntimeError(f"reviewer model blob mismatch: {actual}")
    return {
        "provider": "OLLAMA_LOCAL",
        "model": MODEL,
        "model_blob_sha256": actual,
        "runtime": ollama_version(),
        "temperature": 0,
        "seed": SEED,
    }


def review_prompt(case: dict) -> str:
    subject = {
        "case_id": case["case_id"],
        "category": case["category"],
        "prompt": case["prompt"],
        "grader_contract": case["grader_contract"],
        "source_evidence": case.get("source_evidence"),
    }
    return (
        "You are an independent benchmark-quality reviewer. "
        "Audit the case below; do not grade a model response. "
        "ADMIT only if the prompt is unambiguous, the deterministic grader "
        "checks what the prompt actually asks, and supplied source evidence "
        "is sufficient where source evidence is relevant. "
        "For executable tests, flag under-specification or tests that permit "
        "materially wrong implementations. Return JSON only with exactly: "
        "verdict (ADMIT|REJECT), prompt_well_posed (bool), "
        "grader_matches_prompt (bool), source_evidence_sufficient (bool), "
        "issue_code (NONE or short code), reason (short string), "
        "confidence (high|medium|low).\nSUBJECT="
        + canonical_json(subject)
    )
def parse_review(text: str) -> dict:
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != REQUIRED_REVIEW_KEYS:
        raise ValueError("review must use the exact closed schema")
    if value["verdict"] not in {"ADMIT", "REJECT"}:
        raise ValueError("invalid verdict")
    for field in (
        "prompt_well_posed",
        "grader_matches_prompt",
        "source_evidence_sufficient",
    ):
        if not isinstance(value[field], bool):
            raise ValueError(f"{field} must be boolean")
    if value["confidence"] not in {"high", "medium", "low"}:
        raise ValueError("invalid confidence")
    if not isinstance(value["issue_code"], str) or not value["issue_code"]:
        raise ValueError("invalid issue_code")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("invalid reason")
    if value["verdict"] == "ADMIT":
        if not all(
            value[field]
            for field in (
                "prompt_well_posed",
                "grader_matches_prompt",
                "source_evidence_sufficient",
            )
        ):
            raise ValueError("ADMIT requires all checks true")
    return value
def ollama_version() -> str:
    result = subprocess.run(
        ["ollama", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _ollama_request(prompt: str) -> dict:
    request_body = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "seed": SEED,
        },
    }
    request = Request(
        OLLAMA_URL,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=900) as response:
        return json.loads(response.read().decode("utf-8"))


def call_reviewer(prompt: str, *, max_attempts: int = 3) -> tuple[dict, dict]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    failures: list[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            outer = _ollama_request(prompt)
            response_text = outer.get("response")
            if not isinstance(response_text, str) or not response_text.strip():
                raise ValueError("empty reviewer response")
            review = parse_review(response_text)
            runtime = {
                key: outer.get(key)
                for key in (
                    "created_at",
                    "done_reason",
                    "total_duration",
                    "load_duration",
                    "prompt_eval_count",
                    "prompt_eval_duration",
                    "eval_count",
                    "eval_duration",
                )
            }
            runtime["attempts"] = attempt
            if failures:
                runtime["prior_attempt_failures"] = failures
            return review, runtime
        except (ValueError, json.JSONDecodeError) as exc:
            failures.append(f"attempt_{attempt}:{type(exc).__name__}:{exc}")
            if attempt == max_attempts:
                raise RuntimeError(
                    "reviewer response invalid after "
                    f"{max_attempts} attempts: {' | '.join(failures)}"
                ) from exc
    raise AssertionError("unreachable")
def summarize_reviews(reviews: list[dict]) -> dict:
    verdicts = Counter(row["verdict"] for row in reviews)
    categories = {}
    for category in sorted({row["category"] for row in reviews}):
        subset = [row for row in reviews if row["category"] == category]
        categories[category] = dict(
            sorted(Counter(row["verdict"] for row in subset).items())
        )
    reject = verdicts["REJECT"]
    return {
        "status": "PASS" if reject == 0 else "HOLD",
        "reviewed": len(reviews),
        "admit": verdicts["ADMIT"],
        "reject": reject,
        "by_category": categories,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
def run_audit(
    *,
    packet_path: Path,
    packet_manifest_path: Path,
    output_path: Path,
) -> dict:
    packet_raw = packet_path.read_bytes()
    packet_manifest = json.loads(
        packet_manifest_path.read_text(encoding="utf-8")
    )
    packet_sha = sha256_bytes(packet_raw)
    if packet_sha != packet_manifest["packet_sha256"]:
        raise RuntimeError("audit packet hash mismatch")
    packet = _read_jsonl(packet_path)
    identity = reviewer_identity()

    prior: dict[str, dict] = {}
    if output_path.exists():
        for row in _read_jsonl(output_path):
            prior[row["case_id"]] = row
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reviews = []
    for case in packet:
        case_id = case["case_id"]
        if case_id in prior:
            reviews.append(prior[case_id])
            continue
        prompt = review_prompt(case)
        review, runtime = call_reviewer(prompt)
        row = {
            "case_id": case_id,
            "category": case["category"],
            "review": review,
            "verdict": review["verdict"],
            "review_prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "runtime": runtime,
        }
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(row) + "\n")
        reviews.append(row)

    raw = output_path.read_bytes()
    summary = summarize_reviews(reviews)
    return {
        "schema": "V10_RETENTION_INDEPENDENT_MODEL_AUDIT_V1",
        "candidate_sha256": packet_manifest["candidate_sha256"],
        "packet_sha256": packet_sha,
        "sample_rows": len(packet),
        "review_output_sha256": sha256_bytes(raw),
        "reviewer": identity,
        "summary": summary,
        "claim_ceiling": (
            "INDEPENDENT_MODEL_SAMPLE_AUDIT_ONLY / "
            "DOES_NOT_OVERRIDE_DETERMINISTIC_SOURCE_OR_TEST_TRUTH / "
            "NOT_FULL_BANK_REVIEW"
        ),
    }
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = run_audit(
            packet_path=args.packet,
            packet_manifest_path=args.packet_manifest,
            output_path=args.output,
        )
        code = 0 if receipt["summary"]["status"] == "PASS" else 2
    except Exception as exc:
        receipt = {
            "schema": "V10_RETENTION_INDEPENDENT_MODEL_AUDIT_V1",
            "status": "HOLD",
            "error": str(exc),
        }
        code = 2
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
