from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

from successor.experiments.build_retention_audit_arch_v2_packet import (
    packet_bytes,
)
from successor.experiments.retention_audit_arch_v2_reviewer import (
    ollama_text,
    reviewer_identity,
    run_batched_reviewer_qualification,
)
from successor.experiments.run_retention_audit_arch_v2_semantic import (
    run_semantic_audit,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_execution_binding(
    binding: dict,
    *,
    root: Path,
) -> dict[str, str]:
    artifacts = binding.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("execution binding artifacts missing")
    verified: dict[str, str] = {}
    for name, record in sorted(artifacts.items()):
        if not isinstance(record, dict):
            raise ValueError(f"binding artifact invalid:{name}")
        rel = record.get("path")
        expected = record.get("sha256")
        if not isinstance(rel, str) or not rel:
            raise ValueError(f"binding artifact path invalid:{name}")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"binding artifact hash invalid:{name}")
        path = root / rel
        if not path.is_file():
            raise ValueError(f"binding artifact missing:{name}:{rel}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"binding hash mismatch:{name}:{actual}!={expected}"
            )
        verified[name] = actual
    return verified


def _validate_packet_binding(
    packet: list[dict],
    manifest: dict,
) -> None:
    payload = packet_bytes(packet)
    actual_sha = hashlib.sha256(payload).hexdigest()
    if actual_sha != manifest.get("packet_sha256"):
        raise ValueError("packet hash mismatch")
    if len(packet) != manifest.get("sample_rows"):
        raise ValueError("packet row count mismatch")
    case_ids = [row.get("case_id") for row in packet]
    if case_ids != manifest.get("sample_case_ids"):
        raise ValueError("packet case IDs do not match manifest")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("packet case IDs are not unique")
    if manifest.get("disjoint_from_all_predecessors") is not True:
        raise ValueError("packet is not predecessor-disjoint")


def run_live_review(
    packet: list[dict],
    manifest: dict,
    *,
    call_text: Callable[[str], str] = ollama_text,
    identity_fn: Callable[[], dict] = reviewer_identity,
    qualification_batch_size: int = 8,
    semantic_batch_size: int = 5,
    max_attempts: int = 3,
    expected_reviewer_identity: dict | None = None,
) -> dict:
    _validate_packet_binding(packet, manifest)
    identity = identity_fn()
    if (
        expected_reviewer_identity is not None
        and identity != expected_reviewer_identity
    ):
        raise ValueError("reviewer identity mismatch")

    qualification = run_batched_reviewer_qualification(
        call_text=call_text,
        batch_size=qualification_batch_size,
        max_attempts=max_attempts,
    )
    qualification["reviewer"] = identity

    if qualification.get("status") != "REVIEWER_QUALIFIED":
        return {
            "reviewer": identity,
            "reviewer_qualification": qualification,
            "semantic": {
                "status": "SEMANTIC_AUDIT_HOLD",
                "reviewed": 0,
                "expected_count": len(packet),
                "observation_counts": {},
                "reasons": ["reviewer_unqualified"],
                "rows": [],
            },
        }

    semantic = run_semantic_audit(
        packet,
        call_reviewer=lambda prompt, batch: call_text(prompt),
        expected_count=len(packet),
        max_attempts=max_attempts,
        batch_size=semantic_batch_size,
    )
    semantic["reviewer"] = identity
    return {
        "reviewer": identity,
        "reviewer_qualification": qualification,
        "semantic": semantic,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument(
        "--reviewer-qualification-output",
        type=Path,
        required=True,
    )
    parser.add_argument("--semantic-output", type=Path, required=True)
    parser.add_argument("--qualification-batch-size", type=int, default=8)
    parser.add_argument("--semantic-batch-size", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--execution-binding", type=Path, required=True)
    args = parser.parse_args()

    binding = json.loads(
        args.execution_binding.read_text(encoding="utf-8")
    )
    root = Path(__file__).resolve().parents[2]
    verify_execution_binding(binding, root=root)

    packet = _read_jsonl(args.packet)
    manifest = json.loads(
        args.packet_manifest.read_text(encoding="utf-8")
    )
    result = run_live_review(
        packet,
        manifest,
        qualification_batch_size=args.qualification_batch_size,
        semantic_batch_size=args.semantic_batch_size,
        max_attempts=args.max_attempts,
        expected_reviewer_identity=binding.get("reviewer"),
    )

    args.reviewer_qualification_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.semantic_output.parent.mkdir(parents=True, exist_ok=True)
    args.reviewer_qualification_output.write_text(
        json.dumps(
            result["reviewer_qualification"],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    args.semantic_output.write_text(
        json.dumps(result["semantic"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(json.dumps({
        "reviewer_qualification_status": (
            result["reviewer_qualification"].get("status")
        ),
        "semantic_status": result["semantic"].get("status"),
        "semantic_reviewed": result["semantic"].get("reviewed"),
        "reviewer": result["reviewer"],
        "reviewer_qualification_output": (
            args.reviewer_qualification_output.as_posix()
        ),
        "semantic_output": args.semantic_output.as_posix(),
    }, sort_keys=True))

    return (
        0
        if (
            result["reviewer_qualification"].get("status")
            == "REVIEWER_QUALIFIED"
            and result["semantic"].get("status")
            == "SEMANTIC_AUDIT_COMPLETE"
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
