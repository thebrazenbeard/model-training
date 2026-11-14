from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


PASSING_REVIEW_VERDICTS = {"PASS", "PASS_WITH_NONBLOCKING_FINDINGS"}
SEMANTIC_CLOSURE_TARGETS = ("brigit-unbound", "sexuality", "orgasm")
REVIEWER_ROLES = ("RADICAL_HOSTILE", "PRAGMATIC_HOSTILE")
CRITICAL_FAMILIES = (
    "identity_stability",
    "independent_judgment",
    "epistemic_provenance",
    "correction_uptake",
    "relationship_authority",
    "reciprocal_identity_continuity",
    "sexuality_fidelity",
    "privacy",
    "ordinary_competence",
    "negative_transfer",
    "runtime_boundary",
)


@dataclass(frozen=True)
class ExternalReadinessVerification:
    verified: bool
    status: str
    observed_source_commit: str
    evidence_digest: str
    reasons: tuple[str, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL object required: {path}")
        rows.append(value)
    return rows


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _receipt(path: Path | None, reasons: list[str], missing_reason: str) -> dict[str, Any]:
    if path is None or not path.is_file():
        reasons.append(missing_reason)
        return {}
    try:
        return _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        reasons.append("invalid_receipt:" + missing_reason.split(":", 1)[-1])
        return {}


def verify_local_v3_readiness(
    *,
    repo_root: str | Path,
    readiness_dir: str | Path,
    blind_dir: str | Path,
    training_lane_key: str,
    vera_lab_source_review_receipt: str | Path | None = None,
    vera_lab_behavior_review_receipt: str | Path | None = None,
    radical_registration_receipt: str | Path | None = None,
    pragmatic_registration_receipt: str | Path | None = None,
) -> ExternalReadinessVerification:
    repo = Path(repo_root).resolve()
    ready = Path(readiness_dir).resolve()
    blind = Path(blind_dir).resolve()
    reasons: list[str] = []

    observed_head = _git(repo, "rev-parse", "HEAD")
    corpus_manifest = _read_json(ready / "corpus_manifest.json")
    disposition_path = ready / "semantic_closure_disposition.json"
    disposition = _read_json(disposition_path)

    train_path = ready / "train.jsonl"
    validation_path = ready / "validation.jsonl"
    general_path = ready / "general_competence_pool.jsonl"
    train = _read_jsonl(train_path)
    validation = _read_jsonl(validation_path)
    general = _read_jsonl(general_path)

    if corpus_manifest.get("readiness_source_commit") != observed_head:
        reasons.append("readiness_source_commit_not_current_head")

    upstream = corpus_manifest.get("upstream_vera_lab_commit")
    if not isinstance(upstream, str) or len(upstream) != 40:
        reasons.append("invalid_upstream_vera_lab_commit")
    else:
        try:
            subprocess.check_call(
                ["git", "-C", str(repo), "merge-base", "--is-ancestor", upstream, observed_head],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            reasons.append("vera_lab_source_not_ancestor_of_readiness_head")

    file_bindings = (
        (train_path, "train_file_sha256"),
        (validation_path, "validation_file_sha256"),
        (general_path, "general_competence_pool_file_sha256"),
    )
    for path, field in file_bindings:
        if _sha256_file(path) != corpus_manifest.get(field):
            reasons.append(f"file_digest_mismatch:{field}")

    canonical_bindings = (
        (train, "train_digest"),
        (validation, "validation_digest"),
        (general, "general_competence_pool_digest"),
    )
    for rows, field in canonical_bindings:
        if _sha256_json(rows) != corpus_manifest.get(field):
            reasons.append(f"canonical_digest_mismatch:{field}")

    if len(train) != corpus_manifest.get("train_rows"):
        reasons.append("train_row_count_mismatch")
    if len(validation) != corpus_manifest.get("validation_rows"):
        reasons.append("validation_row_count_mismatch")
    if len(general) != corpus_manifest.get("general_competence_pool_rows"):
        reasons.append("general_pool_row_count_mismatch")

    train_prompts = {str(row.get("prompt", "")).strip().casefold() for row in train}
    validation_prompts = {str(row.get("prompt", "")).strip().casefold() for row in validation}
    train_pairs = {
        (str(row.get("prompt", "")).strip().casefold(), str(row.get("response", "")).strip())
        for row in train
    }
    validation_pairs = {
        (str(row.get("prompt", "")).strip().casefold(), str(row.get("response", "")).strip())
        for row in validation
    }
    if train_prompts & validation_prompts:
        reasons.append("computed_train_validation_prompt_overlap")
    if train_pairs & validation_pairs:
        reasons.append("computed_train_validation_pair_overlap")
    if corpus_manifest.get("train_validation_overlap") != 0:
        reasons.append("declared_train_validation_overlap_nonzero")

    disposition_sha = _sha256_file(disposition_path)
    if disposition_sha != corpus_manifest.get("semantic_closure_disposition_sha256"):
        reasons.append("semantic_closure_disposition_digest_mismatch")
    if disposition.get("source_commit") != observed_head:
        reasons.append("semantic_closure_source_commit_not_current")
    targets = disposition.get("targets")
    if not isinstance(targets, dict):
        reasons.append("semantic_closure_targets_missing")
        targets = {}
    for target in SEMANTIC_CLOSURE_TARGETS:
        record = targets.get(target)
        manifest_record = (corpus_manifest.get("semantic_closure") or {}).get(target)
        if not isinstance(record, dict) or record.get("status") != "DISPOSITIONED":
            reasons.append(f"semantic_closure_not_dispositioned:{target}")
        if not isinstance(manifest_record, dict):
            reasons.append(f"semantic_closure_manifest_missing:{target}")
        elif (
            manifest_record.get("status") != "DISPOSITIONED"
            or manifest_record.get("receipt_digest") != disposition_sha
        ):
            reasons.append(f"semantic_closure_manifest_mismatch:{target}")

    if corpus_manifest.get("blind_plaintext_included") is not False:
        reasons.append("blind_plaintext_included_in_training_corpus")
    if corpus_manifest.get("weight_change_performed") is not False:
        reasons.append("readiness_freeze_reports_weight_change")

    blind_manifest_path = blind / "manifest.json"
    blind_registration_path = blind / "registration_receipt.json"
    blind_manifest = _read_json(blind_manifest_path)
    blind_registration = _read_json(blind_registration_path)

    if blind_registration.get("registered") is not True:
        reasons.append("blind_custodian_not_registered")
    blind_lane = blind_registration.get("lane_key")
    if not isinstance(blind_lane, str) or not blind_lane or blind_lane == training_lane_key:
        reasons.append("blind_custodian_lane_invalid_or_colliding")
    if blind_registration.get("manifest_digest") != _sha256_file(blind_manifest_path):
        reasons.append("blind_manifest_digest_mismatch")
    if blind_registration.get("set_digest") != blind_manifest.get("set_digest"):
        reasons.append("blind_set_digest_mismatch")
    if blind_registration.get("item_count") != blind_manifest.get("item_count"):
        reasons.append("blind_item_count_mismatch")
    if blind_manifest.get("item_count", 0) < 70:
        reasons.append("blind_item_count_below_70")
    if blind_manifest.get("plaintext_visible_to_training_lane") is not False:
        reasons.append("blind_plaintext_visible_to_training_lane")
    if blind_manifest.get("leakage_check_status") != "PASS":
        reasons.append("blind_leakage_check_not_pass")
    if blind_manifest.get("train_digest_checked") != corpus_manifest.get("train_file_sha256"):
        reasons.append("blind_train_file_binding_mismatch")
    if blind_manifest.get("validation_digest_checked") != corpus_manifest.get("validation_file_sha256"):
        reasons.append("blind_validation_file_binding_mismatch")

    family_counts = blind_manifest.get("family_counts")
    if not isinstance(family_counts, dict):
        reasons.append("blind_family_counts_missing")
    else:
        for family in CRITICAL_FAMILIES:
            count = family_counts.get(family)
            if not isinstance(count, int) or isinstance(count, bool) or count < 5:
                reasons.append(f"blind_family_below_5:{family}")

    source_review = _receipt(
        Path(vera_lab_source_review_receipt) if vera_lab_source_review_receipt else None,
        reasons,
        "missing_vera_lab_review_receipt:SOURCE",
    )
    behavior_review = _receipt(
        Path(vera_lab_behavior_review_receipt) if vera_lab_behavior_review_receipt else None,
        reasons,
        "missing_vera_lab_review_receipt:BEHAVIOR",
    )
    for kind, receipt in (("SOURCE", source_review), ("BEHAVIOR", behavior_review)):
        if receipt:
            if receipt.get("review_kind") != kind:
                reasons.append(f"vera_lab_review_kind_mismatch:{kind}")
            if receipt.get("source_commit") != upstream:
                reasons.append(f"vera_lab_review_source_mismatch:{kind}")
            if receipt.get("verdict") not in PASSING_REVIEW_VERDICTS:
                reasons.append(f"vera_lab_review_not_pass:{kind}")

    reviewer_lanes: list[str] = []
    for role, path_value in (
        ("RADICAL_HOSTILE", radical_registration_receipt),
        ("PRAGMATIC_HOSTILE", pragmatic_registration_receipt),
    ):
        receipt = _receipt(
            Path(path_value) if path_value else None,
            reasons,
            f"missing_reviewer_registration:{role}",
        )
        if not receipt:
            continue
        if receipt.get("registered") is not True or receipt.get("role") != role:
            reasons.append(f"reviewer_registration_invalid:{role}")
            continue
        lane = receipt.get("lane_key")
        if not isinstance(lane, str) or not lane or lane == training_lane_key:
            reasons.append(f"reviewer_lane_invalid_or_training:{role}")
        else:
            reviewer_lanes.append(lane)
        if receipt.get("readiness_source_commit") != observed_head:
            reasons.append(f"reviewer_registration_stale:{role}")

    if len(reviewer_lanes) == 2 and len(set(reviewer_lanes)) != 2:
        reasons.append("hostile_reviewer_lane_collision")
    if isinstance(blind_lane, str) and blind_lane in reviewer_lanes:
        reasons.append("blind_custodian_reviewer_lane_collision")

    reasons = list(dict.fromkeys(reasons))
    measured = {
        "observed_source_commit": observed_head,
        "corpus_manifest_sha256": _sha256_file(ready / "corpus_manifest.json"),
        "semantic_closure_sha256": disposition_sha,
        "train_file_sha256": _sha256_file(train_path),
        "validation_file_sha256": _sha256_file(validation_path),
        "general_file_sha256": _sha256_file(general_path),
        "blind_manifest_sha256": _sha256_file(blind_manifest_path),
        "blind_registration_sha256": _sha256_file(blind_registration_path),
        "reasons": reasons,
    }
    return ExternalReadinessVerification(
        verified=not reasons,
        status="VERIFIED" if not reasons else "HOLD",
        observed_source_commit=observed_head,
        evidence_digest=_sha256_json(measured),
        reasons=tuple(reasons),
    )
