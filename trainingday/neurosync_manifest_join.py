"""Source-only cross-manifest consistency check, with permanent authority HOLD.

A TRAININGDAY Neuro-Sync transfer preflight and a seven-arm preregistration
can each validate while describing *different experiments*. This module binds
their canonical JSON and compares overlapping source/model identities.

Dataset split-byte hashes and study dataset-manifest hashes describe different
objects. A caller's mapping claim can be checked for internal reference
consistency, but is not independent proof of the relationship. Therefore this
module never certifies a joint experiment, authorizes a physical launch, or
authorizes persistent state changes. A future authenticated verifier must
attest the actual mapped bytes and artifact custody separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib
import json
import re
from typing import Any, Mapping


MAPPING_SCHEMA = "trainingday.neurosync.dataset-mapping-claim.v1"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_DOC_BYTES = 65536
_MAPPING_KEYS = frozenset({
    "schema", "study_id", "preflight_train_split_sha256",
    "study_train_manifest_sha256", "preflight_eval_split_sha256",
    "study_heldout_manifest_sha256", "preflight_rights_receipt_sha256",
    "study_rights_reference_sha256", "mapping_receipt_sha256",
})


@dataclass(frozen=True, slots=True)
class ManifestJoinResult:
    """Descriptive source result, never a capability or execution grant."""

    preflight_canonical_sha256: str | None
    study_canonical_sha256: str | None
    mapping_claim_canonical_sha256: str | None
    structural_matches: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def same_experiment_qualified(self) -> bool:
        return False

    @property
    def physical_launch_authorized(self) -> bool:
        return False

    @property
    def state_commit_authorized(self) -> bool:
        return False

    @property
    def disposition(self) -> str:
        if self.blockers:
            return "UNJOINED_HOLD"
        return "STRUCTURALLY_MATCHED_UNVERIFIED_HOLD"

    def as_dict(self) -> dict[str, Any]:
        return {
            "preflight_canonical_sha256": self.preflight_canonical_sha256,
            "study_canonical_sha256": self.study_canonical_sha256,
            "mapping_claim_canonical_sha256": self.mapping_claim_canonical_sha256,
            "structural_matches": self.structural_matches,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "same_experiment_qualified": False,
            "physical_launch_authorized": False,
            "state_commit_authorized": False,
            "disposition": self.disposition,
        }


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _canonical_document(
    document: Mapping[str, Any] | str | bytes | None,
    label: str, blockers: list[str],
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        if isinstance(document, bytes):
            if len(document) > _MAX_DOC_BYTES:
                raise ValueError("document exceeds 65536 bytes")
            document = document.decode("utf-8", errors="strict")
        if isinstance(document, str):
            if len(document.encode("utf-8")) > _MAX_DOC_BYTES:
                raise ValueError("document exceeds 65536 bytes")
            document = json.loads(
                document, object_pairs_hook=_no_duplicate_keys,
                parse_constant=_reject_constant,
            )
        if not isinstance(document, dict):
            raise ValueError("JSON object required")
        encoded = json.dumps(document, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")
        if len(encoded) > _MAX_DOC_BYTES:
            raise ValueError("canonical document exceeds 65536 bytes")
        return document, sha256(encoded).hexdigest()
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        blockers.append(f"{label}:INVALID_CANONICAL_JSON:{type(exc).__name__}")
        return None, None


def _load_validator(module_name: str, function_name: str):
    """Load sibling contract or its source-layout version; no code generation."""
    for candidate in (f"trainingday.{module_name}", module_name):
        try:
            module = importlib.import_module(candidate)
        except ModuleNotFoundError as exc:
            if exc.name not in (candidate, "trainingday"):
                raise
            continue
        return getattr(module, function_name)
    raise ModuleNotFoundError(module_name)


def validate_manifest_join(
    preflight_document: Mapping[str, Any] | str | bytes,
    study_document: Mapping[str, Any] | str | bytes,
    *,
    dataset_mapping_claim: Mapping[str, Any] | str | bytes | None = None,
) -> ManifestJoinResult:
    """Compare two source plans without pretending their datasets are mapped.

    ``dataset_mapping_claim`` is a proposed cross-reference, not a verified
    receipt. Even a fully matched mapping claim cannot eliminate the HOLD:
    external host-custodied verification of the split/manifest relationship,
    independently sealed heldout set, rights, and artifacts remains absent.
    """
    blockers: list[str] = []
    warnings = [
        "DATASET_MAPPING_NOT_INDEPENDENTLY_VERIFIED",
        "JOIN_IS_SOURCE_ONLY_NO_SHARED_EXECUTION_EVIDENCE",
        "CHECKPOINT_IDENTITIES_NOT_AVAILABLE_IN_PREFLIGHT",
        "STUDY_OMITS_CHAT_TEMPLATE_IDENTITY",
        "NO_PHYSICAL_LAUNCH_OR_STATE_COMMIT_AUTHORITY",
    ]
    pre, pre_hash = _canonical_document(preflight_document, "preflight", blockers)
    study, study_hash = _canonical_document(study_document, "study", blockers)
    if dataset_mapping_claim is None:
        mapping, mapping_hash = None, None
    else:
        mapping, mapping_hash = _canonical_document(
            dataset_mapping_claim, "mapping", blockers)

    if pre is not None:
        try:
            validate_preflight = _load_validator(
                "neurosync_transfer_preflight", "validate_transfer_manifest")
            pre_result = validate_preflight(pre)
            if not pre_result.compatible:
                blockers.append("preflight:VALIDATOR_REJECTED")
                blockers.extend(f"preflight:{code}" for code in pre_result.blockers)
        except (ImportError, AttributeError) as exc:
            blockers.append(f"preflight:VALIDATOR_UNAVAILABLE:{type(exc).__name__}")

    if study is not None:
        try:
            validate_study = _load_validator(
                "neurosync_experiment_design", "validate_study_plan")
            validate_study(study)
        except (ImportError, AttributeError) as exc:
            blockers.append(f"study:VALIDATOR_UNAVAILABLE:{type(exc).__name__}")
        except ValueError as exc:
            blockers.append(f"study:VALIDATOR_REJECTED:{type(exc).__name__}")

    if pre is not None and study is not None:
        try:
            pins = study["source_pins"]
            sources = pre["source_revisions"]
            comparison = (
                ("neuro_sync_head", pins["neuro_sync_head"], sources["neuro_sync_git_sha"]),
                ("model_training_head", pins["model_training_head"], sources["model_training_git_sha"]),
                ("shared_base_sha256", study["models"]["shared_base_sha256"], pre["donor"]["base_sha256"]),
                ("recipient_base_sha256", study["models"]["shared_base_sha256"], pre["recipient"]["base_sha256"]),
                ("tokenizer_sha256", study["models"]["tokenizer_sha256"], pre["donor"]["tokenizer_sha256"]),
                ("recipient_tokenizer_sha256", study["models"]["tokenizer_sha256"], pre["recipient"]["tokenizer_sha256"]),
            )
            for key, left, right in comparison:
                if left != right:
                    blockers.append(f"cross_manifest.{key}:MISMATCH")
        except (TypeError, KeyError) as exc:
            blockers.append(f"cross_manifest:INVALID_DOCUMENT_STRUCTURE:{type(exc).__name__}")

    if mapping is not None:
        try:
            if set(mapping) != _MAPPING_KEYS:
                blockers.append("mapping:SCHEMA_FIELDS_MISMATCH")
            if mapping.get("schema") != MAPPING_SCHEMA:
                blockers.append("mapping.schema:UNSUPPORTED")
            for key in _MAPPING_KEYS - {"schema", "study_id"}:
                if not isinstance(mapping.get(key), str) or not _HEX64.fullmatch(mapping[key]):
                    blockers.append(f"mapping.{key}:INVALID_SHA256")
            if pre is not None and study is not None:
                references = (
                    ("study_id", study["study_id"]),
                    ("preflight_train_split_sha256", pre["dataset"]["train_split_sha256"]),
                    ("study_train_manifest_sha256", study["dataset"]["approved_train_manifest_sha256"]),
                    ("preflight_eval_split_sha256", pre["dataset"]["eval_split_sha256"]),
                    ("study_heldout_manifest_sha256", study["dataset"]["sealed_heldout_manifest_sha256"]),
                    ("preflight_rights_receipt_sha256", pre["dataset"]["rights_receipt_sha256"]),
                    ("study_rights_reference_sha256", study["dataset"]["data_rights_reference_sha256"]),
                )
                for key, expected in references:
                    if mapping.get(key) != expected:
                        blockers.append(f"mapping.{key}:REFERENCE_MISMATCH")
        except (TypeError, KeyError) as exc:
            blockers.append(f"mapping:INVALID_DOCUMENT_STRUCTURE:{type(exc).__name__}")

    # A self-asserted digest relationship is not a host-verified mapping.
    # This is the material gap the join is explicitly designed to expose.
    if dataset_mapping_claim is None:
        blockers.append("mapping:INDEPENDENTLY_VERIFIED_RELATION_REQUIRED")

    return ManifestJoinResult(
        preflight_canonical_sha256=pre_hash,
        study_canonical_sha256=study_hash,
        mapping_claim_canonical_sha256=mapping_hash,
        structural_matches=not blockers,
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


__all__ = ["MAPPING_SCHEMA", "ManifestJoinResult", "validate_manifest_join"]
