from __future__ import annotations

import re

from successor.experiments.sealed_final_bank_commitment import (
    FORBIDDEN_PLAINTEXT_KEYS,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _nonempty(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(value) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def _find_plaintext_keys(value, *, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text.casefold() in FORBIDDEN_PLAINTEXT_KEYS:
                found.append(f"{path}.{key_text}")
            found.extend(
                _find_plaintext_keys(child, path=f"{path}.{key_text}")
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(
                _find_plaintext_keys(
                    child,
                    path=f"{path}[{index}]",
                )
            )
    return found


def _human_pool(
    value,
    *,
    label: str,
    reasons: list[str],
) -> set[str]:
    if not isinstance(value, list) or not value:
        reasons.append(f"{label}_pool_missing")
        return set()

    actor_ids: set[str] = set()
    for index, actor in enumerate(value):
        if not isinstance(actor, dict):
            reasons.append(f"{label}_entry_not_object:{index}")
            continue
        actor_id = actor.get("actor_id")
        if not _nonempty(actor_id):
            reasons.append(f"{label}_actor_id_invalid:{index}")
            continue
        if actor_id in actor_ids:
            reasons.append(f"{label}_duplicate_actor_id:{actor_id}")
        actor_ids.add(actor_id)
        if not _sha256(
            actor.get("private_identity_attestation_sha256")
        ):
            reasons.append(
                f"{label}_identity_attestation_invalid:{actor_id}"
            )
    return actor_ids


def _validate_synthetic(
    label: str,
    value,
    reasons: list[str],
) -> dict:
    summary = {
        "actor_id": None,
        "provider": None,
        "model": None,
        "model_family": None,
        "runtime": None,
    }
    if not isinstance(value, dict):
        reasons.append(f"synthetic_{label}_missing")
        return summary

    for field in (
        "actor_id",
        "provider",
        "model",
        "model_family",
        "revision",
        "runtime",
    ):
        if not _nonempty(value.get(field)):
            reasons.append(f"synthetic_{label}_{field}_invalid")

    summary = {
        key: value.get(key)
        for key in summary
    }

    lineage = (
        str(value.get("model", ""))
        + " "
        + str(value.get("model_family", ""))
    ).casefold()
    if "qwen" in lineage:
        reasons.append(f"synthetic_{label}_qwen_lineage_forbidden")

    if not isinstance(value.get("seed"), int):
        reasons.append(f"synthetic_{label}_seed_invalid")
    sampling = value.get("sampling")
    if not isinstance(sampling, dict) or not sampling:
        reasons.append(f"synthetic_{label}_sampling_invalid")
    if not _sha256(value.get("identity_receipt_sha256")):
        reasons.append(f"synthetic_{label}_identity_receipt_invalid")

    return summary


def validate_custodian_binding(binding: dict) -> dict:
    reasons: list[str] = []

    if not isinstance(binding, dict):
        return {
            "schema": "V10_FINAL_BANK_CUSTODIAN_BINDING_CHECK_V1",
            "status": "HOLD",
            "reasons": ["binding_not_object"],
        }

    for path in _find_plaintext_keys(binding):
        reasons.append(f"plaintext_field_present:{path}")

    if (
        binding.get("schema")
        != "V10_FINAL_BANK_CUSTODIAN_BINDING_V1"
    ):
        reasons.append(f"schema:{binding.get('schema')}")
    if (
        binding.get("status")
        != "BOUND_FOR_PRETRAINING_SEALED_BANK"
    ):
        reasons.append(f"binding_status:{binding.get('status')}")

    subject = binding.get("composition_subject")
    if not isinstance(subject, dict):
        reasons.append("composition_subject_missing")
        subject = {}
    proposal_blob = subject.get("proposal_blob_sha")
    if not (
        isinstance(proposal_blob, str)
        and bool(_GIT_SHA.fullmatch(proposal_blob))
    ):
        reasons.append("composition_proposal_blob_sha_invalid")
    for field in (
        "generation_spec_sha256",
        "grader_spec_sha256",
    ):
        if not _sha256(subject.get(field)):
            reasons.append(f"composition_{field}_invalid")
    for field in (
        "family_manifest_spec_sha256",
        "diversity_diagnostic_spec_sha256",
    ):
        if field in subject and not _sha256(subject.get(field)):
            reasons.append(f"composition_{field}_invalid")

    synthetic = binding.get("synthetic_custodians")
    if not isinstance(synthetic, dict):
        reasons.append("synthetic_custodians_missing")
        synthetic = {}
    if set(synthetic) != {"A", "B"}:
        reasons.append("synthetic_custodian_key_set_mismatch")
    a = _validate_synthetic("A", synthetic.get("A"), reasons)
    b = _validate_synthetic("B", synthetic.get("B"), reasons)

    if (
        _nonempty(a.get("actor_id"))
        and a.get("actor_id") == b.get("actor_id")
    ):
        reasons.append("synthetic_actor_ids_not_distinct")
    if (
        _nonempty(a.get("model_family"))
        and a.get("model_family") == b.get("model_family")
    ):
        reasons.append("synthetic_model_families_not_distinct")
    if (
        _nonempty(a.get("model"))
        and a.get("model") == b.get("model")
    ):
        reasons.append("synthetic_models_not_distinct")
    if (
        _nonempty(a.get("provider"))
        and _nonempty(a.get("runtime"))
        and (a.get("provider"), a.get("runtime"))
        == (b.get("provider"), b.get("runtime"))
    ):
        reasons.append("synthetic_provider_runtime_not_distinct")

    authors = _human_pool(
        binding.get("human_authors"),
        label="human_author",
        reasons=reasons,
    )
    reviewers = _human_pool(
        binding.get("human_reviewers"),
        label="human_reviewer",
        reasons=reasons,
    )
    for actor_id in sorted(authors & reviewers):
        reasons.append(
            f"human_author_reviewer_overlap:{actor_id}"
        )

    synthetic_ids = {
        actor_id
        for actor_id in (a.get("actor_id"), b.get("actor_id"))
        if _nonempty(actor_id)
    }
    for actor_id in sorted(
        (authors | reviewers) & synthetic_ids
    ):
        reasons.append(
            f"human_synthetic_actor_overlap:{actor_id}"
        )

    custody = binding.get("custody")
    if not isinstance(custody, dict):
        reasons.append("custody_missing")
        custody = {}
    if not _nonempty(custody.get("surface_id")):
        reasons.append("custody_surface_id_invalid")
    if (
        custody.get("surface_class")
        != "INDEPENDENT_CUSTODY_EVALUATION_LANE"
    ):
        reasons.append("custody_surface_class_invalid")
    if custody.get("training_lane_access") is not False:
        reasons.append("custody_training_lane_access_not_false")
    if custody.get("candidate_development_access") is not False:
        reasons.append(
            "custody_candidate_development_access_not_false"
        )
    if (
        custody.get(
            "hash_only_publication_before_candidate_freeze"
        )
        is not True
    ):
        reasons.append("custody_hash_only_publication_not_true")
    if (
        custody.get("plaintext_sealed_until_candidate_freeze")
        is not True
    ):
        reasons.append("custody_plaintext_seal_not_true")
    if not _sha256(
        custody.get("access_control_receipt_sha256")
    ):
        reasons.append("custody_access_control_receipt_invalid")

    reasons = sorted(set(reasons))
    return {
        "schema": "V10_FINAL_BANK_CUSTODIAN_BINDING_CHECK_V1",
        "status": (
            "CUSTODIAN_BINDING_PASS"
            if not reasons
            else "HOLD"
        ),
        "reasons": reasons,
        "synthetic_actor_ids": sorted(synthetic_ids),
        "human_author_count": len(authors),
        "human_reviewer_count": len(reviewers),
        "custody_surface_id": custody.get("surface_id"),
        "claim_ceiling": (
            "CUSTODIAN_IDENTITY_AND_ACCESS_BINDING_ONLY / "
            "NO_FINAL_PLAINTEXT GENERATED / "
            "NO_BANK_ADMISSION / NO_TRAINING_AUTHORITY"
        ),
    }



def validate_custodian_handoff(
    *,
    binding: dict,
    contract: dict,
    proposal_blob_sha: str,
    generation_spec_sha256: str,
    grader_spec_sha256: str,
    family_manifest_spec_sha256: str | None = None,
    diversity_diagnostic_spec_sha256: str | None = None,
) -> dict:
    reasons: list[str] = []

    binding_check = validate_custodian_binding(binding)
    if binding_check["status"] != "CUSTODIAN_BINDING_PASS":
        reasons.extend(
            "binding:" + reason
            for reason in binding_check["reasons"]
        )

    if not isinstance(contract, dict):
        reasons.append("contract_not_object")
        contract = {}
    if (
        contract.get("schema")
        != "V10_FINAL_BANK_CUSTODIAN_CONTRACT_V1"
    ):
        reasons.append(f"contract_schema:{contract.get('schema')}")
    if (
        contract.get("status")
        != "FROZEN_REQUIREMENTS_NO_CUSTODIANS_BOUND"
    ):
        reasons.append(f"contract_status:{contract.get('status')}")

    contract_subject = contract.get("composition_subject")
    if not isinstance(contract_subject, dict):
        reasons.append("contract_composition_subject_missing")
        contract_subject = {}

    binding_subject = binding.get("composition_subject")
    if not isinstance(binding_subject, dict):
        reasons.append("binding_composition_subject_missing")
        binding_subject = {}

    expected = {
        "proposal_blob_sha": proposal_blob_sha,
        "generation_spec_sha256": generation_spec_sha256,
        "grader_spec_sha256": grader_spec_sha256,
    }
    if "family_manifest_spec_sha256" in contract_subject:
        expected["family_manifest_spec_sha256"] = (
            family_manifest_spec_sha256
        )
    if "diversity_diagnostic_spec_sha256" in contract_subject:
        expected["diversity_diagnostic_spec_sha256"] = (
            diversity_diagnostic_spec_sha256
        )

    for field, actual in (
        ("proposal_blob_sha", proposal_blob_sha),
        ("generation_spec_sha256", generation_spec_sha256),
        ("grader_spec_sha256", grader_spec_sha256),
        (
            "family_manifest_spec_sha256",
            family_manifest_spec_sha256,
        ),
        (
            "diversity_diagnostic_spec_sha256",
            diversity_diagnostic_spec_sha256,
        ),
    ):
        if field not in contract_subject:
            continue
        contract_value = contract_subject.get(field)
        if contract_value != actual:
            reasons.append(
                f"{field}_file_subject_mismatch:"
                f"{actual}!={contract_value}"
            )

    for field, expected_value in expected.items():
        if binding_subject.get(field) != expected_value:
            reasons.append(f"binding_subject_mismatch:{field}")

    if not isinstance(proposal_blob_sha, str) or not _GIT_SHA.fullmatch(
        proposal_blob_sha
    ):
        reasons.append("proposal_blob_sha_invalid")
    if not _sha256(generation_spec_sha256):
        reasons.append("generation_spec_file_sha256_invalid")
    if not _sha256(grader_spec_sha256):
        reasons.append("grader_spec_file_sha256_invalid")
    if (
        "family_manifest_spec_sha256" in contract_subject
        and not _sha256(family_manifest_spec_sha256)
    ):
        reasons.append("family_manifest_spec_file_sha256_invalid")
    if (
        "diversity_diagnostic_spec_sha256" in contract_subject
        and not _sha256(diversity_diagnostic_spec_sha256)
    ):
        reasons.append("diversity_diagnostic_spec_file_sha256_invalid")

    if (
        generation_spec_sha256
        != contract_subject.get("generation_spec_sha256")
    ):
        reasons.append("generation_spec_file_sha256_mismatch")
    if (
        grader_spec_sha256
        != contract_subject.get("grader_spec_sha256")
    ):
        reasons.append("grader_spec_file_sha256_mismatch")
    if (
        "family_manifest_spec_sha256" in contract_subject
        and family_manifest_spec_sha256
        != contract_subject.get("family_manifest_spec_sha256")
    ):
        reasons.append("family_manifest_spec_file_sha256_mismatch")
    if (
        "diversity_diagnostic_spec_sha256" in contract_subject
        and diversity_diagnostic_spec_sha256
        != contract_subject.get("diversity_diagnostic_spec_sha256")
    ):
        reasons.append("diversity_diagnostic_spec_file_sha256_mismatch")

    reasons = sorted(set(reasons))
    return {
        "schema": "V10_FINAL_BANK_CUSTODIAN_HANDOFF_CHECK_V1",
        "status": (
            "HANDOFF_READY_FOR_INDEPENDENT_CUSTODY"
            if not reasons
            else "HOLD"
        ),
        "reasons": reasons,
        "proposal_blob_sha": proposal_blob_sha,
        "generation_spec_sha256": generation_spec_sha256,
        "grader_spec_sha256": grader_spec_sha256,
        "family_manifest_spec_sha256": family_manifest_spec_sha256,
        "diversity_diagnostic_spec_sha256": (
            diversity_diagnostic_spec_sha256
        ),
        "synthetic_actor_ids": binding_check.get(
            "synthetic_actor_ids", []
        ),
        "human_author_count": binding_check.get(
            "human_author_count", 0
        ),
        "human_reviewer_count": binding_check.get(
            "human_reviewer_count", 0
        ),
        "custody_surface_id": binding_check.get(
            "custody_surface_id"
        ),
        "claim_ceiling": (
            "CUSTODIAN_HANDOFF_SUBJECT_AND_IDENTITIES_VERIFIED / "
            "NO_FINAL_PLAINTEXT GENERATED / "
            "NO_BANK ADMISSION / NO TRAINING AUTHORITY"
        ),
    }
