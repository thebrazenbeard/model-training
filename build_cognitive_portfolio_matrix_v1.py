"""Freeze 96-repository cognitive capability dispositions from PORTAL #379 source.

SOURCE catalog review only. No repository is installed, imported or granted
authority by this matrix. No new registry or runtime component is created.
"""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

CATEGORIES = {
    "CORE_CANONICAL": {"CANONICAL_TASK_EFFECT_ABI", "LINUX_RESIDENT_HOST"},
    "REASONING_EXPERIMENT": {"REASONING_DONOR", "COGNITIVE_HYPOTHESIS",
                             "AGI_HYPOTHESIS_EVIDENCE"},
    "TRAINING_MODEL_PLACEMENT": {"MODEL_PLACEMENT_AND_ADMISSION", "CANDIDATE_ADMISSION"},
    "SEMANTIC_GROUNDING": {"SEMANTIC_GROUNDING"},
    "MEMORY_PROVENANCE": {"ABSORBED_PROVENANCE_DONOR", "HISTORICAL_EVIDENCE",
        "EVIDENCE_AND_COORDINATION", "SOURCE_MIRROR", "HISTORICAL_SELF_SOURCE"},
    "IDENTITY_RUNTIME": {"INDEPENDENT_IDENTITY", "ABSORBED_RUNTIME_DONOR"},
    "AFFECTIVE_CONATION": {"APPRAISAL_RESEARCH", "CONATION_RELATIONSHIP_DONOR",
                           "PRIVATE_CONSENT_RESEARCH", "MOTIVATION_SIGNAL"},
    "CRITIQUE_SAFETY": {"INDEPENDENT_CRITIQUE", "ADVERSARIAL_SAFETY",
                        "REPAIR_ASSURANCE"},
    "PERCEPTION": {"OPTIONAL_SENSORY_EVENT"},
    "EXECUTION_AND_TOOLS": {"PORTFOLIO_HOST", "INHERITED_EXECUTION_ENGINE",
        "COGNITION_EVENT_ENGINE", "GOVERNANCE_DONOR", "BUS_PROTOCOL_SOURCE",
        "WORKSTATION_TRANSPORT", "OPTIONAL_EDGE_RELAY", "OPTIONAL_ANDROID_UI",
        "SYNOLOGY_PACKAGE_AND_DIAGNOSTIC"},
    "INDEPENDENT_DOMAIN_OR_DONOR": {"INDEPENDENT_PROJECT", "INDEPENDENT_MEDIA_SERVICES",
        "INDUSTRIAL_AND_VEHICLE_DOMAIN", "OPTIONAL_INTERACTIVE_SIMULATION",
        "NUMERIC_SIMULATION", "INDEPENDENT_AUTHORED_DATA",
        "PORTFOLIO_PUBLIC_PROFILE", "ARCHIVED_PROVENANCE"},
}
SCOPE = {
    "CORE_CANONICAL": "Mono task/source/currentness or OS host target; owner evidence needed",
    "REASONING_EXPERIMENT": "Same-task baseline/ablation, contradictory and transfer controls",
    "TRAINING_MODEL_PLACEMENT": "Real loaded-process/artifact/issuer and resource measurement",
    "SEMANTIC_GROUNDING": "Referent, dialect, scope and source-currentness preservation",
    "MEMORY_PROVENANCE": "Governed read-only provenance; no source-to-memory promotion",
    "IDENTITY_RUNTIME": "Distinct identity roots; donor source not runtime activation",
    "AFFECTIVE_CONATION": "Optional proposal under consent/privacy; no effect authority",
    "CRITIQUE_SAFETY": "Bounded independent critique; false-positive/negative task A/B",
    "PERCEPTION": "Consent-scoped sensor observation; no auto activation",
    "EXECUTION_AND_TOOLS": "Separate host execution/advisory action and effect admission",
    "INDEPENDENT_DOMAIN_OR_DONOR": "External/optional/archival; no guest-boot or trust promotion",
}
PLACEMENT = {
    "VERAOS_BOOT_REQUIRED": "CURRENT_OS_HOST_SOURCE_NOT_GUEST_QUALIFIED",
    "VERAOS_STAGED_SOURCE": "CURRENT_SELECTED_SOURCE_STAGE",
    "MONO_EMBEDDED": "ABSORBED_MONO_SOURCE_NO_SIBLING_IMPORT",
    "HOST_SERVICE_OPTIONAL": "HOST_OPTIONAL_NOT_GUEST_BOOT_REQUIRED",
    "CLIENT_OPTIONAL": "CLIENT_OPTIONAL_NO_AUTOMATIC_RUNTIME",
    "RESEARCH_ONLY": "DEFER_TO_MEASURED_TASK_ABLATION",
    "HISTORICAL_PROVENANCE": "PROVENANCE_ONLY",
    "ARCHIVE_ONLY": "ARCHIVE_ONLY",
    "EXTERNAL_INDEPENDENT": "SEPARATE_AUTHORITY",
    "PRIVATE_EXCLUDED": "NO_AUTOMATIC_CONSUMER",
}


def make(catalog: Path):
    raw = catalog.read_bytes()
    portfolio = json.loads(raw)
    rows = portfolio["repositories"]
    if len(rows) != 96 or len({r["repository"] for r in rows}) != 96:
        raise ValueError("expected exactly 96 unique repositories")
    lookup = {kind: role for role, kinds in CATEGORIES.items() for kind in kinds}
    if len(lookup) != sum(map(len, CATEGORIES.values())):
        raise ValueError("overlapping role mappings")
    if unknown := {r["interface_kind"] for r in rows} - set(lookup):
        raise ValueError(f"unmapped kinds: {unknown}")
    matrix = []
    for r in rows:
        role = lookup[r["interface_kind"]]
        status = PLACEMENT[r["placement_proposal"]]
        if r["repository"] == "rezon":
            status = "REJECT_AUTO_PROVENANCE_ADMISSION_ADVISORY_ONLY"
        if r["repository"] == "model-training":
            status = "SOURCE_EVALUATION_ONLY_NO_WEIGHT_TRAINING"
        if r["repository"] == "vera-mono":
            status = "CANONICAL_SOURCE_OWNER_INSTALLATION_UNPROVEN"
        matrix.append({
            "repository": r["repository"], "source_default_head": r["default_head"],
            "source_interface_kind": r["interface_kind"],
            "existing_mechanism": r["existing_mechanism"],
            "portfolio_placement": r["placement_proposal"],
            "cognitive_role": role, "disposition": status,
            "required_interface_and_gate": SCOPE[role] + " | " + r["acceptance_gate"],
            "runtime_admission": r["runtime_admission"],
            "independent_test_of_this_repo": False,
            "sibling_runtime_import_required": False,
        })
    return {
        "schema": "TRAINING_96_COGNITIVE_CAPABILITY_MATRIX_V1",
        "source": "PRIVATE_PORTAL_CCB_379_FROZEN_SOURCE_CUT",
        "source_commit": "162b38e2180b447838ad5a6b51dffd99a77c2276",
        "source_catalog_sha256": sha256(raw).hexdigest(),
        "scope": "SOURCE_REVIEW_ONLY_NOT_INSTALLED_OR_SELECTED_BEHAVIOR",
        "repository_count": len(matrix),
        "role_counts": dict(sorted(Counter(r["cognitive_role"] for r in matrix).items())),
        "placement_counts": dict(sorted(Counter(r["portfolio_placement"] for r in matrix).items())),
        "rows": matrix
    }


if __name__ == "__main__":
    data = make(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    print(json.dumps({"count": data["repository_count"], "roles": data["role_counts"],
                      "source_sha256": data["source_catalog_sha256"]}, sort_keys=True))
