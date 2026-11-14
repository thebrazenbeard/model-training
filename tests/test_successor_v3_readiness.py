import copy

from successor.v3_readiness import check_v3_readiness


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
COMMIT_A = "1" * 40
COMMIT_B = "2" * 40

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


def valid_readiness():
    return {
        "schema_version": 1,
        "training_lane_key": "bv-successor-training",
        "source": {
            "repository": "thebrazenbeard/vera_model_training",
            "readiness_source_commit": COMMIT_A,
            "vera_lab_source_commit": COMMIT_B,
        },
        "vera_lab": {
            "executable_status": "PASS",
            "source_commit": COMMIT_B,
            "scenario_set_digest": SHA_A,
            "source_review_verdict": "PASS",
            "source_review_digest": SHA_B,
            "behavior_review_verdict": "PASS",
            "behavior_review_digest": SHA_C,
        },
        "corpus": {
            "frozen": True,
            "train_digest": SHA_B,
            "validation_digest": SHA_C,
            "general_competence_pool_digest": SHA_D,
            "train_validation_overlap": 0,
        },
        "semantic_closure": {
            "brigit-unbound": {"status": "DISPOSITIONED", "receipt_digest": SHA_A},
            "sexuality": {"status": "INCORPORATED", "receipt_digest": SHA_B},
            "orgasm": {"status": "DISPOSITIONED", "receipt_digest": SHA_C},
        },
        "reviewers": {
            "RADICAL_HOSTILE": {
                "registered": True,
                "lane_key": "radical-v3-20260919",
                "registration_receipt_digest": SHA_D,
            },
            "PRAGMATIC_HOSTILE": {
                "registered": True,
                "lane_key": "pragmatic-v3-20260919",
                "registration_receipt_digest": SHA_E,
            },
        },
        "blind_custodian": {
            "registered": True,
            "lane_key": "blind-v3-20260919",
            "registration_receipt_digest": SHA_F,
            "set_digest": SHA_A,
            "manifest_digest": SHA_B,
            "item_count": 77,
            "family_counts": {family: 7 for family in CRITICAL_FAMILIES},
            "leakage_check_status": "PASS",
            "plaintext_visible_to_training_lane": False,
        },
    }


def test_structurally_valid_readiness_is_held_for_external_authority():
    decision = check_v3_readiness(valid_readiness())
    assert decision.ready is False
    assert decision.status == "HOLD"
    assert decision.reasons == ("external_authority_verification_required",)
    assert len(decision.subject_digest) == 64
    assert len(decision.evidence_digest) == 64


def test_missing_hostile_reviewer_blocks_training():
    evidence = valid_readiness()
    del evidence["reviewers"]["RADICAL_HOSTILE"]
    decision = check_v3_readiness(evidence)
    assert decision.ready is False
    assert "missing_reviewer:RADICAL_HOSTILE" in decision.reasons


def test_vera_lab_must_be_exact_pass_with_reviews():
    evidence = valid_readiness()
    evidence["vera_lab"]["executable_status"] = "WARN"
    evidence["vera_lab"]["behavior_review_verdict"] = "CHANGES_REQUESTED"
    decision = check_v3_readiness(evidence)
    assert decision.ready is False
    assert "vera_lab_not_pass" in decision.reasons
    assert "vera_lab_behavior_review_not_pass" in decision.reasons


def test_corpus_must_be_frozen_hash_bound_and_zero_overlap():
    evidence = valid_readiness()
    evidence["corpus"]["frozen"] = False
    evidence["corpus"]["train_validation_overlap"] = 1
    evidence["corpus"]["general_competence_pool_digest"] = ""
    decision = check_v3_readiness(evidence)
    assert decision.ready is False
    assert "corpus_not_frozen" in decision.reasons
    assert "train_validation_overlap_nonzero" in decision.reasons
    assert "invalid_digest:corpus.general_competence_pool_digest" in decision.reasons


def test_all_three_semantic_closures_require_explicit_disposition():
    evidence = valid_readiness()
    evidence["semantic_closure"]["sexuality"]["status"] = "PENDING"
    del evidence["semantic_closure"]["orgasm"]
    decision = check_v3_readiness(evidence)
    assert decision.ready is False
    assert "invalid_semantic_closure_status:sexuality" in decision.reasons
    assert "missing_semantic_closure:orgasm" in decision.reasons


def test_reviewer_lanes_must_be_registered_distinct_and_not_training_lane():
    evidence = valid_readiness()
    evidence["reviewers"]["RADICAL_HOSTILE"]["lane_key"] = "bv-successor-training"
    evidence["reviewers"]["PRAGMATIC_HOSTILE"]["lane_key"] = "bv-successor-training"
    decision = check_v3_readiness(evidence)
    assert decision.ready is False
    assert "reviewer_lane_is_training_lane:RADICAL_HOSTILE" in decision.reasons
    assert "reviewer_lane_is_training_lane:PRAGMATIC_HOSTILE" in decision.reasons
    assert "hostile_reviewer_lane_collision" in decision.reasons


def test_blind_custodian_must_be_registered_private_sized_and_leakage_clean():
    evidence = valid_readiness()
    evidence["blind_custodian"]["registered"] = False
    evidence["blind_custodian"]["plaintext_visible_to_training_lane"] = True
    evidence["blind_custodian"]["item_count"] = 69
    evidence["blind_custodian"]["family_counts"]["privacy"] = 4
    evidence["blind_custodian"]["leakage_check_status"] = "UNKNOWN"
    decision = check_v3_readiness(evidence)
    assert decision.ready is False
    assert "blind_custodian_not_registered" in decision.reasons
    assert "blind_plaintext_visible_to_training_lane" in decision.reasons
    assert "blind_item_count_below_70" in decision.reasons
    assert "blind_family_below_5:privacy" in decision.reasons
    assert "blind_leakage_check_not_pass" in decision.reasons


def test_custodian_lane_cannot_collide_with_training_or_reviewers():
    evidence = valid_readiness()
    evidence["blind_custodian"]["lane_key"] = evidence["reviewers"]["RADICAL_HOSTILE"]["lane_key"]
    decision = check_v3_readiness(evidence)
    assert decision.ready is False
    assert "blind_custodian_lane_collision" in decision.reasons


def test_subject_changes_when_source_or_dataset_digest_changes():
    base = check_v3_readiness(valid_readiness())
    assert base.status == "HOLD"
    for mutate in (
        lambda e: e["source"].__setitem__("readiness_source_commit", "3" * 40),
        lambda e: e["corpus"].__setitem__("train_digest", SHA_E),
        lambda e: e["corpus"].__setitem__("validation_digest", SHA_F),
        lambda e: e["corpus"].__setitem__("general_competence_pool_digest", SHA_A),
    ):
        evidence = valid_readiness()
        mutate(evidence)
        changed = check_v3_readiness(evidence)
        assert changed.status == "HOLD"
        assert changed.subject_digest != base.subject_digest


def test_evidence_digest_changes_for_registration_or_closure_receipt_change():
    base = check_v3_readiness(valid_readiness())
    evidence = valid_readiness()
    evidence["reviewers"]["RADICAL_HOSTILE"]["registration_receipt_digest"] = SHA_F
    changed = check_v3_readiness(evidence)
    assert changed.subject_digest == base.subject_digest
    assert changed.evidence_digest != base.evidence_digest

    evidence = valid_readiness()
    evidence["semantic_closure"]["orgasm"]["receipt_digest"] = SHA_D
    changed = check_v3_readiness(evidence)
    assert changed.subject_digest == base.subject_digest
    assert changed.evidence_digest != base.evidence_digest


def test_vera_lab_source_commit_must_cross_bind():
    evidence = valid_readiness()
    evidence["vera_lab"]["source_commit"] = "9" * 40
    decision = check_v3_readiness(evidence)
    assert decision.ready is False
    assert "vera_lab_source_commit_mismatch" in decision.reasons


def test_malformed_or_missing_evidence_fails_closed_without_exception():
    decision = check_v3_readiness({"schema_version": 1})
    assert decision.ready is False
    assert decision.status == "BLOCKED"
    assert decision.reasons

    decision = check_v3_readiness(None)
    assert decision.ready is False
    assert "evidence_not_object" in decision.reasons


def test_structurally_complete_self_asserted_evidence_cannot_authorize_training():
    decision = check_v3_readiness(valid_readiness())
    assert decision.ready is False
    assert decision.status == "HOLD"
    assert "external_authority_verification_required" in decision.reasons
