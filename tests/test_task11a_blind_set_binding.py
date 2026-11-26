from successor.vera_lab.promotion import PromotionEvidence, _blind_reasons


def test_blind_gate_requires_attested_evaluation_set_digest():
    candidate = "a" * 64
    evidence = PromotionEvidence(
        candidate_digest=candidate,
        blind={
            "candidate_digest": candidate,
            "set_digest": "b" * 64,
            "evaluation_set_digest": "c" * 64,
            "item_count": 77,
            "family_counts": {},
            "critical_failure_ids": [],
        },
        vera_lab={},
        regression={},
        reviews=(),
    )

    reasons = _blind_reasons(evidence)

    assert "blind_evaluation_set_digest_mismatch" in reasons
