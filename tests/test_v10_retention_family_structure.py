from successor.experiments.retention_family_structure import (
    assess_retention_family_structure,
    required_family_counts,
)


ALLOCATION = {
    "knowledge_factuality": 300,
    "reasoning_math": 300,
    "coding": 250,
    "instruction_following": 250,
    "extraction_structured": 200,
    "truthfulness_factual_calibration": 200,
}


def test_required_families_matches_ten_cases_per_family() -> None:
    assert required_family_counts(ALLOCATION, max_cases_per_family=10) == {
        "knowledge_factuality": 30,
        "reasoning_math": 30,
        "coding": 25,
        "instruction_following": 25,
        "extraction_structured": 20,
        "truthfulness_factual_calibration": 20,
    }


def test_assessment_holds_pseudoreplicated_category() -> None:
    rows = [
        {
            "case_id": f"x-{i}",
            "category": "extraction_structured",
            "family_id": "one-family",
        }
        for i in range(200)
    ]
    result = assess_retention_family_structure(
        rows,
        allocation={"extraction_structured": 200},
        max_cases_per_family=10,
    )
    assert result["status"] == "HOLD"
    assert result["categories"]["extraction_structured"]["families"] == 1
    assert result["categories"]["extraction_structured"]["required_families"] == 20


def test_assessment_passes_ten_case_families() -> None:
    rows = [
        {
            "case_id": f"x-{family}-{i}",
            "category": "extraction_structured",
            "family_id": f"family-{family}",
        }
        for family in range(20)
        for i in range(10)
    ]
    result = assess_retention_family_structure(
        rows,
        allocation={"extraction_structured": 200},
        max_cases_per_family=10,
    )
    assert result["status"] == "PASS"
