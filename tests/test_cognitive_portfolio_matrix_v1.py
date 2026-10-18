"""Frozen 96-repo catalog matrix integrity and no silent runtime promotions."""
import json
from pathlib import Path
from build_cognitive_portfolio_matrix_v1 import make

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs/evidence/TRAINING_96_COGNITIVE_CAPABILITY_MATRIX_V1.json"
SOURCE = "6b097eaed2451175f33fdbee157a3a1bcc9f204078c48857efddb3758c907c65"


def matrix():
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def test_exact_96_rows_11_roles_immutable_source_digest():
    data = matrix()
    rows = data["rows"]
    assert data["source_catalog_sha256"] == SOURCE
    assert data["source_commit"] == "162b38e2180b447838ad5a6b51dffd99a77c2276"
    assert len(rows) == data["repository_count"] == 96
    assert len({r["repository"] for r in rows}) == 96
    assert sum(data["role_counts"].values()) == 96
    assert len(data["role_counts"]) == 11


def test_no_catalog_row_self_certifies_installation_or_effect():
    for row in matrix()["rows"]:
        assert row["source_default_head"]
        assert row["existing_mechanism"] and row["required_interface_and_gate"]
        assert row["independent_test_of_this_repo"] is False
        assert row["sibling_runtime_import_required"] is False
        status = row["runtime_admission"]
        assert status["installed"] == "UNVERIFIED"
        assert status["selected_route"] == "UNVERIFIED"
        assert status["live_consumption"] == "UNVERIFIED"
        assert status["effect_authority"] == "NONE_GRANTED"


def test_canonical_selected_and_research_stay_separate():
    r = {x["repository"]: x for x in matrix()["rows"]}
    assert r["vera-mono"]["disposition"] == "CANONICAL_SOURCE_OWNER_INSTALLATION_UNPROVEN"
    assert r["vera-os"]["disposition"] == "CURRENT_OS_HOST_SOURCE_NOT_GUEST_QUALIFIED"
    assert r["pre-active"]["disposition"] == "CURRENT_SELECTED_SOURCE_STAGE"
    assert r["volition"]["disposition"] == "CURRENT_SELECTED_SOURCE_STAGE"
    assert r["rezon"]["disposition"] == "REJECT_AUTO_PROVENANCE_ADMISSION_ADVISORY_ONLY"
    assert r["model-training"]["disposition"] == "SOURCE_EVALUATION_ONLY_NO_WEIGHT_TRAINING"
    assert r["cricket"]["cognitive_role"] == "CRITIQUE_SAFETY"
    assert r["mosaic"]["cognitive_role"] == "TRAINING_MODEL_PLACEMENT"
    assert r["vision"]["cognitive_role"] == "PERCEPTION"
    assert r["brigit"]["disposition"] == "SEPARATE_AUTHORITY"


def test_saved_rows_regenerate_from_exact_private_catalog_only_if_supplied(tmp_path):
    # Optional exact-source regeneration; ordinary regression is self-contained.
    import os
    path = os.environ.get("VERA_PORTAL_379_CATALOG")
    if path:
        source = make(Path(path))
        assert source == matrix()
