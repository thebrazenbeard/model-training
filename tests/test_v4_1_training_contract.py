from successor import build_v4_1_hf_rehearsal as rehearsal
from successor import build_v4_1_training_corpus as training


def test_v41_rehearsal_pool_is_42500() -> None:
    assert rehearsal.EXPECTED_ROWS == 42_500
    assert sum(rehearsal.TARGETS.values()) == 42_500
    assert rehearsal.TARGETS["smoltalk_smollm3_explore_instruct_rewriting_no_think"] == 7_500
    assert rehearsal.TARGETS["smoltalk_smollm3_smol_rewrite_no_think"] == 6_500
    assert rehearsal.TARGETS["tulu_3_sft_personas_instruction_following_no_think"] == 7_500
    assert rehearsal.TARGETS["table_gpt_no_think"] == 5_000


def test_v41_training_arithmetic() -> None:
    assert training.EXPECTED_TRAIN_ROWS == 50_000
    assert training.EXPECTED_VALIDATION_ROWS == 2_500
    assert training.CUSTOM_VALIDATION_PER_FAMILY == 50
    assert training.REHEARSAL_VALIDATION_ROWS == 2_000
    assert training.EXPECTED_TRAIN_ROWS + training.EXPECTED_VALIDATION_ROWS == 52_500

def test_v41_training_manifest_tracks_custom_corpus_lineage() -> None:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "successor" / "corpus" / "v4_1"
    custom_manifest = json.loads(
        (root / "custom" / "manifest.json").read_text(encoding="utf-8")
    )
    training_manifest = json.loads(
        (root / "training_manifest.json").read_text(encoding="utf-8")
    )

    expected_training_id = custom_manifest["corpus_id"].replace(
        "10K_DIVERSE_CORE", "50K"
    )
    assert training_manifest["corpus_id"] == expected_training_id
    assert training_manifest["custom_source"]["corpus_id"] == custom_manifest["corpus_id"]
    assert training_manifest["custom_source"]["manifest_digest"] == custom_manifest["manifest_digest"]
    assert training_manifest["custom_source"]["rows"] == custom_manifest["rows"]
    assert training_manifest["custom_source"]["families"] == {
        family: meta["rows"]
        for family, meta in custom_manifest["families"].items()
    }

