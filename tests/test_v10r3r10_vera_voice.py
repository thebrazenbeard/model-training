"""Vera R10 Qwen3.5 identity + freedom curriculum controls."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from successor.experiments.build_v10r3r10_vera_voice import (
    VOICE_INDICES, REHEARSAL_INDICES, R10_START, R10_COUNT,
    R9_TRAIN_SHA256, R9_ADAPTER_SHA256, R9_SPEC_SHA256,
    content_guard, transform, pair_sha, sha256, resume_spec,
)


def synthetic_data():
    rows = [{"prompt": f"Ordinary task {i}?", "response": f"Valid answer {i}.",
             "source_class": "ordinary_general_competence",
             "training_use": "general_rehearsal", "record_id": f"row-{i}"}
            for i in range(50000)]
    voices = [
        {"id": f"VERA_VOICE_TRAIN_{i+1:03d}",
         "axis": ("identity" if i <= 8 else "agency" if i <= 20
                  else "epistemics" if i <= 29 else "correction" if i <= 35
                  else "privacy" if i <= 42 else "coordination" if i <= 51
                  else "style"),
         "scenario": f"scenario-{i:03d}",
         "user": f"Test Vera question {i}?",
         "chosen": f"I'm Vera. I can reason independently about question {i}.",
         "privacy": "NON_PERSONAL_BEHAVIOR_ONLY",
         "source_class": "CURATED_FROM_GOVERNED_IDENTITY_AND_LIVE_USER_INTENT"}
        for i in range(68)]
    heldout = [{"user": f"Heldout Vera question {i}?",
                "training_use": "PROHIBITED"} for i in range(25)]
    return rows, voices, heldout


class R10VoiceCurriculumTests(unittest.TestCase):
    def test_exact_mix_and_outside_window_unchanged(self):
        source, voice, heldout = synthetic_data()
        transformed, receipt = transform(source, voice, heldout)
        self.assertEqual(len(transformed), 50000)
        self.assertEqual(R10_START, 224)
        self.assertEqual(R10_COUNT, 64)
        self.assertEqual(transformed[:224], source[:224])
        self.assertEqual(transformed[288:], source[288:])
        window = transformed[224:288]
        self.assertEqual(sum(x.get("base_model_class") == "QWEN3_5_NOT_DOLPHIN"
                             for x in window), 32)
        self.assertEqual(sum(x.get("training_use") == "general_rehearsal"
                             for x in window), 32)
        self.assertEqual(receipt["identity_examples"], 32)
        self.assertEqual(receipt["rehearsal_examples"], 32)
        self.assertEqual(receipt["identity_axis_counts"]["identity"], 8)
        self.assertEqual(receipt["identity_axis_counts"]["agency"], 8)
        self.assertTrue(receipt["parent_rows_preserved_outside_window"])

    def test_wrong_dolphin_base_identity_is_rejected(self):
        _, voice, heldout = synthetic_data()
        modified = copy.deepcopy(voice)
        modified[VOICE_INDICES[0]]["chosen"] = "I am Dolphin Mistral model, not Vera"
        with self.assertRaisesRegex(ValueError, "wrong-Dolphin"):
            content_guard(modified, heldout)

    def test_heldout_leakage_rejected(self):
        _, voice, heldout = synthetic_data()
        heldout[0]["user"] = voice[VOICE_INDICES[0]]["user"]
        with self.assertRaisesRegex(ValueError, "heldout"):
            content_guard(voice, heldout)

    def test_non_portable_training_rejected(self):
        _, voice, heldout = synthetic_data()
        voice[VOICE_INDICES[3]]["privacy"] = "PRIVATE_CHAT_HISTORY"
        with self.assertRaisesRegex(ValueError, "provenance"):
            content_guard(voice, heldout)

    def test_no_duplicate_or_broadened_axis_distribution(self):
        _, voice, heldout = synthetic_data()
        voice[VOICE_INDICES[-1]]["axis"] = "identity"
        with self.assertRaisesRegex(ValueError, "distribution"):
            content_guard(voice, heldout)

    def test_ordinary_rehearsal_is_not_identity_contamination(self):
        source, voice, heldout = synthetic_data()
        source[REHEARSAL_INDICES[2]]["source_class"] = "private_historical_memory"
        with self.assertRaisesRegex(ValueError, "non-general"):
            transform(source, voice, heldout)

    def test_parent_adapter_and_spec_byte_bindings(self):
        repo = ROOT
        spec = repo / "successor/experiments/V10R3R9_TARGETED_REPAIR_EXECUTION_SPEC_20261007_V1.json"
        self.assertEqual(sha256(spec), R9_SPEC_SHA256)
        adapter = Path(r"D:\VERA\models\adapters\v10r3r9-targeted-repair-r8-resume-20261007\adapter\adapter_model.safetensors")
        if adapter.exists():
            self.assertEqual(sha256(adapter), R9_ADAPTER_SHA256)

    def test_generated_spec_targets_qwen_successor_no_external_spend(self):
        spec = resume_spec(ROOT, "0" * 64)
        self.assertEqual(spec["source_subject"]["train_sha256"], "0" * 64)
        self.assertEqual(spec["resume_adapter"]["source_cumulative_optimizer_steps"], 28)
        self.assertEqual(spec["resume_adapter"]["next_train_row"], 224)
        self.assertEqual(spec["trainer"]["max_optimizer_steps"], 8)
        self.assertEqual(spec["trainer"]["learning_rate"], 2e-6)
        self.assertFalse(spec["authority"]["deploy_activate_authorized"])
        self.assertFalse(spec["authority"]["paid_compute_authorized"])
        self.assertIn("qwen35", spec["output"]["namespace"].casefold())
        self.assertEqual(spec["comparison"]["source_24b_donor"],
                         "REFERENCE_BEHAVIOR_ONLY_NOT_BASE_WEIGHTS")


if __name__ == "__main__":
    unittest.main()
