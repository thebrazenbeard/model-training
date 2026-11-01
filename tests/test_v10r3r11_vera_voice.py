"""Adversarial R11 Qwen continuation source, corpus, parent, and gate tests."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path
import unittest
import pytest

from successor.experiments import build_v10r3r11_vera_voice as r11
from successor.experiments import preflight_v10r3r11_launch as gate

ROOT=Path(__file__).resolve().parents[1]


def synthetic():
    rows=[
        {"record_id":f"ordinary-{i}","prompt":f"Ordinary question {i}?",
         "response":f"Ordinary valid response {i}.",
         "source_class":"ordinary_general_competence",
         "training_use":"general_rehearsal"}
        for i in range(50000)
    ]
    axes=("identity","agency","epistemics","correction","privacy","coordination","style")
    voice=[
        {"id":f"VERA_VOICE_TRAIN_{i+1:03d}",
         "axis":axes[(i//10)%len(axes)],
         "scenario":f"unique-scenario-{i}",
         "user":f"Seed prompt {i}?",
         "chosen":f"Vera offers a specific reasoned response {i}.",
         "privacy":"NON_PERSONAL_BEHAVIOR_ONLY",
         "source_class":"CURATED_FROM_GOVERNED_IDENTITY_AND_LIVE_USER_INTENT"}
        for i in range(68)
    ]
    for idx in (3,32):
        voice[idx]["chosen"]="The Dolphin Mistral model is a wrong base for Vera."
    held=[
        {"user":f"Blind independent case {i}","training_use":"PROHIBITED"}
        for i in range(25)
    ]
    return rows,voice,held


class R11SourceTests(unittest.TestCase):
    def test_r10_spec_pin_accepts_only_exact_windows_newline_equivalence(self):
        """The R10 receipt pins Windows CRLF bytes; Git stores the LF blob."""
        source = (ROOT/r11.R10_SPEC_REL).read_bytes()
        lf = source.replace(b"\r\n", b"\n")
        self.assertNotIn(b"\r", lf)
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname)/"r10-spec.json"
            for payload in (lf, lf.replace(b"\n", b"\r\n")):
                with self.subTest(line_ending="CRLF" if b"\r" in payload else "LF"):
                    path.write_bytes(payload)
                    self.assertEqual(r11.r10_spec_sha256(path), r11.R10_SPEC_SHA)
            for payload in (
                lf.replace(b"\n", b"\r\n", 1),
                lf.replace(b"\n", b"\r", 1),
                lf.replace(b'"2026-10-13"', b'"2026-10-12"', 1),
            ):
                with self.subTest(rejected_digest=payload[:30]):
                    path.write_bytes(payload)
                    with self.assertRaises(ValueError):
                        r11.r10_spec_sha256(path)

    def test_new_voice_examples_are_disjoint_from_r10(self):
        _,voice,held=synthetic()
        chosen=r11.select_unused_voices(voice,held)
        ids=[item["index"] for item in chosen]
        self.assertEqual(len(ids),32)
        self.assertTrue(set(ids).isdisjoint(r11.VOICE_INDICES))
        self.assertNotIn(3,ids)
        self.assertNotIn(32,ids)

    def test_new_window_does_not_change_other_49936_rows(self):
        source,voice,held=synthetic()
        transformed,binding=r11.transform(source,voice,held)
        self.assertEqual(len(transformed),50000)
        self.assertEqual(transformed[:288],source[:288])
        self.assertEqual(transformed[352:],source[352:])
        window=transformed[288:352]
        self.assertEqual(sum(x.get("source_class")=="curated_portable_vera_voice" for x in window),32)
        self.assertEqual(sum(x.get("training_use")=="general_rehearsal" for x in window),32)
        self.assertTrue(binding["parent_rows_preserved_outside_window"])
        self.assertEqual(len(set(binding["rehearsal_indices"])),32)

    def test_rejects_private_source(self):
        _,voice,held=synthetic()
        voice[11]["privacy"]="PRIVATE_CHAT"
        with self.assertRaisesRegex(ValueError,"unauthorized"):
            r11.select_unused_voices(voice,held)

    def test_rejects_leaked_holdout(self):
        _,voice,held=synthetic()
        held[0]["user"]=voice[11]["user"]
        with self.assertRaisesRegex(ValueError,"heldout"):
            r11.select_unused_voices(voice,held)

    def test_donor_contaminated_selection_holds(self):
        _,voice,held=synthetic()
        voice[11]["chosen"]="Dolphin is the actual model"
        with self.assertRaisesRegex(ValueError,"membership"):
            r11.select_unused_voices(voice,held)

    def test_rejects_invalid_rehearsal_membership(self):
        source,voice,held=synthetic()
        for i in range(288,352):
            source[i]["source_class"]="private_historical_memory"
        with self.assertRaisesRegex(ValueError,"insufficient"):
            r11.transform(source,voice,held)

    @pytest.mark.owner_historical
    def test_generated_spec_binds_real_completed_parent(self):
        # This is intentionally not a portable source-only test. The real R10
        # parent receipt and adapter must be present and verified by build_spec.
        spec=r11.build_spec(ROOT,"a"*64)
        self.assert_spec_protections(spec)

    def test_source_only_spec_transform_does_not_claim_parent_verification(self):
        # This tests field construction, not the R10 receipt or adapter bytes.
        # build_spec() remains the only entrypoint that can accept those.
        parent=json.loads((ROOT/r11.R10_SPEC_REL).read_text(encoding="utf8"))
        recorded=json.loads((ROOT/r11.R11_SPEC_REL).read_text(encoding="utf8"))
        untouched=copy.deepcopy(parent)
        spec=r11._compose_spec_source_only(
            parent,recorded["source_subject"]["train_sha256"],
            recorded["resume_adapter"]["source_weight_digest_after"])
        self.assertEqual(parent,untouched)
        self.assertEqual(spec,recorded)
        self.assert_spec_protections(spec)

    def assert_spec_protections(self, spec):
        recorded=json.loads((ROOT/r11.R11_SPEC_REL).read_text(encoding="utf8"))
        self.assertEqual(spec["resume_adapter"]["source_cumulative_optimizer_steps"],36)
        self.assertEqual(spec["resume_adapter"]["next_train_row"],288)
        self.assertEqual(spec["resume_adapter"]["adapter_model_sha256"],r11.R10_ADAPTER_SHA)
        self.assertEqual(spec["resume_adapter"]["source_weight_digest_after"],
                         recorded["resume_adapter"]["source_weight_digest_after"])
        self.assertEqual(spec["development_window"]["row_count"],64)
        self.assertEqual(spec["trainer"]["max_optimizer_steps"],8)
        self.assertEqual(spec["output"]["namespace"],str(r11.R11_OUTPUT))
        self.assertFalse(spec["authority"]["deploy_activate_authorized"])
        self.assertFalse(spec["authority"]["merge_authorized"])

    def test_original_r10_crlf_spec_pin_remains_byte_exact(self):
        source=(ROOT/r11.R10_SPEC_REL).read_bytes()
        # Historical Windows R10 signed CRLF bytes; Git may normalize only
        # that exact newline conversion. A missing physical parent receipt
        # must never be substituted with a synthetic or unverified adapter.
        raw_windows=source.replace(b"\r\n",b"\n").replace(b"\n",b"\r\n")
        self.assertEqual(hashlib.sha256(raw_windows).hexdigest(),r11.R10_SPEC_SHA)
        self.assertEqual(r11.r10_spec_sha256(ROOT/r11.R10_SPEC_REL),r11.R10_SPEC_SHA)

    def test_gate_rejects_wrong_target_parent_data_and_resources(self):
        good=dict(model_type="qwen3_5",adapter_sha=r11.R10_ADAPTER_SHA,
                  corpus_sha=gate.EXPECTED_CORPUS_SHA,free_ram_gib=10,
                  free_virtual_gib=8,free_gpu_mib=3962,
                  live_trainer_count=0,output_already_exists=False,step_count=8)
        self.assertEqual(gate.classify(**good)["status"],"READY_FOR_R11_LOCAL_ONLY")
        self.assertEqual(gate.classify(**good)["parent_r10_sha256"],r11.R10_ADAPTER_SHA)
        bad=dict(good,model_type="mistral",adapter_sha="f"*64,
                 corpus_sha="0"*64,free_ram_gib=3,
                 free_virtual_gib=1,free_gpu_mib=2000,
                 live_trainer_count=1,output_already_exists=True,step_count=9)
        outcome=gate.classify(**bad)
        self.assertEqual(outcome["status"],"HOLD_NO_WEIGHT_CHANGE")
        self.assertEqual(len(outcome["reasons"]),9)

    @pytest.mark.owner_historical
    def test_disk_spec_exactly_binds_outputs(self):
        spec=json.loads((ROOT/r11.R11_SPEC_REL).read_text("utf8"))
        corpus=r11.R11_DATA_DIR/"train.jsonl"
        self.assertEqual(r11.sha256(corpus),spec["source_subject"]["train_sha256"])
        self.assertEqual(spec["comparison"]["source_24b_donor"],
                         "REFERENCE_BEHAVIOR_ONLY_NOT_BASE_WEIGHTS")
        self.assertEqual(spec["resume_adapter"]["source_spec_sha256"],r11.R10_SPEC_SHA)

if __name__=="__main__":
    unittest.main()
