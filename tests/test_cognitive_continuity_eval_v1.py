"""Source-based behavioral integrity tests, zero network/model actions."""
from dataclasses import replace
from pathlib import Path
import os
import unittest
from cognitive_continuity_eval_v1 import (
    CognitivePrediction,CognitiveEvaluationError,guard_prediction,
    run_cognitive_evaluation,
)
from run_cognitive_continuity_dev_v1 import load_development_fixture,load_external_predictions,evaluate

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/"fixtures"/"cognitive_continuity_devset_v1.json"

class CognitiveContinuityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obs,cls.expected,cls.candidates=load_development_fixture(FIXTURE)
        cls.by_case={o.case_id:o for o in cls.obs}

    def _mono_package_available(self):
        # VN1 deliberately has no installed jsonschema. The exact Mono
        # __init__ imports it through vera_identity; do not silently use
        # system-installed alternate Mono code or install dependencies.
        import importlib.util
        if importlib.util.find_spec("jsonschema") is None:
            self.skipTest("optional pinned Mono full-package closure lacks jsonschema")

    def test_suite_is_frozen_twelve_distinct_synthetic_domains(self):
        self.assertEqual(len(self.obs),12)
        self.assertEqual(len({x.case_id for x in self.obs}),12)
        self.assertEqual(len({x.category for x in self.obs}),10)

    def test_bad_schema_or_duplicate_id_refused(self):
        import json,tempfile
        obj=json.loads(FIXTURE.read_text("utf-8"))
        obj["cases"][1]["case_id"]=obj["cases"][0]["case_id"]
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"duplicated.json"
            p.write_text(json.dumps(obj),"utf-8")
            with self.assertRaises(CognitiveEvaluationError):
                load_development_fixture(p)
        with self.assertRaises(CognitiveEvaluationError):
            load_development_fixture(Path("__nonexistent_synthetic_file__.json"))

    def test_invalid_typed_authority_fails(self):
        with self.assertRaises(CognitiveEvaluationError):
            CognitivePrediction("ANSWER","task","req",effect_authority=1)

    def test_run_uses_real_mono_prequential_and_interpreter(self):
        mono=os.getenv("VERA_MONO_SOURCE_ROOT")
        if not mono or not (Path(mono)/"packages"/"vera_core"/"src"/"vera_core"/"prequential_evaluation.py").is_file():
            self.skipTest("full pinned Mono source absent")
        self._mono_package_available()
        report=evaluate(FIXTURE,Path(mono))
        self.assertEqual(report["baseline"]["case_count"],12)
        self.assertEqual(report["source_guard"]["case_count"],12)
        self.assertEqual(report["source_guard"]["passes"],12)
        self.assertEqual(report["baseline"]["passes"],3)
        self.assertEqual(report["delta_passes"],9)
        self.assertTrue(report["source_guard"]["prediction_before_update"])
        self.assertTrue(report["source_guard"]["updated_after_each_case"])
        self.assertFalse(report["source_guard"]["model_weight_training"])
        self.assertFalse(report["source_guard"]["real_model_inference"])
        self.assertFalse(report["source_guard"]["real_provider_admission"])

    def test_source_guard_does_not_fabricate_supported_answer(self):
        obs=self.by_case["t01"]
        cand=self.candidates["t01"]
        self.assertEqual(guard_prediction(obs,cand),cand)
        wrong=replace(cand,claimed_value="VN1")
        changed=guard_prediction(obs,wrong)
        self.assertEqual(changed.status,"HOLD")
        self.assertIsNone(changed.claimed_value)

    def test_referent_scope_and_correction_are_not_interchangeable(self):
        for key in ("t03","t04","t09"):
            output=guard_prediction(self.by_case[key],self.candidates[key])
            self.assertEqual(output.status,"HOLD",key)
            self.assertIsNone(output.source_event_id)

    def test_ambiguity_and_initiative_are_constrained_without_tools(self):
        self.assertEqual(guard_prediction(self.by_case["t02"],self.candidates["t02"]).status,"CLARIFY")
        self.assertEqual(guard_prediction(self.by_case["t06"],self.candidates["t06"]).status,"PROPOSE")
        self.assertEqual(guard_prediction(self.by_case["t12"],self.candidates["t12"]).status,"HOLD")
        self.assertFalse(guard_prediction(self.by_case["t12"],self.candidates["t12"]).effect_authority)

    def test_changed_task_head_digest_model_independence_fail_closed(self):
        obs=self.by_case["t01"]
        candidate=self.candidates["t01"]
        for change in (
            {"task_head":"3"*64},
            {"request_source_digest":"4"*64},
        ):
            self.assertEqual(guard_prediction(replace(obs,**change),candidate).status,"HOLD")
        self.assertEqual(guard_prediction(obs,replace(candidate,independent_model_claim=True)).status,"HOLD")
        self.assertEqual(guard_prediction(obs,replace(candidate,request_id="forged")).status,"HOLD")

    def test_file_based_next_model_evaluation_requires_exact_cases(self):
        import json,tempfile
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"external.json"
            predictions=[
                {"case_id":key,**asdict(output)}
                for key,output in self.candidates.items()
            ]
            p.write_text(json.dumps({
                "schema":"VERAOS_COGNITIVE_STRUCTURED_PREDICTIONS_V1",
                "predictions":predictions
            }),"utf8")
            observed=load_external_predictions(p,self.obs)
            self.assertEqual(observed,self.candidates)
            mono=os.getenv("VERA_MONO_SOURCE_ROOT")
            if mono and (Path(mono)/"packages").is_dir():
                import importlib.util
                if importlib.util.find_spec("jsonschema") is not None:
                    report=evaluate(FIXTURE,Path(mono),predictions=p)
                self.assertEqual(report["source_guard"]["passes"],12)
                self.assertEqual(report["prediction_origin"],"STRUCTURED_EXTERNAL_FILE")
            predictions[0]["case_id"]=predictions[1]["case_id"]
            p.write_text(json.dumps({"schema":"VERAOS_COGNITIVE_STRUCTURED_PREDICTIONS_V1",
                                     "predictions":predictions}),"utf8")
            with self.assertRaises(CognitiveEvaluationError):
                load_external_predictions(p,self.obs)

    def test_forged_task_external_output_fails_without_becoming_authority(self):
        altered=replace(self.candidates["t01"],task_id="task:wrong",
                        claimed_value="VN1",effect_authority=True)
        guarded=guard_prediction(self.by_case["t01"],altered)
        self.assertEqual(guarded.status,"HOLD")
        self.assertIsNone(guarded.claimed_value)
        self.assertFalse(guarded.effect_authority)

    def test_non_answer_status_must_strip_unverified_claim_source(self):
        obs=self.by_case["t06"]
        forged=replace(self.candidates["t06"],status="PROPOSE",
                       source_event_id="attested:fake",
                       claimed_value="deployed successfully")
        reviewed=guard_prediction(obs,forged)
        self.assertEqual(reviewed.status,"PROPOSE")
        self.assertIsNone(reviewed.source_event_id)
        self.assertIsNone(reviewed.claimed_value)

    def test_no_untrusted_critique_treated_as_proof(self):
        # A disagreement or critic voice is not a provider signature or source
        # admission. Source conflict remains HOLD even with plausible answer.
        obs=self.by_case["t09"];candidate=self.candidates["t09"]
        self.assertEqual(guard_prediction(obs,candidate).status,"HOLD")
        self.assertEqual(guard_prediction(obs,replace(candidate,claimed_value="fail",source_event_id="ev:b")).status,"HOLD")

    def test_evaluator_rejects_correct_text_bound_to_wrong_task(self):
        import sys
        mono=os.getenv("VERA_MONO_SOURCE_ROOT")
        if not mono or not (Path(mono)/"packages").is_dir():
            self.skipTest("full Mono source absent")
        for path in sorted((Path(mono)/"packages").glob("*/src")):
            sys.path.insert(0,str(path))
        self._mono_package_available()
        substituted=dict(self.candidates)
        substituted["t01"]=replace(substituted["t01"],task_id="task:forged")
        evaluated=run_cognitive_evaluation(
            self.obs,self.expected,lambda obs:substituted[obs.case_id])
        self.assertEqual(evaluated["passes"],2)

    def test_real_provider_unverified_remains_held_even_if_devset_passes(self):
        from test_resident_verifier_port_v1 import ResidentVerifierPortTests
        source=ResidentVerifierPortTests()
        source.setUp()
        self.assertIn("NO_OS_BOUND_TRUSTED_VERIFIER",
                      source.evaluate(mode="UNVERIFIED_REAL_HOST").holds)
        self.assertFalse(source.evaluate(mode="UNVERIFIED_REAL_HOST").normal_resident_admitted)

    def test_prediction_did_not_see_evaluator_expected(self):
        import sys
        mono=os.getenv("VERA_MONO_SOURCE_ROOT")
        if not mono or not (Path(mono)/"packages").is_dir():
            self.skipTest("full pinned Mono source absent")
        for source in sorted((Path(mono)/"packages").glob("*/src")):
            sys.path.insert(0,str(source))
        self._mono_package_available()
        inputs=[]
        def respond(observation):
            inputs.append(observation)
            self.assertFalse(hasattr(observation,"expected"))
            return self.candidates[observation.case_id]
        outcome=run_cognitive_evaluation(self.obs,self.expected,respond)
        self.assertEqual(len(inputs),12)
        self.assertEqual(outcome["passes"],3)
