"""No-effect VeraOS local-text model admission preflight, synthetic only."""
from dataclasses import replace
import hashlib
import unittest

from controlled_resident_trial_v1 import (
    ControlledTrialTarget, ServingSession, ArtifactResourceSample,
    evaluate_controlled_trial, prepare_resident_admission_handoff,
    validate_resident_admission_handoff,
)
from test_participating_cognition_v1 import compose
from test_veraos_development_evidence_v1 import VeraOSDevelopmentCutTests
from veraos_development_evidence_v1 import bind_participation_development_cut


class ControlledResidentTrialTests(unittest.TestCase):
    def setUp(self):
        self.bytes=b"tiny synthetic GGUF-shaped test fixture, NOT model weights"
        self.digest=hashlib.sha256(self.bytes).hexdigest()
        self.fixture_nonce_ledger=set()
        self.session=ServingSession(
            host_id="windows:ollama-host", generation=2,
            session_id="host-session:2", nonce="a"*32)
        self.target=ControlledTrialTarget(
            os_source="5ddac1459c3a7b06f251721a14048fc26e60bda0",
            task_id="task:one", model_name="qwen3:4b-instruct",
            model_lineage="QWEN_DEVELOPMENT",
            serving_host_id="windows:ollama-host",
            artifact_sha256=self.digest,
            artifact_size_bytes=len(self.bytes),
            minimum_free_ram_mib=512,
            route_class="COGNITION_ONLY",
            intended_endpoint="http://127.0.0.1:18766/v1")
        self.sample=ArtifactResourceSample(
            session=self.session, observed_at_epoch_s=1800000000,
            sample_id="sample:artifact-byte-readback",
            sha256=self.digest, size_bytes=len(self.bytes),
            cpu_threads=8, available_ram_mib=4096)
        d=VeraOSDevelopmentCutTests()
        d.setUp()
        self.development=d.review()
        self.episode=compose()
        self.cognition=bind_participation_development_cut(
            self.episode,self.development,expected_task_id="task:one")

    def evaluate(self, *, sample=None, before=None, after=None,
                 content=None, target=None, cognition=None, episode=None,
                 now=1800000001, nonce_ledger=None):
        return evaluate_controlled_trial(
            target or self.target,
            before or self.session,
            sample or self.sample,
            after or self.session,
            self.bytes if content is None else content,
            self.development,
            cognition or self.cognition,
            episode=self.episode if episode is None else episode,
            now_epoch_s=now,
            evidence_scope="SYNTHETIC_NO_EFFECT_FIXTURE",
            fixture_nonce_ledger=(self.fixture_nonce_ledger if nonce_ledger is None
                                  else nonce_ledger))

    def test_eligible_synthetic_trial_proposes_os_provider_HELD_not_PASS(self):
        good=self.evaluate()
        proposal=prepare_resident_admission_handoff(self.target,good)
        self.assertEqual(proposal["readiness"],"SYNTHETIC_WIRING_ELIGIBLE")
        self.assertTrue(proposal["synthetic_wiring_eligible"])
        self.assertFalse(proposal["real_resident_admitted"])
        self.assertFalse(proposal["provider_authenticated"])
        self.assertEqual(proposal["provider"]["qualification"]["status"],"HOLD")
        self.assertEqual(proposal["provider"]["admission"]["status"],"HELD")
        self.assertEqual(proposal["provider"]["transport"]["attachment"],
                         "http://127.0.0.1:18766/v1")
        self.assertEqual(proposal["serving_host"],"windows:ollama-host")
        self.assertTrue(validate_resident_admission_handoff(self.target,good,proposal))
        self.assertIn("receiver_owned_one_use_nonce_and_generation",
                      proposal["real_admission_requires"])

    def test_forged_PASS_and_self_claimed_runtime_attestation_rejected(self):
        good=self.evaluate()
        baseline=prepare_resident_admission_handoff(self.target,good)
        attempts=[
            {**baseline,"real_resident_admitted":True},
            {**baseline,"provider_authenticated":True},
            {**baseline,"readiness":"GENUINE_RESIDENT_ADMITTED"},
            {**baseline,"provider":{**baseline["provider"],
               "qualification":{**baseline["provider"]["qualification"],"status":"PASS"},
               "admission":{**baseline["provider"]["admission"],
                            "status":"VERAOS_ROUTE_ELIGIBLE"}}},
            {**baseline,"provider":{**baseline["provider"],
               "qualification":{**baseline["provider"]["qualification"],
                                "subject_sha256":"0"*64}}},
        ]
        for attempt in attempts:
            with self.subTest(field=list(attempt)):
                with self.assertRaises(ValueError):
                    validate_resident_admission_handoff(self.target,good,attempt)

    def test_unverified_real_host_missing_bytes_and_trained_lineage_stay_HELD(self):
        real=evaluate_controlled_trial(
            self.target,self.session,self.sample,self.session,self.bytes,
            self.development,self.cognition,episode=self.episode,
            now_epoch_s=1800000001,evidence_scope="REAL_HOST_UNATTESTED")
        real_proposal=prepare_resident_admission_handoff(self.target,real)
        self.assertEqual(real_proposal["readiness"],"HOLD_UNVERIFIED_REAL_ADMISSION")
        self.assertFalse(real_proposal["synthetic_wiring_eligible"])
        self.assertEqual(real_proposal["provider"]["qualification"]["status"],"HOLD")
        self.assertTrue(validate_resident_admission_handoff(self.target,real,real_proposal))

        absent=self.evaluate(content=None)  # this helper intentionally defaults bytes
        self.assertTrue(absent.source_preflight_accepted)
        no_file=evaluate_controlled_trial(
            self.target,self.session,self.sample,self.session,None,
            self.development,self.cognition,episode=self.episode,
            now_epoch_s=1800000001,evidence_scope="SYNTHETIC_NO_EFFECT_FIXTURE",
            fixture_nonce_ledger=set())
        self.assertIn("SYNTHETIC_ARTIFACT_BYTES_REQUIRED",no_file.holds)
        self.assertFalse(prepare_resident_admission_handoff(
            self.target,no_file)["synthetic_wiring_eligible"])
        renamed=replace(self.target,model_lineage="VERA_TRAINED")
        held=self.evaluate(target=renamed)
        self.assertFalse(prepare_resident_admission_handoff(
            renamed,held)["real_resident_admitted"])

    def test_synthetic_own_host_artifact_bytes_and_cognition_accepts_preflight_only(self):
        r=self.evaluate()
        self.assertEqual(r.status,"ACCEPT_SYNTHETIC_DEV_PREFLIGHT")
        self.assertEqual(r.source_model_lineage,"QWEN_DEVELOPMENT")
        self.assertEqual(r.target_host,"windows:ollama-host")
        self.assertTrue(r.source_preflight_accepted)
        self.assertFalse(r.model_identity_authenticated)
        self.assertFalse(r.normal_resident_admitted)
        self.assertFalse(r.inference_executed)
        self.assertFalse(r.effect_authority)
        self.assertFalse(r.trained_vera_model_proven)

    def test_changes_in_host_generation_session_nonce_fail_closed(self):
        for changed in (
            replace(self.session,host_id="guest:qemu"),
            replace(self.session,host_id="vn1:executor"),
            replace(self.session,generation=3),
            replace(self.session,session_id="host-session:3"),
            replace(self.session,nonce="b"*32),
        ):
            with self.subTest(changed=changed):
                r=self.evaluate(after=changed)
                self.assertEqual(r.status,"HOLD_SOURCE_PREFLIGHT")
                self.assertIn("SERVING_SESSION_CHANGED",r.holds)

    def test_artifact_bytes_hash_metadata_and_resource_substitution_fail(self):
        for sample,content,expected_hold in (
            (self.sample,b"edited model bytes", "ARTIFACT_BYTES_MISMATCH"),
            (replace(self.sample,sha256="0"*64),self.bytes,
             "ARTIFACT_BYTES_MISMATCH"),
            (replace(self.sample,size_bytes=len(self.bytes)+1),self.bytes,
             "ARTIFACT_BYTES_MISMATCH"),
            (replace(self.sample,available_ram_mib=128),self.bytes,
             "INSUFFICIENT_HOST_RAM"),
            (replace(self.sample,session=replace(self.session,generation=3)),self.bytes,
             "SERVING_SESSION_CHANGED"),
            (replace(self.sample,observed_at_epoch_s=1799999600),self.bytes,
             "SAMPLE_OUTSIDE_WINDOW"),
        ):
            with self.subTest(hold=expected_hold):
                r=self.evaluate(sample=sample,content=content)
                self.assertEqual(r.status,"HOLD_SOURCE_PREFLIGHT")
                self.assertIn(expected_hold,r.holds)

    def test_forged_vera_trained_identity_task_and_effect_cannot_admit(self):
        forged=replace(self.target,model_lineage="VERA_TRAINED")
        self.assertIn("MODEL_LINEAGE_NOT_QUALIFIED",self.evaluate(target=forged).holds)
        forged=replace(self.target,task_id="task:other")
        self.assertIn("COGNITION_TASK_MISMATCH",self.evaluate(target=forged).holds)
        forged_cognition={**self.cognition,"effect_authority":True}
        self.assertIn("COGNITION_NOT_SOURCE_ONLY",
                      self.evaluate(cognition=forged_cognition).holds)

    def test_cognition_digest_participants_and_semantics_cannot_be_relabelled(self):
        for altered in (
            {**self.cognition,"participants":["agent:vn1","agent:forged"]},
            {**self.cognition,"source_cognition_sha256":"f"*64},
            {**self.cognition,"development_subject":"f"*40},
        ):
            r=self.evaluate(cognition=altered)
            self.assertEqual(r.status,"HOLD_SOURCE_PREFLIGHT")
            self.assertIn("COGNITION_RECEIPT_RELABELED",r.holds)
        changed=self.episode.as_dict()
        changed["core_semantic_plan"]["fidelity"]="LOSSY"
        result=self.evaluate(episode=changed)
        self.assertIn("COGNITION_RECEIPT_RELABELED",result.holds)

    def test_replayed_nonce_or_sample_id_is_not_accepted_twice_in_fixture(self):
        self.assertEqual(self.evaluate().status,"ACCEPT_SYNTHETIC_DEV_PREFLIGHT")
        same=self.evaluate()
        self.assertEqual(same.status,"HOLD_SOURCE_PREFLIGHT")
        self.assertIn("REPLAYED_FIXTURE_NONCE_OR_SAMPLE",same.holds)
        new_nonce=replace(self.session,nonce="b"*32)
        reused_sample=replace(self.sample,session=new_nonce)
        denial=self.evaluate(sample=reused_sample,before=new_nonce,after=new_nonce)
        self.assertEqual(denial.status,"HOLD_SOURCE_PREFLIGHT")
        self.assertIn("REPLAYED_FIXTURE_NONCE_OR_SAMPLE",denial.holds)

    def test_fixture_without_replay_ledger_cannot_accept(self):
        r=self.evaluate(nonce_ledger=False)
        self.assertEqual(r.status,"HOLD_SOURCE_PREFLIGHT")
        self.assertIn("FIXTURE_NONCE_LEDGER_MISSING",r.holds)

    def test_production_or_live_mode_always_held_even_when_shapes_match(self):
        r=evaluate_controlled_trial(
            self.target,self.session,self.sample,self.session,self.bytes,
            self.development,self.cognition,episode=self.episode,
            now_epoch_s=1800000001,
            evidence_scope="REAL_HOST_UNATTESTED")
        self.assertEqual(r.status,"HOLD_SOURCE_PREFLIGHT")
        self.assertIn("HOST_CAPTURE_NOT_RECEIVER_AUTHENTICATED",r.holds)
        self.assertFalse(r.normal_resident_admitted)


if __name__=="__main__":
    unittest.main()
