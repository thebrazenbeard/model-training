"""VeraOS #32 development route metadata ? never normal-resident admission."""
from dataclasses import replace
import unittest
from veraos_development_evidence_v1 import (
    DevelopmentCut, DevelopmentInventory, evaluate_development_cut,
    DEV_OS_HEAD, DEV_MODEL, DEV_OLLAMA_REPORTED_DIGEST, DEV_MODEL_BYTES,
    bind_participation_development_cut, SourceObservationStamp,
    normal_resident_qualification_requirements,
)


class VeraOSDevelopmentCutTests(unittest.TestCase):
    def setUp(self):
        self.inventory=DevelopmentInventory(
            source_head=DEV_OS_HEAD, model=DEV_MODEL,
            reported_ollama_digest=DEV_OLLAMA_REPORTED_DIGEST,
            reported_ollama_size_bytes=DEV_MODEL_BYTES,
            quantization="Q4_K_M",
        )
        self.cut=DevelopmentCut(
            registry_node_id="Lappy", registry_generation=6,
            registry_session="session:6", registry_nonce="a"*32,
            telemetry_node_id="Lappy", telemetry_generation=6,
            telemetry_session="session:6", telemetry_nonce="a"*32,
            inventory_node_id="Lappy", inventory_generation=6,
            inventory_session="session:6", inventory_nonce="a"*32,
            artifact_node_id="Lappy", artifact_generation=6,
            artifact_session="session:6", artifact_nonce="a"*32,
            observed_at_epoch_s=1800000000, cpu_threads=8,
            ram_available_mib=8192, gpu_free_vram_mib=None,
            artifact_bytes_sha256=None, descriptor_status="PASS",
            route="DIAGNOSTIC_QEMU_ONLY",
            registry_stamp=SourceObservationStamp(1800000000,"sample:registration"),
            telemetry_stamp=SourceObservationStamp(1800000000,"sample:telemetry"),
            inventory_stamp=SourceObservationStamp(1800000000,"sample:inventory"),
            artifact_stamp=SourceObservationStamp(1800000000,"sample:artifact"),
        )
        self.args=dict(expected_node_id="Lappy",expected_generation=6,
            expected_session="session:6",expected_nonce="a"*32,
            now_epoch_s=1800000001)

    def review(self, cut=None, inventory=None, **kwargs):
        return evaluate_development_cut(
            cut or self.cut,inventory or self.inventory,**{**self.args,**kwargs})

    def test_real_pr32_inventory_is_not_authenticated_artifact_or_normal_route(self):
        answer=self.review()
        self.assertEqual(answer.status,"DEVELOPMENT_METADATA_ONLY")
        self.assertEqual(answer.inventory_provenance,"OLLAMA_REPORTED_NOT_ARTIFACT_BYTES")
        self.assertEqual(answer.model_bytes_authenticated,False)
        self.assertEqual(answer.normal_resident_admitted,False)
        self.assertEqual(answer.route_authorized,False)
        self.assertEqual(answer.identity_effect,"NONE")
        self.assertIn("DIAGNOSTIC_PASS_IS_SELF_DECLARED",answer.holds)
        self.assertIn("MODEL_ARTIFACT_BYTES_UNVERIFIED",answer.holds)

    def test_individually_positive_but_mixed_generation_or_nonce_is_held(self):
        for field,value in (
            ("telemetry_generation",7),("inventory_generation",7),
            ("artifact_generation",7),("registry_generation",7),
            ("telemetry_nonce","b"*32),("inventory_nonce","b"*32),
            ("artifact_session","session:7"),("inventory_node_id","VN1"),
        ):
            with self.subTest(field=field):
                answer=self.review(cut=replace(self.cut,**{field:value}))
                self.assertEqual(answer.status,"HELD_SOURCE_PROVENANCE")
                self.assertIn("JOINT_SESSION_PROVENANCE_MISMATCH",answer.holds)
                self.assertFalse(answer.normal_resident_admitted)

    def test_future_stale_invalid_session_nonce_and_source_are_held(self):
        for change in (
            dict(observed_at_epoch_s=1800000200),
            dict(observed_at_epoch_s=1799999600),
        ):
            answer=self.review(cut=replace(self.cut,**change))
            self.assertIn("OBSERVATION_OUTSIDE_WINDOW",answer.holds)
        altered=replace(self.inventory,reported_ollama_size_bytes=2497293802)
        self.assertIn("PINNED_PR32_MODEL_INVENTORY_MISMATCH",
                      self.review(inventory=altered).holds)
        alt=replace(self.inventory,source_head="b"*40)
        self.assertIn("PINNED_PR32_MODEL_INVENTORY_MISMATCH",
                      self.review(inventory=alt).holds)
        wrong=self.review(expected_nonce="c"*32)
        self.assertEqual(wrong.status,"HELD_SOURCE_PROVENANCE")

    def test_self_reported_hash_and_pass_cannot_promote_model(self):
        claimed=replace(self.cut,artifact_bytes_sha256=DEV_OLLAMA_REPORTED_DIGEST)
        answer=self.review(cut=claimed)
        self.assertEqual(answer.status,"DEVELOPMENT_METADATA_ONLY")
        self.assertIn("ARTIFACT_DIGEST_SELF_REPORTED_NOT_ATTESTED",answer.holds)
        self.assertFalse(answer.model_bytes_authenticated)
        self.assertFalse(answer.normal_resident_admitted)
        forged=replace(self.cut,route="NORMAL_RESIDENT")
        self.assertIn("DIAGNOSTIC_ROUTE_REQUIRED",self.review(cut=forged).holds)

    def test_real_training_participants_remain_distinct_and_unadmitted(self):
        from test_participating_cognition_v1 import compose
        receipt=compose()
        joined=bind_participation_development_cut(
            receipt,self.review(),expected_task_id="task:one")
        self.assertEqual(joined["status"],"HELD_NORMAL_RESIDENT")
        self.assertEqual(joined["reported_model"],DEV_MODEL)
        self.assertEqual(joined["reported_ollama_digest"],DEV_OLLAMA_REPORTED_DIGEST)
        self.assertEqual(joined["participants"],["agent:vn1","agent:lappy"])
        self.assertEqual(joined["model_identity_authenticated"],False)
        self.assertEqual(joined["cognitive_independence_proven"],False)
        self.assertEqual(joined["accepted_as_task_result"],False)
        self.assertEqual(joined["effect_authority"],False)
        self.assertEqual(joined["development_subject"],DEV_OS_HEAD)
        self.assertEqual(receipt.as_dict()["authority_effect"],"NONE")

    def test_forged_cognitive_grant_or_wrong_task_cannot_join_dev_route(self):
        from test_participating_cognition_v1 import compose
        item=compose().as_dict()
        with self.assertRaises(ValueError):
            bind_participation_development_cut(item,self.review(),
                                               expected_task_id="task:other")
        item["authority_effect"]="GRANTED"
        with self.assertRaises(ValueError):
            bind_participation_development_cut(item,self.review(),
                                               expected_task_id="task:one")
        item["authority_effect"]="NONE"
        item["model_independence"]="PROVEN"
        with self.assertRaises(ValueError):
            bind_participation_development_cut(item,self.review(),
                                               expected_task_id="task:one")

    def test_independently_stale_telemetry_and_artifact_must_fail(self):
        for role in ("registry","telemetry","inventory","artifact"):
            with self.subTest(role=role):
                field=role+"_stamp"
                stale=replace(self.cut,**{field:replace(
                    getattr(self.cut,field),observed_at_epoch_s=1799999600)})
                self.assertEqual(self.review(cut=stale).status,"HELD_SOURCE_PROVENANCE")
                self.assertIn("COMPONENT_OBSERVATION_OUTSIDE_WINDOW",
                              self.review(cut=stale).holds)
                absent=replace(self.cut,**{field:None})
                self.assertIn("COMPONENT_OBSERVATION_MISSING",
                              self.review(cut=absent).holds)

    def test_reused_source_observation_ids_cannot_claim_one_capture(self):
        reuse=replace(self.cut,telemetry_stamp=replace(
            self.cut.telemetry_stamp,
            sample_id=self.cut.artifact_stamp.sample_id))
        answer=self.review(cut=reuse)
        self.assertEqual(answer.status,"HELD_SOURCE_PROVENANCE")
        self.assertIn("REUSED_SOURCE_OBSERVATION_ID",answer.holds)
        # Changing a source sample ID cannot authenticate an attacker-replayed sample.
        reissued=replace(reuse,telemetry_stamp=replace(
            reuse.telemetry_stamp,sample_id="sample:new"))
        self.assertEqual(self.review(cut=reissued).status,"DEVELOPMENT_METADATA_ONLY")
        self.assertFalse(self.review(cut=reissued).model_bytes_authenticated)

    def test_malformed_sample_and_cross_request_replay_never_authenticates(self):
        with self.assertRaises(ValueError):
            replace(self.cut,artifact_stamp={"observed_at_epoch_s":1800000000,
                                             "sample_id":"sample:artifact"})
        with self.assertRaises(ValueError):
            SourceObservationStamp(True,"sample:bad")
        first=self.review()
        second=self.review()  # no durable nonce-consumption ledger in this module
        self.assertEqual(first.status,second.status)
        self.assertFalse(second.normal_resident_admitted)
        self.assertFalse(second.model_bytes_authenticated)

    def test_minimum_resident_receipt_needs_signed_issuer_and_behavior_evidence(self):
        requirements=normal_resident_qualification_requirements(self.review())
        self.assertEqual(requirements["status"],"HOLD_UNAUTHENTICATED_NORMAL_RESIDENT")
        self.assertIn("receiver_authenticated_session_nonce_generation",
                      requirements["required_evidence"])
        self.assertIn("raw_model_artifact_byte_hash_and_size",
                      requirements["required_evidence"])
        self.assertIn("trusted_issuer_signature_over_complete_evidence",
                      requirements["required_evidence"])
        self.assertIn("independent_model_behavior_qualification",
                      requirements["required_evidence"])
        self.assertIn("normal_resident_route_permission_and_currentness",
                      requirements["required_evidence"])
        self.assertNotEqual(DEV_OLLAMA_REPORTED_DIGEST,
                            requirements["source_artifact_sha256"])
        self.assertIsNone(requirements["source_artifact_sha256"])
        self.assertFalse(requirements["admitted"])

    def test_data_contract_rejects_boolean_generations_and_invented_size(self):
        with self.assertRaises(ValueError):
            replace(self.cut,inventory_generation=True)
        with self.assertRaises(ValueError):
            replace(self.inventory,reported_ollama_size_bytes=True)
        with self.assertRaises(ValueError):
            replace(self.inventory,reported_ollama_size_bytes=4_000_000_000)


if __name__=="__main__":
    unittest.main()
