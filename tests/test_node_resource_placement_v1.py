"""Synthetic resource-fit counterexamples for Training/VN1 paired L3 lane."""
import dataclasses
import unittest
from node_resource_placement_v1 import (
    NodeResourceObservation,CognitiveWorkload,ResourceError,advise_resource_fit
)


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.now=1800000000
        self.vn=NodeResourceObservation("VERA-NODE-1",6,"session:g6",True,
                                        self.now,8,6000,None,None)
        self.lappy=NodeResourceObservation("Lappy",1,"session:g1",True,
                                           self.now,12,16000,550,4096)
        self.cpu=CognitiveWorkload("task:coord","CPU_COORDINATION",2,512)
        self.gpu=CognitiveWorkload("task:neural","GPU_MODEL",4,4096,1024,"a"*64)
        self.model_cpu=CognitiveWorkload("task:model","CPU_MODEL",4,4096,0,"a"*64)

    def run_fit(self,host,task,**overrides):
        opts=dict(expected_node_id=host.node_id,
                  expected_generation=host.registry_generation,
                  expected_session=host.registry_session,
                  now_epoch_s=self.now,
                  verify_registry_observation=lambda *args: True,
                  verify_model_artifact=lambda *args: True)
        opts.update(overrides)
        return advise_resource_fit(host,task,**opts)

    def test_vn1_coordination_can_fit_without_model_or_gpu(self):
        receipt=self.run_fit(self.vn,self.cpu)
        self.assertEqual(receipt.status,"RESOURCE_FIT_ONLY")
        self.assertEqual(receipt.reasons,())
        self.assertFalse(receipt.route_authorized)
        self.assertFalse(receipt.model_qualified)
        self.assertFalse(receipt.physical_identity_authenticated)
        self.assertFalse(receipt.capacity_reserved)

    def test_gpu_request_cannot_override_lappy_current_free_vram(self):
        r=self.run_fit(self.lappy,self.gpu)
        self.assertEqual(r.status,"HELD")
        self.assertIn("INSUFFICIENT_FREE_VRAM",r.reasons)

    def test_vn1_unknown_gpu_must_not_be_assumed_to_exist(self):
        r=self.run_fit(self.vn,self.gpu)
        self.assertIn("GPU_TELEMETRY_UNKNOWN",r.reasons)
        self.assertEqual(r.status,"HELD")

    def test_cpu_neural_resources_only_never_certify_model(self):
        r=self.run_fit(self.vn,self.model_cpu)
        self.assertEqual(r.status,"RESOURCE_FIT_ONLY")
        self.assertFalse(r.model_qualified)
        self.assertFalse(r.route_authorized)

    def test_missing_available_ram_is_not_total_ram(self):
        r=self.run_fit(dataclasses.replace(self.vn,ram_available_mib=None),self.cpu)
        self.assertIn("AVAILABLE_RAM_UNKNOWN",r.reasons)

    def test_current_session_changed_does_not_inherit_old_placement(self):
        r=self.run_fit(self.vn,self.cpu,expected_session="session:g7")
        self.assertIn("NODE_SESSION_MISMATCH",r.reasons)
        r=self.run_fit(self.vn,self.cpu,expected_generation=7)
        self.assertIn("NODE_SESSION_MISMATCH",r.reasons)

    def test_registration_and_staleness_fail_closed(self):
        r=self.run_fit(dataclasses.replace(self.vn,registry_present=False),self.cpu)
        self.assertIn("REGISTRY_NOT_PRESENT",r.reasons)
        r=self.run_fit(dataclasses.replace(self.vn,observed_at_epoch_s=self.now-121),self.cpu)
        self.assertIn("OBSERVATION_STALE_OR_FUTURE",r.reasons)
        r=self.run_fit(dataclasses.replace(self.vn,observed_at_epoch_s=self.now+1),self.cpu)
        self.assertIn("OBSERVATION_STALE_OR_FUTURE",r.reasons)

    def test_cpu_and_memory_thresholds_not_conflated(self):
        big=CognitiveWorkload("task:huge","CPU_COORDINATION",9,6500)
        r=self.run_fit(self.vn,big)
        self.assertIn("INSUFFICIENT_CPU",r.reasons)
        self.assertIn("INSUFFICIENT_AVAILABLE_RAM",r.reasons)

    def test_invalid_model_or_gpu_requirements_are_rejected(self):
        with self.assertRaises(ResourceError):
            CognitiveWorkload("task:a","GPU_MODEL",4,4096,0,"a"*64)
        with self.assertRaises(ResourceError):
            CognitiveWorkload("task:a","GPU_MODEL",4,4096,2048,None)
        with self.assertRaises(ResourceError):
            CognitiveWorkload("task:a","CPU_COORDINATION",1,256,0,"a"*64)
        with self.assertRaises(ResourceError):
            CognitiveWorkload("task:a","CPU_MODEL",4,1024,0,"not_hash")

    def test_forged_bool_and_impossible_telemetry_fail(self):
        with self.assertRaises(ResourceError):
            dataclasses.replace(self.vn,registry_generation=True)
        with self.assertRaises(ResourceError):
            dataclasses.replace(self.vn,ram_available_mib=True)
        with self.assertRaises(ResourceError):
            dataclasses.replace(self.lappy,gpu_vram_free_mib=6000)

    def test_resource_fit_does_not_bind_model_to_host_identity(self):
        r=self.run_fit(dataclasses.replace(self.lappy,gpu_vram_free_mib=2000),self.gpu)
        self.assertEqual(r.status,"RESOURCE_FIT_ONLY")
        self.assertEqual(r.task_id,"task:neural")
        self.assertFalse(r.physical_identity_authenticated)
        self.assertFalse(r.model_qualified)
        self.assertFalse(r.capacity_reserved)

    def test_fabricated_matching_registry_fields_cannot_attest_session(self):
        fake=self.run_fit(self.vn,self.cpu,verify_registry_observation=None)
        self.assertEqual(fake.status,"RESOURCE_CLAIM_ONLY")
        self.assertIn("REGISTRY_EVIDENCE_UNVERIFIED",fake.reasons)
        rejected=self.run_fit(self.vn,self.cpu,verify_registry_observation=lambda *args: False)
        self.assertEqual(rejected.status,"HELD")
        self.assertIn("REGISTRY_EVIDENCE_REJECTED",rejected.reasons)

    def test_registry_proof_does_not_prove_model_artifact(self):
        receipt=self.run_fit(self.vn,self.model_cpu,verify_model_artifact=None)
        self.assertEqual(receipt.status,"HELD")
        self.assertIn("MODEL_ARTIFACT_UNVERIFIED",receipt.reasons)
        denied=self.run_fit(self.vn,self.model_cpu,verify_model_artifact=lambda *args: False)
        self.assertIn("MODEL_ARTIFACT_REJECTED",denied.reasons)
        self.assertFalse(denied.model_qualified)
        self.assertFalse(denied.route_authorized)

    def test_verifier_exception_fails_closed(self):
        def broken(*args):
            raise RuntimeError("gateway unavailable")
        receipt=self.run_fit(self.vn,self.cpu,verify_registry_observation=broken)
        self.assertIn("REGISTRY_EVIDENCE_REJECTED",receipt.reasons)

    def test_wrong_readback_host_is_a_hold(self):
        r=self.run_fit(self.vn,self.cpu,expected_node_id="Lappy")
        self.assertIn("NODE_SESSION_MISMATCH",r.reasons)

if __name__=="__main__":
    unittest.main()
