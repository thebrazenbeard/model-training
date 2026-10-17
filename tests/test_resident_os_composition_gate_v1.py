"""Real Mono QualifiedVeraRuntime/trust/tasks in disposable state; zero inference."""
from dataclasses import replace
from hashlib import sha256
import hmac
import os
from pathlib import Path
import sys
import tempfile
import unittest

from resident_os_composition_gate_v1 import (
    QualifiedRuntimeModelGate, ScopedResidentTurn,
    ResidentSourceHold, canonical_turn, make_held_resident_conversation_handler,
)
from resident_verifier_port_v1 import SignedHostEvidence
from test_resident_verifier_port_v1 import LocalPort


class TestHostPort(LocalPort):
    def reobserve(self, request):
        return {
            **super().reobserve(request),
            "loaded_process_pid":4242,
            "model_source":"SERVING_PROCESS_ATTESTATION",
        }


class ActualMonoQualifiedRuntimeGateTests(unittest.TestCase):
    def setUp(self):
        mono=os.getenv("VERA_MONO_SOURCE_ROOT")
        if not mono or not (Path(mono)/"packages/vera_core/src/vera_core/qualified_runtime.py").is_file():
            self.skipTest("exact Vera Mono source-only checkout absent")
        for path in sorted((Path(mono)/"packages").glob("*/src")):
            sys.path.insert(0,str(path))
        from vera_core import VeraStateDirectory, QualifiedVeraRuntime
        from vera_core.task_execution import TaskPacket
        self.temp=tempfile.TemporaryDirectory(prefix="TRAINING-MONO-RESIDENT-SOURCE-")
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self._release)
        state=VeraStateDirectory(Path(self.temp.name)/"vera-state",
            project_id="veraos",identity_id="VERA")
        from test_resident_verifier_port_v1 import ResidentVerifierPortTests
        fixture=ResidentVerifierPortTests()
        fixture.setUp()
        self.port=TestHostPort()
        self.request=fixture.request
        state.outbound_trust_registry().register(
            authority_id="os-owned:local-verifier",
            role="PROVIDER",provider_id=self.request.provider_id,
            key_id="fixture-key",key_digest="2"*64,
            expected_registry_generation=0)
        self.runtime=QualifiedVeraRuntime.from_state_directory(state)
        task_packet=TaskPacket(
            purpose="Synthetic Qwen source boundary proof",
            subject="Bounded local question?",
            completion_state="Synthetic source evidence only; no model execution",
            evidence_requirements=("task provenance","host verification"),
            writable_scope=(),
            non_targets=("network model inference","protected effects"),
            forbidden_shortcuts_or_effects=("test token is not trusted admission",),
            priority_order=("task source","host","scope"),
            unknowns=("model loading unverified",),
            return_shape=("HOLD",),
            relevant_surfaces=("source",),
        )
        task_state=self.runtime.tasks.open_task(
            self.request.task_id,task_packet,
            lifecycle_evidence_digest=self.request.task_digest)
        self.turn=ScopedResidentTurn(
            model_request=self.request,request_id="ui:resident:1",
            task_text=task_packet.subject,source_digest=self.request.task_digest,
            expected_task_head=task_state.journal_head_digest)
        self.signed=self.sign(self.turn)
        self.gate=QualifiedRuntimeModelGate(self.runtime,self.port)

    def _release(self):
        # Current Mono state holds SQLite connections, which on Windows can
        # stay live until runtime wrappers are collected.
        import gc
        self.gate=None
        self.runtime=None
        self.port=None
        gc.collect()

    def sign(self,turn):
        return SignedHostEvidence(
            issuer_id="os-owned:local-verifier",
            key_id="fixture-key",key_digest="2"*64,
            signature=hmac.new(self.port.secret,canonical_turn(turn),"sha256").hexdigest())

    def grant(self,turn=None,signed=None):
        return self.gate.reserve_source_trial(
            turn or self.turn, signed or self.signed,
            now_epoch_s=1800000010)

    def test_source_trial_consumes_actual_task_cas_no_provider_grant(self):
        lease=self.grant()
        self.assertEqual(lease.status,"SCOPED_SOURCE_TRIAL_RESERVED")
        self.assertNotEqual(lease.issued_checkpoint_head,self.turn.expected_task_head)
        self.assertFalse(lease.network_dispatch_authorized)
        self.assertFalse(lease.real_model_admitted)
        self.assertFalse(lease.protected_effect_authority)
        events=self.runtime.tasks.events(self.request.task_id)
        self.assertEqual(events[-1].event_type,"TASK_CHECKPOINT")
        self.assertIn("model-source-trial-",events[-1].event_id)
        with self.assertRaises(ResidentSourceHold):
            self.grant()

    def test_changed_task_head_and_close_are_held_before_nonce(self):
        self.runtime.tasks.checkpoint(
            self.request.task_id,"another-process",
            completed_evidence=("synthetic changed task",),
            blockers=("no execution",),
            protected_effects_still_gated=("COGNITION_TURN",),
            next_frontier="not admitted",
            lifecycle_evidence_digest="4"*64)
        with self.assertRaisesRegex(ResidentSourceHold,"TASK_CHANGED_OR_CANCELLED"):
            self.grant()
        self.assertEqual(self.port.used,set())

    def test_revocation_and_changed_host_session_fail_closed(self):
        self.runtime.outbound_trust.revoke(
            authority_id="os-owned:local-verifier",role="PROVIDER",
            provider_id=self.request.provider_id,
            expected_registry_generation=1,expected_revocation_epoch=0)
        with self.assertRaises(ResidentSourceHold):
            self.grant()
        self.assertEqual(self.port.used,set())

    def test_forged_signature_wrong_scope_and_missing_loaded_process_held(self):
        with self.assertRaisesRegex(ResidentSourceHold,"HOST_SIGNATURE_INVALID"):
            self.grant(signed=replace(self.signed,signature="f"*64))
        switched=replace(self.turn,task_text="Different instruction?")
        with self.assertRaisesRegex(ResidentSourceHold,"TASK_CHANGED_OR_CANCELLED"):
            self.grant(turn=switched,signed=self.sign(switched))
        self.port.actual_session="different:session"
        with self.assertRaisesRegex(ResidentSourceHold,"HOST_MODEL_OR_SESSION_SUBSTITUTED"):
            self.grant()
        self.port.actual_session="session:1"
        def inventoried_only(request):
            item=TestHostPort.reobserve(self.port,request)
            item.pop("loaded_process_pid")
            item["model_source"]="OLLAMA_INVENTORY_ONLY"
            return item
        self.port.reobserve=inventoried_only
        with self.assertRaisesRegex(ResidentSourceHold,"STORED_BYTES_NOT_LOADED_PROCESS_PROOF"):
            self.grant()
        self.assertEqual(self.port.used,set())

    def test_network_tool_and_task_source_elevation_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.turn,actions=("COGNITION_TURN","BROWSER"))
        with self.assertRaises(ValueError):
            replace(self.turn,source_digest="a"*64)
        with self.assertRaises(TypeError):
            QualifiedRuntimeModelGate(object(),self.port)
        self.assertEqual(self.port.used,set())

    def test_closed_task_rejects_valid_host_signature_without_nonce(self):
        self.runtime.tasks.close_task(
            self.request.task_id,"source-finished",
            surfaces={"source":"verified-current"},
            evidence_refs=("synthetic termination evidence",),
            blockers=(),
            claim_ceiling="source only",next_frontier="no provider",
            lifecycle_evidence_digest="5"*64)
        with self.assertRaisesRegex(ResidentSourceHold,"TASK_CHANGED_OR_CANCELLED"):
            self.grant()
        self.assertEqual(self.port.used,set())

    def test_restarted_verifier_cannot_replay_a_committed_same_task_nonce(self):
        lease=self.grant()
        from vera_core.task_execution import TaskExecutionLedger
        reopened=TaskExecutionLedger(self.runtime.tasks.path)
        self.assertEqual(
            reopened.read(self.request.task_id).journal_head_digest,
            lease.issued_checkpoint_head)
        fresh_port=TestHostPort()
        fresh_gate=QualifiedRuntimeModelGate(self.runtime,fresh_port)
        with self.assertRaisesRegex(ResidentSourceHold,"TASK_CHANGED_OR_CANCELLED"):
            fresh_gate.reserve_source_trial(
                self.turn,self.signed,now_epoch_s=1800000010)
        self.assertEqual(fresh_port.used,set())

    def test_swapped_signed_task_head_or_host_generation_never_reserved(self):
        swapped=replace(self.turn,expected_task_head="e"*64)
        with self.assertRaisesRegex(ResidentSourceHold,"TASK_CHANGED_OR_CANCELLED"):
            self.grant(turn=swapped,signed=self.sign(swapped))
        req=replace(self.request,generation=2)
        changed=replace(self.turn,model_request=req)
        with self.assertRaisesRegex(ResidentSourceHold,"HOST_MODEL_OR_SESSION_SUBSTITUTED"):
            self.grant(turn=changed,signed=self.sign(changed))
        self.assertEqual(self.port.used,set())

    def test_nonce_reuse_after_source_checkpoint_is_held(self):
        lease=self.grant()
        with self.assertRaises(ResidentSourceHold):
            self.grant()
        self.assertFalse(lease.network_dispatch_authorized)

class SourceOnlyNoNetworkCallbackIntegration(unittest.TestCase):
    def test_exact_http_callback_shape_blocks_even_after_signed_source_gate(self):
        # Importing a test harness does not register a resident provider.
        case=ActualMonoQualifiedRuntimeGateTests(methodName="test_source_trial_consumes_actual_task_cas_no_provider_grant")
        try:
            case.setUp()
        except unittest.SkipTest:
            self.skipTest("pinned actual Mono source absent")
        try:
            handler=make_held_resident_conversation_handler(
                case.gate,case.turn,case.signed,now_epoch_s=1800000010)
            before=len(case.runtime.tasks.events(case.request.task_id))
            self.assertEqual(handler("not same task","ui:resident:1")["status"],"BLOCKED")
            self.assertEqual(len(case.runtime.tasks.events(case.request.task_id)),before)
            self.assertEqual(handler("Bounded local question?","ui:resident:1"),
                             {"status":"BLOCKED","run_id":None,"final_text":None})
            self.assertEqual(len(case.runtime.tasks.events(case.request.task_id)),before+1)
            self.assertEqual(handler("Bounded local question?","ui:resident:1")["status"],"BLOCKED")
            self.assertEqual(len(case.runtime.tasks.events(case.request.task_id)),before+1)
        finally:
            for cleanup in reversed(case._cleanups):
                cleanup[0](*cleanup[1],**cleanup[2])
            case._cleanups.clear()
