"""Synthetic source-readback regression against pinned PreActive DB schema."""
from dataclasses import replace
from hashlib import sha256
import json
import sqlite3
import unittest

from resident_model_turn_readback_v1 import (
    ExpectedTextTurn, ModelTransportEvidence, verify_resident_model_turn)

class ResidentModelTurnReadbackTests(unittest.TestCase):
    def setUp(self):
        self.db=sqlite3.connect(":memory:")
        self.db.row_factory=sqlite3.Row
        self.db.executescript("""
            CREATE TABLE events(id TEXT PRIMARY KEY,kind TEXT,payload_json TEXT,
                dedup_key TEXT UNIQUE,status TEXT);
            CREATE TABLE runs(id TEXT PRIMARY KEY,task TEXT,status TEXT,final_text TEXT,
                source_event_id TEXT UNIQUE,capabilities_json TEXT,last_error TEXT);
            CREATE TABLE run_messages(id INTEGER PRIMARY KEY,run_id TEXT,role TEXT,
                content TEXT);
            CREATE TABLE journal(seq INTEGER PRIMARY KEY,event_type TEXT,
                subject_id TEXT,payload_json TEXT);
        """)
        self.task="Explain the system state without authority claims"
        self.response="Qwen development model: source-only fixture response."
        self.db.execute("INSERT INTO events VALUES (?,?,?,?,?)",(
            "event-1","task.requested",
            json.dumps({"task":self.task,"capabilities":[]}),
            "veraos-text:turn-one","DONE"))
        self.db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?)",(
            "run-1",self.task,"COMPLETED",self.response,"event-1","[]",None))
        self.db.execute("INSERT INTO run_messages(run_id,role,content) VALUES (?,?,?)",(
            "run-1","assistant",self.response))
        self.db.execute("INSERT INTO journal VALUES (?,?,?,?)",(
            1,"VERAOS_COGNITION_ROUTE_SELECTED","event-1",
            json.dumps({"provider_id":"dev:virtio-qwen4b",
                        "model":"qwen3:4b-instruct","route_id":"dev:virtio-qwen4b",
                        "effect_class":"COGNITION_ONLY"})))
        self.expected=ExpectedTextTurn(
            request_id="turn-one",task_id="task-one",
            semantic_sha256="c"*64,task_text=self.task,
            provider_id="dev:virtio-qwen4b",model_id="qwen3:4b-instruct",
            endpoint="http://127.0.0.1:18766/v1",host_id="Lappy",
            host_generation=1,host_session="session:1",
            model_blob_sha256="85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9",
            model_manifest_sha256="0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0")
        self.transport=ModelTransportEvidence(
            request_id="turn-one",provider_id=self.expected.provider_id,
            model_id=self.expected.model_id,endpoint=self.expected.endpoint,
            host_id="Lappy",host_generation=1,host_session="session:1",
            artifact_sha256=self.expected.model_blob_sha256,
            manifest_sha256=self.expected.model_manifest_sha256,
            final_text_sha256=sha256(self.response.encode()).hexdigest(),
            verified_by_os=False)
        self.db.commit()
        self.claim={"status":"COMPLETED","event_id":"event-1",
                    "run_id":"run-1","final_text":self.response,
                    "provider_id":"dev:virtio-qwen4b",
                    "route_id":"dev:virtio-qwen4b"}
    def tearDown(self):
        self.db.close()
    def assess(self,*,claim=None,transport=None,expected=None):
        # The fixture simulates an already committed PreActive state.
        self.db.commit()
        return verify_resident_model_turn(
            self.db, expected or self.expected,
            self.claim if claim is None else claim,
            self.transport if transport is None else transport)

    def test_uncommitted_run_state_cannot_be_called_durable(self):
        self.db.execute("UPDATE runs SET status='RUNNING' WHERE id='run-1'")
        held=verify_resident_model_turn(
            self.db,self.expected,self.claim,self.transport)
        self.assertEqual(held.status,"HOLD")
        self.assertIn("UNCOMMITTED_STORE_READBACK",held.holds)
        self.db.rollback()
        self.assertEqual(self.assess().status,"SYNTHETIC_STORE_LINK_VERIFIED")

    def test_synthetic_persisted_result_accurate_but_not_real_model_attribution(self):
        r=self.assess()
        self.assertEqual(r.status,"SYNTHETIC_STORE_LINK_VERIFIED")
        self.assertEqual(r.display_text,self.response)
        self.assertEqual(r.attribution,"UNVERIFIED_MODEL_SOURCE")
        self.assertFalse(r.real_model_consumption_verified)
        self.assertFalse(r.canonical_task_completed)
        self.assertFalse(r.effect_authority)

    def test_fake_verified_transport_boolean_does_not_elevate(self):
        r=self.assess(transport=replace(self.transport,verified_by_os=True))
        self.assertEqual(r.status,"HOLD")
        self.assertIn("UNTRUSTED_TRANSPORT_PROMOTION",r.holds)

    def test_forged_completed_or_failed_run_cannot_claim_response(self):
        self.db.execute("UPDATE runs SET status='FAILED' WHERE id='run-1'")
        result=self.assess()
        self.assertEqual(result.status,"HOLD")
        self.assertIsNone(result.display_text)
        self.db.execute("UPDATE runs SET status='COMPLETED' WHERE id='run-1'")
        self.db.execute("UPDATE runs SET final_text=NULL WHERE id='run-1'")
        self.assertEqual(self.assess().status,"HOLD")

    def test_corrupted_response_or_model_transport_digest_is_held(self):
        for mutation in (
            dict(final_text="tampered"),
            dict(provider_id="other:local"),
            dict(event_id="event-2"),
            dict(run_id="run-2"),
            dict(status="PENDING"),
        ):
            with self.subTest(mutation=mutation):
                r=self.assess(claim={**self.claim,**mutation})
                self.assertEqual(r.status,"HOLD")
                self.assertIsNone(r.display_text)
        fake=replace(self.transport,final_text_sha256="f"*64)
        self.assertEqual(self.assess(transport=fake).status,"HOLD")

    def test_wrong_task_and_reused_request_key_fail(self):
        self.db.execute("UPDATE events SET dedup_key='veraos-text:other' WHERE id='event-1'")
        self.assertEqual(self.assess().status,"HOLD")
        self.db.execute("UPDATE events SET dedup_key='veraos-text:turn-one' WHERE id='event-1'")
        self.db.execute("UPDATE runs SET task='different user request' WHERE id='run-1'")
        self.assertEqual(self.assess().status,"HOLD")

    def test_journal_route_model_and_task_effect_changes_fail(self):
        for payload in (
            {"provider_id":"fake", "model":"qwen3:4b-instruct",
             "route_id":"dev:virtio-qwen4b","effect_class":"COGNITION_ONLY"},
            {"provider_id":"dev:virtio-qwen4b","model":"vera-local",
             "route_id":"dev:virtio-qwen4b","effect_class":"COGNITION_ONLY"},
            {"provider_id":"dev:virtio-qwen4b","model":"qwen3:4b-instruct",
             "route_id":"dev:virtio-qwen4b","effect_class":"PROTECTED_EFFECT"},
        ):
            with self.subTest(payload=payload):
                self.db.execute("UPDATE journal SET payload_json=? WHERE seq=1",
                                (json.dumps(payload),))
                self.assertEqual(self.assess().status,"HOLD")

    def test_wrong_host_session_artifact_and_endpoint_fail(self):
        for kw in (
            dict(host_id="VERA-NODE-1"),dict(host_generation=7),
            dict(host_session="session:other"),
            dict(artifact_sha256="f"*64),
            dict(endpoint="http://127.0.0.1:11434/v1"),
            dict(model_id="vera-local"),
        ):
            with self.subTest(kw=kw):
                self.assertEqual(self.assess(
                    transport=replace(self.transport,**kw)).status,"HOLD")

    def test_last_assistant_message_must_equal_final_text(self):
        self.db.execute("UPDATE run_messages SET content='different response'")
        self.assertEqual(self.assess().status,"HOLD")
        self.db.execute("DELETE FROM run_messages")
        self.assertEqual(self.assess().status,"HOLD")

    def test_no_journal_selection_or_extra_actions_fail(self):
        self.db.execute("DELETE FROM journal")
        self.assertEqual(self.assess().status,"HOLD")
        self.db.execute("UPDATE events SET payload_json=?",
                        (json.dumps({"task":self.task,"capabilities":["tools"]}),))
        self.assertEqual(self.assess().status,"HOLD")


if __name__=="__main__":
    unittest.main()
