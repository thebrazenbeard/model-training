"""Optional real pinned PreActive Store test (no Engine or model invoked)."""
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest

from resident_model_turn_readback_v1 import (
    ExpectedTextTurn,ModelTransportEvidence,verify_resident_model_turn)

class ActualPreActiveStoreTests(unittest.TestCase):
    def test_stored_synthetic_source_turn_and_terminal_corruption(self):
        source=os.environ.get("VERA_PREACTIVE_SOURCE_ROOT")
        if not source or not (Path(source)/"pre_active"/"store.py").is_file():
            self.skipTest("actual independently pinned PreActive source unavailable")
        sys.path.insert(0,source)
        from pre_active.store import Store
        with tempfile.TemporaryDirectory(prefix="training-preactive-source-") as td:
            store=Store(Path(td)/"source.sqlite3")
            try:
                task="One short bounded source-only test turn"
                response="synthetic result from a manual PreActive Store record"
                eid=store.enqueue_event(
                    kind="task.requested",payload={"task":task,"capabilities":[]},
                    dedup_key="veraos-text:source-task:1",now=1000.0)
                rid=store.create_run(
                    task=task,capabilities=set(),source_event_id=eid,now=1000.0)
                store.record_run_message(run_id=rid,role="assistant",
                                         content=response,now=1001.0)
                store.update_run(rid,now=1001.0,status="COMPLETED",final_text=response)
                store.connection.execute(
                    "UPDATE events SET status='DONE' WHERE id=?",(eid,))
                store.append_journal(event_type="VERAOS_COGNITION_ROUTE_SELECTED",
                    subject_id=eid,payload={
                        "provider_id":"dev:virtio-qwen4b",
                        "route_id":"dev:virtio-qwen4b",
                        "model":"qwen3:4b-instruct",
                        "effect_class":"COGNITION_ONLY"},now=1001.0)
                store.connection.commit()
                expected=ExpectedTextTurn(
                    request_id="real-store-1",task_id="source-task:1",
                    semantic_sha256="c"*64,task_text=task,
                    provider_id="dev:virtio-qwen4b",model_id="qwen3:4b-instruct",
                    endpoint="http://127.0.0.1:18766/v1",
                    host_id="Lappy",host_generation=1,host_session="session:1",
                    model_blob_sha256="85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9",
                    model_manifest_sha256="0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0")
                transport=ModelTransportEvidence(
                    request_id=expected.request_id,provider_id=expected.provider_id,
                    model_id=expected.model_id,endpoint=expected.endpoint,
                    host_id=expected.host_id,host_generation=1,host_session="session:1",
                    artifact_sha256=expected.model_blob_sha256,
                    manifest_sha256=expected.model_manifest_sha256,
                    final_text_sha256=hashlib.sha256(response.encode()).hexdigest())
                claim={"status":"COMPLETED","event_id":eid,"run_id":rid,
                       "final_text":response,"route_id":expected.provider_id,
                       "provider_id":expected.provider_id}
                result=verify_resident_model_turn(store,expected,claim,transport)
                self.assertEqual(result.status,"SYNTHETIC_STORE_LINK_VERIFIED")
                self.assertFalse(result.real_model_consumption_verified)
                store.close()
                store=Store(Path(td)/"source.sqlite3")
                again=verify_resident_model_turn(store,expected,claim,transport)
                self.assertEqual(again.status,"SYNTHETIC_STORE_LINK_VERIFIED")
                store.update_run(rid,now=1002.0,status="FAILED",final_text="not actual")
                store.connection.commit()
                held=verify_resident_model_turn(store,expected,claim,transport)
                self.assertEqual(held.status,"HOLD")
                self.assertIsNone(held.display_text)
            finally:
                store.close()

if __name__=="__main__":
    unittest.main()
