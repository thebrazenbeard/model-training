"""Exact VeraOS PR37 malformed terminal HTTP falsifier, fake service only."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading
import types
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from resident_http_completion_gate_v1 import normalize_http_result, HttpResultHold

OS37="e134dd7881a68b719b27e3c0fdd627ee3dad241c"
BLOB="91b68d5adb49751dd6eca6e1a6b9c5d84c120f80"
OS37_CORRECTED="59a4d8f0d26c00104cf1ea6d87249d94fc5a58ad"
BLOB_CORRECTED="ee20af7fb5de2b48d179bdfa22f518b3b619202b"

class ExactVeraOS37HttpSourceTests(unittest.TestCase):
    def test_false_completed_http_200_is_held_by_training_guard(self):
        root=os.environ.get("VERAOS_GIT_ROOT")
        if not root:
            self.skipTest("VeraOS exact Git source unavailable")
        try:
            real=subprocess.check_output(["git","-C",root,"rev-parse",
                OS37+":src/veraos/desktop.py"],text=True).strip()
            if real!=BLOB:
                self.skipTest("VeraOS source changed; requires renewed audit")
            source=subprocess.check_output(["git","-C",root,"show",
                OS37+":src/veraos/desktop.py"])
        except (OSError,subprocess.CalledProcessError):
            self.skipTest("no pinned OS37 source")
        mod=types.ModuleType("pinned_veraos37_desktop")
        exec(compile(source,"pinned_veraos37_desktop.py","exec"),mod.__dict__)
        bogus=dict(status="COMPLETED",run_id=None,event_id="event:one",
            final_text=None,provider_id="dev:virtio-qwen4b",
            route_id="dev:virtio-qwen4b")
        server=ThreadingHTTPServer(("127.0.0.1",0),
            mod.make_handler(lambda: {},conversation_handler=lambda task,rid:bogus))
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            url=f"http://127.0.0.1:{server.server_port}/api/conversation"
            req=urllib.request.Request(
                url,json.dumps({"task":"test fake turn","request_id":"fake:1"}).encode(),
                headers={"Origin":f"http://127.0.0.1:{server.server_port}",
                         "Content-Type":"application/json"},
                method="POST")
            with urllib.request.urlopen(req,timeout=4) as resp:
                self.assertEqual(resp.status,200)
                payload=json.load(resp)
            self.assertEqual(payload["status"],"COMPLETED")
            self.assertIsNone(payload["run_id"])
            self.assertIsNone(payload["final_text"])
            with self.assertRaises(HttpResultHold):
                normalize_http_result(bogus,request_id="fake:1",
                    expected_provider="dev:virtio-qwen4b",fixture_mode=True)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(3)

class ExactVeraOS37CorrectionTests(unittest.TestCase):
    def test_current_pr37_http_rejects_incomplete_fake_terminal_result(self):
        root=os.environ.get("VERAOS_GIT_ROOT")
        if not root:
            self.skipTest("exact current VeraOS source unavailable")
        try:
            blob=subprocess.check_output(["git","-C",root,"rev-parse",
                OS37_CORRECTED+":src/veraos/desktop.py"],text=True).strip()
            if blob!=BLOB_CORRECTED:
                self.skipTest("VeraOS source head changed; re-audit")
            source=subprocess.check_output(["git","-C",root,"show",
                OS37_CORRECTED+":src/veraos/desktop.py"])
        except (OSError,subprocess.CalledProcessError):
            self.skipTest("corrected VeraOS Git object absent")
        mod=types.ModuleType("pinned_veraos37_corrected")
        exec(compile(source,"pinned_veraos37_corrected.py","exec"),mod.__dict__)
        server=ThreadingHTTPServer(("127.0.0.1",0),
            mod.make_handler(lambda:{}, conversation_handler=lambda task,rid:
              dict(status="COMPLETED",run_id=None,event_id="event:one",
                   final_text=None)))
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            url=f"http://127.0.0.1:{server.server_port}/api/conversation"
            req=urllib.request.Request(
                url,json.dumps({"task":"no model","request_id":"fake:2"}).encode(),
                headers={"Origin":f"http://127.0.0.1:{server.server_port}",
                         "Content-Type":"application/json"},
                method="POST")
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(req,timeout=4)
            self.assertEqual(raised.exception.code,502)
            try:
                self.assertEqual(json.load(raised.exception)["status"],"OUTCOME_UNKNOWN")
            finally:
                raised.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(3)
