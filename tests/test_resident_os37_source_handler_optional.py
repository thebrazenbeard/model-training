"""Exact PORTAL PR37 browser HTTP consumes a HELD Mono-bound gate, no model."""
import json
import os
from pathlib import Path
import subprocess
import threading
import types
import unittest
import urllib.request

from resident_os_composition_gate_v1 import make_held_resident_conversation_handler

OS37_SHA="59a4d8f0d26c00104cf1ea6d87249d94fc5a58ad"
DESKTOP_SHA="ee20af7fb5de2b48d179bdfa22f518b3b619202b"


class VeraOS37ActualSourceNoNetworkTests(unittest.TestCase):
    def test_http_conversation_consumes_real_runtime_source_lease_but_no_model(self):
        root=os.environ.get("VERAOS_GIT_ROOT")
        mono=os.environ.get("VERA_MONO_SOURCE_ROOT")
        if not (root and mono):
            self.skipTest("pinned OS37 or Mono checkout absent")
        try:
            blob=subprocess.check_output(["git","-C",root,"rev-parse",
                    OS37_SHA+":src/veraos/desktop.py"],text=True).strip()
            if blob!=DESKTOP_SHA:
                self.skipTest("PORTAL source differs from exact audited head")
            source=subprocess.check_output(["git","-C",root,"show",
                    OS37_SHA+":src/veraos/desktop.py"])
        except (OSError,subprocess.CalledProcessError):
            self.skipTest("pinned exact PR37 source unavailable")
        from test_resident_os_composition_gate_v1 import ActualMonoQualifiedRuntimeGateTests
        case=ActualMonoQualifiedRuntimeGateTests(methodName="test_source_trial_consumes_actual_task_cas_no_provider_grant")
        try:
            case.setUp()
        except unittest.SkipTest:
            self.skipTest("exact Mono composition dependencies unavailable")
        module=types.ModuleType("exact_veraos37_desktop")
        exec(compile(source,"exact_veraos37_desktop.py","exec"),module.__dict__)
        from http.server import ThreadingHTTPServer
        handler=make_held_resident_conversation_handler(
            case.gate,case.turn,case.signed,now_epoch_s=1800000010)
        server=ThreadingHTTPServer(("127.0.0.1",0),
            module.make_handler(lambda:{"status":"COMPOSED_NO_COGNITION"},
                                conversation_handler=handler))
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            endpoint=f"http://127.0.0.1:{server.server_port}/api/conversation"
            def ask(question):
                req=urllib.request.Request(
                    endpoint,json.dumps({"task":question,
                          "request_id":"ui:resident:1"}).encode(),
                    method="POST",headers={
                      "Origin":f"http://127.0.0.1:{server.server_port}",
                      "Content-Type":"application/json"})
                with urllib.request.urlopen(req,timeout=4) as answer:
                    return answer.status,json.load(answer)
            bad,first=ask("substituted question")
            self.assertEqual(bad,202)
            self.assertEqual(first["status"],"BLOCKED")
            before=len(case.runtime.tasks.events(case.request.task_id))
            for _ in range(2):
                code,body=ask("Bounded local question?")
                self.assertEqual(code,202)
                self.assertEqual(body["status"],"BLOCKED")
                self.assertIsNone(body["final_text"])
            after=len(case.runtime.tasks.events(case.request.task_id))
            self.assertEqual(after,before+1)
            self.assertEqual(case.port.used,{(
                "Lappy","session:1","a"*32,"task:one")})
        finally:
            server.shutdown();server.server_close();thread.join(3)
            for action,args,kwargs in reversed(case._cleanups):
                action(*args,**kwargs)
            case._cleanups.clear()
