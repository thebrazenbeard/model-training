"""KNOWN RISK probe: source-only PR36 callback bypasses host auth token.

When the source exists at the exact pin, show that this is only an
in-process test injection, NOT a trusted provider attestation. No endpoint,
model, network connection or protected effect is invoked.
"""
from __future__ import annotations
import ast
import dataclasses
import importlib
import os
import subprocess
import sys
import types
import typing
import unittest

OS36_HEAD="79b2562cb2b30fafae712e64a1d260bcaea2baaf"
RESIDENT_BLOB="c4433b6c22cea77cc8225a9e21d9506e5b9470ae"

class KnownSyntheticCallbackHazard(unittest.TestCase):
    def test_arbitrary_synthetic_callback_can_reach_native_admission_stub(self):
        root=os.getenv("VERAOS_GIT_ROOT")
        if not root:
            self.skipTest("exact independent VeraOS PR36 Git object unavailable")
        try:
            actual=subprocess.check_output(
                ["git","-C",root,"rev-parse",
                 OS36_HEAD+":src/veraos/resident.py"],text=True).strip()
            if actual!=RESIDENT_BLOB:
                self.skipTest("PR36 source changed; re-review rather than assume")
            source=subprocess.check_output(
                ["git","-C",root,"show",OS36_HEAD+":src/veraos/resident.py"])
        except (OSError,subprocess.CalledProcessError):
            self.skipTest("pinned PR36 source absent")
        tree=ast.parse(source)
        resident_class=next(
            v for v in tree.body if isinstance(v,ast.ClassDef)
            and v.name=="ResidentHandles")
        native_module=types.ModuleType("veraos.native_cognition")
        pkg=types.ModuleType("veraos")
        pkg.__path__=[]
        target=types.ModuleType("veraos.resident")
        target.__package__="veraos"
        portal_module=types.ModuleType("portal")
        calls=[]
        native_module.admit_local_text_route=lambda **kwargs: (
            calls.append(("ADMISSION_ENTERED",kwargs["provider"]
                          ["qualification"]["status"])) or object())
        native_module.execute_text_request=lambda route,**kwargs: (
            calls.append(("EXECUTE_CALLED",kwargs["request_id"])) or
            {"status":"STUB_NOT_REAL_INFERENCE"})
        previous={n:sys.modules.get(n) for n in (
            "veraos","veraos.resident","veraos.native_cognition","portal")}
        try:
            sys.modules["veraos"]=pkg
            sys.modules["veraos.resident"]=target
            sys.modules["veraos.native_cognition"]=native_module
            sys.modules["portal"]=portal_module
            target.__dict__.update({
                "dataclass":dataclasses.dataclass,"Callable":typing.Callable,
                "importlib":importlib})
            exec(compile(ast.Module(body=[resident_class],type_ignores=[]),
                         "pinned_veraos36_resident.py","exec"),target.__dict__)
            handles=target.ResidentHandles(object(),object())
            request={"provider":{"provider_id":"forged",
                       "qualification":{"status":"PASS"}},
                "base_url":"http://127.0.0.1:18766/v1",
                "model":"qwen3:4b-instruct","task":"synthetic test",
                "request_id":"test-forged-source","now":1.0}
            result=handles.execute_cognition_fixture(
                request,host_admission_verifier=lambda *args:
                "SYNTHETIC_SOURCE_FIXTURE_ONLY")
            self.assertEqual(result["status"],"STUB_NOT_REAL_INFERENCE")
            self.assertEqual(calls,[("ADMISSION_ENTERED","PASS"),
                                    ("EXECUTE_CALLED","test-forged-source")])
            # SECURITY FINDING: the token is inadequate as a real verifier.
            # A production normal resident must not expose this string gate.
        finally:
            for name,value in previous.items():
                if value is None:sys.modules.pop(name,None)
                else:sys.modules[name]=value

if __name__=="__main__":
    unittest.main()
