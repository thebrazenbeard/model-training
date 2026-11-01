"""Keep dependencies required by the full CI test collection explicit."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CIDependencyContractTests(unittest.TestCase):
    def test_openwebui_testclient_dependencies_are_pinned(self):
        # The full test suite imports fastapi.testclient.TestClient, which
        # depends on HTTPX even though FastAPI does not require it by default.
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        pins = {
            line.split("==", 1)[0].lower(): line.split("==", 1)[1]
            for line in requirements.splitlines()
            if "==" in line and not line.lstrip().startswith("#")
        }
        # These are executable test dependencies, not merely optional model
        # runtime extras: R13's real FP32 restore test imports safetensors,
        # and the Lane B state-tree test uses torchao. A CI green result must
        # not substitute mocks or silently deselect either test.
        for dependency in ("fastapi", "httpx", "safetensors", "torchao"):
            with self.subTest(dependency=dependency):
                self.assertIn(dependency, pins)
                self.assertTrue(pins[dependency])


if __name__ == "__main__":
    unittest.main()
