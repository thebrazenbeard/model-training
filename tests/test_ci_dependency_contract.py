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
        for dependency in ("fastapi", "httpx"):
            with self.subTest(dependency=dependency):
                self.assertIn(dependency, pins)
                self.assertTrue(pins[dependency])

    def test_cpu_training_compatibility_test_dependencies_follow_torch(self):
        # The CPU-only CI tests import safetensors.torch and torchao's
        # OptimState8bit, but must not pull a second default PyTorch wheel.
        workflow = (ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")
        torch_install = "torch==2.14.0+cpu"
        extra_install = "--no-deps safetensors==0.8.0 torchao==0.18.0"
        self.assertIn(torch_install, workflow)
        self.assertIn(extra_install, workflow)
        self.assertLess(workflow.index(torch_install), workflow.index(extra_install))


if __name__ == "__main__":
    unittest.main()
