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


if __name__ == "__main__":
    unittest.main()
