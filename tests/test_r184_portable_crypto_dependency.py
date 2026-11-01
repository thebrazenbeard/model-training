"""R184: portable source CI must collect the owner-controlled crypto suite.

This test does not admit training data, verify signatures from real owners,
qualify a host, or lift the intentionally red host-integration lane.
"""
from pathlib import Path
import re
import unittest

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/portable-source-ci.yml"


class PortableRightsCryptoDependencyTests(unittest.TestCase):
    def test_exact_pinned_crypto_dependency_precedes_portable_collection(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        start = text.index("  portable-source-only:")
        end = text.index("  host-integration-required:", start)
        source_job = text[start:end]
        self.assertIn("runs-on: ubuntu-24.04", source_job)
        self.assertIn("persist-credentials: false", source_job)
        self.assertIn("cryptography==46.0.3", source_job)
        self.assertLess(
            source_job.index("cryptography==46.0.3"),
            source_job.index("python -m pytest -q -p ci_host_boundary --portable-source-only"),
        )
        self.assertNotIn("continue-on-error", source_job)
        self.assertNotIn("|| true", source_job)

    def test_real_host_integration_still_fails_closed(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        host_job = text[text.index("  host-integration-required:"):]
        self.assertIn("HOST_INTEGRATION_NOT_QUALIFIED", host_job)
        self.assertRegex(host_job, r"(?m)^\s+exit 1\s*$")


if __name__ == "__main__":
    unittest.main()
