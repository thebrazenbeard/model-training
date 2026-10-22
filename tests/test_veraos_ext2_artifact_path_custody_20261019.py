"""Training adversarial synthetic ext2 artifact-custody test (NOT Buildroot)."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

MEMBER = "/opt/veraos/components/vera_mono/provenance.py"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ext2_read(path):
    run = subprocess.run(["debugfs", "-R", "cat " + MEMBER, str(path)],
                         capture_output=True, timeout=12, check=True)
    return run.stdout

class ArtifactIdentityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="training-ext2-swap-")
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.stage_a = root / "stage-A"
        self.stage_b = root / "stage-B"
        for stage, contents in ((self.stage_a, b"CANONICAL_GENERATION_A\n"),
                                (self.stage_b, b"FOREIGN_GENERATION_B\n")):
            dest = stage / MEMBER.lstrip("/")
            dest.parent.mkdir(parents=True)
            dest.write_bytes(contents)
        self.selected = root / "selected.ext2"
        self.foreign = root / "foreign.ext2"
        for stage, image in ((self.stage_a, self.selected),
                             (self.stage_b, self.foreign)):
            run = subprocess.run(["mke2fs", "-q", "-F", "-t", "ext2",
                                  "-d", str(stage), str(image), "8M"],
                                 capture_output=True, timeout=25, check=True)
        self.expected_bytes = (self.stage_a / MEMBER.lstrip("/")).read_bytes()

    def test_unmodified_selected_image_reads_exact_stage(self):
        self.assertEqual(ext2_read(self.selected), self.expected_bytes)
        self.assertNotEqual(sha(self.selected), sha(self.foreign))

    def test_old_verification_does_not_bind_future_path_consumer(self):
        initial_member = ext2_read(self.selected)
        audited_image_sha = sha(self.selected)
        self.assertEqual(initial_member, self.expected_bytes)
        with self.selected.open("rb") as pinned_old_inode:
            old_inode = os.fstat(pinned_old_inode.fileno()).st_ino
            os.replace(self.foreign, self.selected)
            current_image_sha = sha(self.selected)
            consumed_member = ext2_read(self.selected)
            pinned_old_inode.seek(0)
            original_open_sha = hashlib.sha256(pinned_old_inode.read()).hexdigest()
        print("CUSTODY_EVIDENCE", "pid", os.getpid(),
              "old_inode", old_inode, "audited", audited_image_sha,
              "consumer", current_image_sha,
              "old_fd_digest", original_open_sha, flush=True)
        self.assertNotEqual(consumed_member, initial_member)
        self.assertEqual(original_open_sha, audited_image_sha)
        # A source/audit digest is only safe for the SAME bytes consumed.
        self.assertNotEqual(current_image_sha, audited_image_sha)
        self.assertNotEqual(consumed_member, self.expected_bytes)
        self.assertNotEqual(os.stat(self.selected).st_ino, old_inode)
        decision = ("HOLD_ARTIFACT_IDENTITY_MISMATCH"
                    if current_image_sha != audited_image_sha else "POTENTIAL_MATCH")
        self.assertEqual(decision, "HOLD_ARTIFACT_IDENTITY_MISMATCH")

if __name__ == "__main__":
    unittest.main(verbosity=2)