"""Offline, synthetic file-byte verification tests; no model or host effects."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

if Path(__file__).resolve().parent.name == "tests":
    from trainingday import neurosync_artifact_bytes as m
else:
    import neurosync_artifact_bytes as m


class LocalArtifactBytesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "authorized"
        self.root.mkdir()
        self.payloads = {
            "donor_trained_checkpoint": b"checkpoint-ONLY-synthetic",
            "recipient_base_checkpoint": b"recipient-base-ONLY-synthetic",
            "adapter_weights": b"fake-safetensors-BYTES",
            "adapter_config": b'{"r":8,"fake":true}',
            "tokenizer": b"tokenizer-fixture-bytes",
        }
        self.manifest = {"schema": m.SCHEMA, "profile": "FULL_FIVE_ROLE", "artifacts": {}}
        for role, payload in self.payloads.items():
            name = role + ".bin"
            (self.root / name).write_bytes(payload)
            self.manifest["artifacts"][role] = {
                "relative_path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }

    def held(self, code, *, root=None, manifest=None):
        with self.assertRaises(m.ArtifactHold) as out:
            m.verify_local_artifact_bytes(
                self.root if root is None else root,
                self.manifest if manifest is None else manifest,
            )
        self.assertEqual(code, out.exception.code)

    def test_reads_exact_known_bytes_without_claiming_origin_or_training(self):
        witness = m.verify_local_artifact_bytes(self.root, self.manifest)
        self.assertEqual(tuple(self.payloads), tuple(x.role for x in witness.observations))
        self.assertEqual(
            [hashlib.sha256(x).hexdigest() for x in self.payloads.values()],
            [x.sha256 for x in witness.observations],
        )
        self.assertEqual("LOCAL_FILE_BYTES_MATCH_DECLARATIONS__ORIGIN_UNVERIFIED_HOLD",
                         witness.status)
        self.assertFalse(witness.physical_launch_authorized)
        self.assertFalse(witness.state_commit_authorized)
        self.assertFalse(witness.rights_verified)
        self.assertFalse(witness.independent_host_witness)
        self.assertFalse(witness.trained_model_verified)
        self.assertFalse(witness.as_dict()["independent_host_witness"])
        self.assertEqual("LOCAL_READBACK_UNATTESTED", witness.as_dict()["evidence_class"])

    def test_cannot_construct_launch_permission(self):
        w = m.verify_local_artifact_bytes(self.root, self.manifest)
        with self.assertRaises(TypeError):
            m.LocalByteEvidence(w.observations, w.profile,
                                physical_launch_authorized=True)
        with self.assertRaises(TypeError):
            m.LocalByteEvidence(w.observations, w.profile, status="HOST_ATTESTED")
        with self.assertRaises(m.ArtifactHold) as empty:
            m.LocalByteEvidence(tuple(), "FULL_FIVE_ROLE")
        self.assertEqual("BYTE_EVIDENCE_REQUIRES_LOCAL_SCANNER", empty.exception.code)
        with self.assertRaises((AttributeError, TypeError)):
            w.physical_launch_authorized = True

    def test_digest_mismatch(self):
        self.manifest["artifacts"]["adapter_weights"]["sha256"] = "0" * 64
        self.held("ARTIFACT_SHA256_MISMATCH")

    def test_peft_only_profile_never_implies_full_checkpoint(self):
        candidate = deepcopy(self.manifest)
        candidate["profile"] = "PEFT_ADAPTER_ONLY_PARTIAL_HOLD"
        for role in ("donor_trained_checkpoint", "recipient_base_checkpoint"):
            del candidate["artifacts"][role]
        observation = m.verify_local_artifact_bytes(self.root, candidate)
        self.assertEqual(m.PEFT_ONLY_ROLES,
                         tuple(x.role for x in observation.observations))
        self.assertEqual("PEFT_ADAPTER_ONLY_PARTIAL_HOLD__ORIGIN_UNVERIFIED",
                         observation.status)
        self.assertFalse(observation.trained_model_verified)
        self.assertFalse(observation.physical_launch_authorized)

    def test_size_mismatch(self):
        self.manifest["artifacts"]["adapter_config"]["size_bytes"] += 1
        self.held("ARTIFACT_SIZE_MISMATCH")

    def test_bad_root_is_rejected(self):
        self.held("ROOT_MUST_BE_ABSOLUTE_CLEAN_PATH", root=Path("relative-root"))
        self.held("ROOT_DIRECTORY_UNAVAILABLE_OR_LINKED", root=self.root / "missing")
        self.held("ALLOWROOT_TOO_BROAD", root=Path("/"))

    def test_reject_symlink_root_itself(self):
        linked_root = Path(self.temp.name) / "root-link"
        try:
            linked_root.symlink_to(self.root, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        self.held("ROOT_DIRECTORY_UNAVAILABLE_OR_LINKED", root=linked_root)

    def test_reject_symlink_file_to_outside(self):
        name = "adapter_config.bin"
        outside = Path(self.temp.name) / "outside"
        outside.write_bytes(self.payloads["adapter_config"])
        (self.root / name).unlink()
        try:
            (self.root / name).symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        self.held("ARTIFACT_UNAVAILABLE_OR_LINKED")

    def test_reject_symlink_parent_directory(self):
        nested = self.root / "nested"
        nested.mkdir()
        original = self.root / "tokenizer.bin"
        original.rename(nested / original.name)
        linked = self.root / "alias"
        try:
            linked.symlink_to(nested, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        self.manifest["artifacts"]["tokenizer"]["relative_path"] = "alias/tokenizer.bin"
        self.held("ARTIFACT_UNAVAILABLE_OR_LINKED")

    def test_reject_path_traversal_and_windows_unc(self):
        entry = self.manifest["artifacts"]["tokenizer"]
        for path in ("../outside", "/tmp/elsewhere", "a/../tokenizer.bin",
                     "..\\outside", "C:\\path", "//server/share", "a//b",
                     "name\n.bin", "a/./tokenizer.bin"):
            with self.subTest(path=path):
                candidate = deepcopy(self.manifest)
                candidate["artifacts"]["tokenizer"]["relative_path"] = path
                self.held("INVALID_RELATIVE_PATH", manifest=candidate)

    def test_reject_unexpected_or_missing_roles_and_fields(self):
        candidate = deepcopy(self.manifest)
        candidate["artifacts"].pop("adapter_config")
        self.held("ARTIFACT_ROLES_REQUIRED", manifest=candidate)
        candidate = deepcopy(self.manifest)
        candidate["artifacts"]["adapter_config"]["role"]="owner"
        self.held("INVALID_ARTIFACT_FIELDS", manifest=candidate)
        candidate = deepcopy(self.manifest)
        candidate["launch"]="yes"
        self.held("INVALID_MANIFEST_FIELDS", manifest=candidate)

    def test_reject_aliased_path_and_digest(self):
        candidate = deepcopy(self.manifest)
        candidate["artifacts"]["tokenizer"]["relative_path"] = "adapter_config.bin"
        self.held("DUPLICATE_ARTIFACT_PATH", manifest=candidate)
        candidate = deepcopy(self.manifest)
        candidate["artifacts"]["tokenizer"]["sha256"] = candidate["artifacts"]["adapter_config"]["sha256"]
        self.held("ALIASED_ARTIFACT_DIGEST", manifest=candidate)

    def test_reject_nonregular_fifo_without_blocking(self):
        if not hasattr(os, "mkfifo"):
            self.skipTest("mkfifo unavailable")
        name = self.root / "tokenizer.bin"
        name.unlink()
        os.mkfifo(name)
        self.held("ARTIFACT_NOT_EXCLUSIVE_REGULAR_FILE")

    def test_reject_oversize_claim_without_allocating(self):
        candidate = deepcopy(self.manifest)
        candidate["artifacts"]["tokenizer"]["size_bytes"] = 32 * 1024**3 + 1
        self.held("INVALID_SIZE_CEILING", manifest=candidate)
        candidate = deepcopy(self.manifest)
        for entry in candidate["artifacts"].values():
            entry["size_bytes"] = 16 * 1024**3
        self.held("TOTAL_SIZE_CEILING", manifest=candidate)

    def test_reject_mutation_while_reading(self):
        original = m.os.read
        target = self.root / "donor_trained_checkpoint.bin"
        changed = False

        def mutate_after_read(fd, amount):
            nonlocal changed
            block = original(fd, amount)
            if not changed and block:
                changed = True
                target.write_bytes(b"X" * len(self.payloads["donor_trained_checkpoint"]))
                os.utime(target, ns=(42_000_000_000, 42_000_000_000))
            return block

        with mock.patch.object(m.os, "read", side_effect=mutate_after_read):
            self.held("ARTIFACT_CHANGED_DURING_READ")
        self.assertTrue(changed)

    def test_unsupported_secure_open_fails_closed(self):
        with mock.patch.object(m, "_require_secure_open",
                               side_effect=m.ArtifactHold("PLATFORM_NO_ATOMIC_PATH_CONTAINMENT")):
            self.held("PLATFORM_NO_ATOMIC_PATH_CONTAINMENT")

    def test_time_budget_exceeded_is_hold(self):
        # A monotonic deadline provides a bounded CPU/read-loop budget;
        # it cannot interrupt a single underlying stalled filesystem syscall.
        with mock.patch.object(m.time, "monotonic",
                               side_effect=[0.0, 601.0]):
            self.held("ARTIFACT_TIME_BUDGET_EXCEEDED")


if __name__ == "__main__":
    unittest.main()
