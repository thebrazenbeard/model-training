"""R27 public synthetic checkpoint portability; never load real model artifacts."""

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from trainingday.r27_checkpoint_portability import (
    PortabilityError,
    assess_declared_cross_host_readbacks,
    build_public_f32_safetensors,
    inspect_public_f32_safetensors,
    normalize_artifact_inventory,
    read_bounded_local_file,
)


class PublicPortabilityTests(unittest.TestCase):
    def setUp(self):
        self.payload = build_public_f32_safetensors({
            "lora_A.weight": ((2, 2), (0.0, 0.5, -0.25, 1.0)),
            "lora_B.weight": ((2, 2), (0.75, -0.5, 0.125, 0.0)),
        })

    def test_deterministic_little_endian_golden_fixture(self):
        other = build_public_f32_safetensors({
            "lora_B.weight": ((2, 2), (0.75, -0.5, 0.125, 0.0)),
            "lora_A.weight": ((2, 2), (0.0, 0.5, -0.25, 1.0)),
        })
        self.assertEqual(self.payload, other)
        size = struct.unpack("<Q", self.payload[:8])[0]
        header = json.loads(self.payload[8:8 + size])
        start = 8 + size
        self.assertEqual(self.payload[start:start + 4], struct.pack("<f", 0.0))
        self.assertEqual(header["lora_A.weight"]["dtype"], "F32")
        info = inspect_public_f32_safetensors(self.payload)
        self.assertEqual(info["tensor_count"], 2)
        self.assertEqual(info["bytes_sha256"], hashlib.sha256(self.payload).hexdigest())
        self.assertFalse(info["optimizer_resume_authorized"])

    def test_blob_corruption_detected_by_pinned_hash(self):
        tamper = bytearray(self.payload)
        tamper[-1] ^= 1
        pinned = hashlib.sha256(self.payload).hexdigest()
        with self.assertRaises(PortabilityError):
            inspect_public_f32_safetensors(bytes(tamper), expected_sha256=pinned)

    def test_header_bounds_endian_and_truncation(self):
        for broken in (b"", b"\0" * 7, b"\xff" * 8, self.payload[:-1],
                       struct.pack(">Q", 32) + self.payload[8:]):
            with self.subTest(broken=broken[:8]), self.assertRaises(PortabilityError):
                inspect_public_f32_safetensors(broken)

    def test_duplicate_json_keys_rejected(self):
        header = b'{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
        blob = struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)
        with self.assertRaises(PortabilityError):
            inspect_public_f32_safetensors(blob)

    def test_noncontiguous_overlapping_and_trailing_tensor_bytes_rejected(self):
        body = struct.pack("<f", 1.0)
        for offsets, tail in [([1, 5], b"x"), ([0, 5], b"x"), ([0, 4], b"x")]:
            header = json.dumps({"t": {"dtype": "F32", "shape": [1],
                                       "data_offsets": offsets}}, separators=(",", ":")).encode()
            bad = struct.pack("<Q", len(header)) + header + body + tail
            with self.subTest(offsets=offsets, tail=tail), self.assertRaises(PortabilityError):
                inspect_public_f32_safetensors(bad)

    def test_nan_inf_and_huge_shape_rejected(self):
        for weights in (((1,), (float("nan"),)), ((1,), (float("inf"),)),
                        ((10 ** 1000,), (1.0,))):
            with self.subTest(weights=weights[0]), self.assertRaises(PortabilityError):
                build_public_f32_safetensors({"t": weights})

    def test_malicious_container_iteration_cannot_extend_tensor_bytes(self):
        class EvilList(list):
            def __iter__(self):
                return iter([0.0, 1.0])

        for tensor in (((1,), EvilList([0.0])),
                       (EvilList([1]), [0.0]),
                       (1, [0.0])):
            with self.subTest(tensor=tensor), self.assertRaises(PortabilityError):
                build_public_f32_safetensors({"t": tensor})

    def test_windows_linux_relative_names_and_case_alias_guard(self):
        a = normalize_artifact_inventory(["adapter\\adapter_model.safetensors",
                                          "meta/state.json"])
        b = normalize_artifact_inventory(["adapter/adapter_model.safetensors",
                                          "meta\\state.json"])
        self.assertEqual(a, b)
        for bad in (["../state.json"], ["C:\\state.json"], ["/tmp/state.json"],
                    ["\\\\server\\share\\x"], ["safe\\..\\x"], ["CON.txt"],
                    ["name:alt"], ["a. \\b"], ["A/x", "a/X"],
                    ["bad<name"], ["odd|name"], ["prefix\\optimizer.pt"],
                    ["rng_state.pth"], ["cafe\u0301.txt"],
                    ["COM¹.txt"], ["LPT².prn"], ["COM³"],
                    ["nested\\com¹.TXT"], ["nested/lpt³"]):
            with self.subTest(path=bad), self.assertRaises(PortabilityError):
                normalize_artifact_inventory(bad)

    def test_declared_readbacks_never_become_authenticated(self):
        sha = hashlib.sha256(self.payload).hexdigest()
        all_three = [{"host": host, "sha256": sha}
                     for host in ("Lappy", "VERA-NODE-1", "VN2")]
        assessment = assess_declared_cross_host_readbacks(self.payload, all_three)
        self.assertTrue(assessment["declared_digest_consistent"])
        self.assertFalse(assessment["actual_cross_host_verified"])
        self.assertFalse(assessment["optimizer_resume_authorized"])
        self.assertFalse(assessment["qwen4b_release_authorized"])
        all_three[-1]["sha256"] = "0" * 64
        self.assertFalse(assess_declared_cross_host_readbacks(self.payload, all_three)["declared_digest_consistent"])

    def test_missing_duplicate_and_fake_host_rejected(self):
        sha = hashlib.sha256(self.payload).hexdigest()
        for hosts in (["Lappy", "VN2"], ["Lappy"] * 3,
                      ["Lappy", "VERA-NODE-1", "bogus-host"]):
            with self.subTest(hosts=hosts), self.assertRaises(PortabilityError):
                assess_declared_cross_host_readbacks(self.payload,
                    [{"host": host, "sha256": sha} for host in hosts])

    def test_symlink_and_oversize_local_file_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "fixture.safetensors"
            p.write_bytes(self.payload)
            self.assertEqual(read_bounded_local_file(p), self.payload)
            link = Path(tmp) / "link.safetensors"
            link.symlink_to(p)
            with self.assertRaises(PortabilityError):
                read_bounded_local_file(link)
            (Path(tmp) / "huge.safetensors").write_bytes(b"x" * (1024 * 1024 + 1))
            with self.assertRaises(PortabilityError):
                read_bounded_local_file(Path(tmp) / "huge.safetensors")

    def test_fresh_process_public_fixture_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "fixture.safetensors"
            p.write_bytes(self.payload)
            proc = subprocess.run([sys.executable, "-m", "trainingday.r27_checkpoint_portability",
                                   "--read-public-file", str(p)], capture_output=True, text=True,
                                  check=True, timeout=10)
            receipt = json.loads(proc.stdout)
            self.assertEqual(receipt["bytes_sha256"], hashlib.sha256(self.payload).hexdigest())
            self.assertEqual(receipt["tensor_count"], 2)
            self.assertFalse(receipt["actual_cross_host_verified"])


if __name__ == "__main__":
    unittest.main()
