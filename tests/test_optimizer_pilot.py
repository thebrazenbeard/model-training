"""No torch/model downloads/broker calls: CPU-only synthetic optimizer rehearsal."""

from pathlib import Path
import hashlib
import os
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from trainingday.lowmem_experiment import LowMemoryExperimentError
from trainingday.official_public_base import expected_source_record
from trainingday.optimizer_pilot import (
    CheckpointCommitIndeterminate, PilotBounds, memory_preflight,
    physical_optimizer_launch, replay_synthetic_adapter,
    inspect_crashed_fixture_staging,
    validate_official_plan,
)
from trainingday.physical_launch_authority import PhysicalLaunchAuthorityUnavailable
from trainingday.pinned_base import BASE_ID, BASE_REVISION


TARGETS = ("model.layers.3.self_attn.q_proj", "model.layers.3.self_attn.v_proj")


def declaration() -> dict:
    return {
        "model_id": BASE_ID, "revision": BASE_REVISION,
        "loader_class": "Qwen3_5ForCausalLM", "quantization": "NF4_DOUBLE_BF16",
        "files": {name: "a" * 64 for name in (
            "config.json", "tokenizer_config.json", "tokenizer.json",
            "model.safetensors.index.json",
            "model.safetensors-00001-of-00002.safetensors")},
    }


def replay(path: Path, **kwargs) -> dict:
    options = dict(declaration=declaration(), source_record=expected_source_record(),
                   module_names=TARGETS, bounds=PilotBounds(), free_mib=4096,
                   estimated_peak_mib=3000, checkpoint_path=path)
    options.update(kwargs)
    return replay_synthetic_adapter(**options)


class Qwen35OptimizerPilotTests(unittest.TestCase):
    def test_exact_official_source_identity_and_text_architecture(self):
        actual = validate_official_plan(declaration(), expected_source_record(), TARGETS)
        self.assertEqual(actual, tuple(sorted(TARGETS)))
        source = expected_source_record()
        self.assertEqual(source["identity"]["checkpoint_architecture"],
                         "Qwen3_5ForConditionalGeneration")
        self.assertEqual(source["identity"]["text_loader"], "Qwen3_5ForCausalLM")
        self.assertEqual(source["identity"]["revision"], BASE_REVISION)

    def test_reject_mutated_revision_tokenizer_and_multimodal_identity(self):
        for modify in ("model_id", "revision", "tokenizer", "architecture", "loader"):
            with self.subTest(modify=modify):
                binding = declaration()
                source = expected_source_record()
                if modify == "model_id":
                    binding["model_id"] = "Someone/Other"
                if modify == "revision":
                    binding["revision"] = "f" * 40
                if modify == "tokenizer":
                    binding["files"]["tokenizer.json"] = "not-sha256"
                if modify == "architecture":
                    source["identity"]["checkpoint_architecture"] = "Qwen3ForCausalLM"
                if modify == "loader":
                    source["identity"]["text_loader"] = "Qwen3ForCausalLM"
                with self.assertRaises(ValueError):
                    validate_official_plan(binding, source, TARGETS)

    def test_reject_vision_mtp_foreign_module_and_missing_pair(self):
        for targets in (
            ("model.visual.blocks.0.self_attn.q_proj", TARGETS[1]),
            ("mtp.layers.0.self_attn.q_proj", TARGETS[1]),
            ("model.layers.3.self_attn.q_proj", "model.layers.4.self_attn.v_proj"),
            (TARGETS[0], TARGETS[0]),
            ("model.layers.3.mlp.gate_proj", TARGETS[1]),
            ("model.layers.999.self_attn.q_proj", "model.layers.999.self_attn.v_proj"),
            ("model.layers.32.self_attn.q_proj", "model.layers.32.self_attn.v_proj"),
            ("model.layers.2.self_attn.q_proj", "model.layers.2.self_attn.v_proj"),
            (TARGETS[0],),
        ):
            with self.subTest(targets=targets), self.assertRaises(LowMemoryExperimentError):
                validate_official_plan(declaration(), expected_source_record(), targets)
        self.assertEqual(validate_official_plan(
            declaration(), expected_source_record(),
            ("model.layers.31.self_attn.q_proj", "model.layers.31.self_attn.v_proj")),
            ("model.layers.31.self_attn.q_proj", "model.layers.31.self_attn.v_proj"))

    def test_arbitrary_hex_digest_does_not_become_trusted_tokenizer(self):
        candidate = declaration()
        candidate["files"]["tokenizer.json"] = "f" * 64
        with TemporaryDirectory() as tmp:
            result = replay(Path(tmp) / "fixture.json", declaration=candidate)
        self.assertEqual(result["identity"]["tokenizer_sha256_declared_unverified"],
                         "f" * 64)
        self.assertEqual(result["identity"]["tokenizer_digest_trust"],
                         "UNVERIFIED_DECLARATION")
        self.assertFalse(result["identity"]["trusted_tokenizer_pin_matched"])
        self.assertEqual(result["source_identity_trust"], "UNVERIFIED_DECLARATION")
        self.assertEqual(result["module_inventory_trust"], "UNVERIFIED_DECLARATION")
        self.assertFalse(result["tokenizer_bytes_verified"])

    def test_rank_sequence_steps_memory_and_flags_are_bounded(self):
        for field, value in (
            ("steps", 4), ("sequence_length", 128), ("rank", 8),
            ("batch_size", 2), ("seed", -1), ("device_total_mib", 8192),
            ("reserve_mib", 0), ("steps", True),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(
                LowMemoryExperimentError
            ):
                PilotBounds(**{field: value})
        with self.assertRaises(LowMemoryExperimentError):
            PilotBounds(device_total_mib=256, reserve_mib=512)
        for free, peak in ((-1, 3000), (4097, 3000), (4096, 0), (True, 1)):
            with self.subTest(free=free, peak=peak), self.assertRaises(
                LowMemoryExperimentError
            ):
                memory_preflight(bounds=PilotBounds(), free_mib=free,
                                 estimated_peak_mib=peak)
        fit = memory_preflight(bounds=PilotBounds(), free_mib=4096,
                               estimated_peak_mib=3000)
        self.assertTrue(fit["estimated_fit_from_untrusted_inputs"])
        self.assertFalse(fit["physical_headroom_measured"])
        self.assertFalse(fit["physical_training_authorized"])
        self.assertEqual(fit["input_trust"], "UNVERIFIED_DECLARATION")
        self.assertFalse(memory_preflight(bounds=PilotBounds(), free_mib=4096,
                                          estimated_peak_mib=3800)
                         ["estimated_fit_from_untrusted_inputs"])

    def test_three_deterministic_nonzero_synthetic_updates_persist_reload(self):
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            a = replay(Path(first) / "fixture.json")
            b = replay(Path(second) / "fixture.json")
            self.assertEqual(a, b)
            self.assertEqual(a["synthetic_updates"], 3)
            self.assertEqual(a["gradient_source"], "SYNTHETIC_PRNG_NOT_MODEL_AUTOGRAD")
            self.assertEqual(a["loss_source"], "SYNTHETIC_SCHEDULE_NOT_MODEL_FORWARD")
            self.assertEqual(a["physical_pretrained_optimizer_steps"], 0)
            self.assertEqual(a["status"], "SOURCE_ONLY_HARD_HOLD")
            self.assertNotEqual(a["initial_adapter_sha256"], a["final_adapter_sha256"])
            self.assertTrue(a["checkpoint_reload_equal"])
            self.assertTrue(a["gradient_norms_finite_positive"])
            self.assertTrue(all(step["adapter_tensors_changed"] == 2
                                for step in a["step_witnesses"]))
            self.assertEqual(a["step_witnesses"][0]["before_sha256"],
                             a["initial_adapter_sha256"])
            self.assertEqual(a["step_witnesses"][-1]["after_sha256"],
                             a["final_adapter_sha256"])
            self.assertTrue((Path(first) / "fixture.json").is_file())
            self.assertFalse(a["base_bytes_verified"])
            self.assertFalse(a["tokenizer_bytes_verified"])
            self.assertFalse(a["official_graph_verified"])
            self.assertFalse(a["rights_verified"])
            self.assertIsNone(a["gpu_lease_id"])

    def test_different_seed_changes_adapter(self):
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            a = replay(Path(first) / "fixture.json")
            b = replay(Path(second) / "fixture.json", bounds=PilotBounds(seed=99))
            self.assertNotEqual(a["final_adapter_sha256"], b["final_adapter_sha256"])

    def test_estimated_oom_aborts_without_persisting(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "fixture.json"
            with self.assertRaisesRegex(LowMemoryExperimentError, "VRAM budget"):
                replay(target, estimated_peak_mib=3900)
            self.assertFalse(target.exists())

    def test_checkpoint_existing_file_not_overwritten(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "fixture.json"
            target.write_text("USER_OWNED_FILE", encoding="utf-8")
            with self.assertRaisesRegex(LowMemoryExperimentError, "checkpoint"):
                replay(target)
            self.assertEqual(target.read_text(encoding="utf-8"), "USER_OWNED_FILE")

    def test_checkpoint_symlink_refused(self):
        with TemporaryDirectory() as tmp:
            victim = Path(tmp) / "victim"
            victim.write_text("SAFE", encoding="utf-8")
            alias = Path(tmp) / "alias"
            alias.symlink_to(victim)
            with self.assertRaisesRegex(LowMemoryExperimentError, "checkpoint"):
                replay(alias)
            self.assertEqual(victim.read_text(encoding="utf-8"), "SAFE")

    def test_checkpoint_ancestor_symlink_refused_and_no_orphan(self):
        with TemporaryDirectory() as tmp:
            real = Path(tmp) / "real"
            real.mkdir()
            alias = Path(tmp) / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(LowMemoryExperimentError, "checkpoint"):
                replay(alias / "fixture.json")
            self.assertEqual(list(real.iterdir()), [])

    def test_checkpoint_partial_write_failure_rolls_back_owned_temp_only(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "fixture.json"
            user_file = root / "owned.txt"
            user_file.write_text("KEEP", encoding="utf-8")
            with patch("trainingday.optimizer_pilot.os.write",
                       side_effect=[10, OSError("injected partial write")]):
                with self.assertRaisesRegex(LowMemoryExperimentError, "checkpoint"):
                    replay(target)
            self.assertFalse(target.exists())
            self.assertEqual(user_file.read_text(encoding="utf-8"), "KEEP")
            self.assertEqual(list(root.glob(".trainingday-fixture-*")), [])

    def test_parent_directory_rename_cannot_report_success_for_old_path(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_parent = root / "original"
            original_parent.mkdir(mode=0o700)
            moved_parent = root / "moved"
            target = original_parent / "fixture.json"
            original_link = os.link

            def rename_before_publish(*args, **kwargs):
                original_parent.rename(moved_parent)
                return original_link(*args, **kwargs)

            with patch("trainingday.optimizer_pilot.os.link",
                       side_effect=rename_before_publish):
                with self.assertRaisesRegex(CheckpointCommitIndeterminate,
                                            "INDETERMINATE"):
                    replay(target)
            self.assertFalse(target.exists())
            self.assertTrue((moved_parent / "fixture.json").is_file())
            self.assertFalse(list(moved_parent.glob(".trainingday-fixture-*")))

    def test_post_publish_directory_fsync_failure_is_indeterminate(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "fixture.json"
            original_fsync = os.fsync

            def fail_directory_fsync(fd):
                if stat.S_ISDIR(os.fstat(fd).st_mode):
                    raise OSError("injected directory fsync failure")
                return original_fsync(fd)

            with patch("trainingday.optimizer_pilot.os.fsync",
                       side_effect=fail_directory_fsync):
                with self.assertRaisesRegex(CheckpointCommitIndeterminate,
                                            "INDETERMINATE"):
                    replay(target)
            self.assertTrue(target.is_file())
            self.assertFalse(list(Path(tmp).glob(".trainingday-fixture-*")))
            with self.assertRaises(LowMemoryExperimentError):
                replay(target)  # never silently overwrite an ambiguous commit

    def test_process_crash_orphan_is_listed_but_never_deleted(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "fixture.json"
            # Simulate abrupt process death immediately before hard-link
            # publication: neither `finally` nor ordinary cleanup can run.
            script = (
                "import os; from pathlib import Path; "
                "import trainingday.optimizer_pilot as pilot; "
                "from tests.test_optimizer_pilot import replay; "
                "pilot.os.link=lambda *a,**kw: os._exit(73); "
                "replay(Path(" + repr(str(target)) + "))"
            )
            child = subprocess.run([sys.executable, "-c", script],
                                   capture_output=True, text=True, check=False)
            self.assertEqual(child.returncode, 73, child.stderr)
            self.assertFalse(target.exists())
            leftovers = list(root.glob(".trainingday-fixture-*"))
            self.assertEqual(len(leftovers), 1)
            self.assertEqual(inspect_crashed_fixture_staging(target)["candidates"], [])
            stale = time.time() - 4000
            os.utime(leftovers[0], (stale, stale))
            other = root / "user_owned.txt"
            other.write_text("SAFE", encoding="utf-8")
            inventory = inspect_crashed_fixture_staging(target)
            self.assertEqual(inventory["status"],
                             "READ_ONLY_UNVERIFIED_CANDIDATE_INVENTORY")
            self.assertEqual(len(inventory["candidates"]), 1)
            self.assertEqual(inventory["candidates"][0]["basename"], leftovers[0].name)
            self.assertFalse(inventory["candidates"][0]["source_verified"])
            self.assertEqual(inventory["deletions_performed"], 0)
            self.assertFalse(inventory["recovery_authorized"])
            self.assertTrue(leftovers[0].is_file())
            self.assertEqual(other.read_text(encoding="utf-8"), "SAFE")
            self.assertEqual(len(inspect_crashed_fixture_staging(target)["candidates"]), 1)

    def test_fixture_directory_must_be_owner_exclusive(self):
        with TemporaryDirectory() as tmp:
            shared = Path(tmp) / "shared"
            shared.mkdir(mode=0o755)
            path = shared / "fixture.json"
            with self.assertRaisesRegex(LowMemoryExperimentError, "checkpoint"):
                replay(path)
            with self.assertRaisesRegex(LowMemoryExperimentError, "owner-only"):
                inspect_crashed_fixture_staging(path)
            self.assertEqual(list(shared.iterdir()), [])

    def test_inspection_preserves_multilink_and_symlink_candidates(self):
        with TemporaryDirectory() as tmp:
            parent = Path(tmp)
            target = parent / "fixture.json"
            prefix = ".trainingday-fixture-" + hashlib.sha256(
                target.name.encode("utf-8")).hexdigest()[:16] + "-"
            linked = parent / (prefix + "a" * 24)
            linked.write_text("USER_OWNED", encoding="utf-8")
            linked.chmod(0o600)
            other_hardlink = parent / "owner_alias"
            os.link(linked, other_hardlink)
            symlink = parent / (prefix + "b" * 24)
            symlink.symlink_to(other_hardlink)
            stale = time.time() - 4000
            os.utime(linked, (stale, stale))
            self.assertEqual(inspect_crashed_fixture_staging(target)["candidates"], [])
            self.assertEqual(linked.read_text(encoding="utf-8"), "USER_OWNED")
            self.assertTrue(symlink.is_symlink())

    def test_forged_old_same_owner_stage_is_reported_not_deleted(self):
        with TemporaryDirectory() as tmp:
            parent = Path(tmp)
            target = parent / "fixture.json"
            prefix = ".trainingday-fixture-" + hashlib.sha256(
                target.name.encode("utf-8")).hexdigest()[:16] + "-"
            unrelated_user_file = parent / (prefix + "c" * 24)
            unrelated_user_file.write_bytes(b"FORGED_USER_DATA")
            unrelated_user_file.chmod(0o600)
            stale = time.time() - 4000
            os.utime(unrelated_user_file, (stale, stale))
            first = inspect_crashed_fixture_staging(target)
            second = inspect_crashed_fixture_staging(target)
            self.assertEqual(first, second)
            self.assertEqual(len(first["candidates"]), 1)
            self.assertEqual(first["candidates"][0]["basename"],
                             unrelated_user_file.name)
            self.assertEqual(first["candidates"][0]["classification"],
                             "UNVERIFIED_NAME_AND_METADATA_MATCH_ONLY")
            self.assertFalse(first["proves_staging_provenance"])
            self.assertFalse(first["recovery_authorized"])
            self.assertEqual(first["deletions_performed"], 0)
            self.assertEqual(unrelated_user_file.read_bytes(), b"FORGED_USER_DATA")

    def test_exclusive_temp_nonce_collision_never_deletes_user_file(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "fixture.json"
            prefix = ".trainingday-fixture-" + hashlib.sha256(
                target.name.encode("utf-8")).hexdigest()[:16] + "-"
            sentinel = root / (prefix + "d" * 24)
            sentinel.write_bytes(b"UNRELATED_USER_CONTENT")
            sentinel.chmod(0o600)
            before = sentinel.stat()
            with patch("trainingday.optimizer_pilot.secrets.token_hex",
                       return_value="d" * 24):
                with self.assertRaisesRegex(LowMemoryExperimentError,
                                            "checkpoint save/reload failed"):
                    replay(target)
            after = sentinel.stat()
            self.assertEqual(sentinel.read_bytes(), b"UNRELATED_USER_CONTENT")
            self.assertEqual((after.st_dev, after.st_ino),
                             (before.st_dev, before.st_ino))
            self.assertFalse(target.exists())

    def test_physical_launch_holds_without_reading_model(self):
        with self.assertRaisesRegex(PhysicalLaunchAuthorityUnavailable, "HARD HOLD"):
            physical_optimizer_launch(model="Qwen/Qwen3.5-4B", lease="self-asserted")

    def test_import_no_torch_cuda_or_model_side_effects(self):
        statement = ("import sys; import trainingday.optimizer_pilot; "
                     "assert 'torch' not in sys.modules; "
                     "assert 'transformers' not in sys.modules")
        done = subprocess.run([sys.executable, "-c", statement],
                              text=True, capture_output=True, check=False)
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == "__main__":
    unittest.main()
