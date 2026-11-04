"""No torch/model downloads/broker calls: CPU-only synthetic optimizer rehearsal."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from trainingday.lowmem_experiment import LowMemoryExperimentError
from trainingday.official_public_base import expected_source_record
from trainingday.optimizer_pilot import (
    PilotBounds, memory_preflight,
    physical_optimizer_launch, replay_synthetic_adapter,
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
            (TARGETS[0],),
        ):
            with self.subTest(targets=targets), self.assertRaises(LowMemoryExperimentError):
                validate_official_plan(declaration(), expected_source_record(), targets)

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
        self.assertFalse(memory_preflight(bounds=PilotBounds(), free_mib=4096,
                                          estimated_peak_mib=3800)
                         ["estimated_fit_from_untrusted_inputs"])

    def test_three_deterministic_nonzero_synthetic_updates_persist_reload(self):
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            a = replay(Path(first) / "fixture.json")
            b = replay(Path(second) / "fixture.json")
            self.assertEqual(a, b)
            self.assertEqual(a["synthetic_updates"], 3)
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
            with self.assertRaisesRegex(LowMemoryExperimentError, "symlink"):
                replay(alias)
            self.assertEqual(victim.read_text(encoding="utf-8"), "SAFE")

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
