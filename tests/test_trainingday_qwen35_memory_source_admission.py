"""CPU/std-lib, synthetic metadata only: no official weights or GPU."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from trainingday.qwen35_memory_source_admission import (
    MetadataAdmissionError, OFFICIAL_CONFIG_SHA256, OFFICIAL_INDEX_SHA256,
    OFFICIAL_SHARD_SHA256, admit_pinned_official_metadata, audit_metadata,
    missing_source_report,
)


class ArchitectureAdmissionTests(unittest.TestCase):
    def setUp(self):
        types = (["linear_attention"] * 3 + ["full_attention"]) * 8
        self.config = {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 32, "layer_types": types,
                "vocab_size": 248320, "hidden_size": 2560,
                "tie_word_embeddings": True, "mtp_num_hidden_layers": 1,
            },
            "vision_config": {"depth": 24},
        }
        self.shards = (
            "model.safetensors-00001-of-00002.safetensors",
            "model.safetensors-00002-of-00002.safetensors",
        )
        self.keys = {
            "model.language_model.embed_tokens.weight": self.shards[0],
            "model.visual.patch_embed.weight": self.shards[0],
            "mtp.layers.0.self_attn.q_proj.weight": self.shards[1],
        }
        for layer, kind in enumerate(types):
            prefix = f"model.language_model.layers.{layer}."
            suffixes = (
                ("linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
                 "linear_attn.out_proj.weight")
                if kind == "linear_attention" else
                ("self_attn.q_proj.weight", "self_attn.v_proj.weight")
            )
            for suffix in suffixes:
                self.keys[prefix + suffix] = self.shards[layer % 2]
            self.keys[prefix + "mlp.down_proj.weight"] = self.shards[layer % 2]
        self.index = {"metadata": {"total_size": 9_319_737_856},
                      "weight_map": self.keys}

    @staticmethod
    def encode(obj):
        return json.dumps(obj, sort_keys=True).encode("utf-8")

    def audit(self, *, conf=None, index=None):
        config_bytes = self.encode(self.config if conf is None else conf)
        index_bytes = self.encode(self.index if index is None else index)
        return audit_metadata(config_bytes, index_bytes,
                              config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                              index_sha256=hashlib.sha256(index_bytes).hexdigest())

    def test_complete_synthetic_shape_yields_source_only_hard_unknown(self):
        result = self.audit()
        self.assertEqual(result["decoder_layers"], 32)
        self.assertEqual(result["gated_deltanet_layers"], 24)
        self.assertEqual(result["full_attention_layers"], 8)
        self.assertEqual(result["vision_layers"], 24)
        self.assertEqual(result["mtp_layers"], 1)
        self.assertEqual(result["minimal_lora_gdn_layers"], 0)
        self.assertEqual(result["minimal_lora_full_attention_layers"], 8)
        self.assertEqual(result["minimal_lora_scope"],
                         "trainingday.lowmem_experiment.nf4_minimal_gpu_ONLY")
        self.assertTrue(result["architecture_shape_checked"])
        self.assertEqual(result["bf16_tied_embedding_bytes"], 1_271_398_400)
        self.assertEqual(result["idealized_full_checkpoint_nf4_plus_bf16_embedding_bytes"],
                         3_283_483_264)
        self.assertEqual(result["hardware_fit"], "UNKNOWN")
        self.assertFalse(result["official_origin_authenticated"])
        self.assertFalse(result["official_weight_shards_authenticated"])
        self.assertFalse(result["pinned_official_metadata_bytes_verified"])
        self.assertFalse(result["training_authorized"])
        self.assertEqual(result["optimizer_steps_observed"], 0)

    def test_missing_metadata_reports_hard_hold(self):
        report = missing_source_report()
        self.assertEqual(report["status"], "OFFICIAL_METADATA_BYTES_MISSING_HARD_HOLD")
        self.assertFalse(report["architecture_shape_checked"])
        self.assertFalse(report["training_authorized"])
        command = subprocess.run(
            [sys.executable, "-m", "trainingday.qwen35_memory_source_admission"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(command.returncode, 2)
        self.assertEqual(json.loads(command.stdout)["status"], report["status"])

    def test_explicit_wrong_digests_rejected_even_with_parseable_json(self):
        conf = self.encode(self.config)
        index = self.encode(self.index)
        with self.assertRaises(MetadataAdmissionError):
            audit_metadata(conf, index, config_sha256="0" * 64,
                           index_sha256=hashlib.sha256(index).hexdigest())

    def test_caller_self_hashed_forgery_cannot_pass_official_pin_gate(self):
        """A synthetic but self-consistent index is not an official source."""
        conf = self.encode(self.config)
        index = self.encode(self.index)
        self.assertNotEqual(hashlib.sha256(conf).hexdigest(), OFFICIAL_CONFIG_SHA256)
        self.assertNotEqual(hashlib.sha256(index).hexdigest(), OFFICIAL_INDEX_SHA256)
        with self.assertRaises(MetadataAdmissionError):
            admit_pinned_official_metadata(conf, index)
        self.assertEqual(set(OFFICIAL_SHARD_SHA256), set(self.shards))
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "config.json"
            index_path = Path(temp) / "index.json"
            config_path.write_bytes(conf)
            index_path.write_bytes(index)
            command = subprocess.run(
                [sys.executable, "-m", "trainingday.qwen35_memory_source_admission",
                 "--config", str(config_path), "--index", str(index_path)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(command.returncode, 1)
            self.assertEqual(json.loads(command.stdout)["status"],
                             "SOURCE_METADATA_REJECTED_HARD_HOLD")

    def test_derivative_identity_cannot_qualify_as_official_shape(self):
        """Catalog manifest has no official model architecture / index schema."""
        derivative = {"kind": "public_upstream_base_model", "source": {
            "repository": "rodrigomt/Qwen3.5-4B-Uncensored-Aggressive",
            "revision": "d61dd146c8fd44c9a49cdb7f59f34e17b61902d8",
        }}
        with self.assertRaises(MetadataAdmissionError):
            self.audit(conf=derivative)

    def test_missing_or_wrong_decoder_layers_rejected(self):
        self.config["text_config"]["layer_types"][0] = "full_attention"
        with self.assertRaises(MetadataAdmissionError):
            self.audit()
        self.config["text_config"]["layer_types"][0] = "linear_attention"
        self.keys.pop("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
        with self.assertRaises(MetadataAdmissionError):
            self.audit()

    def test_vision_mtp_or_foreign_namespace_rejected(self):
        self.keys.pop("mtp.layers.0.self_attn.q_proj.weight")
        with self.assertRaises(MetadataAdmissionError):
            self.audit()
        self.keys["mtp.layers.0.self_attn.q_proj.weight"] = self.shards[1]
        self.keys["foreign.layer.weight"] = self.shards[1]
        with self.assertRaises(MetadataAdmissionError):
            self.audit()

    def test_wrong_full_weight_size_or_shard_rejected(self):
        self.index["metadata"]["total_size"] -= 2
        with self.assertRaises(MetadataAdmissionError):
            self.audit()
        self.index["metadata"]["total_size"] += 2
        self.keys["model.visual.patch_embed.weight"] = "model.safetensors-99999-of-99999.safetensors"
        with self.assertRaises(MetadataAdmissionError):
            self.audit()

    def test_duplicate_json_or_symlink_input_rejected(self):
        config = b'{"model_type":"qwen3_5","model_type":"qwen3_5"}'
        index = self.encode(self.index)
        with self.assertRaises(MetadataAdmissionError):
            audit_metadata(config, index,
                           config_sha256=hashlib.sha256(config).hexdigest(),
                           index_sha256=hashlib.sha256(index).hexdigest())
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "config.json"
            target.write_bytes(self.encode(self.config))
            link = Path(temp) / "alias.json"
            link.symlink_to(target)
            # The CLI must reject the alias even with matching claimed hashes.
            index_file = Path(temp) / "index.json"
            index_file.write_bytes(index)
            command = subprocess.run(
                [sys.executable, "-m", "trainingday.qwen35_memory_source_admission",
                 "--config", str(link), "--index", str(index_file)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(command.returncode, 1)
            self.assertEqual(json.loads(command.stdout)["status"],
                             "SOURCE_METADATA_REJECTED_HARD_HOLD")

    @unittest.skipUnless(hasattr(os, "mkfifo") and hasattr(os, "O_NONBLOCK"),
                         "POSIX FIFO nonblocking open only")
    def test_fifo_metadata_is_rejected_without_blocking_open(self):
        with tempfile.TemporaryDirectory() as temp:
            fifo = Path(temp) / "config.fifo"
            os.mkfifo(fifo)
            index_path = Path(temp) / "index.json"
            index_path.write_bytes(self.encode(self.index))
            command = subprocess.run(
                [sys.executable, "-m", "trainingday.qwen35_memory_source_admission",
                 "--config", str(fifo), "--index", str(index_path)],
                capture_output=True, text=True, check=False, timeout=2,
            )
            self.assertEqual(command.returncode, 1)
            self.assertEqual(json.loads(command.stdout)["status"],
                             "SOURCE_METADATA_REJECTED_HARD_HOLD")


if __name__ == "__main__":
    unittest.main()
