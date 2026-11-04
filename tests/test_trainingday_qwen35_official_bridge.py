"""No-shard official Qwen3.5 multimodal-to-text key-mapping qualification.

The hosted path fetches only pinned config/index metadata. This does not load
pretrained weight shards, train Qwen, authorize GPU access, or measure fit.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import unittest

from trainingday.qwen_text_checkpoint import (
    TextCheckpointProjectionError,
    anchored_weight_key_mapping,
    prepare_text_loader,
)


CONFIG_SHA256 = "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"
INDEX_SHA256 = "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d"
SHARDS = {
    "model.safetensors-00001-of-00002.safetensors",
    "model.safetensors-00002-of-00002.safetensors",
}


class WeightMappingEscapeTests(unittest.TestCase):
    def test_literal_keys_are_anchored_not_regex_wildcards(self):
        raw = {"model.language_model.layers.1.mlp.down_proj.weight":
               "model.layers.1.mlp.down_proj.weight"}
        result = anchored_weight_key_mapping(raw)
        self.assertEqual(len(result), 1)
        pattern = next(iter(result))
        self.assertEqual(result[pattern], next(iter(raw.values())))
        self.assertTrue(re.search(pattern, next(iter(raw))))
        for foreign in ("modelXlanguage_model.layers.1.mlp.down_proj.weight",
                        "model.language_model.layers.1.mlp.down_proj.weight.foreign",
                        "foreign.model.language_model.layers.1.mlp.down_proj.weight",
                        "model.language_model.layers.1.mlp.down_proj.weight\n"):
            with self.subTest(foreign=foreign):
                self.assertIsNone(re.search(pattern, foreign))

    def test_mismatched_source_to_target_is_not_admitted(self):
        with self.assertRaises(TextCheckpointProjectionError):
            anchored_weight_key_mapping({
                "model.language_model.layers.1.mlp.down_proj.weight":
                "model.layers.2.mlp.down_proj.weight",
            })


@unittest.skipUnless(os.environ.get("TRAININGDAY_OFFICIAL_META_DIR"),
                     "dedicated pinned-metadata workflow owns full bridge test")
class OfficialMetadataBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Absence is a hard CI error, never a test skip or implicit fallback.
        value = os.environ.get("TRAININGDAY_OFFICIAL_META_DIR")
        if not value:
            raise RuntimeError("required pinned official metadata directory missing")
        cls.root = Path(value)
        cls.config = (cls.root / "config.json").read_bytes()
        cls.index = (cls.root / "model.safetensors.index.json").read_bytes()
        if (len(cls.config) > 65_536 or len(cls.index) > 1_048_576
                or hashlib.sha256(cls.config).hexdigest() != CONFIG_SHA256
                or hashlib.sha256(cls.index).hexdigest() != INDEX_SHA256):
            raise RuntimeError("official config/index bytes are not independently pinned")

    def test_all_426_official_text_keys_exactly_map_to_text_skeleton(self):
        import torch
        from accelerate import init_empty_weights
        from transformers import Qwen3_5ForCausalLM
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import WeightRenaming

        if torch.version.cuda is not None:
            self.fail("this test requires a CPU-only PyTorch build")
        index = json.loads(self.index)
        all_keys = index["weight_map"]
        text_keys = sorted(key for key in all_keys
                           if key.startswith("model.language_model."))
        self.assertEqual(len(text_keys), 426)
        self.assertEqual(sum(k.startswith("model.visual.") for k in all_keys), 297)
        self.assertEqual(sum(k.startswith("mtp.") for k in all_keys), 15)
        self.assertEqual(set(all_keys.values()), SHARDS)
        self.assertEqual(index["metadata"]["total_size"], 9_319_737_856)

        text_config, loader_mapping = prepare_text_loader(
            self.root,
            expected_index_sha256=INDEX_SHA256,
            expected_shards=SHARDS,
        )
        self.assertEqual(type(text_config).__name__, "Qwen3_5TextConfig")
        self.assertEqual(len(loader_mapping), len(text_keys))
        with init_empty_weights(include_buffers=True):
            skeleton = Qwen3_5ForCausalLM(text_config)
        if any(parameter.device.type != "meta" for parameter in skeleton.parameters()):
            self.fail("text skeleton escaped the meta device")

        transforms = get_model_conversion_mapping(skeleton, loader_mapping)
        direct = transforms[:len(loader_mapping)]
        self.assertEqual(len(direct), 426)
        self.assertTrue(all(isinstance(obj, WeightRenaming) for obj in direct))
        expected_targets = set(skeleton.state_dict()) - {"lm_head.weight"}
        self.assertEqual(expected_targets, set(loader_mapping.values()))

        for source in text_keys:
            target = "model." + source[len("model.language_model."):]
            matched = []
            for transform in direct:
                actual, pattern = transform.rename_source_key(source)
                if pattern is not None:
                    matched.append(actual)
            self.assertEqual(matched, [target], f"not a unique exact rename: {source}")

            # All mapping sources are regexes in Transformers 5.22.1.
            # A '.' in a raw key must not match arbitrary characters.
            for forged in (source.replace(".", "X", 1), source + ".foreign",
                           source + "\n"):
                self.assertFalse(any(transform.rename_source_key(forged)[1] is not None
                                     for transform in direct),
                                 f"foreign key was accidentally remapped: {forged}")
        print("OFFICIAL_QWEN35_TEXT_BRIDGE_EVIDENCE=" + json.dumps({
            "status": "PINNED_OFFICIAL_METADATA_TEXT_MAPPING_REGEX_VERIFIED_ONLY",
            "config_sha256": CONFIG_SHA256,
            "index_sha256": INDEX_SHA256,
            "text_tensor_mapping_count": len(loader_mapping),
            "target_model_class": type(skeleton).__name__,
            "pretrained_weight_shards_loaded": False,
            "optimizer_steps_observed": 0,
            "cuda_used": False,
            "hardware_fit": "UNKNOWN",
            "physical_training_authorized": False,
        }, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
