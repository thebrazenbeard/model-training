"""Adversarial shape/name coverage independent of heavyweight PyTorch CI."""

import unittest

from trainingday.qwen35_hybrid_rank4_meta_budget import (
    HybridMetaBudgetError,
    hybrid_projection_targets,
    validate_rank4_adapter_inventory,
)


class _MetaShape:
    def __init__(self, shape):
        self.shape = shape


class AdapterInventoryGuardTests(unittest.TestCase):
    @staticmethod
    def fixture():
        targets = hybrid_projection_targets(["linear_attention"] * 24 +
                                            ["full_attention"] * 8)
        expected_base_shapes = {target: (2, 3) for target in targets}
        actual = {}
        for target in targets:
            actual[f"base_model.model.{target}.lora_A.default.weight"] = _MetaShape((4, 3))
            actual[f"base_model.model.{target}.lora_B.default.weight"] = _MetaShape((2, 4))
        return actual, expected_base_shapes

    def test_exact_a_b_pairs_match_all_152_hybrid_base_projections(self):
        actual, shapes = self.fixture()
        self.assertEqual(validate_rank4_adapter_inventory(actual, shapes), 3040)

    def test_304_arbitrary_a_only_shapes_cannot_masquerade_as_peft(self):
        actual, shapes = self.fixture()
        fake = {f"base_model.model.model.layers.{index}.lora_A.default.weight":
                _MetaShape((4, 3)) for index in range(304)}
        self.assertEqual(len(fake), len(actual))
        with self.assertRaises(HybridMetaBudgetError):
            validate_rank4_adapter_inventory(fake, shapes)

    def test_missing_replaced_and_wrong_rank_shapes_fail_closed(self):
        actual, shapes = self.fixture()
        first = next(iter(actual))
        for invalid in (
            dict(list(actual.items())[:-1]),
            {**{k: v for k, v in actual.items() if k != first}, "foreign.lora_A.default.weight": actual[first]},
            {**actual, first: _MetaShape((3, 4))},
        ):
            with self.subTest(size=len(invalid)), self.assertRaises(HybridMetaBudgetError):
                validate_rank4_adapter_inventory(invalid, shapes)

    def test_malformed_base_target_inventory_rejected(self):
        actual, shapes = self.fixture()
        target = next(iter(shapes))
        for invalid in (
            dict(list(shapes.items())[:-1]),
            {**shapes, target: (0, 3)},
            {**shapes, target: (True, 3)},
            {**shapes, "foreign.proj": (2, 3)},
        ):
            with self.subTest(size=len(invalid)), self.assertRaises(HybridMetaBudgetError):
                validate_rank4_adapter_inventory(actual, invalid)


if __name__ == "__main__":
    unittest.main()
