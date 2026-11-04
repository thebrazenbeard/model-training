"""Independent numeric/state oracle for public R203 counterfactual TRAIN."""

import hashlib
import json
import re
import unittest
from collections import Counter, defaultdict

from trainingday import r203_counterfactual_toolstate as source


_STATE = re.compile(
    r"Initial: A=(-?\d+); B=(-?\d+); Q=(\[[^\]]+\]); floor=(-?\d+)\. "
    r"Events: A_delta=(-?\d+); Q_rotate_left=(\d+)\."
)


def independently_score(row):
    """Oracle parses public text and executes tools without source private helpers."""
    match = _STATE.search(row["prompt"])
    if match is None:
        raise AssertionError("unparseable state record")
    a, b = int(match[1]), int(match[2])
    initial = json.loads(match[3])
    floor, delta, left = map(int, (match[4], match[5], match[6]))
    if len(initial) != 3 or len(set(initial)) != 3:
        raise AssertionError("queue must hold three distinct values")
    updated_a = a + delta
    moved = initial[left % 3:] + initial[:left % 3]
    if updated_a + b < floor:
        return "sum_ints", {"a": updated_a, "b": b}, updated_a + b
    if moved != sorted(moved):
        return "sort_ints", {"values": moved}, sorted(moved)
    return "no_tool", {}, None


class R203CounterfactualTests(unittest.TestCase):
    def test_all_rows_match_independent_executable_tool_oracle(self):
        rows = source.train_rows()
        self.assertEqual(len(rows), 192)
        self.assertEqual(len({r["id"] for r in rows}), 192)
        self.assertEqual(len({r["prompt"] for r in rows}), 192)
        classes = Counter()
        for row in rows:
            self.assertEqual(set(row), {"id", "prompt", "answer", "split"})
            self.assertEqual(row["split"], "train")
            self.assertRegex(row["id"], r"^r203-toolstate-[0-9a-f]{20}$")
            self.assertLessEqual(len(row["prompt"]), 1024)
            action = json.loads(row["answer"])
            self.assertEqual(set(action), {"tool", "args", "result"})
            self.assertEqual(row["answer"], json.dumps(action, sort_keys=True,
                                                        separators=(",", ":")))
            tool, args, result = independently_score(row)
            self.assertEqual(action, {"tool": tool, "args": args, "result": result})
            classes[tool] += 1
        self.assertEqual(classes, {"sum_ints": 64, "sort_ints": 64, "no_tool": 64})
        self.assertEqual(hashlib.sha256(source.train_bytes()).hexdigest(),
                         "ab04ddf9eb88aa0c809404aada24b5784c3c7de11fb3cd9cb62b0b5d4752d389")
        self.assertEqual(source.train_rows(), rows)

    def test_64_minimal_contrast_triples(self):
        surfaces = Counter()
        for case in range(64):
            first, second, third = source.contrastive_triple(case)
            self.assertEqual([json.loads(r["answer"])["tool"]
                              for r in (first, second, third)],
                             ["sum_ints", "sort_ints", "no_tool"])
            for row in (first, second, third):
                independently_score(row)
            # A SUM/SORT contrast changes only sensor delta; sort/no-tool
            # changes only queue rotation. Every other lexical cue, initial
            # record, rule and catalog stays identical in each pair.
            strip_delta = lambda text: re.sub(r"A_delta=-?\d+", "A_delta=?", text)
            strip_shift = lambda text: re.sub(r"Q_rotate_left=\d+", "Q_rotate_left=?", text)
            self.assertEqual(strip_delta(first["prompt"]),
                             strip_delta(second["prompt"]))
            self.assertEqual(strip_shift(second["prompt"]),
                             strip_shift(third["prompt"]))
            surfaces[sum(text in first["prompt"] for text in source.SURFACE_TEMPLATES)] += 1
        self.assertEqual(surfaces, {1: 64})

    def test_shortcut_baselines_and_opaque_row_order(self):
        rows = source.train_rows()
        classes = [json.loads(r["answer"])["tool"] for r in rows]
        # The exact R202 lexical rule sees none of its former task cues and
        # defaults to no_tool. Old source had tool-class accuracy 256/256.
        def old_r202_rule(prompt):
            if "Select a tool to add integers" in prompt or "A report claims" in prompt:
                return "sum_ints"
            if "Select a tool to sort" in prompt or "Arrange the list" in prompt:
                return "sort_ints"
            return "no_tool"
        self.assertEqual(sum(old_r202_rule(r["prompt"]) == actual
                             for r, actual in zip(rows, classes)), 64)
        self.assertEqual(max(Counter(classes).values()), 64)
        # This public row list is hash-shuffled, not the triple's SUM/SORT/NO
        # order; a modulo-3 learner should be close to chance, not 100%.
        cycle = ("sum_ints", "sort_ints", "no_tool")
        self.assertLess(sum(actual == cycle[i % 3] for i, actual
                            in enumerate(classes)), 90)
        # A one-field shift lookup cannot consistently classify each case's
        # paired SUM/SORT actions because both share the exact same shift.
        by_shift = defaultdict(Counter)
        by_delta_sign = defaultdict(Counter)
        by_surface = defaultdict(Counter)
        for row, answer in zip(rows, classes):
            state = _STATE.search(row["prompt"])
            by_shift[int(state[6])][answer] += 1
            delta = int(state[5])
            by_delta_sign["negative" if delta < 0 else
                          "zero" if delta == 0 else "positive"][answer] += 1
            surface = next(i for i, phrase in enumerate(source.SURFACE_TEMPLATES)
                           if phrase in row["prompt"])
            by_surface[surface][answer] += 1
        shift_baseline = sum(max(counts.values()) for counts in by_shift.values())
        delta_sign_baseline = sum(max(counts.values())
                                  for counts in by_delta_sign.values())
        template_baseline = sum(max(counts.values())
                                for counts in by_surface.values())
        self.assertEqual(shift_baseline, 70)       # 36.5%, not 100%.
        self.assertEqual(delta_sign_baseline, 73)  # 38.0%.
        self.assertEqual(template_baseline, 64)    # 33.3%.

    def test_source_only_hard_hold_and_index_validation(self):
        for invalid in (-1, 64, True, 2.0, "2", None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(source.R203ToolstateError):
                    source.contrastive_triple(invalid)
        receipt = source.source_commitment()
        self.assertEqual(receipt["status"], "SOURCE_ONLY_UNAUTHENTICATED_HOLD")
        for key in ("private_dev_included", "data_rights_granted",
                    "independent_evaluator_qualified", "model_tokenizer_fit_verified",
                    "gpu_lease_granted", "authorizes_training"):
            self.assertIs(receipt[key], False)
        self.assertEqual(receipt["train_count"], 192)
        self.assertEqual(receipt["triples"], 64)


if __name__ == "__main__":
    unittest.main()
