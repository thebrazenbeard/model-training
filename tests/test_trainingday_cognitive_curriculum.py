"""Independent per-row oracle tests for R202's original TRAIN-only source."""

import hashlib
import itertools
import json
import re
import unittest

from trainingday import cognitive_curriculum as candidate


_MATH = re.compile(
    r"Start at (-?\d+); add (\d+); multiply the result by (\d+); "
    r"subtract (\d+). Is the final integer divisible by (\d+)\? "
    r"Answer YES or NO only\."
)
_LANGUAGE = re.compile(
    r"(\w+) holds the ([a-z ]+)\. (\w+) holds the ([a-z ]+)\. "
    r"(\w+) gives the ([a-z ]+) to (\w+)\. "
    r"(\w+) gives the ([a-z ]+) to (\w+)\. "
    r"Who now holds the ([a-z ]+)\? Answer with one name\."
)
_TOOL = re.compile(r"Available tools: sum_ints\(a,b\), sort_ints\(values\), "
                   r"or no_tool\. Return only compact JSON with keys tool and args\. (.*)")
_PLAN = re.compile(r"([VWXYZ]) requires ([VWXYZ,]+|none) and takes (\d+) unit\(s\)")
_ERROR = re.compile(
    r"Starting value (-?\d+), operations in order are \+(\d+), \*(\d+), -(\d+)\. "
    r"A work log claims step1=(-?\d+), step2=(-?\d+), step3=(-?\d+)\. "
    r"Identify the earliest incorrect step and its correct value\. "
    r"Answer 'STEP N: VALUE' only\."
)
_UNCERTAINTY = re.compile(
    r"Access requires ALL: level >= (\d+), completed (\w+) training, "
    r"and open_orders <= (\d+)\. .*Record: level=(unknown|-?\d+); "
    r"training=(unknown|incomplete|complete); open_orders=(unknown|-?\d+)\. "
    r"Answer ACCEPT, REJECT, or INSUFFICIENT only\."
)


class CognitiveCurriculumTests(unittest.TestCase):
    def test_frozen_size_identity_and_integrity(self):
        rows = candidate.train_rows()
        self.assertEqual(len(rows), 1536)
        self.assertEqual(len({r["id"] for r in rows}), len(rows))
        self.assertEqual(len({r["prompt"] for r in rows}), len(rows))
        for family in candidate.FAMILIES:
            self.assertEqual(sum(r["id"].startswith(f"r202-{family}-train-")
                                 for r in rows), 256)
        for row in rows:
            self.assertEqual(set(row), {"id", "prompt", "answer", "split"})
            self.assertEqual(row["split"], "train")
            self.assertTrue(0 < len(row["answer"]) <= 512)
            self.assertTrue(0 < len(row["prompt"]) <= 512)
            self.assertNotIn("<eval_answer>", row["prompt"])
        self.assertEqual(hashlib.sha256(candidate.train_bytes()).hexdigest(),
                         "e95563a1672d531b884b5b6003f4f38412e42a09e7aa4685adec50338c294bd1")
        self.assertEqual(candidate.train_rows(), rows)

    def test_math_oracle_independent(self):
        for i in range(256):
            row = candidate.train_row("math_logic", i)
            m = _MATH.fullmatch(row["prompt"])
            self.assertIsNotNone(m)
            start, add, multiplier, subtract, divisor = map(int, m.groups())
            number = start
            for op, parameter in ((lambda x, y: x + y, add),
                                  (lambda x, y: x * y, multiplier),
                                  (lambda x, y: x - y, subtract)):
                number = op(number, parameter)
            self.assertEqual(row["answer"], "YES" if number % divisor == 0 else "NO")

    def test_language_object_tracking_independent(self):
        for i in range(256):
            row = candidate.train_row("language_tracking", i)
            m = _LANGUAGE.fullmatch(row["prompt"])
            self.assertIsNotNone(m)
            owner1, item1, owner2, item2, src1, item3, dst1, src2, item4, dst2, query = m.groups()
            self.assertNotEqual(item1, item2)
            holders = {item1: owner1, item2: owner2}
            for source, item, dest in ((src1, item3, dst1), (src2, item4, dst2)):
                self.assertEqual(holders[item], source)
                holders[item] = dest
            self.assertEqual(row["answer"], holders[query])

    def test_tool_choice_schema_and_arguments_independent(self):
        for i in range(256):
            row = candidate.train_row("tool_selection", i)
            m = _TOOL.fullmatch(row["prompt"])
            self.assertIsNotNone(m)
            task = m.group(1)
            obj = json.loads(row["answer"])
            self.assertEqual(set(obj), {"tool", "args"})
            self.assertEqual(row["answer"], json.dumps(obj, separators=(",", ":"), sort_keys=True))
            if "Select a tool to add integers" in task:
                n = re.fullmatch(r"Select a tool to add integers (-?\d+) and (-?\d+) "
                                 r"to check whether the claimed sum (-?\d+) is correct\.", task)
                self.assertIsNotNone(n)
                a, b, _claim = map(int, n.groups())
                self.assertEqual(obj, {"tool": "sum_ints", "args": {"a": a, "b": b}})
            elif "Select a tool to sort" in task:
                n = re.fullmatch(r"Select a tool to sort the integers (\[[^\]]+\]) ascending\.", task)
                self.assertIsNotNone(n)
                self.assertEqual(obj, {"tool": "sort_ints",
                                       "args": {"values": json.loads(n.group(1))}})
            else:
                self.assertIn("No weather tool is available.", task)
                self.assertEqual(obj, {"tool": "no_tool", "args": {}})

    def test_planning_brute_force_lex_policy(self):
        for i in range(256):
            row = candidate.train_row("dependency_planning", i)
            specs = _PLAN.findall(row["prompt"])
            self.assertEqual(len(specs), 5)
            requirements = {name: set() if raw == "none" else set(raw.split(","))
                            for name, raw, _duration in specs}
            durations = {name: int(duration) for name, _deps, duration in specs}
            self.assertEqual(set(requirements), set("VWXYZ"))
            target = row["answer"].split(",")
            self.assertEqual(len(target), 5)
            self.assertEqual(set(target), set(requirements))
            for j, name in enumerate(target):
                self.assertTrue(requirements[name].issubset(set(target[:j])))
                remaining = set(requirements) - set(target[:j])
                eligible = [task for task in remaining
                            if requirements[task].issubset(set(target[:j]))]
                self.assertEqual(name, min(eligible, key=lambda task: (durations[task], task)))
            self.assertTrue(any(all(requirements[name].issubset(set(order[:j]))
                                    for j, name in enumerate(order))
                                for order in itertools.permutations("VWXYZ")))

    def test_error_detection_independent(self):
        for i in range(256):
            row = candidate.train_row("error_correction", i)
            m = _ERROR.fullmatch(row["prompt"])
            self.assertIsNotNone(m)
            start, add, mult, sub, shown1, shown2, shown3 = map(int, m.groups())
            actual = [start + add, (start + add) * mult, (start + add) * mult - sub]
            shown = [shown1, shown2, shown3]
            bad = [j + 1 for j in range(3) if shown[j] != actual[j]]
            self.assertEqual(len(bad), 1)
            self.assertEqual(row["answer"], f"STEP {bad[0]}: {actual[bad[0] - 1]}")

    def test_uncertainty_open_world_three_valued_oracle(self):
        for i in range(256):
            row = candidate.train_row("evidence_uncertainty", i)
            m = _UNCERTAINTY.fullmatch(row["prompt"])
            self.assertIsNotNone(m)
            threshold, _module, max_orders, level, training, orders = m.groups()
            facts = [None if level == "unknown" else int(level) >= int(threshold),
                     None if training == "unknown" else training == "complete",
                     None if orders == "unknown" else int(orders) <= int(max_orders)]
            decision = ("REJECT" if False in facts else
                        "ACCEPT" if all(fact is True for fact in facts) else "INSUFFICIENT")
            self.assertEqual(row["answer"], decision)

    def test_bad_indices_or_families_and_hard_hold(self):
        for name, index in (("invalid", 0), ("math_logic", -1),
                            ("math_logic", 256), ("math_logic", True),
                            ("math_logic", 0.5), ("", 1)):
            with self.subTest(name=name, index=index), self.assertRaises(candidate.CognitiveCurriculumError):
                candidate.train_row(name, index)
        receipt = candidate.source_commitment()
        for key in ("rights_holder_authenticated", "data_rights_granted",
                    "evaluation_custody_verified", "semantic_disjointness_independently_verified",
                    "model_tokenizer_fit_verified", "gpu_lease_granted", "authorizes_training",
                    "private_dev_included"):
            self.assertIs(receipt[key], False)
        self.assertEqual(receipt["status"], "SOURCE_ONLY_UNAUTHENTICATED_HOLD")
        self.assertNotIn("private_dev_sha256", receipt)


if __name__ == "__main__":
    unittest.main()
