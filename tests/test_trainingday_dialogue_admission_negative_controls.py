"""Adversarial, fictional dialogue-candidate admission boundary regressions.

These tests do not grant rights, issue signatures, or run a model. They catch
two public-dataset laundering routes before any future training consumer can
mistake synthetic TRAIN/DEV rows for admitted or independent material.
"""

import unittest

from trainingday import dialogue_curriculum as dialogue


class DialogueAdmissionNegativeControls(unittest.TestCase):
    def test_normalized_train_dev_prompt_leakage_is_rejected(self):
        rows = dialogue.candidate_rows()
        train = next(row for row in rows if row["split"] == "train")
        dev = next(row for row in rows if row["split"] == "dev")

        # Same question after NFKC/case/whitespace normalization, despite
        # different exact bytes and untouched opaque case IDs.
        prompt = train["messages"][-2]["content"]
        dev["messages"][-2]["content"] = " \tＷ" + prompt[1:].upper() + "  "
        self.assertNotEqual(prompt, dev["messages"][-2]["content"])

        with self.assertRaisesRegex(ValueError, "normalized prompt collision"):
            dialogue.validate(rows)

    def test_default_ignorables_cannot_hide_train_dev_prompt_aliases(self):
        invisible_variants = (
            "\u200b",  # zero-width space
            "\ufeff",  # zero-width no-break space
            "\u2060",  # word joiner
            "\u200d",  # zero-width joiner: reject, never silently strip
            "\u034f",  # combining grapheme joiner
            "\ufe0f",  # variation selector 16
            "\u3164",  # Hangul filler
            "\U000e0100",  # supplementary variation selector
        )
        for invisible in invisible_variants:
            with self.subTest(codepoint=f"U+{ord(invisible):04X}"):
                rows = dialogue.candidate_rows()
                train = next(row for row in rows if row["split"] == "train")
                dev = next(row for row in rows if row["split"] == "dev")
                original = train["messages"][-2]["content"]
                dev["messages"][-2]["content"] = original[:3] + invisible + original[3:]

                with self.assertRaisesRegex(ValueError, "default-ignorable"):
                    dialogue.validate(rows)
                # The guard must reject uncertain text, not silently remove a
                # semantically meaningful joiner or selector from the input.
                self.assertIn(invisible, dev["messages"][-2]["content"])

    def test_visible_non_ascii_text_is_not_blanket_rejected(self):
        rows = dialogue.candidate_rows()
        dev = next(row for row in rows if row["split"] == "dev")
        dev["messages"][-2]["content"] = "A fictional cafe\u0301 has a naïve sign."
        dialogue.validate(rows)

    def test_unicode_18_default_ignorable_range_pin_is_bounded(self):
        ranges = dialogue._DEFAULT_IGNORABLE_U18
        self.assertEqual(sum(end - start + 1 for start, end in ranges), 4174)
        self.assertTrue(all(start <= end for start, end in ranges))
        self.assertTrue(all(left[1] < right[0]
                            for left, right in zip(ranges, ranges[1:])))

    def test_unrecognized_promotion_claims_cannot_be_smuggled_into_rows(self):
        for forged_claim in ("rights_admitted", "promotion_allowed",
                             "protected_evaluation_passed"):
            with self.subTest(forged_claim=forged_claim):
                rows = dialogue.candidate_rows()
                rows[0][forged_claim] = True
                self.assertIs(rows[0]["training_authorized"], False)
                self.assertEqual(rows[0]["rights_status"], "NOT_ADMITTED")

                with self.assertRaisesRegex(ValueError, "candidate row fields"):
                    dialogue.validate(rows)

        rows = dialogue.candidate_rows()
        rows[0]["messages"][-1]["rights_admitted"] = True
        with self.assertRaisesRegex(ValueError, "dialogue message fields"):
            dialogue.validate(rows)


if __name__ == "__main__":
    unittest.main()
