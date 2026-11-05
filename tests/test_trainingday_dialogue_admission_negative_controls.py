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
