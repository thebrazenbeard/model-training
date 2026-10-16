"""R13 corpus must be frozen, balanced and independent from protected prompts."""
import collections
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from successor.experiments import build_r13_behavior_corrective as build


class R13CorpusContractTests(unittest.TestCase):
    def test_new_128_targeted_rows_balance_identity_and_judgment(self):
        rows, sources, donors, indices = build.prepare()
        self.assertEqual(len(rows), 50000)
        self.assertEqual(len(sources), 128)
        self.assertEqual(len(donors), 128)
        self.assertEqual(len(indices), 128)
        self.assertEqual(collections.Counter(x["axis"] for x in sources),
                         {"identity":64,"judgment":64})
        self.assertEqual(collections.Counter(x["training_use"] for x in rows[416:672]),
                         {"targeted_behavioral_sft":128,"general_rehearsal":128})

    def test_training_window_alternates_and_never_changes_outer_rows(self):
        rows,sources,donors,indices = build.prepare()
        self.assertTrue(all(rows[416+4*i]["record_id"]==sources[i]["id"] for i in range(64)))
        self.assertTrue(all(rows[418+4*i]["record_id"]==sources[64+i]["id"] for i in range(64)))
        self.assertTrue(all(rows[417+4*i]["record_id"]==donors[2*i]["record_id"] for i in range(64)))
        self.assertTrue(all(rows[419+4*i]["record_id"]==donors[2*i+1]["record_id"] for i in range(64)))
        self.assertTrue(all(i>=672 for i in indices))
        self.assertEqual(len(set(x["prompt"].casefold().strip() for x in rows[416:672])),256)

    def test_exact_heldout_prompt_excluded(self):
        rows,_,_,_=build.prepare()
        frozen={x["user"].casefold().strip() for x in build.jsonl(build.HOLDOUT)}
        self.assertFalse(frozen & {x["prompt"].casefold().strip() for x in rows[416:672]})
        self.assertEqual(build.sha(build.R12_CORPUS), build.R12_CORPUS_SHA)

    def test_frozen_source_rebuilds_bit_identical_spec_and_corpus(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            with patch.multiple(build, OUT_DIR=root/"new-corpus",
                                SPEC=root/"new-spec.json", BINDING=root/"new-binding.json"):
                # A separate test proves output collision rejection; suppress
                # just that gate while rebuilding source into this disposable
                # location, retaining the exact frozen adapter namespace.
                with patch.object(build,"require_fresh_outputs",return_value=None):
                    evidence=build.build()
                self.assertEqual(evidence["corpus_sha256"],
                    "90de3879113a99f85002ec884d504ee487e5ed699d9deaa0146290c47597b3c5")
                self.assertEqual(evidence["spec_sha256"],
                    "ad2a3d56c9a5ee648764e9469676b78155bb993de690edda9627189a767b0fdf")
                self.assertNotIn(b"\r\n",build.SPEC.read_bytes())

    def test_builder_must_not_overwrite_existing_artifacts(self):
        with TemporaryDirectory() as td:
            blocker = Path(td)/"r13-output"
            blocker.mkdir()
            with patch.object(build,"NEW_ADAPTER",blocker):
                with self.assertRaisesRegex(ValueError,"output"):
                    build.require_fresh_outputs()


if __name__=="__main__":
    unittest.main()
