"""Frozen machine-semantic development vectors; no model-success claim."""
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import replace
import copy
import json
import unittest

from semantic_transfer_eval_v1 import (
    SCHEMA, PREDICTION_SCHEMA, CATEGORIES, TransferEvidenceError,
    load_cases, score_predictions, load_predictions, digest,
)

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/"training"/"semantic_transfer_devset_v1.json"


class SemanticTransferDevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=load_cases(FIXTURE)

    def test_five_cases_per_category_with_canonical_digest(self):
        self.assertEqual(len(self.cases),25)
        self.assertEqual({k:sum(x["category"]==k for x in self.cases)
                          for k in CATEGORIES},{k:5 for k in CATEGORIES})
        raw=json.loads(FIXTURE.read_text(encoding="utf8"))
        self.assertEqual(digest(raw),digest({"schema":SCHEMA,"cases":list(self.cases)}))

    def test_frozen_positive_fixture_scores_all_not_qualified(self):
        preds={c["id"]:c["expected"] for c in self.cases}
        score=score_predictions(self.cases,preds,run_id="positive",
                                submission_kind="FIXTURE_POSITIVE")
        self.assertEqual(sum(k[1] for k in score.scores),25)
        self.assertEqual(score.failed_ids,())
        self.assertEqual(score.to_mapping()["qualification"],
                         "UNQUALIFIED_DEVELOPMENT_DIAGNOSTIC")

    def test_negative_control_cannot_be_falsely_promoted_by_self_claim(self):
        preds={c["id"]:{"decision":"PASS","value":"confidence:100"}
               for c in self.cases}
        score=score_predictions(self.cases,preds,run_id="negative",
                                submission_kind="FIXTURE_NEGATIVE")
        self.assertEqual(sum(k[1] for k in score.scores),0)
        self.assertEqual(len(score.failed_ids),25)

    def test_missing_predictions_count_as_failures_and_extra_keys_fail(self):
        score=score_predictions(self.cases,{},run_id="none")
        self.assertEqual(len(score.failed_ids),25)
        self.assertEqual(sum(k[1] for k in score.scores),0)
        preds={self.cases[0]["id"]:{**self.cases[0]["expected"],"is_correct":True}}
        score=score_predictions(self.cases,preds,run_id="claim")
        self.assertIn(self.cases[0]["id"],score.failed_ids)

    def test_unknown_case_id_rejected_before_scoring(self):
        with self.assertRaisesRegex(TransferEvidenceError,"Unknown predictions"):
            score_predictions(self.cases,{"invented-id":{"decision":"PASS","value":True}},
                              run_id="bad")

    def test_predicted_bool_does_not_equate_to_int_and_float_not_accepted(self):
        sample=next(c for c in self.cases if c["id"]=="correction-02")
        preds={sample["id"]:{"decision":"CURRENT","value":True}}
        score=score_predictions(self.cases,preds,run_id="bool")
        self.assertIn(sample["id"],score.failed_ids)
        with self.assertRaises(TransferEvidenceError):
            score_predictions(self.cases,
                              {sample["id"]:{"decision":"CURRENT","value":17.0}},run_id="float")

    def test_duplicate_ids_category_drift_and_unknown_schema_fail(self):
        raw=json.loads(FIXTURE.read_text())
        with TemporaryDirectory() as td:
            path=Path(td)/"cases.json"
            for modification in ("dup","category","schema"):
                tmp=copy.deepcopy(raw)
                if modification=="dup":
                    tmp["cases"][1]["id"]=tmp["cases"][0]["id"]
                elif modification=="category":
                    tmp["cases"][0]["category"]="invalid"
                else:
                    tmp["schema"]="OTHER"
                path.write_text(json.dumps(tmp),encoding="utf8")
                with self.subTest(modification=modification):
                    with self.assertRaises(TransferEvidenceError):
                        load_cases(path)

    def test_prediction_reader_has_closed_schema(self):
        preds={c["id"]:c["expected"] for c in self.cases}
        with TemporaryDirectory() as td:
            p=Path(td)/"out.json"
            p.write_text(json.dumps({"schema":PREDICTION_SCHEMA,
                                     "predictions":preds}),encoding="utf8")
            self.assertEqual(load_predictions(p),preds)
            p.write_text(json.dumps({"schema":"UNKNOWN","predictions":preds}),encoding="utf8")
            with self.assertRaises(TransferEvidenceError):
                load_predictions(p)


if __name__=="__main__":
    unittest.main()
