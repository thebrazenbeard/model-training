"""Independent internal-consistency tests of local diagnostic receipts."""
from pathlib import Path
from tempfile import TemporaryDirectory
import copy
import json
import unittest

from verify_semantic_transfer_receipt_v1 import verify_receipt
from semantic_transfer_eval_v1 import TransferEvidenceError

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"training"/"semantic_dev_evidence"
CASES=ROOT/"training"/"semantic_transfer_devset_v1.json"


class LocalReceiptTests(unittest.TestCase):
    def test_two_real_local_receipts_recompute_from_frozen_case_set(self):
        expected={"minimal":1,"codebook":13}
        for profile,total in expected.items():
            with self.subTest(profile=profile):
                path=DATA/f"lappy_qwen3_5_4b_{profile}25_v1.json"
                result=verify_receipt(path,CASES)
                self.assertEqual(result["exact_matches"],total)
                self.assertEqual(result["valid_json_responses"],25)
                self.assertEqual(result["integrity"],"CONSISTENT_UNATTESTED")
                self.assertEqual(result["qualification"],"NONE")
                self.assertEqual(result["profile"],profile)

    def test_tampered_prediction_is_rejected(self):
        path=DATA/"lappy_qwen3_5_4b_codebook25_v1.json"
        raw=json.loads(path.read_text(encoding="utf8"))
        with TemporaryDirectory() as td:
            temp=Path(td)/"changed.json"
            for kind in ("prediction", "raw", "prompt", "score", "case_order"):
                payload=copy.deepcopy(raw)
                if kind=="prediction":
                    payload["prediction_receipt"]["predictions"]["scope-01"]={"decision":"PASS","value":True}
                elif kind=="raw":
                    payload["model_case_results"][0]["model_response"]='{"decision":"DONE","value":true}'
                elif kind=="prompt":
                    payload["prompt_profile_sha256"]="0"*64
                elif kind=="score":
                    payload["report"]["scores"][0]["passed"]=5
                elif kind=="case_order":
                    payload["model_case_results"][0]["id"]="scope-01"
                temp.write_text(json.dumps(payload,sort_keys=True),encoding="utf8")
                with self.subTest(kind=kind), self.assertRaises(TransferEvidenceError):
                    verify_receipt(temp,CASES)

    def test_cannot_relabel_local_development_as_qualification(self):
        raw=json.loads((DATA/"lappy_qwen3_5_4b_codebook25_v1.json").read_text())
        raw["report"]["qualification"]="QUALIFIED"
        with TemporaryDirectory() as td:
            p=Path(td)/"forged.json"
            p.write_text(json.dumps(raw),encoding="utf8")
            with self.assertRaises(TransferEvidenceError):
                verify_receipt(p,CASES)

if __name__=="__main__":
    unittest.main()
