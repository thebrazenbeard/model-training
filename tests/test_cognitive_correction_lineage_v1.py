"""Developer-controlled correction ancestry falsifier. No model/runtime effects."""
from dataclasses import replace
import unittest
from cognitive_continuity_eval_v1 import (
    CognitiveObservation, CognitivePrediction, CognitiveSource, guard_prediction,
)

class CorrectionLineageTests(unittest.TestCase):
    def setUp(self):
        self.original=CognitiveSource("event:old","purpose","role","personal")
        self.corrected=CognitiveSource("event:new","purpose","role","organizational",
            "CORRECTION","event:old")
        self.obs=CognitiveObservation(
            case_id="lineage-1", category="correction",
            task_id="task:one",request_id="request:one",
            utterance="Actually I meant the role, not my own purpose.",
            sources=(self.original,self.corrected),referent="purpose",scope="role",
            task_head="a"*64,accepted_task_head="a"*64,
            request_source_digest="b"*64,accepted_source_digest="b"*64,
            previous_interaction_id="event:old")
        self.pred=CognitivePrediction("ANSWER","task:one","request:one",
            "purpose","role","organizational","event:new")

    def test_valid_correction_of_present_same_scope_source_is_preserved(self):
        self.assertEqual(guard_prediction(self.obs,self.pred).status,"ANSWER")

    def test_correction_without_named_predecessor_cannot_bootstrap_truth(self):
        orphan=replace(self.corrected,supersedes=None)
        self.assertEqual(guard_prediction(
            replace(self.obs,sources=(orphan,),previous_interaction_id=None),
            self.pred).status,"HOLD")

    def test_missing_original_cannot_bootstrap_a_correction(self):
        self.assertEqual(guard_prediction(
            replace(self.obs,sources=(self.corrected,)),self.pred).status,"HOLD")

    def test_untyped_statement_cannot_supersede_an_original(self):
        forged=replace(self.corrected,kind="USER_STATEMENT")
        self.assertEqual(guard_prediction(
            replace(self.obs,sources=(self.original,forged)),self.pred).status,"HOLD")

    def test_wrong_scope_original_cannot_authorize_correction(self):
        foreign=replace(self.original,scope="personal")
        self.assertEqual(guard_prediction(
            replace(self.obs,sources=(foreign,self.corrected)),self.pred).status,"HOLD")

    def test_wrong_referent_original_cannot_authorize_correction(self):
        foreign=replace(self.original,referent="person")
        self.assertEqual(guard_prediction(
            replace(self.obs,sources=(foreign,self.corrected)),self.pred).status,"HOLD")

    def test_corrupted_previous_interaction_id_cannot_be_silently_ignored(self):
        self.assertEqual(guard_prediction(
            replace(self.obs,previous_interaction_id="event:unrelated"),
            self.pred).status,"HOLD")

    def test_stale_task_still_refused_despite_valid_correction(self):
        self.assertEqual(guard_prediction(
            replace(self.obs,task_head="c"*64),self.pred).status,"HOLD")

    def test_superseded_original_claim_not_displayed(self):
        prior=replace(self.pred,claimed_value="personal",source_event_id="event:old")
        out=guard_prediction(self.obs,prior)
        self.assertEqual(out.status,"HOLD")
        self.assertIsNone(out.claimed_value)

    def test_two_agreeing_unsuperseded_sources_remain_valid(self):
        agree=CognitiveSource("event:agree","purpose","role","organizational")
        self.assertEqual(guard_prediction(
            replace(self.obs,sources=(self.original,self.corrected,agree)),
            self.pred).status,"ANSWER")

    def test_cross_task_request_remains_held(self):
        self.assertEqual(guard_prediction(
            self.obs,replace(self.pred,request_id="another")).status,"HOLD")
