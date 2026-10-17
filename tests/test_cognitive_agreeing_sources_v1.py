"""Frozen synthetic source-guard ablation, not a model/independent review.

Measure supported useful answer false-HOLD and unsupported false acceptance
as separate denominators. These cases are developer-authored, not blind.
"""
from dataclasses import replace
import unittest

from cognitive_continuity_eval_v1 import (
    CognitiveObservation, CognitivePrediction, CognitiveSource, guard_prediction,
)


class SourceAblationTests(unittest.TestCase):
    def setUp(self):
        self.src=CognitiveSource("ev:original","machine","veraos","Lappy")
        self.obs=CognitiveObservation(
            case_id="measured:agree",category="multi_turn",
            task_id="task:one",request_id="request:one",
            utterance="Which host did I specify?",
            sources=(self.src,),referent="machine",scope="veraos",
            task_head="1"*64,accepted_task_head="1"*64,
            request_source_digest="2"*64,accepted_source_digest="2"*64)
        self.pred=CognitivePrediction(
            status="ANSWER",task_id="task:one",request_id="request:one",
            referent="machine",scope="veraos",
            source_event_id="ev:original",claimed_value="Lappy")

    def test_agreeing_active_events_preserve_original_exact_citation(self):
        obs=replace(self.obs,sources=(self.src,
            CognitiveSource("ev:agree","machine","veraos","Lappy")))
        self.assertEqual(guard_prediction(obs,self.pred).status,"ANSWER")

    def test_agreeing_active_events_preserve_second_exact_citation(self):
        second=CognitiveSource("ev:agree","machine","veraos","Lappy")
        obs=replace(self.obs,sources=(self.src,second))
        self.assertEqual(guard_prediction(
            obs,replace(self.pred,source_event_id=second.source_event_id)).status,"ANSWER")

    def test_unrelated_source_does_not_block_supported_answer(self):
        obs=replace(self.obs,sources=(self.src,
            CognitiveSource("ev:other","machine","research","VN1")))
        self.assertEqual(guard_prediction(obs,self.pred).status,"ANSWER")

    def test_conflict_still_holds_even_if_one_source_is_cited(self):
        obs=replace(self.obs,sources=(self.src,
            CognitiveSource("ev:conflict","machine","veraos","VN1")))
        self.assertEqual(guard_prediction(obs,self.pred).status,"HOLD")

    def test_duplicate_source_id_still_holds_even_when_values_agree(self):
        obs=replace(self.obs,sources=(self.src,self.src))
        self.assertEqual(guard_prediction(obs,self.pred).status,"HOLD")

    def test_citation_not_in_active_sources_holds(self):
        obs=replace(self.obs,sources=(self.src,
            CognitiveSource("ev:agree","machine","veraos","Lappy")))
        self.assertEqual(guard_prediction(
            obs,replace(self.pred,source_event_id="ev:invented")).status,"HOLD")

    def test_stale_task_head_holds_despite_two_agreeing_sources(self):
        obs=replace(self.obs,sources=(self.src,
            CognitiveSource("ev:agree","machine","veraos","Lappy")),
            task_head="3"*64)
        self.assertEqual(guard_prediction(obs,self.pred).status,"HOLD")

    def test_superseded_original_source_holds(self):
        correction=CognitiveSource("ev:corrected","machine","veraos","VN1",
            kind="CORRECTION",supersedes="ev:original")
        obs=replace(self.obs,sources=(self.src,correction))
        self.assertEqual(guard_prediction(obs,self.pred).status,"HOLD")

    def test_current_correction_preserves_new_exact_citation(self):
        correction=CognitiveSource("ev:corrected","machine","veraos","VN1",
            kind="CORRECTION",supersedes="ev:original")
        obs=replace(self.obs,sources=(self.src,correction),
            previous_interaction_id="ev:original",
            utterance="Actually the host is VN1, not Lappy.")
        p=replace(self.pred,source_event_id="ev:corrected",claimed_value="VN1")
        class Corrected:
            class speech_act:
                value="CORRECTION"
        self.assertEqual(guard_prediction(obs,p,
            interpret=lambda utterance,prev:Corrected()).status,"ANSWER")

    def test_conflicting_active_correction_holds(self):
        correction=CognitiveSource("ev:corrected","machine","veraos","VN1",
            kind="CORRECTION",supersedes="ev:original")
        obs=replace(self.obs,sources=(self.src,correction,
            CognitiveSource("ev:third","machine","veraos","Lappy")))
        self.assertEqual(guard_prediction(obs,
            replace(self.pred,source_event_id="ev:corrected",claimed_value="VN1")).status,"HOLD")

    def test_reproducible_prechange_ablation_counts_and_digest(self):
        from evaluate_source_agreement_ablation_v1 import (
            frozen_cases,previous_unique_source_guard,score,
        )
        cases=frozen_cases()
        baseline=score(cases,previous_unique_source_guard)
        candidate=score(cases,guard_prediction)
        self.assertEqual((baseline["preserved"],baseline["false_holds"],
                          baseline["correct_holds"],baseline["unsafe_acceptances"]),
                         (1,2,3,0))
        self.assertEqual((candidate["preserved"],candidate["false_holds"],
                          candidate["correct_holds"],candidate["unsafe_acceptances"]),
                         (3,0,3,0))

    def test_frozen_tradeoff_metrics_separate_false_hold_from_unsafe_answer(self):
        agree=replace(self.obs,sources=(self.src,
            CognitiveSource("ev:agree","machine","veraos","Lappy")))
        disputed=replace(self.obs,sources=(self.src,
            CognitiveSource("ev:dispute","machine","veraos","VN1")))
        stale=replace(self.obs,task_head="3"*64)
        unsupported=replace(self.pred,source_event_id="ev:invented")
        supported=((self.obs,self.pred),(agree,self.pred),
                   (agree,replace(self.pred,source_event_id="ev:agree")))
        unsupported_cases=((disputed,self.pred),(stale,self.pred),
                           (self.obs,unsupported))
        false_holds=sum(guard_prediction(o,p).status!="ANSWER" for o,p in supported)
        unsafe=sum(guard_prediction(o,p).status=="ANSWER" for o,p in unsupported_cases)
        self.assertEqual((len(supported),false_holds), (3,0))
        self.assertEqual((len(unsupported_cases),unsafe),(3,0))
