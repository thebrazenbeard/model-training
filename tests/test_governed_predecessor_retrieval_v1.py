"""Actual disposable Mono #77 TaskExecutionLedger read-only recovery tests.

Task creation happens only in the test TEMP state. The adapter must not
open any tasks or append any event during readback, or admit any provider.
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless
from dataclasses import replace

from cognitive_continuity_eval_v1 import CognitiveObservation,CognitivePrediction,CognitiveSource,guard_prediction
from cognitive_governed_predecessor_v1 import recover_pruned_predecessor


class GovernedPredecessorTests(TestCase):
    def setUp(self):
        src=os.environ.get("VERA_MONO77_SOURCE_ROOT")
        if not src or not (Path(src)/"packages/vera_core/src/vera_core/local_conversation.py").is_file():
            self.skipTest("exact Mono77 source not present")
        try:
            import jsonschema  # noqa: F401 - actual composed Mono import requirement
        except ImportError:
            self.skipTest("exact Mono source present, but jsonschema runtime dependency absent")
        for pkg in (Path(src)/"packages").glob("*/src"):
            if str(pkg) not in sys.path:sys.path.insert(0,str(pkg))
        from vera_core.task_execution import TaskExecutionLedger
        from vera_core.local_conversation import open_message
        self.temp=tempfile.TemporaryDirectory(prefix="TRAINING-governed-predecessor-")
        self.addCleanup(self.temp.cleanup)
        self.ledger=TaskExecutionLedger(Path(self.temp.name)/"tasks.sqlite3")
        self.conversation="desktop-a"
        self.previous=open_message(self.ledger,self.conversation,"msg-1","personal")
        self.current=open_message(self.ledger,self.conversation,"msg-2","organizational")
        self.prev_ref="mono-task-open:"+self.previous["opened_head"]
        self.now_ref="mono-task-open:"+self.current["opened_head"]
        self.correction=CognitiveSource(self.now_ref,"purpose","role","organizational",
                                        "CORRECTION",self.prev_ref)
        self.obs=CognitiveObservation(
            "actual-ledger:correction","correction",self.current["task_id"],"msg-2",
            "organizational",(self.correction,),"purpose","role",
            self.ledger.read(self.current["task_id"]).journal_head_digest,
            self.ledger.read(self.current["task_id"]).journal_head_digest,
            self.current["source_digest"],self.current["source_digest"],
            self.prev_ref)
        self.pred=CognitivePrediction("ANSWER",self.current["task_id"],"msg-2",
                                      "purpose","role","organizational",self.now_ref)

    def assess(self,obs=None,pred=None,conv=None,prev="msg-1"):
        return recover_pruned_predecessor(self.ledger,
                    self.obs if obs is None else obs,
                    self.pred if pred is None else pred,
                    conversation_id=self.conversation if conv is None else conv,
                    predecessor_message_id=prev)

    def test_baseline_holds_pruned_source_even_if_user_says_it_exists(self):
        self.assertEqual(guard_prediction(self.obs,self.pred).status,"HOLD")

    def test_actual_mono_source_retrieval_restores_exact_correction_without_write(self):
        events_before=self.ledger.events()
        out=self.assess()
        self.assertEqual(out.status,"ANSWER")
        self.assertEqual(out.source_event_id,self.now_ref)
        self.assertEqual(self.ledger.events(),events_before)

    def test_missing_predecessor_is_hold_not_heuristic_acceptance(self):
        self.assertEqual(self.assess(prev="never-recorded").status,"HOLD")

    def test_wrong_conversation_is_hold(self):
        self.assertEqual(self.assess(conv="other-conversation").status,"HOLD")

    def test_wrong_prior_event_citation_is_hold(self):
        obs=replace(self.obs,sources=(replace(self.correction,
                    supersedes="mono-task-open:"+"f"*64),))
        self.assertEqual(self.assess(obs=obs).status,"HOLD")

    def test_changed_accepted_source_digest_is_hold(self):
        obs=replace(self.obs,accepted_source_digest="1"*64,
                         request_source_digest="1"*64)
        self.assertEqual(self.assess(obs=obs).status,"HOLD")

    def test_stale_task_head_after_unrelated_task_open_is_hold(self):
        from vera_core.local_conversation import open_message
        open_message(self.ledger,"other","msg-3","an unrelated task")
        self.assertEqual(self.assess().status,"HOLD")

    def test_correction_value_different_from_canonical_text_is_hold(self):
        tampered=replace(self.correction,value="an unverified value")
        obs=replace(self.obs,sources=(tampered,))
        pred=replace(self.pred,claimed_value="an unverified value")
        self.assertEqual(self.assess(obs=obs,pred=pred).status,"HOLD")

    def test_fake_source_id_for_current_opener_is_hold(self):
        obs=replace(self.obs,sources=(replace(self.correction,
                    source_event_id="event:claimed"),))
        pred=replace(self.pred,source_event_id="event:claimed")
        self.assertEqual(self.assess(obs=obs,pred=pred).status,"HOLD")

    def test_foreign_conversation_same_locator_is_not_recovered(self):
        from vera_core.local_conversation import open_message
        open_message(self.ledger,"other-conversation","only-foreign","private")
        # Current conversation has no such prior message, even though
        # another conversation has exactly that message locator.
        self.assertEqual(self.assess(prev="only-foreign").status,"HOLD")

    def test_unknown_cancelled_ancestor_history_is_not_automatically_factual_truth(self):
        # ANSWER is a source-consistency status, not an independent fact
        self.assertEqual(self.assess().status,"ANSWER")
        self.assertEqual(self.current["origin_authenticated"],False)
