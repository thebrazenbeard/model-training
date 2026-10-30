"""Matched repeatability and provenance for toy CPU optimizer study."""
import unittest
from trainingday.mezo_multiseed import summarize, run_experiment

class MultiSeedTests(unittest.TestCase):
    def test_no_gpu_or_official_model_and_frozen_same_for_each_seed(self):
        report = run_experiment(seeds=[71], step_budgets=[2])
        self.assertEqual(report["status"], "TOY_ONLY_CPU_MEZO_ADAMW_ABLATION")
        records=report["arms"]
        self.assertEqual(len(records), 6)
        self.assertEqual({r["initial_sha256"] for r in records}.__len__(),1)
        self.assertEqual({r["heldout_sha256"] for r in records}.__len__(),1)
        self.assertTrue(all(not r["official_4b_model_trained"] for r in records))
        self.assertTrue(all(r["cuda_allocations"]==0 for r in records))
    def test_zeroupdate_digests_and_forward_counts(self):
        report = run_experiment(seeds=[72], step_budgets=[3])
        rows = report["arms"]
        zeros=[r for r in rows if r["mode"]=="zero"]
        self.assertEqual(len(zeros),2)
        self.assertTrue(all(r["initial_sha256"] == r["final_sha256"] for r in zeros))
        self.assertEqual(next(r["forward_evaluations"] for r in rows if r["method"]=="mezo" and r["mode"]=="correct"),6)
        self.assertEqual(next(r["forward_evaluations"] for r in rows if r["method"]=="adamw" and r["mode"]=="correct"),3)
    def test_summary_does_not_falsely_claim_model_learning(self):
        report = run_experiment(seeds=[73,74], step_budgets=[2,3])
        summary=summarize(report)
        self.assertEqual(len(report["arms"]),20)
        self.assertFalse(summary["vera_neural_qualification"])
        self.assertEqual(len(summary["metrics"]),8)
