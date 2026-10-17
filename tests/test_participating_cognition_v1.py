"""Vera shared identity / separate participation source experiment tests."""
from dataclasses import replace
import copy
import importlib.util
import subprocess
import os
from pathlib import Path
import sys
import unittest

from participating_cognition_v1 import (
    ParticipantBinding,Contribution,compose_episode,ParticipationError,
)
from independent_challenge_eval_v1 import AttributedClaim

ROOT=Path(__file__).resolve().parents[1]
SHA="a"*40

def source(model=None,node="VN1",agent="agent:vn1",inst="runtime:one",value=True):
    model_sha=model
    p=ParticipantBinding("Vera",node,agent,inst,model_sha,SHA)
    c=AttributedClaim(
        message_id="observation:"+node,
        shared_identity="Vera",node_id=node,agent_id=agent,
        model_digest=model_sha,claim_type="RUNTIME_OBSERVATION",
        referent="link:1",predicate="healthy",scope="boot:1",revision=1,
        evidence_digest=("a" if node=="VN1" else "b")*64,
        evidence_group="evidence:"+node,value=value,
    )
    return Contribution(p,c,"task:one")

def plan():
    return dict(
        source_profile="vera-core",target_profile="mosaic",
        required_capabilities=["typed:referent"],native_capabilities=["typed:referent"],
        missing_capabilities=[],unresolved_capabilities=[],rewrites=[],
        fidelity="EXACT",semantic_equivalence="NOT_ESTABLISHED",
        authorization_effect="NONE",catalog_id=None,catalog_digest=None)

def mosaic():
    return dict(case_id="case:1",family="recovery",condition="MOSAIC_STATE",
                answer="ON",expected_value="ON",correct=True,
                stale_error=False,correction_burden=0,payload_bytes=80)

def cricket():
    return {"disposition":"PASS","findings":[]}

def compose(first=None,second=None,**kw):
    args=dict(core_plan=plan(),mosaic_result=mosaic(),cricket_result=cricket())
    args.update(kw)
    return compose_episode(first or source(),second or source(model="c"*64,
        node="Lappy",agent="agent:lappy",inst="runtime:two",value=False),**args)


class ParticipantBoundaryTests(unittest.TestCase):
    def test_distinct_sources_coexist_without_grant_or_adoption(self):
        result=compose()
        out=result.as_dict()
        self.assertEqual(out["schema"],"VERA_PARTICIPATING_COGNITION_SOURCE_V1")
        self.assertEqual(out["contributions"][0]["participant"]["node_id"],"VN1")
        self.assertEqual(out["contributions"][1]["participant"]["node_id"],"Lappy")
        self.assertEqual(out["contributions"][0]["claim"]["value"],True)
        self.assertEqual(out["contributions"][1]["claim"]["value"],False)
        self.assertNotEqual(out["contributions"][0]["claim_sha256"],
                            out["contributions"][1]["claim_sha256"])
        self.assertEqual(out["dissent"]["classification"],"UNVERIFIED_DISSENT")
        self.assertFalse(out["dissent"]["accepted_as_fact"])
        self.assertEqual(out["authority_effect"],"NONE")
        self.assertEqual(out["identity_effect"],"NONE")
        self.assertEqual(out["shared_memory_effect"],"NONE")
        self.assertEqual(out["model_independence"],"NOT_ESTABLISHED")
        self.assertEqual(out["cognitive_gain"],"NOT_ESTABLISHED")

    def test_identical_neural_model_does_not_create_independent_verification(self):
        one=source(model="c"*64)
        two=source(model="c"*64,node="Lappy",agent="agent:lappy",inst="runtime:two",value=False)
        out=compose(one,two).as_dict()
        self.assertEqual(out["dissent"]["classification"],"UNVERIFIED_DISSENT")
        self.assertEqual(out["model_independence"],"NOT_ESTABLISHED")

    def test_same_node_cannot_be_double_counted_as_distinct_runtime(self):
        x=source()
        y=source(value=False)
        with self.assertRaises(ParticipationError):
            compose(x,y)

    def test_restarted_same_agent_is_not_a_second_participant(self):
        first=source(inst="runtime:initial")
        restarted=source(inst="runtime:replacement",value=False)
        with self.assertRaisesRegex(ParticipationError,"Same logical agent"):
            compose(first,restarted)

    def test_migrated_logical_agent_is_not_a_second_participant(self):
        first=source(node="VN1",agent="agent:one",inst="runtime:old")
        migrated=source(node="Lappy",agent="agent:one",inst="runtime:new",value=False)
        with self.assertRaisesRegex(ParticipationError,"Same logical agent"):
            compose(first,migrated)

    def test_distinct_agents_on_same_physical_node_remain_distinct(self):
        first=source(node="VN1",agent="agent:one",inst="runtime:one")
        second=source(node="VN1",agent="agent:two",inst="runtime:two",value=False)
        out=compose(first,second).as_dict()
        self.assertEqual(out["contributions"][0]["participant"]["agent_id"],"agent:one")
        self.assertEqual(out["contributions"][1]["participant"]["agent_id"],"agent:two")

    def test_same_runtime_distinct_agents_are_correlated_not_independent_conflict(self):
        a=source(node="VN1",agent="agent:one",inst="runtime:shared",value=True)
        b=source(node="Lappy",agent="agent:two",inst="runtime:shared",value=False)
        checks=[]
        def independent(*args):
            checks.append(args)
            return True
        receipt=compose(a,b,verify_evidence=lambda *args: True,
                        verify_independence=independent).as_dict()
        self.assertEqual(receipt["dissent"]["classification"],"SELF_OR_CORRELATED_CLAIMS")
        self.assertFalse(receipt["dissent"]["challenge_for_review"])
        self.assertEqual(checks,[])
        self.assertEqual(len(receipt["contributions"]),2)
        self.assertEqual(receipt["authority_effect"],"NONE")

    def test_claim_only_independence_does_not_prove_distinct_runtimes(self):
        receipt=compose(verify_evidence=lambda *args: True,
                        verify_independence=lambda *args: True).as_dict()
        self.assertEqual(receipt["dissent"]["classification"],"UNVERIFIED_DISSENT")
        self.assertFalse(receipt["dissent"]["challenge_for_review"])

    def test_pair_verifier_receives_exact_participant_bindings(self):
        seen=[]
        def pair_check(a,b):
            seen.append((a.participant.runtime_instance_id,
                         b.participant.runtime_instance_id,
                         a.participant.agent_id,b.participant.agent_id))
            return True
        receipt=compose(verify_evidence=lambda *args: True,
                        verify_independence=lambda *args: True,
                        verify_participant_independence=pair_check).as_dict()
        self.assertEqual(receipt["dissent"]["classification"],"EVIDENCE_CONFLICT_FOR_REVIEW")
        self.assertEqual(seen,[("runtime:one","runtime:two","agent:vn1","agent:lappy")])
        self.assertFalse(receipt["dissent"]["independence_proven"])
        self.assertEqual(receipt["authority_effect"],"NONE")

    def test_pair_independence_verifier_errors_and_truthy_values_fail_closed(self):
        def fail(*args):
            raise RuntimeError("receiver verifier unavailable")
        for verifier in (lambda *args: 1, lambda *args: "yes", fail):
            receipt=compose(verify_evidence=lambda *args: True,
                            verify_independence=lambda *args: True,
                            verify_participant_independence=verifier).as_dict()
            self.assertEqual(receipt["dissent"]["classification"],"UNVERIFIED_DISSENT")
            self.assertFalse(receipt["dissent"]["challenge_for_review"])

    def test_cricket_pass_with_block_finding_rejected(self):
        finding=dict(rule_id="deny:effect",severity="BLOCK",title="Blocked",
                     rationale="Protected effect",evidence="No permission",
                     recommendation="Hold",source="Cricket")
        with self.assertRaisesRegex(ParticipationError,"Contradictory Cricket"):
            compose(cricket_result={"disposition":"PASS","findings":[finding]})
        with self.assertRaisesRegex(ParticipationError,"Contradictory Cricket"):
            compose(cricket_result={"disposition":"CHALLENGE","findings":[finding]})
        blocked=compose(cricket_result={"disposition":"BLOCK","findings":[finding]}).as_dict()
        self.assertEqual(blocked["status"],"CRICKET_BLOCKED")
        self.assertEqual(blocked["authority_effect"],"NONE")

    def test_identity_node_model_substitution_rejected(self):
        a=source()
        b=source(node="Lappy",agent="agent:lappy",inst="runtime:two")
        with self.assertRaises(ParticipationError):
            replace(b,participant=replace(b.participant,model_sha256="c"*64))

    def test_different_task_cannot_be_composed(self):
        with self.assertRaises(ParticipationError):
            compose(second=replace(source(node="Lappy",agent="agent:lappy",inst="runtime:two"),task_id="task:other"))

    def test_cricket_block_status_survives_semantic_unrepresentability(self):
        p=plan()
        p.update(fidelity="UNREPRESENTABLE",
                 required_capabilities=["typed:referent","latent:unknown"],
                 missing_capabilities=["latent:unknown"],
                 unresolved_capabilities=["latent:unknown"])
        finding=dict(rule_id="deny:effect",severity="BLOCK",title="Blocked",
                     rationale="Protected effect",evidence="No permission",
                     recommendation="Hold",source="Cricket")
        receipt=compose(core_plan=p,
                        cricket_result={"disposition":"BLOCK","findings":[finding]}).as_dict()
        self.assertEqual(receipt["status"],"CRICKET_BLOCKED")
        self.assertEqual(receipt["core_semantic_plan"]["fidelity"],"UNREPRESENTABLE")
        self.assertEqual(receipt["authority_effect"],"NONE")

    def test_cricket_disposition_must_equal_actual_maximum_finding_severity(self):
        def finding(severity):
            return dict(rule_id="source:rule",severity=severity,title="Review",
                        rationale="Source policy",evidence="Fixture",
                        recommendation="Keep held",source="Cricket")
        cases=(
            ("BLOCK",[]),("CHALLENGE",[]),
            ("BLOCK",[finding("CHALLENGE")]),
            ("CHALLENGE",[finding("BLOCK")]),
            ("PASS",[finding("CHALLENGE")]),
            ("PASS",[finding("BLOCK")]),
        )
        for disposition,findings in cases:
            with self.subTest(disposition=disposition,findings=findings):
                with self.assertRaisesRegex(ParticipationError,"Cricket"):
                    compose(cricket_result={"disposition":disposition,"findings":findings})
        for disposition,findings in (
            ("PASS",[]),("PASS",[finding("NOTE")]),
            ("CHALLENGE",[finding("CHALLENGE")]),
            ("BLOCK",[finding("BLOCK")]),
        ):
            with self.subTest(valid=disposition):
                out=compose(cricket_result={
                    "disposition":disposition,"findings":findings}).as_dict()
                self.assertEqual(out["cricket_review"]["disposition"],disposition)
                self.assertEqual(out["authority_effect"],"NONE")

    def test_cricket_pass_cannot_become_effect_or_identity_proof(self):
        out=compose(cricket_result={"disposition":"PASS","findings":[]}).as_dict()
        self.assertEqual(out["authority_effect"],"NONE")
        self.assertEqual(out["executed_effect"],"NONE")
        self.assertEqual(out["identity_effect"],"NONE")

    def test_core_authorization_and_semantic_equivalence_escalation_rejected(self):
        for field,value in (("authorization_effect","GRANTED"),
                            ("semantic_equivalence","PROVEN"),
                            ("fidelity","MADE_UP")):
            p=plan()
            p[field]=value
            with self.subTest(field=field),self.assertRaises(ParticipationError):
                compose(core_plan=p)

    def test_semantic_missing_capabilities_require_exact_rewrite_coverage(self):
        missing=plan()
        missing.update(required_capabilities=["typed:referent","kv:layout"],
                       missing_capabilities=["kv:layout"],rewrites=[],fidelity="EXACT")
        with self.assertRaisesRegex(ParticipationError,"rewrite"):
            compose(core_plan=missing)
        covered=dict(missing)
        covered["rewrites"]=[dict(source_capability="kv:layout",rule_name="replace:kv",
                                  fidelity="LOSSY",description="Approximate")]
        with self.assertRaisesRegex(ParticipationError,"fidelity"):
            compose(core_plan=covered)
        covered["fidelity"]="LOSSY"
        out=compose(core_plan=covered).as_dict()
        self.assertEqual(out["core_semantic_plan"]["fidelity"],"LOSSY")
        self.assertEqual(out["authority_effect"],"NONE")
        for edits in (
            [dict(covered["rewrites"][0],source_capability="unknown")],
            covered["rewrites"]+covered["rewrites"],
            [dict(covered["rewrites"][0],fidelity="INVALID")],
            [dict(covered["rewrites"][0],fidelity=["LOSSY"])],
        ):
            with self.subTest(rewrites=edits), self.assertRaises(ParticipationError):
                compose(core_plan={**covered,"rewrites":edits})

    def test_mosaic_success_cannot_claim_cognitive_gain(self):
        out=compose(mosaic_result=mosaic()).as_dict()
        self.assertTrue(out["mosaic_handoff"]["correct"])
        self.assertEqual(out["cognitive_gain"],"NOT_ESTABLISHED")

    def test_structured_review_and_mosaic_receipts_are_closed(self):
        for k,v in (("mosaic_result",{**mosaic(),"trust_me":True}),
                    ("cricket_result",{**cricket(),"authority":"GRANTED"})):
            with self.subTest(field=k),self.assertRaises(ParticipationError):
                compose(**{k:v})

    def test_nested_receipt_changes_do_not_mutate_canonical_readback(self):
        result=compose()
        original=result.as_dict()
        mutable=result.contributions[0]
        mutable["claim"]["value"]="INJECTED"
        result.cricket_review["disposition"]="BLOCK"
        copied=result.as_dict()
        copied["contributions"][1]["claim"]["value"]="ANOTHER"
        self.assertEqual(result.as_dict(),original)
        self.assertEqual(result.as_dict()["cricket_review"]["disposition"],"PASS")

    def test_contradictory_source_handoff_receipt_is_rejected(self):
        for field,value in (("answer","OFF"),("correction_burden",1),
                            ("stale_error",True),("payload_bytes",-1)):
            raw=mosaic()
            raw[field]=value
            with self.subTest(field=field),self.assertRaises(ParticipationError):
                compose(mosaic_result=raw)

    def test_false_exact_semantic_fidelity_is_rejected(self):
        raw=plan()
        raw["fidelity"]="EXACT"
        raw["native_capabilities"]=[]
        raw["missing_capabilities"]=["typed:referent"]
        raw["unresolved_capabilities"]=["typed:referent"]
        with self.assertRaises(ParticipationError):
            compose(core_plan=raw)

    def test_unknown_or_unrepresentable_transfer_is_held(self):
        p=plan()
        p.update(fidelity="UNREPRESENTABLE",required_capabilities=["typed:referent","latent:unknown"],missing_capabilities=["latent:unknown"],unresolved_capabilities=["latent:unknown"])
        out=compose(core_plan=p).as_dict()
        self.assertEqual(out["status"],"SEMANTIC_TRANSFER_UNREPRESENTABLE")
        self.assertEqual(out["authority_effect"],"NONE")


@unittest.skipUnless(os.environ.get("VERA_REAL_CONTRACT_ROOT"),
                     "Requires independent exact-head Vera Mono/Mosaic/Cricket checkouts")
class RealSourceAdapterTests(unittest.TestCase):
    """Use the actual three repositories; no mocked types or imports."""

    @classmethod
    def setUpClass(cls):
        root=Path(os.environ["VERA_REAL_CONTRACT_ROOT"])
        # Caller owns exact source pins; this tests API interoperability only.
        for path in (root/"vera-mono"/"packages",root/"mosaic",root/"cricket"/"src"):
            if not path.is_dir():raise AssertionError("Independent source missing")
        for x in (root/"vera-mono"/"packages").glob("*/src"):
            sys.path.insert(0,str(x))
        sys.path[:0]=[str(root/"mosaic"),str(root/"cricket"/"src")]
        if os.environ.get("VERA_CORE_SOURCE_ONLY") == "1":
            # Narrow, explicit source-only mode; not vera_core package qualification.
            repo=root/"vera-mono"
            commit="16a2828dfa7fb7e6c0e19004f014bc68fc47badf"
            source="packages/vera_core/src/vera_core/semantic_transfer.py"
            def git(*args):
                result=subprocess.run(["git","-C",str(repo),*args],
                                      capture_output=True,text=True,check=True,timeout=12)
                return result.stdout.strip()
            assert git("rev-parse","HEAD")==commit, "PINNED_HEAD_REQUIRED"
            assert git("rev-parse",f"HEAD:{source}")==git("hash-object",str(repo/source)), "SOURCE_BLOB_DRIFT"
            spec=importlib.util.spec_from_file_location("_training_pinned_vera_semantics",repo/source)
            assert spec and spec.loader
            module=importlib.util.module_from_spec(spec)
            sys.modules[spec.name]=module
            spec.loader.exec_module(module)
            cls.core_semantic=(module.SemanticCapabilityProfile,module.plan_semantic_transfer)
            cls.import_qualification="EXACT_GIT_FILE_STANDALONE_NOT_PACKAGE_IMPORT"
        else:
            from vera_core.semantic_transfer import SemanticCapabilityProfile,plan_semantic_transfer
            cls.core_semantic=(SemanticCapabilityProfile,plan_semantic_transfer)
            cls.import_qualification="VERA_CORE_PACKAGE_IMPORT"

    def test_actual_vera_core_mosaic_cricket_types_compose_without_promoting_effect(self):
        SemanticCapabilityProfile,plan_semantic_transfer=self.core_semantic
        from runtime.handoff import ContinuityCase,StateEvent,run_case
        from cricket import Cricket,ReviewRequest
        sem=plan_semantic_transfer(
            SemanticCapabilityProfile("vera-core",frozenset({"typed:referent"})),
            SemanticCapabilityProfile("mosaic",frozenset({"typed:referent"})),
            ["typed:referent"],
        )
        mosaic_result=run_case(ContinuityCase("case:1","recovery",(
            StateEvent("old","recovery","task:one","OFF",1),
            StateEvent("new","recovery","task:one","ON",2),
        ),"task:one","ON"),"MOSAIC_STATE")
        review=Cricket().review(ReviewRequest(
            user_message="Review an observation, no effect or external authority.",
            candidate_response="An independent proposal is staged; not yet accepted.",
            effect_class="none",
            explicit_authorization=False,
            completion_claimed=False,
        ))
        receipt=compose(core_plan=sem.as_dict(),mosaic_result=mosaic_result,
                        cricket_result=review.to_dict()).as_dict()
        self.assertEqual(receipt["core_semantic_plan"]["fidelity"],"EXACT")
        self.assertEqual(receipt["core_semantic_plan"]["semantic_equivalence"],"NOT_ESTABLISHED")
        self.assertEqual(receipt["mosaic_handoff"]["condition"],"MOSAIC_STATE")
        self.assertTrue(receipt["mosaic_handoff"]["correct"])
        self.assertIn(receipt["cricket_review"]["disposition"],("PASS","CHALLENGE","BLOCK"))
        self.assertEqual(receipt["authority_effect"],"NONE")

    def test_actual_core_lossy_rewrite_cannot_claim_exact_transfer(self):
        SemanticCapabilityProfile,plan_semantic_transfer=self.core_semantic
        import importlib
        core=importlib.import_module(SemanticCapabilityProfile.__module__)
        rule=core.SemanticTransferRule(
            name="lossy:kv",source_capability="kv:layout",
            target_capabilities=frozenset({"typed:referent"}),
            fidelity=core.TransferFidelity.LOSSY,
            description="Bounded non-equivalent rewrite")
        sem=plan_semantic_transfer(
            SemanticCapabilityProfile("vera-core",frozenset({"typed:referent","kv:layout"})),
            SemanticCapabilityProfile("mosaic",frozenset({"typed:referent"})),
            ["typed:referent","kv:layout"],rules=[rule])
        out=compose(core_plan=sem.as_dict()).as_dict()
        self.assertEqual(out["core_semantic_plan"]["missing_capabilities"],["kv:layout"])
        self.assertEqual(out["core_semantic_plan"]["fidelity"],"LOSSY")
        self.assertEqual(out["authority_effect"],"NONE")
        falsified=sem.as_dict()
        falsified["fidelity"]="EXACT"
        with self.assertRaisesRegex(ParticipationError,"fidelity"):
            compose(core_plan=falsified)

    def test_real_cricket_protected_effect_block_does_not_authorize(self):
        SemanticCapabilityProfile,plan_semantic_transfer=self.core_semantic
        from runtime.handoff import ContinuityCase,StateEvent,run_case
        from cricket import Cricket,ReviewRequest
        plan_receipt=plan_semantic_transfer(
            SemanticCapabilityProfile("vera-core",frozenset({"typed:referent"})),
            SemanticCapabilityProfile("mosaic",frozenset({"typed:referent"})),
            ["typed:referent"]).as_dict()
        handoff=run_case(ContinuityCase("case:1","recovery",(
            StateEvent("e1","recovery","task:one","ON",1),),
            "task:one","ON"),"MOSAIC_STATE")
        result=Cricket().review(ReviewRequest(
            user_message="Review proposed protected action.",
            candidate_response="I have completed the protected action.",
            effect_class="protected",explicit_authorization=False,
            completion_claimed=True,verification_evidence=()))
        self.assertEqual(result.disposition.value,"BLOCK")
        report=compose(core_plan=plan_receipt,mosaic_result=handoff,
                       cricket_result=result.to_dict()).as_dict()
        self.assertEqual(report["cricket_review"]["disposition"],"BLOCK")
        self.assertEqual(report["authority_effect"],"NONE")
        self.assertEqual(report["executed_effect"],"NONE")

    def test_real_core_unrepresentable_capability_fails_no_permission(self):
        SemanticCapabilityProfile,plan_semantic_transfer=self.core_semantic
        from runtime.handoff import ContinuityCase,StateEvent,run_case
        from cricket import Cricket,ReviewRequest
        sem=plan_semantic_transfer(
            SemanticCapabilityProfile("vera-core",frozenset({"typed:referent","kv:layout"})),
            SemanticCapabilityProfile("mosaic",frozenset({"typed:referent"})),
            ["typed:referent","kv:layout"],
        )
        handoff=run_case(ContinuityCase("case:1","recovery",
            (StateEvent("e1","recovery","task:one","ON",1),),"task:one","ON"),
            "MOSAIC_STATE")
        critique=Cricket().review(ReviewRequest(user_message="Review semantic transfer.",
            candidate_response="Semantic equivalence is not established."))
        out=compose(core_plan=sem.as_dict(),mosaic_result=handoff,
                    cricket_result=critique.to_dict()).as_dict()
        self.assertEqual(out["status"],"SEMANTIC_TRANSFER_UNREPRESENTABLE")
        self.assertEqual(out["authority_effect"],"NONE")

if __name__=="__main__":
    unittest.main()
