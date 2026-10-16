"""Adversarial controls for the model-native interop eligibility frontier."""
from dataclasses import replace
import unittest

from semantic_capability_gate_v1 import (
    ModelLineage, SemanticTask, CategoryScore, Qualification,
    Handoff, assess_handoff, SemanticGateError,
)

def sha(ch): return ch*64

def model(**overrides):
    values=dict(
        model_sha256=sha("a"), tokenizer_sha256=sha("b"),
        adapter_sha256=sha("c"), prompt_template_sha256=sha("d"),
        decoding_sha256=sha("e"), runtime_layout_sha256=sha("f"),
        semantic_space_sha256=sha("1"),
    )
    values.update(overrides)
    return ModelLineage(**values)

def task(**overrides):
    values=dict(
        task_id="task:100", capability_id="semantic:reasoning",
        ontology_sha256=sha("2"),input_schema_sha256=sha("3"),
        output_schema_sha256=sha("4"),context_sha256=sha("5"),
    )
    values.update(overrides)
    return SemanticTask(**values)

def qualifying(m=None,t=None,**overrides):
    m=m or model()
    t=t or task()
    values=dict(lineage_sha256=m.exact_id,capability_id=t.capability_id,
        ontology_sha256=t.ontology_sha256,
        input_schema_sha256=t.input_schema_sha256,
        output_schema_sha256=t.output_schema_sha256,
        evaluator_sha256=sha("6"),heldout_dataset_sha256=sha("7"),
        scores=tuple(CategoryScore(name,4,5) for name in
                     ("referent","correction","scope","authority","recovery")))
    values.update(overrides)
    return Qualification(**values)

def handoff(*,sender=None,receiver=None,t=None,mode="TYPED_STATE",**kw):
    params=dict(task=t or task(),sender=sender or model(),receiver=receiver or model(),
                mode=mode,peer_node_id="vn1")
    params.update(kw)
    return Handoff(**params)

def gate(h,q=None,**kw):
    params=dict(qualification=q if q is not None else qualifying(h.receiver,h.task),
        verify_qualification=lambda evidence:True,
        authenticated_peer_node_id="vn1")
    params.update(kw)
    return assess_handoff(h,**params)


class MachineSemanticQualificationTests(unittest.TestCase):
    def test_machine_typed_state_qualified_cross_model_without_shared_weights(self):
        h=handoff(sender=model(model_sha256=sha("8")),
            receiver=model(model_sha256=sha("9")))
        result=gate(h)
        self.assertEqual(result.status,"ADVISORY_COMPATIBLE")
        self.assertTrue(result.eligible)
        self.assertEqual(result.task_id,"task:100")
        self.assertNotEqual(h.sender.exact_id,h.receiver.exact_id)

    def test_missing_receiver_qualification_fails_closed(self):
        h=handoff()
        out=assess_handoff(h,qualification=None,verify_qualification=lambda q:True,
                           authenticated_peer_node_id="vn1")
        self.assertEqual(out.status,"HELD")
        self.assertIn("MISSING_QUALIFICATION",out.reasons)
        self.assertIsNone(out.qualification_evidence_sha256)

    def test_sender_self_attestation_is_not_receiver_verification(self):
        h=handoff()
        out=gate(h,verify_qualification=None)
        self.assertIn("QUALIFICATION_NOT_VERIFIED",out.reasons)
        out=gate(h,verify_qualification=lambda q: False)
        self.assertIn("QUALIFICATION_NOT_VERIFIED",out.reasons)

    def test_unverified_peer_and_effects_are_not_route_authority(self):
        h=handoff(t=task(effect_class="PROTECTED_MUTATION"))
        result=gate(h,authenticated_peer_node_id=None)
        self.assertEqual(result.status,"HELD")
        self.assertIn("PEER_NOT_AUTHENTICATED",result.reasons)
        self.assertIn("EFFECT_AUTHORITY_NOT_IN_SCOPE",result.reasons)

    def test_exact_schema_and_ontology_binding(self):
        h=handoff()
        for field in ("lineage_sha256","ontology_sha256","input_schema_sha256",
                      "output_schema_sha256","capability_id"):
            mismatch="wrong:capability" if field=="capability_id" else sha("0")
            with self.subTest(field=field):
                q=replace(qualifying(h.receiver,h.task),**{field:mismatch})
                self.assertIn("QUALIFICATION_BINDING_MISMATCH",gate(h,q).reasons)

    def test_all_categories_required_and_minimum_success_rate(self):
        h=handoff()
        categories=("referent","correction","scope","authority","recovery")
        q=qualifying(scores=tuple(CategoryScore(c,3 if c=="correction" else 5,5)
                                   for c in categories))
        self.assertIn("QUALIFICATION_BELOW_THRESHOLD",gate(h,q).reasons)
        with self.assertRaises(SemanticGateError):
            qualifying(scores=(CategoryScore("referent",5,5),))

    def test_token_id_transfer_requires_tokenizer_lineage(self):
        sender=model(tokenizer_sha256=sha("8"))
        h=handoff(sender=sender,mode="TOKEN_IDS")
        self.assertIn("TOKENIZER_MISMATCH",gate(h).reasons)
        self.assertTrue(gate(handoff(mode="TOKEN_IDS")).eligible)

    def test_latent_transfer_needs_semantic_space_and_independent_pair_eval(self):
        h=handoff(sender=model(model_sha256=sha("8")),mode="LATENT",
                  alignment_evidence_sha256=sha("a"))
        self.assertIn("LATENT_ALIGNMENT_NOT_VERIFIED",gate(h).reasons)
        self.assertTrue(gate(h,verify_alignment=lambda evidence, pair:
            evidence==sha("a") and pair.sender.model_sha256==sha("8")).eligible)
        bad=handoff(sender=model(model_sha256=sha("8"),
                    semantic_space_sha256=sha("9")),mode="LATENT",
                    alignment_evidence_sha256=sha("a"))
        res=gate(bad,verify_alignment=lambda *args:True)
        self.assertIn("SEMANTIC_SPACE_MISMATCH",res.reasons)

    def test_kv_cache_rejects_same_model_wrong_layout_context_and_opt_in(self):
        h=handoff(mode="KV_CACHE",alignment_evidence_sha256=sha("8"))
        self.assertIn("KV_OPT_IN_REQUIRED",gate(h).reasons)
        self.assertIn("KV_CONTEXT_AND_POSITION_NOT_VERIFIED",gate(h).reasons)
        valid=gate(h,allow_kv_cache=True,verify_alignment=lambda ev,request:True)
        self.assertTrue(valid.eligible)
        changed=handoff(sender=model(runtime_layout_sha256=sha("0")),
                        mode="KV_CACHE",alignment_evidence_sha256=sha("8"))
        self.assertIn("KV_RUNTIME_LAYOUT_SHA256_MISMATCH",
                      gate(changed,allow_kv_cache=True,
                           verify_alignment=lambda *args:True).reasons)
        changed_prompt=handoff(sender=model(prompt_template_sha256=sha("0")),
                        mode="KV_CACHE",alignment_evidence_sha256=sha("8"))
        self.assertIn("KV_PROMPT_TEMPLATE_SHA256_MISMATCH",
                      gate(changed_prompt,allow_kv_cache=True,
                           verify_alignment=lambda *args:True).reasons)

    def test_unknown_private_without_transport_is_never_eligible(self):
        restricted=handoff(t=task(privacy_class="RESTRICTED"))
        self.assertIn("RESTRICTED_TRANSPORT_NOT_ATTESTED",gate(restricted).reasons)
        self.assertTrue(gate(restricted,restricted_transport_attested=True).eligible)
        secret=handoff(t=task(privacy_class="SECRET"))
        self.assertIn("SECRET_HANDOFF_UNSUPPORTED",gate(secret).reasons)

    def test_provenance_and_category_scores_are_digest_bound(self):
        a=qualifying()
        other=replace(a,scores=tuple(CategoryScore(s.name,5,s.total) for s in a.scores))
        self.assertNotEqual(a.evidence_id,other.evidence_id)
        self.assertNotEqual(model(model_sha256=sha("a")).exact_id,
                            model(model_sha256=sha("8")).exact_id)

    def test_rejects_bad_identifiers_scores_and_unknown_wire_modes(self):
        with self.assertRaises(SemanticGateError):model(model_sha256="not-an-id")
        with self.assertRaises(SemanticGateError):task(capability_id="free text")
        with self.assertRaises(SemanticGateError):CategoryScore("referent",True,5)
        with self.assertRaises(SemanticGateError):CategoryScore("authority",6,5)
        with self.assertRaises(SemanticGateError):handoff(mode="MAGIC_LATENT")
        with self.assertRaises(SemanticGateError):handoff(peer_node_id="evil\nheader")

    def test_malformed_lineage_and_qualification_fail_closed(self):
        with self.assertRaises(SemanticGateError):
            Handoff(task="not a task",sender=model(),receiver=model(),
                    mode="TYPED_STATE",peer_node_id="vn1")
        with self.assertRaises(SemanticGateError):
            qualifying(scores=(None,))

    def test_unavailable_external_verifier_quarantines_not_exceptions(self):
        h=handoff(mode="LATENT",alignment_evidence_sha256=sha("0"))
        def unavailable(*args):
            raise OSError("Offline verification registry")
        r=gate(h,verify_qualification=unavailable,verify_alignment=unavailable)
        self.assertIn("QUALIFICATION_NOT_VERIFIED",r.reasons)
        self.assertIn("LATENT_ALIGNMENT_NOT_VERIFIED",r.reasons)
        self.assertEqual(r.status,"HELD")
        kv=handoff(mode="KV_CACHE",alignment_evidence_sha256=sha("0"))
        r=gate(kv,verify_alignment=unavailable,allow_kv_cache=True)
        self.assertIn("KV_CONTEXT_AND_POSITION_NOT_VERIFIED",r.reasons)

if __name__=="__main__":
    unittest.main()
