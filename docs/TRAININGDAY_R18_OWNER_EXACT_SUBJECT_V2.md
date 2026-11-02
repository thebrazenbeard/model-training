# R18 exact owner subject V2 — SOURCE-ONLY HARD HOLD

`trainingday.owner_exact_subject_v2.inspect_owner_exact_subject_v2` is a
versioned *structural* Ed25519 verifier for one owner-controlled R18 rights
assertion. It does not change V1's schema, signature domain, rights policy or
legacy R155/R164 distinct-role validators.

V1 binds an owner-supplied manifest, TRAIN, DEV, exclusion and generator
digest. V2 additionally binds three **different** frozen R185 commitments:

1. `frozen_training_packet_sha256`: SHA-256 of the frozen R185 training
   dataset packet **file** (not the self-referential signed owner receipt).
2. `train_content_sha256`: ordered selected TRAIN id/prompt/answer content.
3. `selected_id_v2_sha256`: ordered selected case-ID V2 digest.

The claim also binds `manifest_sha256`, `train_sha256`,
`generator_source_sha256`, `dev_sha256`, and
`protected_exclusion_sha256`. None are substitutes for one another. The
official model scope is `Qwen/Qwen3.5-4B` at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, corpus
`R18_ARITHMETIC`, purpose `QWEN35_VERA_TRAININGDAY`.

The bounded, canonical JSON assertion is Ed25519 signed with a distinct V2
domain. The owner signing-key fingerprint, owner identity, explicit decision,
key epoch, `ACTIVE` state and UTC validity are checked for internal
consistency. A signed `WITHHOLD` remains `OWNER_WITHHELD_HOLD`.

**None of these checks independently authenticates an input.** The key, all
expected hashes, owner identity, minimum epoch and current time are supplied
by the caller. An attacker can invent a key, sign a changed corpus and supply
matching expected hashes. That synthetic pair passes structural verification
but must never become source rights, evaluator custody or host authority.

Every successful result is a status and boolean evidence only, with **all**
owner-key enrollment, frozen-packet authentication, protected evaluator
custody, rights review, revocation/replay, rights ADMIT, GPU and physical
training flags false. No protected DEV rows, exclusion IDs, prompts or gold
answers are accepted, opened, returned or logged by the verifier.

Future host enrollment must pin Patrick's owner key through an independently
authenticated administrator and frozen-source reference, verify evaluator
custody against its independently held private DEV/exclusions, and bind the
result to a broker-owned immutable run plan and real GPU/process lease.
This module has **no** positive-grant API, key store, enrollment endpoint,
model runner or service installation. `physical_launch_authority.py` remains
an unconditional training HOLD.

Run `python -m unittest -q tests.test_owner_exact_subject_v2` for synthetic
negative/structural tests. Test keys are not owner enrollment.
