# TrainingDay Owner-Controlled R18 Rights Evidence V1 — SOURCE CANDIDATE

Status: UNISSUED / HOLD / never training authorization.

This alternative addresses Patrick's correction rejecting a mandatory
unrelated second human steward. The existing two-principal strict R155
verifier remains intact. No law establishing that requirement was found.

The new source-only profile validates one owner-controlled Ed25519
signature of canonical JSON on exact R18_ARITHMETIC corpus IDs,
manifest, TRAIN, DEV, protected exclusion and generator source digests.
It is scoped to the pinned Qwen3.5-4B revision and specific training
purpose and UTC validity interval.

Even a valid owner assertion returns PENDING_HOST_RATIFICATION.
A signed WITHHOLD also grants nothing. The verifier never grants data
admission, evaluation access, a GPU lease or physical model training.

The supplied public key does not prove enrollment or owner identity.
The clock and corpus commitments are caller-provided; genuine source
rights and protected evaluator custody are not yet established.

For any real approval, host administrators must authenticate and enroll the
owner identity, check current revocation and replay state, review generator
use-rights and privacy evidence, bind exact TRAIN/DEV/exclusion custody,
and authorize the immutable run plan separately. The protected evaluator,
NAS route owner and Lappy GPU owner remain separate authority boundaries.
The existing strict R155 checker and physical launch HARD HOLD are unchanged.

Synthetic-only checks: python -m unittest -q tests.test_owner_controlled_rights_v1
Test keys are ephemeral and confer no real authority.
