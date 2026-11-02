# TrainingDay owner-key enrollment candidate V1 — SOURCE-ONLY HOLD

This portable verifier connects the existing single-owner Ed25519 R18
assertion to a **separately registrar-signed owner-key pin candidate**. It
checks canonical bounded JSON, domain-separated registrar and owner
signatures from distinct registrar and owner keys, exact owner identity and public-key fingerprint, model/purpose,
registry epoch, revocation state and UTC validity. It refuses forged or
substituted keys, stale epochs, expired records and WITHHOLD decisions.

It cannot enroll Patrick's key by itself. All of these are still **caller
supplied**: the purported registrar key and its expected fingerprint, the
minimum epoch, time, owner key, and corpus/split digests. A malicious caller
can generate a registrar and owner key pair, sign both records, and supply a
matching fingerprint. Such a pair passes the *source integrity check* but
never becomes independently authenticated owner consent.

The only success state is
`REGISTRAR_SIGNATURE_VALID_SUPPLIED_PIN_HOST_CUSTODY_PENDING`, with ALL of these false:

- `registrar_host_identity_authenticated`
- `owner_key_enrollment_authenticated`
- `owner_rights_independently_reviewed`
- `protected_evaluator_custody_authenticated`
- `replay_anchor_authenticated`
- `corpus_rights_admitted`
- `gpu_lease_granted`
- `physical_training_authorized`

To turn the source contract into an actual owner ceremony, the Windows/NAS
administrators must provision a **non-caller-controlled, ACL-protected**
registrar trust anchor through an authenticated host-admin operation; keep
the private enrollment-signing key outside this public source and trainer;
hold a monotonic epoch and revocation floor outside the mutable registry;
recompute the frozen R185 packet/TRAIN/selection commitments on Lappy;
and obtain an independent protected evaluator exclusion attestation. The
host broker must separately prove GPU ownership, trainer custody and an
immutable run plan. A NAS-only HMAC ledger and a Python `expected_*`
parameter are not independent authority.

Neither the pre-existing R155/R164 distinct-principal validators nor
`physical_launch_authority.py` is changed. All physical optimizer launches
remain HARD HOLD. The module contains no registration endpoint, private
key, positive grant, persistent key store, network client or model runner.

Synthetic regression: `python -m unittest -q
tests.test_owner_key_enrollment_candidate_v1`. Test-generated keys have no
authority; this is not host installation, data admission or training.
