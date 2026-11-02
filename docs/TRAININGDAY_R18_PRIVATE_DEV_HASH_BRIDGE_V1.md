# R18 private DEV and protected-exclusion hash bridge V1

Status: **SOURCE CANDIDATE / UNDEPLOYED / UNATTESTED / HOLD**.

This source-only bridge is a data-format preflight, not a rights issuer. It is
intended to execute only inside a future separately privileged R18 evaluator
custodian. No private DEV rows, sealed answers, per-case identifiers, or
unredacted exclusion lists belong in this repository, GitHub reviews, Bus,
Blotter, or a training-process invocation.

The function accepts an owner-provided R18 subject (exact manifest, TRAIN,
generator, ordered full TRAIN-content and V2 selected-ID hashes), an **answer-free**
1,024-row TRAIN index of exact `id`, `prompt_sha256`, `answer_sha256` entries,
private canonical DEV JSONL and a private canonical 128-row exclusion index:
64 historical R16 protected records plus 64 R18 external OOD records. The
output includes only aggregate counts and exact SHA-256 commitments, and
always returns `HASHES_ONLY_UNAUTHENTICATED_HOLD` with all rights, custody,
host and training grants `false`.

The explicit `subject.train_content_sha256` is required and checked against the
SHA-256 of canonical JSON for the **ordered** list of the three TRAIN index
fields, matching `r18_qlora.train_content_digest`. Changing a prompt or answer
hash while leaving the separately supplied subject unchanged fails closed,
even when all 1,024 selected IDs and their V2 digest remain unchanged. This
does **not** authenticate the full-content digest: a caller controlling both
the index and the subject can forge both consistently. The existing
`subject.train_sha256` is retained as a distinct declared field; neither the
known R185 packet-file SHA nor the V2 selected-ID digest is accepted as a
substitute for a separately authenticated full TRAIN-content commitment.
The output explicitly marks `train_content_commitment_authenticated=false`.

Case IDs are compared across TRAIN, DEV and protected exclusion indexes using
conservative NFKC-plus-casefold *collision rejection* in addition to raw
identity. For example, `train-0001`, `TRAIN-0001`, and a fullwidth spelling
cannot coexist as apparently disjoint rows. This does not rewrite source IDs
or change their byte-exact ordered selected-ID V2 digest. The normalization
only prevents presentation aliases from evading overlap checks; it is not a
semantic deduplication guarantee, and can reject intentionally distinct
case-sensitive names as a safer false positive.

The R18 external OOD bank exists separately in `r18_eval_source.py`; this
module does **not** create, open, or unseal it. The Labyrinth cognitive
SFT corpus in Draft #110/#111 is a different dataset and cannot substitute
for these R18 arithmetic commitments. A caller must supply the private
exclusion index from independent protected custody without exposing gold.

## Required independent host operation before anything can be admitted

1. A NAS/host administrator establishes an authenticated evaluator service
   identity distinct from the model trainer; enforce private ACLs at the OS
   boundary. Possession of this module or a JSON field is not authentication.
2. The custodian privately verifies true DEV source provenance, correct
   split, generator/use rights and protected R16/R18 exclusion source. It
   computes the answer-free TRAIN index from independently authenticated
   R185 bytes, not from caller-selected values. It must recompute and pin the
   ordered full TRAIN-content digest from the true prompt and answer bytes,
   independently of the caller-controlled `subject` and `train_index`.
3. The custodian consumes its protected data and calls this source bridge,
   then signs an immutable receipt bound to these digests with a host-owned
   enrolled signing key, stable identity, expiry, revocation and replay log.
   This module itself never signs or installs a key.
4. The owner separately reviews the actual source/use rights and the
   exact hash-only packet, then locally signs an owner-controlled assertion.
5. The independent host/NAS ratifier checks both identities, immutable
   corpus/use/time and key/revocation state, then may decide admission.
   This bridge never makes that decision or authorizes training/eval unseal.

## Nonqualification

SHA-256 checks ensure byte-level integrity relative to supplied objects;
they do not authenticate the origin or correctness of the declared full TRAIN
content, independently controlled evaluation,
legal use-rights, OS ACLs or actual separation of principals. Without those
physical gates, `dev_sha256` and `protected_exclusion_sha256` are merely
untrusted candidate commitments.

Synthetic source tests:

```bash
python -m unittest -q tests.test_private_dev_commitment_v1
```

No source tests are evidence of a private bank, owner consent, enrolled key,
deployed service, GPU lease, optimizer step or model improvement.
