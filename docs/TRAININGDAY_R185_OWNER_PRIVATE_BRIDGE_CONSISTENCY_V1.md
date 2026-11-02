# R185 owner V2 ↔ private DEV source-bridge consistency — HARD HOLD

This module checks one previously unchecked *source-level* relationship:
the exact R185 hash commitments in an owner-signed V2 claim must match the
aggregate SHA-256 commitments that the private DEV/exclusion bridge returns.
The source-bridge report must include the ordered TRAIN-content digest
provided by the #206 lineage, as well as selected-ID V2, manifest, TRAIN,
generator, DEV, and protected exclusion digests. The owner receipt's frozen
packet-file SHA is checked against a **caller-supplied** expected packet SHA;
the private bridge cannot independently authenticate it.

`trainingday.owner_private_consistency_v1.inspect_owner_private_consistency_v1`
reuses the V2 Ed25519 signature validator, rejects every mismatched shared
hash, wrong schema/status, withheld owner decision, missing fields, unexpected
payload fields (including private prompts/gold), and any source bridge that
claims positive custody, unseal or training authority. Its output contains
no DEV rows, exclusion IDs, prompts, answers, or gold.

**This is a source-only consistency check, not independent evaluator identity.**
The caller supplies the owner key, signature, expected owner identity,
packet hash, clock, epoch floor, and hash-only evaluator report. One caller
can forge matching source records and a matching signature using a fresh key.
The output field `caller_supplied_hashes_consistent=true` describes only
internal agreement between these untrusted inputs. The new checker returns
`SOURCE_ONLY_MATCH_UNAUTHENTICATED_HOLD` with all owner/evaluator/rights/GPU/
physical grant flags **false** even when internal checks succeed.

The genuine admission path still needs a host-enrolled owner public key and
trusted frozen R185 bytes, a separate evaluator process/OS principal with
protected private DEV/exclusion custody and authenticated output, an external
monotonic witness, and a broker-held GPU/trainer lease. None is implemented
or authorized by this module. The source-only protected bridge must never be
run in the trainer with private held-out material.

Local synthetic validation (no private data):

```bash
python -m unittest -q tests.test_owner_private_consistency_v1
```

The tests exercise mismatch of all seven shared digests, withheld claims,
caller-coforged but still-HOLD records, forged positive custody bits,
unexpected private-content keys, packet mismatch and key/epoch rejection.
This module is layered on the R197 four-way source union; it does not alter
the existing owner V1/V2, private DEV bridge, Windows broker, or NAS witness.
