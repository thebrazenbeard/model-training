# VeraOS normal-resident model qualification receipt V1

Status: SOURCE REQUIREMENTS / HOLD. Not an issued receipt, trusted signer, or
provider admission. Training PR106 owns source provenance, not deployment.

A minimum canonical signed receipt must bind ALL of:

1. Exact VeraOS normal-resident installed image/code digest, selected target
   model, originating source/revision, license, binary format/quantization,
   verified actual model-file SHA-256 and exact byte length. Separately label
   the Ollama inventory digest and size as developer observations. Qwen3 4B
   is not automatically Vera's trained model.
2. Authenticated receiver-host/node identity, gateway generation, enrolled
   session/connectedAt, receiver-minted one-time nonce and request/task ID;
   separate sample IDs and timestamps for registry, CPU/RAM/VRAM, inventory
   and model-file bytes, all within a bounded common capture/expiry interval.
   Explicitly distinguish Windows relay host, QEMU guest, and VN1.
3. Trusted issuer identity and key ID, signature algorithm, canonical payload
   digest and signature covering exact artifact, host/session/nonce, source
   and route. Validate against independently installed trust policy and
   revocation state; self-declared PASS and self-contained keys do not qualify.
4. Independently reproducible model-behavior suite/run receipts bound to that
   exact loaded artifact and runtime environment, with suite/hash and actual
   request/result. Coherent text is not proof of competence, personhood,
   independent mind, shared identity or inference provenance.
5. Fresh normal-resident PORTAL policy/route selection and task-journal CAS,
   nonce-consumption ledger, task/permission expiry and re-read immediately
   before use. Unknown effects must not be replayed. Model qualification
   never confers identity, memory-write or protected-effect authority.

Failure: missing/mismatched session, host, nonce, timestamp, issuer, artifact
byte hash, suite, runtime link, permission or one-use replay ledger => HOLD.
Even a structurally complete receipt is HOLD until an independent trusted
receiver verifies signatures, bytes, one-time challenge and installed route.
The diagnostic /sbin/vera-virtio-probe test-only PASS is NEVER admission
for normal /sbin/vera-init.

Current Training source checks each component sample timestamp and duplicate
sample IDs within one claimed cut. Reuse ACROSS calls cannot be detected
without a separate receiver-owned durable challenge ledger. Matching strings,
a caller callback returning True, or an Ollama inventory hash are not trust.

Exact source context: VeraOS Draft #32
5ddac1459c3a7b06f251721a14048fc26e60bda0;
reported qwen3:4b-instruct Q4_K_M 2,497,293,803 bytes, Ollama digest
0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0.
No inference, signing, trust modification or installation performed.
