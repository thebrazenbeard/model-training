# TrainingDay R22 — adapter lineage source-only negatives

These deterministic CPU cases exercise the existing eight-arm pre-unseal
validator against synthetic receipt and adapter files. They do **not** open
private evaluation data, load Qwen, perform optimizer steps or assert an
authenticated producer.

`PYTHONPATH=. python -m unittest -q tests.test_r22_adapter_lineage`

The tests repin both the per-arm receipt hash and enclosing inventory hash
after altering declared model revision, tokenizer, selected/admitted corpus,
seed, optimizer accounting or completion timing. Source-contract mismatches
must still fail. Independently mutated adapter-model and config bytes must
fail their original receipt hashes.

These checks only establish internal consistency. A privileged process can
rewrite an adapter **and** all of its self-declared hashes; distinct hash
strings do not prove physical weight updates or base-model identity. An
independent attester, exclusive host lease, same-byte loaded-model binding,
and actual optimizer/model reload evidence remain required. No training or
private-bank authority is conferred by passing this suite.
