# TRAININGDAY source-readiness integration preflight

Run `python -m trainingday.integration_preflight` or add `--json` to obtain a
read-only, deterministic integration-gap inventory. It intentionally exits 1
because no complete, independently qualified training-to-serving recovery
chain has been established by this source command.

The thirteen topologically ordered checks distinguish **official model source
metadata** from locally authenticated model bytes; recovered R16 adapter
provenance; independent destination restore; corpus rights; owner-held
process/GPU and anti-replay custody; actual **base-model inference**;
optimizer/checkpoint reload; a **separate independent restore of the newly
trained checkpoint**; evaluator custody; historical **R16 adapter GGUF
conversion** versus a distinct **new-run adapter GGUF conversion**; and actual
**unqualified new-run conversation probe**. The early R16 preservation track remains
independent, but cannot satisfy recovery of a newly produced checkpoint.
Base inference can precede training and protected evaluation; serving
recovered adapter weights does not itself establish generalization.
Synthetic GGUF success never proves official adapter conversion or base
inference.

The new-run conversion dependency is a **source-only graph correction**.
It does not bind optimizer adapter bytes to restored or converted bytes: an
actual one-run controller still needs independently authenticated exact
checkpoint/adapter digest equality across each handoff, evaluator custody,
host lease and recovery readback. Synthetic/untrusted matching hashes do not
constitute those observations. There is still no executable one-run pipeline.

### Exact missing byte-lineage receipts (V2)

The current source modules **cannot** establish a trustworthy new-run
optimizer → restored adapter → converted GGUF → served process identity chain:

- `optimizer_pilot.replay_synthetic_adapter` reports `final_adapter_sha256`
  over its **synthetic float tensors**, not the actual official-model PEFT
  adapter-file bytes. It produces no authenticated physical run ID or
  independent checkpoint/adapter byte witness.
- `qwen35_full32_adapter_reload_cpu` hashes persisted adapter files, but uses
  deliberately random tiny weights, not the official pretrained 4B model.
- `official_qwen35_gguf_workflow.plan` sets converter/quantizer
  `output_sha256: null`; it is a **HOLD plan**, not an authenticated input
  PEFT-adapter digest → output GGUF digest conversion receipt.
- `real_qwen35_inference.verify_manifest` can hash the supplied **local GGUF
  bytes**, but trained mode only accepts the historical R16 source run. Its
  return fields state `conversion_lineage_independently_verified=false` and
  `runtime_started=false`; it does not bind a new optimizer's PEFT adapter
  digest to the converter or a live serving process's loaded file.
- The public evaluator consumes caller-declared predictions rather than
  independently observed, protected heldout evaluations of the exact
  post-restore checkpoint bytes.

An actual controller needs one authenticated physical run ID and official
base/revision pins; independent per-file optimizer checkpoint and PEFT
adapter SHA-256s; destination-restore SHA-256s; converter input PEFT SHA and
output GGUF SHA; and process identity bound to the very same GGUF bytes.
The protected evaluator must bind to the restored checkpoint and independently
attest rights/heldout custody. **Equality of caller-supplied hash strings is
not authentication**; a source-only positive validator around invented
receipts would give false confidence, so this code does not supply one.

The V2 terminal name, `unqualified_new_run_conversation_probe`, is deliberate:
an observed chat is not release qualification or evidence of generalized
persona. Such a research probe need not depend on protected evaluation, but
there is **no `qualified_release` stage** that could truthfully promote it.

The command does not contact GitHub, Lappy, VN1, VN2, the NAS, the broker or a
private evaluator. It does not admit a user's self-declared receipt, mutate
checkpoints or launch training. `SOURCE_ONLY` means exactly that. `UNKNOWN`
does **not** mean ready. `HOLD` is a current local deny-only source finding,
not proof that every external host would also deny. The ordering lists
dependencies; it is not an authority-grant or a self-updating live system.

The upstream official checkpoint is pinned to `Qwen/Qwen3.5-4B` at
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. This is distinct from
the repository's separate public Aggressive/uncensored checkpoint manifest.

The JSON field `source_metadata_pins_enrolled` is **syntactic enrollment only**:
all seven required metadata names must occur exactly once with lowercase,
64-character SHA-256 hex strings, matching the source admission gate's shape
check. A missing or malformed pin leaves the metadata-byte stage at `HOLD`;
even a well-formed source tuple moves that stage only to `UNKNOWN`. Neither
case attests the hashes as independently trusted or validates local bytes.

## A concrete integration criterion

The release remains **HOLD** until independently established evidence can be
read back for **all** thirteen stages on their actual owners; a local CLI cannot
promote the result. To progress the program, land and individually qualify
real host/artifact/data/evaluator/runtime components and only then build a
separate, authority-bearing release controller. No source-only PR or synthetic
CI result may be used as a positive physical training grant.

The command's test suite is `python -m pytest -q
tests/test_trainingday_integration_preflight.py`; full repository CI is still
necessary on the exact PR head.
