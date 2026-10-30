# R22 training-only difficulty audit — source/CPU boundary

The R18 plan requires exactly **1,024 independently admitted Labyrinth training
cases** (`trainingday/r18_benchmark.py::prepare_r18_private_plan`). The selected
rows are produced from a pinned R16 v2 manifest and a local `work_root`. That
complete selected corpus is **not present in this source checkout**. Accordingly,
the real R18 operation-pattern and target-token distributions are **UNKNOWN**;
the numbers below are deliberately **synthetic fixture results only**.

`difficulty_audit.audit_admitted_training()` accepts only the minimal
`id/family/prompt/answer` records returned by `admit_labyrinth` for replayed
verified integer chains. It refuses additional keys (including evaluation
markers), incorrect labels/content IDs, duplicate normalized questions, wrong
selection order, missing or nonpositive target-token observations, and malformed
source/tokenizer hash declarations. It returns *only* grouped counts and digest
commitments, never raw questions, answers or individual case IDs.

Target-token lengths must be measured by the caller using the pinned tokenizer,
then supplied as an ID-to-count mapping. This CPU helper verifies coverage and
hashes the declarations but **does not authenticate the tokenizer or lengths**.
The report therefore unconditionally sets `tokenizer_physically_verified=false`,
`neural_improvement_proven=false`, `private_evaluation_accessed=false`, and
`qualified_for_neural_training=false`. A plan/custodian must separately bind
source manifest and selected IDs; a matching hex digest is not an attestation.

## Reproducible four-case synthetic aggregate

`python -m unittest -q tests.test_r22_training_difficulty_audit` checks a
four-case public synthetic fixture. Its aggregate-only output is:

| Property | Count |
| --- | ---: |
| Operation length 2 | 4 |
| `add>multiply` | 2 |
| `subtract>add` | 1 |
| `multiply>subtract` | 1 |
| Structural duplicate patterns beyond first occurrence | 1 |
| Exact normalized prompt duplicates | 0 |
| Declared target length 1 token | 2 |
| Declared target length 2 tokens | 1 |
| Declared target length 3 tokens | 1 |

Synthetic `aggregate_sha256`:
`27bdf00a938b90c2f343acf59fe22745cf433e62b8f050b2cade3d43fc1c53a8`

This hash is reproducible from the test fixture, **not an R18 training receipt**.
The synthetic manifest/tokenizer hash fields use repeated `a`/`b` characters
and are explicitly not physical file or tokenizer verification.

## Qualifying real-data use

Only an authorized custodian with read access to the *training-only* admitted
`selected_rows` and their prior independent admission can invoke this helper
with `expected_count=1024`, the fixed selection-order hash and measured pinned
tokenizer lengths. No private OOD test bank, hidden labels, or prior heldout
evaluation output should be supplied. A completed real aggregate still requires
independent custody and is not permission to train or promote a model.
