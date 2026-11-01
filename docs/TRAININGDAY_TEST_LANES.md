# TrainingDay test lanes — explicit incomplete host qualification

The published PR #179 hosted Ubuntu Python 3.12 run reported **1,454 passed,
18 failed, 2 skipped**. Several failures require historical Windows `D:\VERA`
corpora/adapters/receipts or a private Lappy development-only heldout file.
Those inputs must never be copied into public CI as convenient fixtures.

## Portable source lane

`python -m pytest -q -p ci_host_boundary --portable-source-only`

This command collects the test suite, validates that **all 12 named owner-host
tests and both native Windows Stage-0 tests are present**, records exactly
14 pytest `deselected` outcomes (with individual names and separate host vs.
Windows categories), and executes the remaining portable tests. No pytest
`skipped` outcomes are allowed. Missing, extra, or duplicated exclusions fail
accounting. The two Windows-only tests are positively exercised in the
separate `Tests/windows-stage0-source` job on a native Windows runner; they
are not classified as missing private owner artifacts. Its output is explicitly labeled
`PORTABLE_SOURCE_ONLY — HOST_INTEGRATION_UNQUALIFIED` and lists each deferred
node ID. It emits machine-readable `CI_HOST_BOUNDARY_RECEIPT_JSON` in the log;
the receipt explicitly disclaims training and qualification. This is never
an unrestricted full-suite PASS.

The separate `portable-source-ci.yml` workflow has an always-failing,
separately named `host-integration-required` job. The overall workflow stays
RED until an owner-approved physical qualification path replaces that hold;
do not mark it `continue-on-error` or exempt the status from review.

The ordinary `python -m pytest -q` is unchanged and still attempts all tests,
including host-dependent tests. The portable command is not the full suite.
Other CI failures (optional dependencies, source mismatch) remain independent
defects until owner-reviewed fixes are published. A green Windows source job
does not qualify a GPU lease or real host integration.

## Restricted host lane

After independent host/GPU lease release, authenticated NAS/Executor routing,
data rights, and the appropriate test-run authorization, run all 12 host manifest
entries against the actual historical artifacts on the authorized host.
Capture their exact source SHA, environment identity, exit code, artifact
digests, and test results in a signed restricted receipt. Do not publish
protected prompts or use a per-row final-bank membership oracle.

Neither a green portable lane nor a signed host artifact inventory establishes
model-weight loading, optimizer/RNG-identical restart, or learning gain.
