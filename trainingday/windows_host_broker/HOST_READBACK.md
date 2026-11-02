# Windows broker readback — operator-run, read-only, always HOLD

When Executor is unavailable, Patrick can inspect the installed host service
directly on **Lappy**, in a local Windows PowerShell session. Use the checkout
under the authoritative `D:\VERA` workspace; `C:\VERA` is obsolete. An installed
service executable under `C:\Program Files\VERA` remains a legitimate Windows
installation path and is inspected from the live service configuration, not
assumed from repository layout.

From the root of the checked-out `model-training` repository:

```powershell
& .\trainingday\windows_host_broker\ReadOnlyHostReadback.ps1
```

The script outputs a single JSON object on stdout. It **does not write a file**,
contact a network endpoint, change Windows settings, install a service,
enroll a key, or start training. If the script cannot run due to local policy,
stop; do not weaken host policy for this diagnostic.

Output schema: `VERA_TRAININGDAY_WINDOWS_READBACK_V1`. Fields distinguish
configuration from runtime evidence:

- `configured_service_account` and `configured_localsystem`: SCM/CIM
  configuration, **not** proof of an active process token.
- `process_owner_sid`, `process_owner_sid_evidence_available`,
  `process_service_pid_stable`, and `process_sid_is_localsystem`: best-effort
  process **owner SID** evidence, not the complete Windows process token.
  The service PID and process creation identity are rechecked after the SID
  query to reduce PID-reuse confusion. The check is not atomic and may remain
  unavailable even when the service is installed.
- `executable_path`, `executable_sha256`, and `executable_is_reparse_point`:
  derived from the live service executable path, without printing raw command
  line arguments; unavailable files remain `null`.
- `host_root_acl`, `host_state_acl`, and `executable_acl`: shallow directory
  checks for existence, reparse points, protected inheritance, and broad write
  ACEs. No SDDL dump. A protected top-level ACL does **not** prove the whole
  path or every child is safe from reparse races or same-SID manipulation.
- `service_probe_error`: a bounded diagnostic category rather than exception
  text that might contain sensitive host information.

Every output has `status=HOLD` and false rights/evaluator/GPU/training flags.
It is an observation artifact, not owner signature authentication, a signed
NAS lease, protected evaluator custody, or permission to run Qwen.

The portable source test is
`python -m unittest -q tests.test_trainingday_windows_readback_source`.
It checks source invariants but cannot execute PowerShell or verify installed
Windows process identity on a Linux CI runner. Actual Windows parser execution
and human review of the JSON are required before treating it as host evidence.
