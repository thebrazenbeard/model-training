# W56 Q4 Windows artifact stager — source candidate, NOT deployed

## Scope

This Windows PowerShell 5.1 code stages precisely:

1. Official ggml-org/llama.cpp CPU x64 release b12125 at commit 7d516e25ead4653169236b3f90d13f7bd8629724: archive 24,730,921 bytes, SHA-256 968091ccb8c230e0cf7e0eed386310703bebae4a4fd4375f0c57db53958307d8.
2. Publisher twoody/Qwen3.5-4B-GGUF Q4_K_M: model 2,783,446,880 bytes, SHA-256 b7508d8b4a652bc020a439aafdd4ce1527d49774865ce3334a43b16dbeaa90ca.

Only the file bytes are pinned. The publisher's claim that Q4 was converted from official Qwen BF16 remains UNVERIFIED. No official Vera model is qualified by this script.

The script streams downloads (not buffering 2.8 GB in memory), requires HTTPS and exact length and SHA-256 before each partial file is moved, rejects unsafe ZIP paths and symbolic links, extracts only allowlisted llama-cli.exe/CPU DLLs, records their hashes, and sets private Windows ACLs on isolated directories. It never executes the model or opens a listener. The supervisor must rehash before invocation.

## Preconditions

- Host identity and user account freshly checked; not Lappy by accident.
- Windows PowerShell 5.1, .NET Framework 4.7.2+, TLS 1.2 connectivity to official GitHub and Hugging Face.
- Existing ordinary parent folder in the current user profile, C:\Users\Vera\BT2ModelArtifacts. The agreed new destination leaf is C:\Users\Vera\BT2ModelArtifacts\w56-q4-oneshot. Atomic CreateDirectoryW refuses an existing/colliding leaf without altering its ACL or contents; parent symlink/junction ancestry is rejected in preflight.
- At least 1 GiB free beyond both downloads.
- A separately qualified JobObject supervisor, idle-workload check and affirmative execution lease before any inference.

## Example commands (do not run without host preflight)

Test offline: powershell.exe -NoProfile -File .\Test-Q4Stager.ps1

Stage downloads: powershell.exe -NoProfile -File .\Stage-Q4Trial.ps1 -Destination 'C:\Users\Vera\BT2ModelArtifacts\w56-q4-oneshot'

Only the designated host owner should run the staging command. Offline tests create and clean up a GUID-scoped temporary fixture.

The destination contains the agreed subdirectories bin, models, downloads, state. The stage receipt is DESTINATION\stage-receipt.json and records binary/model hashes plus Executed=false and LeaseQualified=false. It is not a process lease.

## Limitations

- Source written and statically inspected on a non-Windows scratch host. No PowerShell interpreter was available; tests have NOT been executed.
- Release ZIP integrity is pinned to GitHub's release asset SHA. Individual executable Authenticode signing is unverified.
- Hugging Face URL is pinned to immutable source commit d741a25c4cf8eea74fe876af7631b659b366f126; exact byte SHA must also pass or staging fails.
- ZIP allowlisting may omit an unexpected vendor DLL; stop if dependencies are missing rather than silently broadening extraction.
- Interrupted partial downloads are not resumed. Only each script-generated GUID partial is removed; other evidence in the newly created destination is retained.
- No inference, R16 LoRA conversion, process custody, RAM usage, official lineage or conversational effectiveness was qualified.
