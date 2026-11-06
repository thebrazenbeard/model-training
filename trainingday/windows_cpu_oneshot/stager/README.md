# W56 Q4 Windows artifact stager — source candidate, NOT deployed

## Scope

This Windows PowerShell 5.1 code stages precisely:

1. Official ggml-org/llama.cpp CPU x64 release b12125 at commit 7d516e25ead4653169236b3f90d13f7bd8629724: archive 24,730,921 bytes, SHA-256 968091ccb8c230e0cf7e0eed386310703bebae4a4fd4375f0c57db53958307d8.
2. Publisher twoody/Qwen3.5-4B-GGUF Q4_K_M: model 2,783,446,880 bytes, SHA-256 b7508d8b4a652bc020a439aafdd4ce1527d49774865ce3334a43b16dbeaa90ca.

Only the file bytes are pinned. The publisher's claim that Q4 was converted from official Qwen BF16 remains UNVERIFIED. No official Vera model is qualified by this script.

The script streams downloads (not buffering 2.8 GB in memory), requires HTTPS and exact length and SHA-256 before each partial file is moved, rejects unsafe ZIP paths and symbolic links, extracts only allowlisted `llama-cli.exe`/CPU DLLs, records their hashes, and sets private Windows ACLs on isolated directories. It never executes the model or opens a listener. The supervisor must rehash before invocation.

**Runner dependency repair (source revision 2):** The first staged release had 19 allowed binaries but incorrectly omitted four dynamic-link dependencies. Official b12125 Windows x64 MSVC `llama-cli.exe` is only 9,728 bytes and links to `llama-cli-impl.dll`, which links transitively through `llama-server-impl.dll`, `llama-common.dll` and `mtmd.dll`. The exact official ZIP contains these four files. Revision 2 adds **only those four DLL names** to the extraction allowlist and requires all four before extraction. It also requires `llama-cli.exe`, `llama.dll`, `ggml.dll`, `ggml-base.dll`, and the confirmed fallback `ggml-cpu-x64.dll`. The other original CPU architecture variants remain allowlisted. This is an archive membership check, **not** a successful PE-loader run or model qualification.

Source evidence pinned to [llama.cpp b12125 `tools/cli/CMakeLists.txt`](https://github.com/ggml-org/llama.cpp/blob/7d516e25ead4653169236b3f90d13f7bd8629724/tools/cli/CMakeLists.txt), [`tools/server/CMakeLists.txt`](https://github.com/ggml-org/llama.cpp/blob/7d516e25ead4653169236b3f90d13f7bd8629724/tools/server/CMakeLists.txt) and [`tools/mtmd/CMakeLists.txt`](https://github.com/ggml-org/llama.cpp/blob/7d516e25ead4653169236b3f90d13f7bd8629724/tools/mtmd/CMakeLists.txt). `llama-ui` is a STATIC target, so `llama-ui.dll` is neither present nor required.

## Preconditions

- Host identity and user account freshly checked; not Lappy by accident.
- Windows PowerShell 5.1, .NET Framework 4.7.2+, TLS 1.2 connectivity to official GitHub and Hugging Face.
- Existing ordinary parent folder in the current user profile, `C:\Users\Vera\BT2ModelArtifacts`. The **historical evidence leaf** was `C:\Users\Vera\BT2ModelArtifacts\w56-q4-oneshot`. **Do not rerun against that leaf**: it must be preserved unmodified. Use a new sibling leaf such as `C:\Users\Vera\BT2ModelArtifacts\w56-q4-oneshot-v2` only after verifying it is absent and qualified for a new one-shot lease. Atomic `CreateDirectoryW` refuses an existing/colliding leaf without altering its ACL or contents; parent symlink/junction ancestry is rejected in preflight.
- At least 1 GiB free beyond both downloads.
- A separately qualified JobObject supervisor, idle-workload check and affirmative execution lease before any inference.

## Example commands (do not run without host preflight)

Test offline: powershell.exe -NoProfile -File .\Test-Q4Stager.ps1

Fresh download into new sibling (several GB):

`powershell.exe -NoProfile -File .\Stage-Q4Trial.ps1 -Destination 'C:\Users\Vera\BT2ModelArtifacts\w56-q4-oneshot-v2'`

Repair from the **existing, already staged immutable evidence**, without another network transfer and without changing a single file under the original leaf:

`powershell.exe -NoProfile -File .\Stage-Q4Trial.ps1 -Destination 'C:\Users\Vera\BT2ModelArtifacts\w56-q4-oneshot-v2' -CopyFromVerifiedStage 'C:\Users\Vera\BT2ModelArtifacts\w56-q4-oneshot'`

The copy mode enforces a **different leaf under the exact same canonical parent** as the source, rejecting the original path itself, a nested destination inside the original, and destinations elsewhere. An explicitly bound empty or whitespace `-CopyFromVerifiedStage` fails immediately; it **cannot silently fall back to network downloads**. It rehashes the old ZIP and GGUF against immutable manifests, streams them to GUID `.partial` files inside the fresh sibling, then rehashes each new copy before moving it to its final name. It only reads the old leaf. Do **not** use the source path as `-Destination`, or invoke this against any leaf that already exists. The supervisor must explicitly use the **new** leaf and rehash its complete runner/model receipt before any subsequent executable probe.

Only the designated host owner should run the staging command. Offline tests create and clean up a GUID-scoped temporary fixture.

The destination contains the agreed subdirectories `bin`, `models`, `downloads`, `state`. The v2 stage receipt is `DESTINATION\stage-receipt.json` and records binary/model hashes, source method, `RequiredCliDependenciesPresent=true`, `WindowsPeLoaderVerified=false`, `Executed=false`, and `LeaseQualified=false`. It is not a process lease. Do not manually edit the historical v1 receipt to report repaired dependency closure.

## Limitations

- Source written and statically inspected on a non-Windows scratch host. No PowerShell interpreter was available; tests have NOT been executed.
- Release ZIP integrity is pinned to GitHub's release asset SHA. Individual executable Authenticode signing is unverified.
- Hugging Face URL is pinned to immutable source commit d741a25c4cf8eea74fe876af7631b659b366f126; exact byte SHA must also pass or staging fails.
- ZIP allowlisting covers the source-identified loader chain but may still omit an unexpected PE import, or a runtime-selected dependency. A future isolated version-only probe should capture both signed and unsigned-hex exit codes, stdout and stderr; stop if dependencies are missing rather than silently broadening extraction. The `0xC0000135` loader exit is a **hypothesis pending actual exit-code evidence**, not an observed status.
- Interrupted partial downloads are not resumed. Only each script-generated GUID partial is removed; other evidence in the newly created destination is retained.
- Offline tests include omission of each of the nine mandatory binary entries and check that an incomplete archive fails **before any extraction bytes are written**, plus verified local copying/corruption rejects and same-parent/foreign-parent/descendant stage-boundary cases. Tests have not been run without a Windows PowerShell 5.1 host.
- No inference, R16 LoRA conversion, process custody, RAM usage, PE-loader success, official lineage or conversational effectiveness was qualified by this revision.
