# TrainingDay Windows host broker — deny-only bootstrap

**Evidence:** Installed LocalSystem Windows service and local named pipe. **Not** an authenticated GPU lease, data-rights issuer, NAS witness, trainer, or model qualification.

The C# service in this directory is the exact source of the host-installed bootstrap. It runs outside Git under Windows Service Control Manager. Its pipe has SYSTEM/Administrators-only DACL; it attempts to authenticate the actual client Windows token after a completed pipe read. Every operation deliberately returns `HOLD`, zero authorized launches, no GPU lease, no data rights and no NAS witness. It does not open training data or load model weights.

**Do not** modify `trainingday/physical_launch_authority.py` to interpret this status response as approval. The Python hard hold remains unconditional. Only a separately qualified, host-owned service that owns the immutable run plan, exclusive GPU and process handles, rights grants, durable anti-replay witness, and training child job may launch. The authoritative requirements are in `trainingday/WINDOWS_HOST_BROKER_DEPLOYMENT_BOUNDARY.md`.

## Local build (Windows .NET Framework 4.8)

Compile with local x64 csc.exe using target:exe, references System.ServiceProcess.dll, System.Core.dll and System.Security.dll, and `TrainingDayHostBroker.cs`, `BoundedPipeRequest.cs`, and `BrokerPipeTransport.cs`. The repository contains no executable. Install only after admin review and verified binary hash, with restrictive installation ACL. Hosted source compilation is not a runtime security audit.

## Actual Lappy installation observation (2026-11-01)

- Service `VeraTrainingDayHostBroker`: Running, Automatic, LocalSystem.
- Installed executable: `C:\Program Files\VERA\TrainingDayHostBroker\VeraTrainingDayHostBroker.exe` (directory ACL: Administrators and SYSTEM only).
- Executable SHA-256: `D4885718A9D81FBE4D28AF7232CFAD302DA10763741CB214D5BD9BE4443102FB`.
- Host observation receipt: `C:\ProgramData\VERA\TrainingDayHostBroker\INSTALLATION_OBSERVATION_20261101.json`, SHA-256 `D79E6240857A415657D53A1EF12E7DD38A6605F0694DC01F683C2D9AAF0186D6`.
- Two live admin probes `STATUS_V1` and `PREPARE_V1` returned authenticated caller, exact service PID, `HOLD` and no grant. An early build's premature impersonation and unsupported synchronous-pipe timeouts were corrected before these live tests.
- TLS verification of the existing NAS route failed the server-hostname check; no insecure trust bypass or signed NAS grant.

## Still missing

An independent steward-issued admissible TRAIN dataset decision, an actual R19-01 exclusive GPU and process lease, NAS route owner grant and durable external witness, approved immutable run plan, broker-owned Windows Job process, and hostile end-to-end tests. This prototype is not a positive-grant broker. No Qwen optimizer step or checkpoint has been produced from it.

Service status and source/CI checks cannot be substituted for authenticated physical launch authority.

## Host-owned probe reservation prototype (next source candidate; not deployed)

`HostOwnedProbeReservation.cs` is an **isolated native-test component**, not
imported by the installed service or callable by the Python trainer. On
Windows it opens a journal with an OS-exclusive file handle, validates every
hash-linked append and strictly increasing reservation epoch, durably consumes
nonces before test dispatch intent, refuses concurrent opens and replay, and
keeps an unfinished reservation in permanent `HOLD` after a simulated crash.
It will not automatically clear ambiguous state. A terminal transition
requires the real harmless Windows JobObject child to be killed and observed
exited through a retained kernel process handle, not just a reusable PID.

The exact Windows CI lane compiles and executes its native regression. This
does **not** own CUDA, establish GPU exclusivity or trusted clock/expiry,
authenticate administrator-policy path ACLs, obtain an external monotonic NAS
witness, or ratify R18 corpus rights. An admin changing or rolling back the
local journal remains outside its authority model. Do not wire a positive
training route into this test component; production integration requires the
independent service, identity, rights, immutable plan and NAS checks above.

### Resource-capped harmless Job child (source-only successor)

The native `JobObjectChildCustody` regression additionally configures a
**256 MiB job-wide committed-memory cap** and a **25% CPU hard cap** on its
fixed `ping.exe` loopback child. It reads both limit classes back from the
Windows kernel **before** creating the child suspended; any missing/incorrect
limit fails closed before child creation. The test observes job membership,
limit readback and kill-on-last-job-handle-close using the retained kernel
process handle, avoiding a PID-reuse termination claim. Failure during setup
terminates the still-suspended child and closes owned handles.

An additional native **5 GiB memory-cap representability test** applies that
larger ceiling to the *same tiny ping child*, reads back kernel limits and
confirms owner-handle cleanup. It never allocates 5 GiB or launches GGUF.
Zero/overflow, unaligned memory and invalid CPU quotas must reject **before**
creating a Job. This is API/ABI and fail-closed policy coverage, **not** a
physical availability test or endorsement that Lappy can spare 5 GiB.

This is an OS-isolation *fixture*, not an inference lease or a live Lappy
qualification. Its default cap is intentionally far too small for Qwen. It has no
arbitrary executable/model-path input and does not alter the installed
deny-only service. A separate owner-scoped inference launcher would require:

1. Live host identity and free-memory/competing-process preflight, with
   fail-closed resource margin; user-owned process/session and cancellation.
2. Pinned executable identity, exact artifact hash, tokenizer and adapter
   lineage; no Q4+R13 pairing without matching training-base proof.
3. A newly created, preconfigured, *read-back* Windows Job with appropriate
   job-wide memory/CPU limits, suspended child attachment before resumption,
   retained process/job handles, a strict deadline and witnessed child exit.
4. A tested error/abort/owner-loss cleanup path, plus host-native stress and
   contention regression under an owner-approved maintenance window.

Until those are implemented and **physically verified on the target host**,
bounded GGUF conversation load remains **NO-GO**. WSL `systemd-run` is a
separate possible Linux-process route, not a substitute for Windows host
memory and workload checks or proof that the current distro has usable
delegated cgroup controllers.

## R162 source-only bounded request hardening

The successor source adds `BoundedPipeRequest.Read`: a total 1,500 ms input deadline, maximum 128 printable ASCII bytes, CRLF/LF framing validation, and cooperative service-stop cancellation. On timeout it disposes the active pipe, waits for native overlapped I/O completion before closing its event handle, and never interprets the request as authorization. Windows native tests exercise normal requests, 128/129-byte boundaries, malformed controls, idle and slow-drip clients, stop cancellation, and a healthy request after a timeout.

**Deployment status: NOT INSTALLED.** The current Lappy service and its recorded binary SHA remain those of R161. This separate source candidate must not be deployed or treated as a positive-grant broker without exact-head hosted Windows CI, an independently reviewed host identity/peer binding design, and an owner-controlled maintenance window. The Labyrinth training lane's work remains undisturbed.

## R163 source-only pipe transport boundary

The successor candidate creates the named pipe through Windows `CreateNamedPipeW`, using `PIPE_REJECT_REMOTE_CLIENTS` and `FILE_FLAG_FIRST_PIPE_INSTANCE` plus the inherited protected SYSTEM/Administrators DACL, a non-inheritable overlapped handle, and a single active server instance. It keeps the R162 bounded reader and unconditional `HOLD` responses. Native Windows transport regression checks cover single-instance creation and a local administrative client; an actual remote-SMB negative probe and restart/stop regression remain required on the owner host before deployment. If any pipe or ACL creation fails, the service issues no grant.

**Deployment status: NOT INSTALLED.** These source changes do not alter the running R161 service, issue rights, sign a NAS witness, reserve a GPU, or authorize model training.
