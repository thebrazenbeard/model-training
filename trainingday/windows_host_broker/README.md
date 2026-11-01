# TrainingDay Windows host broker — deny-only bootstrap

**Evidence:** Installed LocalSystem Windows service and local named pipe. **Not** an authenticated GPU lease, data-rights issuer, NAS witness, trainer, or model qualification.

The C# service in this directory is the exact source of the host-installed bootstrap. It runs outside Git under Windows Service Control Manager. Its pipe has SYSTEM/Administrators-only DACL; it attempts to authenticate the actual client Windows token after a completed pipe read. Every operation deliberately returns `HOLD`, zero authorized launches, no GPU lease, no data rights and no NAS witness. It does not open training data or load model weights.

**Do not** modify `trainingday/physical_launch_authority.py` to interpret this status response as approval. The Python hard hold remains unconditional. Only a separately qualified, host-owned service that owns the immutable run plan, exclusive GPU and process handles, rights grants, durable anti-replay witness, and training child job may launch. The authoritative requirements are in `trainingday/WINDOWS_HOST_BROKER_DEPLOYMENT_BOUNDARY.md`.

## Local build (Windows .NET Framework 4.8)

Compile with local x64 csc.exe using target:exe, references System.ServiceProcess.dll, System.Core.dll and System.Security.dll, and `TrainingDayHostBroker.cs`. The repository contains no executable. Install only after admin review and verified binary hash, with restrictive installation ACL. Hosted source compilation is not a runtime security audit.

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
