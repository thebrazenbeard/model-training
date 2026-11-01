# R183 native Windows JobObject child custody

Evidence class: real Windows native source build and harmless child run.
Status: SOURCE CANDIDATE ONLY; not installed in the Lappy broker.

`JobObjectChildCustody.LaunchHarmlessProbe()` creates only a fixed
Windows System32 `ping.exe` loopback test process. The Win32 APIs
create that process suspended, create a native JobObject, set
`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, assign the process to the job,
and resume its thread only after assignment succeeds. On any error,
the child is terminated and owned handles are released.

The native test checks the real PID, Windows in-job membership,
process liveness before owner disposal, and process termination
after JobObject close. Source CI requires this executable to compile
and pass on a Windows GitHub Actions runner. No fake JobObject
backend is accepted as native proof.

This deliberately provides NO generic executable/argument launch
surface. It does not enforce exclusive GPU occupancy, memory caps,
durable reboot-safe lease fencing, principal/issuer authenticity,
signed NAS CAS, data rights, or immutable training-plan binding.
The installed LocalSystem service and its HOLD reply are untouched.
Do not route official Qwen training into this test-only component.

Lappy host check (fixed command, no GPU): compile
`JobObjectChildCustody.cs` with
`tests/native_jobobject_child_custody.cs` using native .NET Framework
`csc.exe`, then run the resulting test executable. Expected result:
`NATIVE_JOB_CUSTODY_PASS child_assigned=true killed_on_close=true`.

Next actual broker integration must authenticate the user and trusted
grant issuers, pin the executable and file identities, prove the GPU
is available/previous processes fenced, and atomically journal a
signed exclusive lease before any supervised training child is created.
