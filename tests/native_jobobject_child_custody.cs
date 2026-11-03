// Native Windows custody regression: a harmless child must run in a real
// JobObject and terminate on owner-handle close. No model or GPU use.
using System;
using System.Diagnostics;
using System.Threading;
using Vera.TrainingDay.HostBroker;

internal static class NativeCustodyTests {
    private static void Require(bool condition, string reason) {
        if (!condition) throw new Exception(reason);
    }
    public static int Main() {
        int pid;
        using (var custody = JobObjectChildCustody.LaunchHarmlessProbe()) {
            pid = custody.ProcessId;
            Require(pid > 0, "child PID absent");
            Require(custody.IsInJob, "child was not bound to native job");
            Require(custody.IsAlive, "child ended before custody close");
            var observed = custody.ReadOwnedKernelIdentityForNativeTestOnly();
            Require(observed.CreationFileTimeUtc > 0, "kernel creation timestamp absent");
            Require(observed.Matches(pid, observed.CreationFileTimeUtc),
                "retained handle identity did not match real child");
            Require(!observed.Matches(pid + 1, observed.CreationFileTimeUtc),
                "forged/reused PID accepted");
            Require(!observed.Matches(pid, observed.CreationFileTimeUtc + 1),
                "same PID with different creation-time accepted");
            var reread = custody.ReadOwnedKernelIdentityForNativeTestOnly();
            Require(reread.Matches(observed.Pid, observed.CreationFileTimeUtc),
                "retained kernel HANDLE identity changed during custody");
        }
        bool alive = true;
        for (int i=0; i<30; i++) {
            Thread.Sleep(100);
            try {
                using (var p=Process.GetProcessById(pid)) {
                    p.Refresh();
                    alive=!p.HasExited;
                }
            } catch (ArgumentException) { alive=false; }
            if (!alive) break;
        }
        Require(!alive, "child survived closing kill-on-close JobObject");
        using (var custody = JobObjectChildCustody.LaunchHarmlessProbe()) {
            var beforeExit = custody.ReadOwnedKernelIdentityForNativeTestOnly();
            Require(custody.CloseJobAndConfirmChildExit(),
                "owned process handle did not observe job-close exit");
            Require(!custody.IsAlive, "job child still active after kernel wait");
            var afterExit = custody.ReadOwnedKernelIdentityForNativeTestOnly();
            Require(afterExit.Matches(beforeExit.Pid, beforeExit.CreationFileTimeUtc),
                "process HANDLE identity changed after observed termination");
        }
        Console.WriteLine("NATIVE_JOB_CUSTODY_PASS child_assigned=true killed_on_close=true kernel_pid_creation_binding=true false_claims_rejected=true exited_handle_identity_stable=true");
        return 0;
    }
}
