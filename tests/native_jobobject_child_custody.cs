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
        Console.WriteLine("NATIVE_JOB_CUSTODY_PASS child_assigned=true killed_on_close=true");
        return 0;
    }
}
