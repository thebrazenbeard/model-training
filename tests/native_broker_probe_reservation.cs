// Native Windows synthetic regression: durable request fencing plus real
// JobObject child-handle custody. Never exercises CUDA or model training.
using System;
using System.IO;
using Vera.TrainingDay.HostBroker;

internal static class NativeProbeReservationTests {
    static void Need(bool value, string reason) {
        if (!value) throw new Exception(reason);
    }
    static void Reject<T>(Action action) where T : Exception {
        try { action(); } catch (T) { return; }
        throw new Exception("expected " + typeof(T).Name + " rejection");
    }
    public static int Main() {
        string directory = Path.Combine(Path.GetTempPath(), "vera-probe-journal-"+Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string journal = Path.Combine(directory, "probe.log");
        string nonceOne = new string('1',64), nonceTwo = new string('2',64);
        string nonceThree = new string('3',64), request = new string('a',64);
        try {
            using (var lease = HostOwnedProbeReservation.OpenForNativeTestOnly(journal)) {
                Need(lease.Reserve(request, nonceOne) == 1, "first monotonic epoch");
                Need(lease.Phase == "RESERVED", "reservation not durably entered");
                Reject<IOException>(() => {
                    using (var competing = HostOwnedProbeReservation.OpenForNativeTestOnly(journal)) { }
                });
                Reject<InvalidOperationException>(() => lease.Reserve(request, nonceTwo));
                lease.MarkDispatchIntent();
                using (var child = JobObjectChildCustody.LaunchHarmlessProbe()) {
                    Need(child.IsInJob && child.IsAlive, "real child not in live job");
                    lease.MarkTerminalAfterOwnedProbeExit(child);
                }
                Need(lease.Phase == "TERMINAL", "terminal not durably entered");
                Reject<InvalidOperationException>(() => lease.Reserve(request, nonceOne));
            }
            using (var lease = HostOwnedProbeReservation.OpenForNativeTestOnly(journal)) {
                Need(lease.Epoch == 1 && lease.Phase == "TERMINAL", "persisted replay floor absent");
                Reject<InvalidOperationException>(() => lease.Reserve(request, nonceOne));
                Need(lease.Reserve(request, nonceTwo) == 2, "epoch failed across process reopen");
                // Simulated process crash: release lock WITHOUT a terminal witness.
            }
            using (var lease = HostOwnedProbeReservation.OpenForNativeTestOnly(journal)) {
                Need(lease.Phase == "RESERVED", "crash-opened reservation not preserved");
                Reject<InvalidOperationException>(() => lease.Reserve(request, nonceThree));
            }
            // A crash after dispatch intent is more consequential than a
            // reservation-only crash: the harmless child might have run.
            // Reopening must not silently count it as terminal or allow a
            // new request without a retained child-exit witness.
            string dispatchedJournal = Path.Combine(directory, "dispatch.log");
            using (var lease = HostOwnedProbeReservation.OpenForNativeTestOnly(dispatchedJournal)) {
                Need(lease.Reserve(request, nonceThree) == 1, "dispatch crash epoch missing");
                lease.MarkDispatchIntent();
            }
            using (var lease = HostOwnedProbeReservation.OpenForNativeTestOnly(dispatchedJournal)) {
                Need(lease.Epoch == 1 && lease.Phase == "DISPATCH_INTENT",
                    "unresolved dispatch intent lost after reopen");
                Reject<InvalidOperationException>(() => lease.Reserve(request, nonceTwo));
                Reject<InvalidOperationException>(() => lease.MarkDispatchIntent());
            }
            // Corrupt trailing byte may not be silently repaired or truncated.
            File.AppendAllText(journal, "incomplete", System.Text.Encoding.ASCII);
            Reject<InvalidDataException>(() => {
                using (var invalid = HostOwnedProbeReservation.OpenForNativeTestOnly(journal)) { }
            });
            Console.WriteLine("NATIVE_PROBE_RESERVATION_PASS monotonic=2 replay_denied=true concurrent_denied=true crash_hold=true dispatch_crash_hold=true job_exit_verified=true corrupt_denied=true");
            return 0;
        } finally {
            // The temporary native-test directory is known and owned by this test.
            if (Directory.Exists(directory)) Directory.Delete(directory, true);
        }
    }
}
