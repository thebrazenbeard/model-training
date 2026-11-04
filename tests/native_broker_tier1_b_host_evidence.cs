// Exact Windows source tests only: retained host-read file bytes, host nonce,
// and existing durable probe journal. Never a signed production grant or GPU run.
using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using Vera.TrainingDay.HostBroker;

internal static class Tier1BHostEvidenceNativeTests {
    static int checks;
    static void Need(bool condition, string why) {
        checks++;
        if(!condition) throw new Exception("Tier1 B host evidence: "+why);
    }
    static void Reject<T>(Action action, string why) where T: Exception {
        checks++;
        try { action(); }
        catch(T) { return; }
        throw new Exception("Tier1 B host evidence unexpectedly accepted "+why);
    }
    static bool IsDigest(string x) {
        if(x==null || x.Length!=64) return false;
        foreach(char ch in x)
            if(!((ch>='0'&&ch<='9')||(ch>='a'&&ch<='f'))) return false;
        return true;
    }
    static void Main() {
        string root=Path.GetFullPath(Path.Combine(Path.GetTempPath(),
                                      "tier1-b-host-evidence-"+Guid.NewGuid().ToString("N")));
        Directory.CreateDirectory(root);
        string plan=Path.Combine(root,"plan.json");
        string manifest=Path.Combine(root,"manifest.json");
        string kernel=Path.Combine(root,"kernel.bin");
        string journal=Path.Combine(root,"journal.log");
        try {
            File.WriteAllText(plan,"abc",Encoding.ASCII);
            File.WriteAllText(manifest,"hello",Encoding.ASCII);
            File.WriteAllText(kernel,"abc",Encoding.ASCII);
            // RED regression: the fixture root itself may be ordinary while
            // a parent directory is an NTFS reparse junction. Current source
            // checked only the final root; it must reject ancestor redirects.
            string actual=Path.Combine(root,"actual");
            string actualChild=Path.Combine(actual,"child");
            Directory.CreateDirectory(actualChild);
            File.Copy(plan,Path.Combine(actualChild,"plan.json"));
            File.Copy(manifest,Path.Combine(actualChild,"manifest.json"));
            File.Copy(kernel,Path.Combine(actualChild,"kernel.bin"));
            string junction=Path.Combine(root,"junction");
            var mklink=new ProcessStartInfo("cmd.exe",
                "/d /c mklink /J \""+junction+"\" \""+actual+"\"") {
                UseShellExecute=false,CreateNoWindow=true,
                RedirectStandardOutput=true,RedirectStandardError=true
            };
            using(var proc=Process.Start(mklink)) {
                string output=proc.StandardOutput.ReadToEnd();
                string error=proc.StandardError.ReadToEnd();
                Need(proc.WaitForExit(5000) && proc.ExitCode==0,
                    "cannot establish harmless native junction fixture: "+output+error);
            }
            Reject<InvalidDataException>(()=>{
                using(var alias=Tier1BHostEvidenceHold.OpenForNativeTestOnly(
                    Path.Combine(junction,"child"))) {}
            },"ancestor NTFS junction into test-owned directory");
            Directory.Delete(junction);
            Reject<ArgumentException>(() => {
                using(var bad=Tier1BHostEvidenceHold.OpenForNativeTestOnly("relative")) {}
            },"caller-relative root");
            Reject<ArgumentException>(() => {
                using(var bad=Tier1BHostEvidenceHold.OpenForNativeTestOnly(
                    @"\\example.invalid\never-an-actual-share")) {}
            },"UNC/network fixture root");
            string nonce1;
            using(var evidence=Tier1BHostEvidenceHold.OpenForNativeTestOnly(root)) {
                Need(evidence.State=="HOLD" && !evidence.LaunchAuthorized
                    && !evidence.OwnerTrustEnrolled && !evidence.GpuLeaseIssued
                    && !evidence.ModelSnapshotAuthenticated
                    && !evidence.WslChildCustodyQualified
                    && !evidence.JournalRollbackResistant,
                    "input capture invented training authority");
                Need(evidence.PlanSha256==
                    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                    "plan bytes not independently hashed");
                Need(evidence.ManifestSha256==
                    "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
                    "manifest bytes not independently hashed");
                Need(evidence.ExecutableSha256==evidence.PlanSha256,
                    "executable bytes not independently hashed");
                Need(evidence.StillMatchesRetainedBytes(),
                    "stable retained bytes did not match");
                nonce1=evidence.NonceSha256;
                Need(IsDigest(nonce1),"host-created nonce has invalid digest");
                Reject<IOException>(()=>File.WriteAllText(plan,"modified",Encoding.ASCII),
                       "concurrent write while retained exclusive input open");
                using(var reserve=HostOwnedProbeReservation.OpenForNativeTestOnly(journal)) {
                    long expected=evidence.ObserveNextTestEpoch(reserve);
                    Need(expected==1,"journal did not supply next local epoch");
                    Need(reserve.Reserve(evidence.PlanSha256,nonce1)==expected,
                         "durable reservation did not use host-derived digest");
                    Reject<InvalidOperationException>(()=>evidence.ObserveNextTestEpoch(reserve),
                        "unresolved reservation before ownership attestation");
                    Reject<InvalidOperationException>(()=>reserve.Reserve(
                         evidence.PlanSha256,nonce1),"nonce replay during active reservation");
                }
            }
            // Simulated broker crash: exact journal remains unresolved after
            // close/reopen, and it cannot hand a replacement request an epoch.
            using(var stillReserved=HostOwnedProbeReservation.OpenForNativeTestOnly(journal)) {
                Need(stillReserved.Phase=="RESERVED" && stillReserved.Epoch==1,
                    "crash recovery discarded unresolved durable reservation");
                using(var later=Tier1BHostEvidenceHold.OpenForNativeTestOnly(root)) {
                    Need(later.NonceSha256!=nonce1,
                        "fresh 256-bit test nonce unexpectedly reused");
                    Reject<InvalidOperationException>(()=>later.ObserveNextTestEpoch(stillReserved),
                        "unresolved journal accepted next grant");
                }
            }
            // Source files are not immutable merely because a test captured
            // them once. Changed bytes produce a different host-derived digest.
            File.WriteAllText(plan,"modified",Encoding.ASCII);
            using(var changed=Tier1BHostEvidenceHold.OpenForNativeTestOnly(root)) {
                Need(changed.PlanSha256!=
                    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                    "after-close source replacement kept stale digest");
                Need(!changed.LaunchAuthorized,"changed file authorized training");
            }
            File.WriteAllText(plan,new string('x',32769),Encoding.ASCII);
            Reject<InvalidDataException>(()=>{
                using(var oversize=Tier1BHostEvidenceHold.OpenForNativeTestOnly(root)) {}
            },"plan larger than 32768-byte bound");
            // Host-local hash chaining is not a rollback-resistant witness.
            // In an isolated fixture, deleting the journal restarts the epoch
            // and allows reuse of a previously consumed nonce. A live broker
            // must never accept this FileMode.OpenOrCreate test mechanism.
            File.Delete(journal);
            using(var recreated=HostOwnedProbeReservation.OpenForNativeTestOnly(journal)) {
                Need(recreated.Epoch==0 && recreated.Phase=="GENESIS",
                     "deleted test journal did not demonstrate missing monotonic anchor");
                Need(recreated.Reserve(
                    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                    nonce1)==1,
                    "test journal replacement no longer demonstrates replay rollback");
            }
            Console.WriteLine("NATIVE_TIER1_B_HOST_EVIDENCE_HOLD_PASS checks="+checks+
                " actual_file_hashes=true retained_handles=true host_nonce=true"+
                " durable_epoch=true crash_hold=true rollback_resistant=false"+
                " owner_trust=false gpu=false");
        } finally {
            // Only the new and unique temporary fixture directory is removed.
            if(Directory.Exists(root)) Directory.Delete(root,true);
        }
    }
}
