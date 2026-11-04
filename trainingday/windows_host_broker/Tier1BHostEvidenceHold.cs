// SOURCE-ONLY Contract B host-input witness for harmless Windows native tests.
// A captured hash is not an owner signature, authenticated pretrained snapshot,
// WDDM lease, or permission to run a Linux/CUDA worker.
using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace Vera.TrainingDay.HostBroker {
internal sealed class Tier1BHostEvidenceHold : IDisposable {
    const int MaxPlanBytes = 32768;
    const int MaxManifestBytes = 4 * 1024 * 1024;
    const int MaxExecutableBytes = 64 * 1024 * 1024;
    readonly FileStream plan, manifest, executable;
    readonly string planSha, manifestSha, executableSha;
    readonly string nonceSha;
    bool disposed;

    internal readonly string State = "HOLD";
    internal readonly bool LaunchAuthorized = false;
    internal readonly bool OwnerTrustEnrolled = false;
    internal readonly bool ModelSnapshotAuthenticated = false;
    internal readonly bool WslChildCustodyQualified = false;
    internal readonly bool GpuLeaseIssued = false;
    internal readonly bool JournalRollbackResistant = false;

    // Caller-selectable fixture root is intentionally available ONLY for
    // isolated source tests. Real host code needs an immutable protected path,
    // ACL/handle identity, signed owner grant, and crash recovery qualification.
    internal static Tier1BHostEvidenceHold OpenForNativeTestOnly(string fixtureRoot) {
        if (String.IsNullOrEmpty(fixtureRoot)
                || !Path.IsPathRooted(fixtureRoot)
                || Path.GetFullPath(fixtureRoot) != fixtureRoot)
            throw new ArgumentException("native fixture root must be absolute and normalized");
        // This is a bounded Windows test fixture, not a supported network or
        // volume-GUID path resolver. Reject UNC/device roots early; do not
        // accidentally ask a remote SMB provider to serve trust-anchor bytes.
        string volumeRoot=Path.GetPathRoot(fixtureRoot);
        if (String.IsNullOrEmpty(volumeRoot) || volumeRoot.Length!=3
                || volumeRoot[1]!=':' || volumeRoot[2]!='\\'
                || !Char.IsLetter(volumeRoot[0]))
            throw new ArgumentException("native fixture root must use local drive syntax");
        if (new DriveInfo(volumeRoot).DriveType!=DriveType.Fixed)
            throw new InvalidDataException("fixture drive is not a fixed local drive");
        // A final `File.GetAttributes(root)` check alone missed an ancestor
        // NTFS junction: root\junction\child resolves through junction even
        // though child itself is not a reparse point. Walk every ancestor.
        for(DirectoryInfo part=new DirectoryInfo(fixtureRoot); part!=null;
            part=part.Parent) {
            if((part.Attributes & FileAttributes.ReparsePoint)!=0)
                throw new InvalidDataException("reparse ancestor rejected");
        }
        FileStream p = null, m = null, e = null;
        try {
            p = OpenReadExclusive(Path.Combine(fixtureRoot, "plan.json"), MaxPlanBytes);
            m = OpenReadExclusive(Path.Combine(fixtureRoot, "manifest.json"), MaxManifestBytes);
            e = OpenReadExclusive(Path.Combine(fixtureRoot, "kernel.bin"), MaxExecutableBytes);
            var result = new Tier1BHostEvidenceHold(p,m,e);
            p = m = e = null;  // ownership transferred to witness
            return result;
        } finally {
            if (p != null) p.Dispose();
            if (m != null) m.Dispose();
            if (e != null) e.Dispose();
        }
    }

    private Tier1BHostEvidenceHold(FileStream p, FileStream m, FileStream e) {
        plan=p; manifest=m; executable=e;
        planSha=DigestCurrentBytes(plan);
        manifestSha=DigestCurrentBytes(manifest);
        executableSha=DigestCurrentBytes(executable);
        // The nonce is generated on the observing host, never supplied in a
        // request or environment variable. A source receipt still cannot sign.
        byte[] nonce = new byte[32];
        using (var rng = new RNGCryptoServiceProvider()) rng.GetBytes(nonce);
        using (SHA256 h=SHA256.Create()) nonceSha=Hex(h.ComputeHash(nonce));
        Array.Clear(nonce,0,nonce.Length);
    }

    static FileStream OpenReadExclusive(string path, int maximumBytes) {
        if ((File.GetAttributes(path) & (FileAttributes.ReparsePoint | FileAttributes.Directory)) != 0)
            throw new InvalidDataException("test input must be a regular file, not a reparse point");
        var stream = new FileStream(path, FileMode.Open, FileAccess.Read,
                                    FileShare.None, 65536, FileOptions.SequentialScan);
        if (stream.Length < 1 || stream.Length > maximumBytes) {
            stream.Dispose();
            throw new InvalidDataException("test input length outside bound");
        }
        return stream;
    }

    static string Hex(byte[] raw) {
        var value=new StringBuilder(raw.Length*2);
        foreach(byte b in raw) value.Append(b.ToString("x2"));
        return value.ToString();
    }
    static string DigestCurrentBytes(FileStream input) {
        input.Position=0;
        using(SHA256 h=SHA256.Create()) {
            string digest=Hex(h.ComputeHash(input));
            input.Position=0;
            return digest;
        }
    }

    // Second read of the same retained handles detects content substitution
    // observable through those handles. It does not establish a protected
    // filesystem, prevent administrator bypass, or cover model weight shards.
    internal bool StillMatchesRetainedBytes() {
        if(disposed) throw new ObjectDisposedException("Tier1BHostEvidenceHold");
        return DigestCurrentBytes(plan)==planSha
            && DigestCurrentBytes(manifest)==manifestSha
            && DigestCurrentBytes(executable)==executableSha;
    }

    internal long ObserveNextTestEpoch(HostOwnedProbeReservation journal) {
        if(disposed) throw new ObjectDisposedException("Tier1BHostEvidenceHold");
        if(journal==null || !(journal.Phase=="GENESIS" || journal.Phase=="TERMINAL")
                || journal.Epoch>=Int64.MaxValue-1)
            throw new InvalidOperationException("Tier1 journal not ready; HOLD");
        return journal.Epoch+1;
    }

    internal string PlanSha256 { get { return planSha; } }
    internal string ManifestSha256 { get { return manifestSha; } }
    internal string ExecutableSha256 { get { return executableSha; } }
    internal string NonceSha256 { get { return nonceSha; } }

    public void Dispose() {
        if(disposed) return;
        disposed=true;
        plan.Dispose(); manifest.Dispose(); executable.Dispose();
    }
}
}
