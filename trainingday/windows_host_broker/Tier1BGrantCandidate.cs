// SOURCE-ONLY Contract B owner-signature candidate. No broker route imports this.
// A signature and scoped manifest can never, by themselves, authorize execution.
using System;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Text;

namespace Vera.TrainingDay.HostBroker {
internal sealed class Tier1BCandidateObservation {
    internal readonly string State = "HOLD";
    internal readonly string Reason;
    internal readonly bool ScopeAndSignatureMatched;
    internal readonly bool LaunchAuthorized = false;
    internal readonly bool Tier2Authorized = false;
    internal readonly bool Promotable = false;
    // Non-authoritative deadline observation: a future host runner must prove
    // fail-safe termination at or before this time before enabling any launch.
    internal readonly long AbortByUnix;
    internal Tier1BCandidateObservation(string reason, bool matched, long abortByUnix=0) {
        Reason = reason; ScopeAndSignatureMatched = matched;
        AbortByUnix = abortByUnix;
    }
}

internal static class Tier1BGrantCandidate {
    // This can be enrolled only by an independently reviewed host deployment.
    // Never read a fingerprint or signer cert from the request or environment.
    const string InstalledOwnerCertificateDerSha256 = "UNENROLLED";
    const string OfficialRevision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a";
    const string OfficialShard1 = "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61";
    const string OfficialShard2 = "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188";
    const int MaxGrantBytes = 4096;
    static readonly string[] Keys = {
        "schema", "tier", "host", "policy", "model_id", "model_revision",
        "shard1_sha256", "shard2_sha256", "plan_sha256", "manifest_sha256",
        "executable_sha256", "nonce_sha256", "epoch", "not_before_unix",
        "not_after_unix", "max_seconds", "vram_target_mib", "protected_access",
        "outputs", "wsl_cleanup"
    };

    internal sealed class HostExpected {
        // These values MUST be computed by a protected host broker from an
        // immutable plan, verified local files and its separate Tier1 journal.
        // Caller-supplied copies are not host evidence.
        internal readonly string Plan, Manifest, Executable, Nonce;
        internal readonly long NextEpoch, NowUnix;
        internal HostExpected(string plan, string manifest, string executable,
                              string nonce, long nextEpoch, long nowUnix) {
            Plan = plan; Manifest = manifest; Executable = executable;
            Nonce = nonce; NextEpoch = nextEpoch; NowUnix = nowUnix;
        }
    }

    static Tier1BCandidateObservation Hold(string reason) {
        return new Tier1BCandidateObservation(reason, false);
    }
    static bool Digest(string s) {
        if (s == null || s.Length != 64) return false;
        for (int i=0;i<s.Length;i++)
            if (!((s[i]>='0' && s[i]<='9') || (s[i]>='a' && s[i]<='f')))
                return false;
        return true;
    }
    static string Sha(byte[] bytes) {
        using (SHA256 h=SHA256.Create()) {
            byte[] raw=h.ComputeHash(bytes);
            var b=new StringBuilder(64);
            foreach (byte value in raw) b.Append(value.ToString("x2"));
            return b.ToString();
        }
    }
    static bool UnsignedNumber(string s, out long n) {
        n=0;
        if (String.IsNullOrEmpty(s) || (s.Length>1 && s[0]=='0'))
            return false;
        for (int i=0;i<s.Length;i++) if(s[i]<'0' || s[i]>'9') return false;
        return Int64.TryParse(s, NumberStyles.None, CultureInfo.InvariantCulture,out n);
    }
    static string[] ParseCanonical(byte[] payload) {
        if (payload==null || payload.Length==0 || payload.Length>MaxGrantBytes)
            throw new InvalidDataException("grant size");
        foreach (byte b in payload)
            if (b!=10 && (b<32 || b>126))
                throw new InvalidDataException("non-canonical byte");
        if (payload[payload.Length-1]!=10)
            throw new InvalidDataException("missing final LF");
        string raw=Encoding.ASCII.GetString(payload);
        string[] lines=raw.Split('\n');
        if (lines.Length!=Keys.Length+1 || lines[Keys.Length]!="")
            throw new InvalidDataException("grant field count");
        string[] values=new string[Keys.Length];
        for(int i=0;i<Keys.Length;i++) {
            string prefix=Keys[i]+"=";
            if (!lines[i].StartsWith(prefix,StringComparison.Ordinal))
                throw new InvalidDataException("grant key order/duplicates");
            values[i]=lines[i].Substring(prefix.Length);
            if(values[i].Length==0)
                throw new InvalidDataException("grant empty field");
        }
        return values;
    }
    static bool MatchesHostScope(string[] f, HostExpected host) {
        if(host==null || !Digest(host.Plan) || !Digest(host.Manifest)
                || !Digest(host.Executable) || !Digest(host.Nonce)
                || host.NextEpoch<=0 || host.NextEpoch>=Int64.MaxValue-1
                || host.NowUnix<=0 || host.NowUnix>Int64.MaxValue-1800)
            return false;
        if(f[0]!="TRAININGDAY_TIER1_B_OWNER_GRANT_V1"
                || f[1]!="TIER1_PUBLIC_SYNTHETIC"
                || f[2]!="LAPPY"
                || f[3]!="WDDM_BOUNDED_COEXISTENCE"
                || f[4]!="Qwen/Qwen3.5-4B"
                || f[5]!=OfficialRevision
                || f[6]!=OfficialShard1 || f[7]!=OfficialShard2
                || f[8]!=host.Plan || f[9]!=host.Manifest
                || f[10]!=host.Executable || f[11]!=host.Nonce
                || f[15]!="1800" || f[16]!="3072"
                || f[17]!="NONE" || f[18]!="UNPROMOTED_ONLY"
                || f[19]!="TARGETED_WSL_PIDTREE_REQUIRED") return false;
        long epoch, before, after;
        if(!UnsignedNumber(f[12],out epoch)
                || !UnsignedNumber(f[13],out before)
                || !UnsignedNumber(f[14],out after)) return false;
        return epoch==host.NextEpoch && before<=host.NowUnix
            && host.NowUnix<after && after>before && after-before<=1800;
    }
    static bool VerifyRsaSignature(byte[] payload, byte[] signature, RSA ownerPublicKey) {
        if (signature==null || ownerPublicKey==null) return false;
        try {
            int rsaBits=ownerPublicKey.KeySize;
            if(rsaBits<2048 || rsaBits>4096 || rsaBits%8!=0
                    || signature.Length!=rsaBits/8) return false;
            return ownerPublicKey.VerifyData(payload, signature,
                HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
        } catch (CryptographicException) { return false; }
    }

    // A CRYPTOGRAPHIC/SCOPING REGRESSION ONLY: it accepts a test public key but
    // never yields authorization and is NEVER callable as a broker approval.
    internal static Tier1BCandidateObservation VerifyForNativeTestOnly(
        byte[] payload, byte[] signature, RSA syntheticPublicKey,
        HostExpected host) {
        string[] fields;
        try { fields=ParseCanonical(payload); }
        catch (InvalidDataException) { return Hold("MALFORMED_CANONICAL_GRANT"); }
        if (!MatchesHostScope(fields,host)) return Hold("SCOPE_OR_REPLAY_MISMATCH");
        if (!VerifyRsaSignature(payload,signature,syntheticPublicKey))
            return Hold("SIGNATURE_INVALID");
        // This observation is not a scheduler or an enforcement mechanism.
        // A later installed host must independently enforce its clock/deadline
        // and kill the nested WSL PID tree, including on broker crash.
        long expires;
        if(!UnsignedNumber(fields[14],out expires))
            return Hold("SCOPE_OR_REPLAY_MISMATCH");
        long abortBy=Math.Min(expires,host.NowUnix+1800);
        return new Tier1BCandidateObservation(
            "SIGNATURE_AND_SCOPE_MATCHED_SOURCE_ONLY_NO_LAUNCH", true, abortBy);
    }

    // Production-facing candidate additionally needs a *compiled and verified*
    // owner public-certificate DER SHA256 pinned at installation. The current
    // source and installed service have NO enrolled anchor. No provided key,
    // callback, environment variable or request metadata may override this.
    internal static Tier1BCandidateObservation VerifyPinnedHostOwnerCandidate(
        byte[] payload, byte[] signature, HostExpected verifiedHost) {
        if (!Digest(InstalledOwnerCertificateDerSha256))
            return Hold("OWNER_CERTIFICATE_NOT_ENROLLED");
        using(var store=new X509Store(StoreName.My,StoreLocation.LocalMachine)) {
            try {
                store.Open(OpenFlags.ReadOnly | OpenFlags.OpenExistingOnly);
                foreach(X509Certificate2 cert in store.Certificates) {
                    if(Sha(cert.RawData)!=InstalledOwnerCertificateDerSha256) continue;
                    DateTime now=DateTime.UtcNow;
                    if(now<cert.NotBefore.ToUniversalTime() ||
                       now>cert.NotAfter.ToUniversalTime())
                        return Hold("OWNER_CERTIFICATE_OUT_OF_DATE");
                    using(RSA publicKey=cert.GetRSAPublicKey()) {
                        if(publicKey==null) return Hold("OWNER_CERTIFICATE_RSA_REQUIRED");
                        return VerifyForNativeTestOnly(payload,signature,publicKey,
                                                      verifiedHost);
                    }
                }
            } catch (CryptographicException) { return Hold("HOST_CERTIFICATE_STORE_UNAVAILABLE"); }
              catch (System.Security.SecurityException) { return Hold("HOST_CERTIFICATE_STORE_UNAVAILABLE"); }
        }
        return Hold("OWNER_CERTIFICATE_NOT_FOUND");
    }
}
}