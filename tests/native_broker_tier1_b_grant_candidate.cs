// Native Windows .NET Framework 4.8 source regression. Ephemeral test RSA
// keys are never enrolled, saved, or accepted by the real host broker.
using System;
using System.Security.Cryptography;
using System.Text;
using Vera.TrainingDay.HostBroker;

internal static class Tier1BGrantCandidateTests {
    static readonly string Plan=new string('a',64);
    static readonly string Manifest=new string('b',64);
    static readonly string Executable=new string('c',64);
    static readonly string Nonce=new string('d',64);
    const long Now=1800000000L;
    static int checks;

    static string[] Fields() {
        return new string[] {
          "TRAININGDAY_TIER1_B_OWNER_GRANT_V1",
          "TIER1_PUBLIC_SYNTHETIC",
          "LAPPY",
          "WDDM_BOUNDED_COEXISTENCE",
          "Qwen/Qwen3.5-4B",
          "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
          "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
          "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
          Plan, Manifest, Executable, Nonce,
          "7", "1799999900", "1800000200", "1800", "3072",
          "NONE", "UNPROMOTED_ONLY", "TARGETED_WSL_PIDTREE_REQUIRED"
        };
    }
    static readonly string[] Keys = {
        "schema", "tier", "host", "policy", "model_id", "model_revision",
        "shard1_sha256", "shard2_sha256", "plan_sha256", "manifest_sha256",
        "executable_sha256", "nonce_sha256", "epoch", "not_before_unix",
        "not_after_unix", "max_seconds", "max_vram_mib", "protected_access",
        "outputs", "wsl_cleanup"
    };
    static byte[] Payload(string[] values) {
        string[] lines=new string[Keys.Length];
        for(int i=0;i<Keys.Length;i++) lines[i]=Keys[i]+"="+values[i];
        return Encoding.ASCII.GetBytes(String.Join("\n",lines)+"\n");
    }
    static Tier1BGrantCandidate.HostExpected Host(long epoch,long now) {
        return new Tier1BGrantCandidate.HostExpected(
            Plan,Manifest,Executable,Nonce,epoch,now);
    }
    static void Assert(bool result,string what) {
        checks++;
        if(!result) throw new Exception("native B grant assertion: "+what);
    }
    static void CheckNoLaunch(Tier1BCandidateObservation receipt,string label) {
        Assert(receipt!=null && receipt.State=="HOLD" &&
            !receipt.LaunchAuthorized && !receipt.Tier2Authorized &&
            !receipt.Promotable,label+" grants authority");
    }
    static void ExpectRejection(RSA fixtureKey, string[] mutated, string label) {
        byte[] b=Payload(mutated);
        byte[] signature=fixtureKey.SignData(b,HashAlgorithmName.SHA256,
                                          RSASignaturePadding.Pkcs1);
        Tier1BCandidateObservation result=Tier1BGrantCandidate
            .VerifyForNativeTestOnly(b,signature,fixtureKey,Host(7,Now));
        CheckNoLaunch(result,label);
        Assert(!result.ScopeAndSignatureMatched,label+" wrongly matched");
    }
    static void Change(RSA key,int index,string value,string label) {
        string[] f=Fields();f[index]=value;
        ExpectRejection(key,f,label);
    }
    static void Main() {
        using(RSA fixture=RSA.Create()) {
            fixture.KeySize=2048;
            byte[] valid=Payload(Fields());
            byte[] sig=fixture.SignData(valid,HashAlgorithmName.SHA256,
                                         RSASignaturePadding.Pkcs1);
            Tier1BCandidateObservation matched=Tier1BGrantCandidate
                .VerifyForNativeTestOnly(valid,sig,fixture,Host(7,Now));
            CheckNoLaunch(matched,"valid fixture");
            Assert(matched.ScopeAndSignatureMatched &&
              matched.Reason=="SIGNATURE_AND_SCOPE_MATCHED_SOURCE_ONLY_NO_LAUNCH",
              "scoped crypto fixture should match without launch");

            byte[] forged=(byte[])sig.Clone();forged[forged.Length/2]^=1;
            Tier1BCandidateObservation badSig=Tier1BGrantCandidate
                .VerifyForNativeTestOnly(valid,forged,fixture,Host(7,Now));
            CheckNoLaunch(badSig,"tampered signature");
            Assert(!badSig.ScopeAndSignatureMatched &&
                badSig.Reason=="SIGNATURE_INVALID","forged signature admitted");
            using(RSA attacker=RSA.Create()) {
                attacker.KeySize=2048;
                byte[] attackerSig=attacker.SignData(valid,HashAlgorithmName.SHA256,
                                                      RSASignaturePadding.Pkcs1);
                var wrongSigner=Tier1BGrantCandidate.VerifyForNativeTestOnly(
                    valid,attackerSig,fixture,Host(7,Now));
                CheckNoLaunch(wrongSigner,"wrong signer");
                Assert(wrongSigner.Reason=="SIGNATURE_INVALID","wrong signer accepted");
            }
            Change(fixture,1,"TIER2_R201","wrong tier");
            Change(fixture,2,"VN1","wrong host");
            Change(fixture,3,"WDDM_PHYSICAL_EXCLUSIVE","wrong policy");
            Change(fixture,4,"rodrigomt/Qwen3.5-4B-Uncensored-Aggressive","modified base");
            Change(fixture,5,"851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0b","wrong revision");
            Change(fixture,6,new string('0',64),"wrong official shard1");
            Change(fixture,7,new string('0',64),"wrong official shard2");
            Change(fixture,8,new string('0',64),"wrong immutable plan");
            Change(fixture,9,new string('0',64),"wrong manifest");
            Change(fixture,10,new string('0',64),"wrong executable");
            Change(fixture,11,new string('0',64),"wrong nonce");
            Change(fixture,12,"8","replayed epoch");
            Change(fixture,12,"07","noncanonical epoch");
            Change(fixture,13,"1800000001","future grant");
            Change(fixture,14,"1799999999","expired grant");
            Change(fixture,14,"1800002001","overlong grant lifetime");
            Change(fixture,15,"1801","overlong runtime");
            Change(fixture,16,"4096","unsafe VRAM target");
            Change(fixture,17,"R201_DEV","protected data");
            Change(fixture,18,"PROMOTABLE","promotion");
            Change(fixture,19,"WSL_TERMINATE_SHARED_DISTRO","unsafe cleanup");
            string[] duplicated=Fields();duplicated[10]="\nnonce_sha256="+Nonce;
            ExpectRejection(fixture,duplicated,"injected duplicate newline");

            byte[] crlf=Encoding.ASCII.GetBytes(Encoding.ASCII.GetString(valid).Replace("\n","\r\n"));
            var badNewline=Tier1BGrantCandidate.VerifyForNativeTestOnly(
                crlf,fixture.SignData(crlf,HashAlgorithmName.SHA256,
                                      RSASignaturePadding.Pkcs1),fixture,Host(7,Now));
            CheckNoLaunch(badNewline,"CRLF");
            Assert(!badNewline.ScopeAndSignatureMatched,"CRLF accepted");
            byte[] noFinal=new byte[valid.Length-1];
            Array.Copy(valid,noFinal,noFinal.Length);
            var trunc=Tier1BGrantCandidate.VerifyForNativeTestOnly(
                noFinal,fixture.SignData(noFinal,HashAlgorithmName.SHA256,
                                         RSASignaturePadding.Pkcs1),fixture,Host(7,Now));
            CheckNoLaunch(trunc,"missing LF");
            Assert(!trunc.ScopeAndSignatureMatched,"truncated accepted");

            var stale=Tier1BGrantCandidate.VerifyForNativeTestOnly(
                valid,sig,fixture,Host(8,Now));
            CheckNoLaunch(stale,"host replay floor");
            Assert(!stale.ScopeAndSignatureMatched,"replayed host epoch accepted");
            var signerUnavailable=Tier1BGrantCandidate.VerifyPinnedHostOwnerCandidate(
                valid,sig,Host(7,Now));
            CheckNoLaunch(signerUnavailable,"unenrolled real broker");
            Assert(signerUnavailable.Reason=="OWNER_CERTIFICATE_NOT_ENROLLED",
                "caller test signer enrolled real broker");
        }
        Console.WriteLine("NATIVE_TIER1_B_SIGNED_SCOPE_HOLD_PASS checks="+checks);
    }
}