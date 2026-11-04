// Native Windows .NET Framework 4.8 source regression. Ephemeral test RSA
// keys are never enrolled, saved, or accepted by the real host broker.
using System;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using Vera.TrainingDay.HostBroker;

internal static class Tier1BGrantCandidateTests {
    const string OwnerPurposeOid="2.25.281124513629607016020108740092472226515";
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
        "not_after_unix", "max_seconds", "vram_target_mib", "protected_access",
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
    static string Sha(byte[] bytes) {
        using(SHA256 h=SHA256.Create()) {
            byte[] digest=h.ComputeHash(bytes);
            return BitConverter.ToString(digest).Replace("-","").ToLowerInvariant();
        }
    }
    static CertificateRequest OwnerRequest(RSA key,bool includePurpose,bool isCa=false) {
        var request=new CertificateRequest("CN=VERA TrainingDay Tier1 Owner TEST",
            key,HashAlgorithmName.SHA256,RSASignaturePadding.Pkcs1);
        request.CertificateExtensions.Add(new X509BasicConstraintsExtension(
            isCa,isCa,0,true));
        request.CertificateExtensions.Add(new X509KeyUsageExtension(
            X509KeyUsageFlags.DigitalSignature,true));
        if(includePurpose) {
            var eku=new OidCollection();
            eku.Add(new Oid(OwnerPurposeOid));
            request.CertificateExtensions.Add(new X509EnhancedKeyUsageExtension(
                eku,true));
        }
        return request;
    }
    static void PublicOnlyMachineCertRegressions() {
        // Fixtures never touch CurrentUser/LocalMachine certificate stores.
        // Only short-lived in-memory keys are created, then disposed.
        using(RSACng ownerKey=new RSACng(3072)) {
            var request=OwnerRequest(ownerKey,true);
            using(X509Certificate2 ownerWithPrivate=request.CreateSelfSigned(
                DateTimeOffset.UtcNow.AddMinutes(-5),DateTimeOffset.UtcNow.AddDays(1))) {
                byte[] payload=Payload(Fields());
                byte[] signature=ownerKey.SignData(payload,HashAlgorithmName.SHA256,
                                                  RSASignaturePadding.Pkcs1);
                string pin=Sha(ownerWithPrivate.RawData);
                Assert(ownerWithPrivate.HasPrivateKey,"owner fixture has no private key");
                var leaked=Tier1BGrantCandidate.VerifyPublicCertForNativeTestOnly(
                    ownerWithPrivate,pin,payload,signature,Host(7,Now));
                CheckNoLaunch(leaked,"owner secret imported into machine store");
                Assert(!leaked.ScopeAndSignatureMatched &&
                    leaked.Reason=="OWNER_PRIVATE_KEY_IN_MACHINE_STORE",
                    "machine store accepted signing key");
                using(X509Certificate2 machinePublicOnly=new X509Certificate2(
                    ownerWithPrivate.Export(X509ContentType.Cert))) {
                    Assert(!machinePublicOnly.HasPrivateKey,
                           "public DER unexpectedly has signing secret");
                    var publicOnly=Tier1BGrantCandidate.VerifyPublicCertForNativeTestOnly(
                        machinePublicOnly,pin,payload,signature,Host(7,Now));
                    CheckNoLaunch(publicOnly,"public-only cert candidate");
                    Assert(publicOnly.ScopeAndSignatureMatched &&
                        publicOnly.Reason==
                          "SIGNATURE_AND_SCOPE_MATCHED_SOURCE_ONLY_NO_LAUNCH",
                        "valid public-only certificate failed source-only observation");
                    var singleton=new X509Certificate2Collection();
                    singleton.Add(machinePublicOnly);
                    var one=Tier1BGrantCandidate.VerifyStoreSnapshotForNativeTestOnly(
                        singleton,pin,payload,signature,Host(7,Now));
                    CheckNoLaunch(one,"single pinned public cert context");
                    Assert(one.ScopeAndSignatureMatched,
                        "unique public-only store context rejected");
                    using(var secondPublicContext=new X509Certificate2(
                        ownerWithPrivate.Export(X509ContentType.Cert))) {
                        var duplicates=new X509Certificate2Collection();
                        duplicates.Add(machinePublicOnly);
                        duplicates.Add(secondPublicContext);
                        var ambiguous=Tier1BGrantCandidate.VerifyStoreSnapshotForNativeTestOnly(
                            duplicates,pin,payload,signature,Host(7,Now));
                        CheckNoLaunch(ambiguous,"duplicate matching public cert contexts");
                        Assert(!ambiguous.ScopeAndSignatureMatched &&
                            ambiguous.Reason=="OWNER_CERTIFICATE_AMBIGUOUS_MACHINE_STORE",
                            "duplicate DER contexts accepted by first-match lookup");
                    }
                    var mixed=new X509Certificate2Collection();
                    mixed.Add(machinePublicOnly);
                    mixed.Add(ownerWithPrivate);
                    var ambiguousSecret=Tier1BGrantCandidate.VerifyStoreSnapshotForNativeTestOnly(
                        mixed,pin,payload,signature,Host(7,Now));
                    CheckNoLaunch(ambiguousSecret,"mixed public/private duplicate contexts");
                    Assert(ambiguousSecret.Reason==
                        "OWNER_CERTIFICATE_AMBIGUOUS_MACHINE_STORE",
                        "duplicate public/private context accepted by first match");
                    var swapped=Tier1BGrantCandidate.VerifyPublicCertForNativeTestOnly(
                        machinePublicOnly,new string('0',64),payload,signature,Host(7,Now));
                    CheckNoLaunch(swapped,"swapped DER trust anchor");
                    Assert(!swapped.ScopeAndSignatureMatched &&
                        swapped.Reason=="OWNER_CERTIFICATE_DER_MISMATCH",
                        "different compiled DER pin was accepted");
                }
                var withoutPurpose=OwnerRequest(ownerKey,false);
                using(var noEkuWithKey=withoutPurpose.CreateSelfSigned(
                    DateTimeOffset.UtcNow.AddMinutes(-5),DateTimeOffset.UtcNow.AddDays(1)))
                using(var noEku=new X509Certificate2(
                    noEkuWithKey.Export(X509ContentType.Cert))) {
                    var noPurpose=Tier1BGrantCandidate.VerifyPublicCertForNativeTestOnly(
                        noEku,Sha(noEku.RawData),payload,signature,Host(7,Now));
                    CheckNoLaunch(noPurpose,"unscoped owner cert purpose");
                    Assert(noPurpose.Reason=="OWNER_CERTIFICATE_PURPOSE_REQUIRED",
                        "ordinary RSA certificate without owner EKU accepted");
                }
                var caRequest=OwnerRequest(ownerKey,true,true);
                using(var caWithKey=caRequest.CreateSelfSigned(
                    DateTimeOffset.UtcNow.AddMinutes(-5),DateTimeOffset.UtcNow.AddDays(1)))
                using(var caPublic=new X509Certificate2(
                    caWithKey.Export(X509ContentType.Cert))) {
                    var ca=Tier1BGrantCandidate.VerifyPublicCertForNativeTestOnly(
                        caPublic,Sha(caPublic.RawData),payload,signature,Host(7,Now));
                    CheckNoLaunch(ca,"owner CA certificate");
                    Assert(ca.Reason=="OWNER_CERTIFICATE_MUST_NOT_BE_CA",
                        "CA certificate accepted for owner-signed launch scope");
                }
            }
        }
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
        PublicOnlyMachineCertRegressions();
        using(RSACryptoServiceProvider fixture=new RSACryptoServiceProvider(3072)) {
            fixture.PersistKeyInCsp=false;
            Assert(fixture.KeySize==3072,"3072-bit fixture unavailable");
            byte[] valid=Payload(Fields());
            byte[] sig=fixture.SignData(valid,HashAlgorithmName.SHA256,
                                         RSASignaturePadding.Pkcs1);
            Tier1BCandidateObservation matched=Tier1BGrantCandidate
                .VerifyForNativeTestOnly(valid,sig,fixture,Host(7,Now));
            CheckNoLaunch(matched,"valid fixture");
            Assert(matched.ScopeAndSignatureMatched &&
              matched.Reason=="SIGNATURE_AND_SCOPE_MATCHED_SOURCE_ONLY_NO_LAUNCH",
              "scoped crypto fixture should match without launch reason="+matched.Reason+
              " rsaBits="+fixture.KeySize+" sigBytes="+sig.Length);

            Assert(matched.AbortByUnix==1800000200L &&
                matched.AbortByUnix<Now+1800,
                "midrun expiry must dominate 1800-second ceiling");
            var oneSecondLeft=Tier1BGrantCandidate.VerifyForNativeTestOnly(
                valid,sig,fixture,Host(7,1800000199L));
            CheckNoLaunch(oneSecondLeft,"expiry mid-run");
            Assert(oneSecondLeft.ScopeAndSignatureMatched &&
                oneSecondLeft.AbortByUnix==1800000200L &&
                oneSecondLeft.AbortByUnix<1800000199L+1800,
                "midrun expiry does not provide an earlier abort deadline");

            byte[] forged=(byte[])sig.Clone();forged[forged.Length/2]^=1;
            Tier1BCandidateObservation badSig=Tier1BGrantCandidate
                .VerifyForNativeTestOnly(valid,forged,fixture,Host(7,Now));
            CheckNoLaunch(badSig,"tampered signature");
            Assert(!badSig.ScopeAndSignatureMatched &&
                badSig.Reason=="SIGNATURE_INVALID","forged signature admitted");
            using(RSACryptoServiceProvider attacker=new RSACryptoServiceProvider(3072)) {
                attacker.PersistKeyInCsp=false;
                Assert(attacker.KeySize==3072,"3072-bit attack fixture unavailable");
                byte[] attackerSig=attacker.SignData(valid,HashAlgorithmName.SHA256,
                                                      RSASignaturePadding.Pkcs1);
                var wrongSigner=Tier1BGrantCandidate.VerifyForNativeTestOnly(
                    valid,attackerSig,fixture,Host(7,Now));
                CheckNoLaunch(wrongSigner,"wrong signer");
                Assert(wrongSigner.Reason=="SIGNATURE_INVALID","wrong signer accepted");
            }
            using(RSACryptoServiceProvider weak=new RSACryptoServiceProvider(1024)) {
                weak.PersistKeyInCsp=false;
                Assert(weak.KeySize==1024,"1024-bit weak fixture unavailable");
                byte[] weakSig=weak.SignData(valid,HashAlgorithmName.SHA256,
                                            RSASignaturePadding.Pkcs1);
                var weakReceipt=Tier1BGrantCandidate.VerifyForNativeTestOnly(
                    valid,weakSig,weak,Host(7,Now));
                CheckNoLaunch(weakReceipt,"weak RSA 1024");
                Assert(!weakReceipt.ScopeAndSignatureMatched &&
                    weakReceipt.Reason=="SIGNATURE_INVALID",
                    "1024-bit RSA owner scope was accepted");
            }
            using(RSACryptoServiceProvider insufficient=new RSACryptoServiceProvider(2048)) {
                insufficient.PersistKeyInCsp=false;
                byte[] weakSignature=insufficient.SignData(valid,HashAlgorithmName.SHA256,
                                                          RSASignaturePadding.Pkcs1);
                var insufficientReceipt=Tier1BGrantCandidate.VerifyForNativeTestOnly(
                    valid,weakSignature,insufficient,Host(7,Now));
                CheckNoLaunch(insufficientReceipt,"RSA 2048 now below owner policy");
                Assert(!insufficientReceipt.ScopeAndSignatureMatched &&
                    insufficientReceipt.Reason=="SIGNATURE_INVALID",
                    "RSA 2048 was admitted despite 3072-bit owner minimum");
            }

            var atExpiry=Tier1BGrantCandidate.VerifyForNativeTestOnly(
                valid,sig,fixture,Host(7,1800000200L));
            CheckNoLaunch(atExpiry,"expiry equals admission");
            Assert(!atExpiry.ScopeAndSignatureMatched && atExpiry.AbortByUnix==0,
                "grant accepted at now==not_after or leaked deadline");
            var clockOverflow=Tier1BGrantCandidate.VerifyForNativeTestOnly(
                valid,sig,fixture,Host(7,Int64.MaxValue-1));
            CheckNoLaunch(clockOverflow,"clock arithmetic overflow");
            Assert(!clockOverflow.ScopeAndSignatureMatched &&
                clockOverflow.AbortByUnix==0,
                "clock overflow matched");
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
