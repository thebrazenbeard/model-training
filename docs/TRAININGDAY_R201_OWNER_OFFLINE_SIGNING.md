# R201 owner signing — offline operator utility, no training authority

**Status: SOURCE-ONLY HOLD.** This is an operator-controlled way to produce
the exact R201 owner-intention envelope accepted by the existing candidate
verifier. It does **not** enroll a key, validate Calc's use rights, authenticate
private DEV custody, authorize evaluation, issue a NAS witness, or launch a GPU.

Patrick may act as **both original rights holder and data steward** for material
he controls. There is no mandatory third human. Independent *evidence custody*
for withheld evaluation and host authorization is a separate requirement.

## What must exist first

1. A *user-selected, dedicated* **existing** Ed25519 PEM private key, kept on
   Patrick's own secured Windows/Linux machine. This utility neither creates a
   key nor discovers, reuses or imports SSH, GitHub, Executor, NAS or other
   existing authentication keys. **Do not send the private key to ChatGPT,
   GitHub, a public CI job, or a chat export.** Signing is a local user action.
2. Two genuine SHA-256 commitments delivered through an independently
   controlled private evaluator: the R201 protected DEV commitment and the
   protected exclusion receipt. The public Python DEV generator can help the
   custodian implement this, but synthetic/test values or an unhashed public
   nonce are **not** independently protected evaluation evidence.
3. Patrick's affirmative decision about the exact Calc TRAIN generator,
   Qwen3.5 model, intended training purpose, and key validity interval. Calc
   at pinned revision `17e12d0...` lacks a root `LICENSE`; this means there is
   no independently inferred third-party license, **not** that ownership can
   be assigned by a model or that a repo-admin account proves source rights.

If Patrick does not already control a **dedicated** PEM signing key, a
separate manual local operation can create one. For example, on his own host
**if OpenSSL is installed**, after establishing a private output directory:

```powershell
# Manual owner action only; the signer never runs this command.
openssl genpkey -algorithm ED25519 -aes-256-cbc -out 'D:\VERA\private\r201-dedicated-owner.pem'
```

The command prompts for a passphrase; do not put it on the command line.
Confirm the output directory/file ACL permits Patrick alone before use (or
use a Linux `umask 077` with private directory). Store a secured backup and
protect against loss or key replacement. Do **not** use an existing SSH,
GitHub, Executor or NAS authentication key for this purpose. Key creation
alone never establishes owner identity or source rights; an independent host
operator must enroll the public key after authenticating Patrick's choice.

## Operator-run example (never executed by the CI tests)

From a local checkout that includes the candidate verifier and this module,
with `cryptography==46.0.3` installed in a private local Python environment:

```powershell
python -m trainingday.r201_owner_offline_sign `
  --existing-key-pem 'D:\VERA\private\r201-dedicated-owner.pem' `
  --owner-identity 'OWNER_CHOSEN_IDENTITY' `
  --private-dev-commitment-sha256 '64_CHARACTER_REAL_CUSTODIAN_HASH' `
  --protected-exclusion-receipt-sha256 '64_CHARACTER_REAL_CUSTODIAN_HASH' `
  --owner-key-epoch 1 `
  --not-before-utc '2026-11-03T00:00:00Z' `
  --expires-utc '2026-11-10T00:00:00Z' `
  --decision ASSERT_RIGHTS_TO_TRAIN `
  --ack-owner-rights `
  --output-prefix 'D:\VERA\private\r201-owner-claim'
```

The placeholders must be replaced with **real owner-selected values**. The
command refuses missing/invalid commitments, stale timestamps, non-Ed25519
keys, or absent affirmative acknowledgment. It will not overwrite output
files. An encrypted PEM prompts for the passphrase without echoing it.

Four local public-evidence files are produced (never the private key):

| File suffix | Contents |
| --- | --- |
| `.claim.json` | Exact canonical R201 corpus/model/purpose/DEV/exclusions/owner bytes |
| `.signature.hex` | Detached Ed25519 signature over domain-prefixed claim bytes |
| `.public_key.hex` | Public key for a separately verified host enrollment |
| `.receipt.json` | SHA-256 identity and `all_training_authority=false` |

For `WITHHOLD`, select that explicit decision and omit `--ack-owner-rights`.
The command still signs a denial, without issuing training permission.

## Required independent readback before any real training

- Authenticated host operator confirms the key is genuinely controlled by
  Patrick, directly acknowledges his rights decision, and pins the exact
  public key, scope, epoch, expiration and revocation policy outside this
  public repository.
- Private evaluator independently owns the secret DEV nonce, exclusion
  provenance, commitment issuance, access controls and blinded results.
- Host-owned immutable run plan, protected NAS Q6 monotonic witness and
  exclusive GPU process/lease are independently qualified and read back.

Only after those separate effects may an authority-bearing host component
consider the signed claim. This repository's candidate verifier and offline
signer **always remain non-authorizing**. A successful signature test or
`R201_OWNER_CLAIM_SIGNED_WITH_SUPPLIED_KEY_SOURCE_ONLY_HOLD` is *not* a grant.
