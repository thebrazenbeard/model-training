"""Opt-in CPU-only R202 token-budget profile against a pinned local tokenizer.

Reads only tokenizer.json and the public R202 source; no model weights, GPUs,
inference, training data grants or protected evaluator material are involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

from trainingday import cognitive_curriculum


OFFICIAL_MODEL = "Qwen/Qwen3.5-4B"
OFFICIAL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
OFFICIAL_TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
BUDGETS = (64, 128, 256)
MAX_TOKENIZER_BYTES = 20_000_000


class TokenizerProbeError(ValueError):
    """Reject unverified tokenizer files and unusable source rows."""


def _hash_local_tokenizer(path: Path) -> str:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not 0 < resolved.stat().st_size <= MAX_TOKENIZER_BYTES:
        raise TokenizerProbeError("tokenizer source absent or outside bounded size")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != OFFICIAL_TOKENIZER_SHA256:
        raise TokenizerProbeError("local tokenizer SHA256 mismatch; refusing profile")
    return observed


def _stats(values: list[int]) -> dict:
    values = sorted(values)
    if not values:
        raise TokenizerProbeError("empty token observations")
    return {
        "min": values[0],
        "median": values[len(values) // 2],
        "p95_nearest_rank": values[(95 * len(values) + 99) // 100 - 1],
        "max": values[-1],
        "exceeds_budget": {str(b): sum(v > b for v in values) for b in BUDGETS},
    }


def profile_counts(encode: Callable[[str], list[int]], rows: list[dict] | None = None) -> dict:
    """Profile public text, never assert model training compatibility.

    `prompt_plus_answer` is only the literal `prompt + '\\n' + answer` token
    budget, with add_special_tokens=False. It excludes model chat-template,
    BOS/EOS, system context, and loss-mask framing.
    """
    rows = cognitive_curriculum.train_rows() if rows is None else rows
    if not rows:
        raise TokenizerProbeError("empty R202 source")
    family_counts = {family: {"prompt": [], "prompt_plus_answer": []}
                     for family in cognitive_curriculum.FAMILIES}
    for row in rows:
        family = next((family for family in family_counts
                       if row.get("id", "").startswith(f"r202-{family}-")), None)
        if family is None or row.get("split") != "train":
            raise TokenizerProbeError("unknown row family or non-TRAIN input")
        prompt = row["prompt"]
        pair = prompt + "\n" + row["answer"]
        for key, value in (("prompt", prompt), ("prompt_plus_answer", pair)):
            ids = encode(value)
            if type(ids) is not list or any(type(t) is not int for t in ids):
                raise TokenizerProbeError("encoder must return integer token IDs")
            family_counts[family][key].append(len(ids))
    all_lengths = {key: [v for family in family_counts.values() for v in family[key]]
                   for key in ("prompt", "prompt_plus_answer")}
    return {
        "schema": "VERA_TRAININGDAY_R202_PINNED_TOKENIZER_PROFILE_V1",
        "corpus_sha256": hashlib.sha256(cognitive_curriculum.train_bytes()).hexdigest(),
        "row_count": len(rows),
        "measurement": "Tokenizer.encode(add_special_tokens=False): prompt; prompt+'\\n'+answer",
        "omits_chat_template_and_special_tokens": True,
        "authorizes_training": False,
        "overall": {key: _stats(values) for key, values in all_lengths.items()},
        "families": {family: {key: _stats(values) for key, values in counts.items()}
                     for family, counts in family_counts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path,
                        help="already-local official Qwen3.5-4B tokenizer.json")
    args = parser.parse_args()
    tokenizer_sha = _hash_local_tokenizer(args.tokenizer)
    try:
        from tokenizers import Tokenizer  # optional local Rust tokenizers wheel
        import tokenizers
    except ImportError as exc:
        raise TokenizerProbeError("optional local tokenizers package unavailable") from exc
    tokenizer = Tokenizer.from_file(str(args.tokenizer.resolve(strict=True)))
    receipt = profile_counts(lambda text: tokenizer.encode(text, add_special_tokens=False).ids)
    receipt.update({"model": OFFICIAL_MODEL, "revision": OFFICIAL_REVISION,
                    "tokenizer_sha256": tokenizer_sha,
                    "tokenizers_version": tokenizers.__version__})
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
