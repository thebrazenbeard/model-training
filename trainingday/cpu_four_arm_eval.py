"""CPU-only four-arm development evaluation for pinned TRAININGDAY 4B adapters.

Does not run optimizer steps. Must execute in a NEW process, separate from
the training run. Never import this module in the trainer: it opens eval rows.
Raw predictions stay in the isolated local output, never in GitHub/Bus.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from trainingday.train_neural import _required_base
from trainingday.structural_eval import (
    EvaluationGateError, select_structural_cases, score_answers,
    compare_control_scores,
)

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def attest(arms, manifest_sha, binding_sha):
    receipts = {}
    for name in ("zero", "correct", "corrupt"):
        root = Path(arms[name])
        p = root / "training_receipt.json"
        if not p.is_file() or p.is_symlink():
            raise RuntimeError("adapter training receipt absent")
        r = json.loads(p.read_text(encoding="utf-8"))
        if (r.get("status") != "CPU_BF16_LORA_SAVED_NOT_INDEPENDENTLY_RELOADED"
                or r.get("mode") != name or r["corpus"]["manifest_sha256"] != manifest_sha
                or r.get("base_binding_sha256") != binding_sha
                or r.get("new_optimizer_steps") != (0 if name == "zero" else 1)):
            raise RuntimeError("unmatched CPU LoRA training receipt")
        for file, field in (("adapter_model.safetensors", "adapter_model_sha256"),
                            ("adapter_config.json", "adapter_config_sha256")):
            target = root / "adapter" / file
            if not target.is_file() or target.is_symlink() or sha(target) != r[field]:
                raise RuntimeError("adapter checksum mismatch")
        receipts[name] = r
    digests = {r["trainable_digest_before"] for r in receipts.values()}
    selected = {r["corpus"]["selected_id_sha256"] for r in receipts.values()}
    seeds = {r["seed"] for r in receipts.values()}
    revisions = {r["base_revision"] for r in receipts.values()}
    if len(digests) != 1 or len(selected) != 1 or len(seeds) != 1 or len(revisions) != 1:
        raise RuntimeError("control arms not matched")
    before = receipts["zero"]["trainable_digest_before"]
    if (receipts["zero"]["trainable_digest_after"] != before
            or receipts["correct"]["trainable_digest_after"] == before
            or receipts["corrupt"]["trainable_digest_after"] == before):
        raise RuntimeError("invalid zero/correct/corrupt tensor digests")
    return receipts

def batch_predict(model, tokenizer, cases, *, batch_size, max_new_tokens):
    import torch
    outputs = {}
    for index in range(0, len(cases), batch_size):
        group = cases[index:index+batch_size]
        messages = [[{"role": "user", "content": case["prompt"]}] for case in group]
        prompts = [
            tokenizer.apply_chat_template(msg, tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=False)
            for msg in messages
        ]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt",
                           add_special_tokens=False)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prefix_length = inputs["input_ids"].shape[1]
        for case, sequence in zip(group, generated):
            outputs[case["id"]] = tokenizer.decode(
                sequence[prefix_length:], skip_special_tokens=True
            ).strip()
        print("GENERATED", index + len(group), "OF", len(cases), flush=True)
    return outputs

def evaluate(args):
    import psutil
    if psutil.virtual_memory().available < 13 * (2**30):
        raise RuntimeError("CPU evaluator system RAM below 13 GiB")
    base, model_path = _required_base(args.base_binding)
    actual_manifest = sha(args.manifest)
    if actual_manifest != args.manifest_sha256:
        raise RuntimeError("manifest SHA mismatch")
    arms = {"zero": args.zero, "correct": args.correct, "corrupt": args.corrupt}
    receipts = attest(arms, actual_manifest, sha(args.base_binding))
    if next(iter(receipts.values()))["base_revision"] != base["revision"]:
        raise RuntimeError("model revision mismatch")
    selection = select_structural_cases(
        args.manifest, expected_manifest_sha256=args.manifest_sha256,
        max_cases=args.cases,
    )
    output = args.output
    if output.exists() or output.is_symlink():
        raise RuntimeError("evaluation output exists")
    output.mkdir(parents=True, exist_ok=False)

    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    from peft import PeftModel
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    frozen = Qwen3_5ForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16,
        device_map="cpu", low_cpu_mem_usage=True, attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(
        frozen, args.zero / "adapter", adapter_name="zero", is_trainable=False,
    )
    model.load_adapter(args.correct / "adapter", adapter_name="correct", is_trainable=False)
    model.load_adapter(args.corrupt / "adapter", adapter_name="corrupt", is_trainable=False)
    model.eval()
    scores = {}
    for arm in ("frozen", "zero", "correct", "corrupt"):
        print("EVALUATING_ARM", arm, flush=True)
        if arm == "frozen":
            with model.disable_adapter():
                answers = batch_predict(
                    model, tokenizer, selection["cases"],
                    batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                )
        else:
            model.set_adapter(arm)
            answers = batch_predict(
                model, tokenizer, selection["cases"],
                batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
            )
        scores[arm] = score_answers(selection, answers, mode=arm)
        predictions = output / f"{arm}.predictions.private.json"
        predictions.write_text(json.dumps(answers, sort_keys=True) + "\n", encoding="utf-8")
        scores[arm]["prediction_sha256"] = sha(predictions)
        print("SCORED_ARM", arm, scores[arm]["correct"], "/", scores[arm]["total"], flush=True)
    try:
        result = compare_control_scores(scores, min_gain_cases=1)
        verdict = "DEV_CANDIDATE_ADVANTAGE_NOT_BLIND_GENERALIZATION"
    except EvaluationGateError:
        result = None
        verdict = "NO_VERIFIED_CORRECT_LABEL_ADVANTAGE"
    report = {
        "status": verdict,
        "base_revision": base["revision"],
        "manifest_sha256": actual_manifest,
        "selected_eval_id_sha256": selection["selected_id_sha256"],
        "selected_eval_case_count": selection["selected_count"],
        "adapter_sha256": {arm: receipts[arm]["adapter_model_sha256"]
                           for arm in receipts},
        "mode_scores": scores,
        "comparison": result,
        "real_model_loaded_in_separate_process": True,
        "real_adapter_loaded": True,
        "independent_final_bank_qualified": False,
        "model_promoted": False,
    }
    receipt = output / "evaluation_receipt.json"
    receipt.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("FINAL_EVAL_RESULT", json.dumps({
        "status": report["status"],
        "scores": {arm: [scores[arm]["correct"], scores[arm]["total"]] for arm in scores},
        "receipt_sha256": sha(receipt),
        "selection_sha256": report["selected_eval_id_sha256"],
    }, sort_keys=True), flush=True)
    return report

def main():
    p = argparse.ArgumentParser()
    for attr in ("base-binding", "manifest", "zero", "correct", "corrupt", "output"):
        p.add_argument("--" + attr, required=True, type=Path)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--cases", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=12)
    args = p.parse_args()
    if not 1 <= args.batch_size <= 4 or not 1 <= args.max_new_tokens <= 24:
        raise ValueError("unsafe generation batch/token bound")
    evaluate(args)

if __name__ == "__main__":
    main()
