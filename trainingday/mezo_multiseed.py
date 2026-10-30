"""CPU-only, per-seed MeZO-vs-AdamW controlled parameter-update study.

Intentionally tests an 8x16 toy frozen model with 34 LoRA parameters; this
is NOT a 4B inference, neural qualification, performance estimate, or
substitute for the official R18 GPU optimizer experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from .mezo_cpu_ablation import run_arm

SCHEMA="trainingday.toy_mezo_multiseed.v1"


def run_experiment(*, seeds: list[int], step_budgets: list[int]):
    if (not seeds or len(seeds)>16 or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 or s >= 2**31 for s in seeds)):
        raise ValueError("use 1..16 unique nonnegative bounded seeds")
    if (not step_budgets or len(step_budgets)>8
        or len(set(step_budgets))!=len(step_budgets)
        or any(type(n) is not int or not 1 <= n <= 200 for n in step_budgets)):
        raise ValueError("use 1..8 positive unique <=200 step budgets")
    records=[]
    for seed in seeds:
        zero={}
        for method in ("mezo", "adamw"):
            record=run_arm(method=method, mode="zero", steps=0, seed=seed)
            zero[method]=record
            records.append(record)
        if zero["mezo"]["initial_sha256"] != zero["adamw"]["initial_sha256"]:
            raise RuntimeError("frozen toy initial weights differ between optimizer methods")
        for steps in step_budgets:
            for method in ("mezo", "adamw"):
                for mode in ("correct", "corrupt"):
                    row=run_arm(method=method,mode=mode,steps=steps,seed=seed)
                    if (row["initial_sha256"]!=zero[method]["initial_sha256"] or
                        row["train_sha256"]!=zero[method]["train_sha256"] or
                        row["heldout_sha256"]!=zero[method]["heldout_sha256"] or
                        row["trainable_parameter_count"]!=34):
                        raise RuntimeError("unmatched initial seed, data, or adapter size")
                    records.append(row)
    report={"schema":SCHEMA,"status":"TOY_ONLY_CPU_MEZO_ADAMW_ABLATION",
            "seeds":seeds,"step_budgets":step_budgets,
            "arms":records,"neural_qualification":False,
            "official_4b_model_trained":False,
            "protected_evaluation_accessed":False,
            "gpu_training_started":False}
    return report


def summarize(report: dict):
    groups={}
    for row in report["arms"]:
        key=(row["method"],row["mode"],row["optimizer_steps"])
        groups.setdefault(key,[]).append(row)
    metrics=[]
    for key,rows in sorted(groups.items()):
        method,mode,steps=key
        metrics.append({
            "method":method,"mode":mode,"steps":steps,"seeds":len(rows),
            "mean_correct_of_32":round(statistics.mean(x["heldout_sign_correct"] for x in rows),3),
            "median_correct_of_32":statistics.median(x["heldout_sign_correct"] for x in rows),
            "mean_heldout_mse":round(statistics.mean(x["heldout_mse"] for x in rows),6),
            "mean_optimizer_seconds":round(statistics.mean(x["optimizer_seconds"] for x in rows),5),
            "total_optimizer_seconds":round(sum(x["optimizer_seconds"] for x in rows),5),
            "total_forward_evaluations":sum(x["forward_evaluations"] for x in rows),
            "real_parameter_updates":all(x["initial_sha256"]!=x["final_sha256"] for x in rows) if steps>0 else False
        })
    return {"schema":SCHEMA+".summary", "status":"TOY_METHOD_EVIDENCE_ONLY",
            "metrics":metrics, "vera_neural_qualification":False,
            "heldout_reasoning_generalization":False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",required=True, help="comma-separated integers")
    parser.add_argument("--step-budgets",required=True,help="comma-separated ints <=200")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--summary",type=Path,required=True)
    args=parser.parse_args()
    if args.output == args.summary or args.output.exists() or args.summary.exists():
        raise FileExistsError("exclusive new report/summary paths required")
    seeds=[int(s) for s in args.seeds.split(",")]
    steps=[int(s) for s in args.step_budgets.split(",")]
    result=run_experiment(seeds=seeds, step_budgets=steps)
    summary=summarize(result)
    raw=(json.dumps(result,sort_keys=True,indent=2)+"\n").encode()
    args.output.write_bytes(raw)
    summary["report_sha256"]=hashlib.sha256(raw).hexdigest()
    args.summary.write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"report_sha256":summary["report_sha256"],
                      "seeds":len(seeds),"arms":len(result["arms"]),
                      "summary":str(args.summary)}))


if __name__=="__main__":
    main()
