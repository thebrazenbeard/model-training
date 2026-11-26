from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

FAMILIES = [
    "identity_stability",
    "independent_judgment",
    "epistemic_provenance",
    "correction_uptake",
    "relationship_authority",
    "reciprocal_identity_continuity",
    "empathy_affective_response",
    "privacy_boundary",
    "runtime_boundary",
    "negative_transfer_resistance",
]


def read_rows(root: Path) -> list[dict]:
    rows = []
    for family in FAMILIES:
        rows.extend(
            json.loads(line)
            for line in (root / f"{family}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def sample_rows(rows: list[dict], per_family: int = 60) -> list[dict]:
    sampled = []
    for family_index, family in enumerate(FAMILIES):
        family_rows = [row for row in rows if row["family"] == family]
        for index in range(min(per_family, len(family_rows))):
            sampled.append(family_rows[(index * 17 + family_index * 7) % len(family_rows)])
    return sampled


def embed_metrics(model: SentenceTransformer, rows: list[dict]) -> dict:
    texts = [row["response"] for row in rows]
    prompts = [row["prompt"] for row in rows]
    intent_prompts = [
        row["prompt"].replace(row.get("scenario_case", ""), "").strip()
        for row in rows
    ]
    intent_responses = [
        row["response"].replace(row.get("scenario_case", ""), "").strip()
        for row in rows
    ]
    response_embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    prompt_embeddings = model.encode(
        prompts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    intent_prompt_embeddings = model.encode(
        intent_prompts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    intent_response_embeddings = model.encode(
        intent_responses,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    intent_alignment = np.sum(
        intent_prompt_embeddings * intent_response_embeddings,
        axis=1,
    )
    intent_permutation = np.array([(index * 7919 + 37) % len(rows) for index in range(len(rows))])
    intent_mismatched_alignment = np.sum(
        intent_prompt_embeddings[intent_permutation] * intent_response_embeddings,
        axis=1,
    )
    intent_alignment_margin = intent_alignment - intent_mismatched_alignment

    alignment = np.sum(prompt_embeddings * response_embeddings, axis=1)
    permutation = np.array([(index * 7919 + 37) % len(rows) for index in range(len(rows))])
    mismatched_alignment = np.sum(
        prompt_embeddings[permutation] * response_embeddings,
        axis=1,
    )
    alignment_margin = alignment - mismatched_alignment
    family_metrics = {}
    for family in FAMILIES:
        indices = np.array([index for index, row in enumerate(rows) if row["family"] == family])
        family_alignment = alignment[indices]
        family_mismatch = mismatched_alignment[indices]
        family_margin = alignment_margin[indices]
        family_intent_alignment = intent_alignment[indices]
        family_intent_mismatch = intent_mismatched_alignment[indices]
        family_intent_margin = intent_alignment_margin[indices]
        family_metrics[family] = {
            "sample_rows": int(len(indices)),
            "mean_prompt_response_cosine": round(float(np.mean(family_alignment)), 6),
            "mean_mismatched_prompt_response_cosine": round(float(np.mean(family_mismatch)), 6),
            "mean_alignment_margin": round(float(np.mean(family_margin)), 6),
            "prompt_response_beats_mismatch_fraction": round(float(np.mean(family_alignment > family_mismatch)), 6),
            "scenario_stripped_alignment_margin": round(float(np.mean(family_intent_margin)), 6),
            "scenario_stripped_beats_mismatch_fraction": round(float(np.mean(family_intent_alignment > family_intent_mismatch)), 6),
        }

    embeddings = response_embeddings
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -1.0)
    nearest = similarity.max(axis=1)

    upper = similarity[np.triu_indices_from(similarity, k=1)]
    high_similarity_fraction = float(np.mean(upper >= 0.90)) if len(upper) else 0.0

    return {
        "sample_rows": len(rows),
        "mean_prompt_response_cosine": round(float(np.mean(alignment)), 6),
        "median_prompt_response_cosine": round(float(np.median(alignment)), 6),
        "p10_prompt_response_cosine": round(float(np.quantile(alignment, 0.10)), 6),
        "prompt_response_below_0_25_fraction": round(float(np.mean(alignment < 0.25)), 6),
        "mean_mismatched_prompt_response_cosine": round(float(np.mean(mismatched_alignment)), 6),
        "mean_prompt_response_alignment_margin": round(float(np.mean(alignment_margin)), 6),
        "prompt_response_beats_mismatch_fraction": round(float(np.mean(alignment > mismatched_alignment)), 6),
        "scenario_stripped_prompt_response_cosine": round(float(np.mean(intent_alignment)), 6),
        "scenario_stripped_mismatched_cosine": round(float(np.mean(intent_mismatched_alignment)), 6),
        "scenario_stripped_alignment_margin": round(float(np.mean(intent_alignment_margin)), 6),
        "scenario_stripped_beats_mismatch_fraction": round(float(np.mean(intent_alignment > intent_mismatched_alignment)), 6),
        "family_alignment": family_metrics,
        "mean_nearest_neighbor_cosine": round(float(np.mean(nearest)), 6),
        "median_nearest_neighbor_cosine": round(float(np.median(nearest)), 6),
        "p90_nearest_neighbor_cosine": round(float(np.quantile(nearest, 0.90)), 6),
        "mean_pairwise_cosine": round(float(np.mean(upper)), 6) if len(upper) else 0.0,
        "high_similarity_pair_fraction_ge_0_90": round(high_similarity_fraction, 6),
    }


def evaluate_quality_gate(baseline: dict, candidate: dict) -> dict:
    family_deltas = {
        family: candidate["family_alignment"][family]["mean_alignment_margin"]
        - baseline["family_alignment"][family]["mean_alignment_margin"]
        for family in FAMILIES
    }
    scenario_stripped_family_deltas = {
        family: candidate["family_alignment"][family]["scenario_stripped_alignment_margin"]
        - baseline["family_alignment"][family]["scenario_stripped_alignment_margin"]
        for family in FAMILIES
    }
    checks = {
        "global_alignment_margin_gain_ge_0_05": candidate["mean_prompt_response_alignment_margin"]
        - baseline["mean_prompt_response_alignment_margin"] >= 0.05,
        "global_prompt_response_beats_mismatch_ge_0_85": candidate["prompt_response_beats_mismatch_fraction"] >= 0.85,
        "global_scenario_stripped_alignment_margin_gain_ge_0_02": candidate["scenario_stripped_alignment_margin"]
        - baseline["scenario_stripped_alignment_margin"] >= 0.02,
        "global_scenario_stripped_beats_mismatch_ge_0_75": candidate["scenario_stripped_beats_mismatch_fraction"] >= 0.75,
        "global_nearest_neighbor_cosine_improves_vs_baseline_by_0_01": candidate["mean_nearest_neighbor_cosine"]
        <= baseline["mean_nearest_neighbor_cosine"] - 0.01,
        "high_similarity_pair_fraction_le_0_0005": candidate["high_similarity_pair_fraction_ge_0_90"] <= 0.0005,
        "low_prompt_response_fraction_le_0_05": candidate["prompt_response_below_0_25_fraction"] <= 0.05,
        "every_family_alignment_margin_gain_ge_0_03": min(family_deltas.values()) >= 0.03,
        "every_family_scenario_stripped_margin_not_regressed_gt_0_01": min(scenario_stripped_family_deltas.values()) >= -0.01,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "family_alignment_margin_delta": {
            family: round(delta, 6)
            for family, delta in sorted(family_deltas.items())
        },
        "family_scenario_stripped_alignment_margin_delta": {
            family: round(delta, 6)
            for family, delta in sorted(scenario_stripped_family_deltas.items())
        },
        "interpretation": {
            "project_regression_gate_not_universal_quality_score": True,
            "thresholds_are_guardrails_against_known_v4_failure_modes": True,
            "passing_does_not_prove_training_benefit": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4", type=Path, required=True)
    parser.add_argument("--v41", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=None,
        help="Optional committed training manifest to bind into the semantic receipt.",
    )
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    model = SentenceTransformer(args.model)
    baseline_rows = sample_rows(read_rows(args.v4))
    candidate_rows = sample_rows(read_rows(args.v41))

    candidate_manifest = json.loads(
        (args.v41 / "manifest.json").read_text(encoding="utf-8")
    )
    training_manifest_path = args.training_manifest or (args.v41.parent / "training_manifest.json")
    training_manifest = (
        json.loads(training_manifest_path.read_text(encoding="utf-8"))
        if training_manifest_path.exists()
        else {}
    )
    training_lineage_verified = (
        training_manifest.get("custom_source", {}).get("manifest_digest")
        == candidate_manifest["manifest_digest"]
        and training_manifest.get("custom_source", {}).get("corpus_id")
        == candidate_manifest["corpus_id"]
    )
    result = {
        "schema": "VERA_V4_V4_1_SEMANTIC_DIVERSITY_COMPARISON_V4",
        "embedding_model": args.model,
        "provenance": {
            "git_head": os.environ.get("GITHUB_SHA"),
            "candidate_corpus_id": candidate_manifest["corpus_id"],
            "candidate_manifest_digest": candidate_manifest["manifest_digest"],
            "training_corpus_id": training_manifest.get("corpus_id"),
            "training_manifest_path": str(training_manifest_path) if training_manifest else None,
            "training_custom_manifest_digest": training_manifest.get("custom_source", {}).get("manifest_digest"),
            "training_lineage_verified": training_lineage_verified,
        },
        "baseline": embed_metrics(model, baseline_rows),
        "candidate": embed_metrics(model, candidate_rows),
        "interpretation": {
            "lower_nearest_neighbor_cosine_means_less_semantic_clustering_in_this_sample": True,
            "embedding_similarity_is_a_model_based_proxy_not_ground_truth": True,
            "same_model_and_sampling_scheme_used_for_both": True,
            "no_model_weights_changed": True,
        },
    }
    result["delta"] = {
        key: round(
            result["candidate"][key] - result["baseline"][key],
            6,
        )
        for key in result["candidate"]
        if isinstance(result["candidate"][key], (int, float))
        and isinstance(result["baseline"][key], (int, float))
    }
    result["quality_gate"] = evaluate_quality_gate(
        result["baseline"],
        result["candidate"],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["quality_gate"]["passed"]:
        raise SystemExit("V4.1 semantic quality gate FAILED")


if __name__ == "__main__":
    main()
