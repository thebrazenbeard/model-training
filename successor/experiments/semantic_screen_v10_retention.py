from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from successor.experiments.v10_qwen35_bank import prompt_fingerprint


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def reconstruct_prompt_map(paths: list[Path]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in paths:
        for row in _rows(path):
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            digest = prompt_fingerprint(prompt)
            mapping.setdefault(digest, prompt)
    return mapping


def verify_registry(mapping: dict[str, str], registry_path: Path) -> dict:
    expected = {
        line.strip()
        for line in registry_path.read_text(encoding="ascii").splitlines()
        if line.strip()
    }
    actual = set(mapping)
    if actual != expected:
        missing = len(expected - actual)
        extra = len(actual - expected)
        raise RuntimeError(
            f"registry mismatch: expected={len(expected)} actual={len(actual)} "
            f"missing={missing} extra={extra}"
        )
    raw = registry_path.read_bytes()
    return {
        "count": len(actual),
        "registry_sha256": hashlib.sha256(raw).hexdigest(),
    }


def max_cosine_scores(
    target_embeddings: np.ndarray,
    source_embeddings: np.ndarray,
    *,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.empty(len(target_embeddings), dtype=np.float32)
    indices = np.empty(len(target_embeddings), dtype=np.int64)
    for start in range(0, len(target_embeddings), batch_size):
        stop = min(start + batch_size, len(target_embeddings))
        sims = target_embeddings[start:stop] @ source_embeddings.T
        local_indices = np.argmax(sims, axis=1)
        scores[start:stop] = sims[np.arange(stop - start), local_indices]
        indices[start:stop] = local_indices
    return scores, indices


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_target_sha256(path: Path, expected: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"target hash mismatch: {actual} != {expected}")
    return actual


def run_screen(
    *,
    target_path: Path,
    source_paths: list[Path],
    registry_path: Path,
    model_dir: Path,
    model_archive: Path,
    expected_model_archive_sha256: str,
    expected_target_sha256: str,
    threshold: float,
    output_path: Path,
) -> dict:
    target_sha = verify_target_sha256(target_path, expected_target_sha256)
    model_sha = _sha256(model_archive)
    if model_sha != expected_model_archive_sha256:
        raise RuntimeError(
            f"model archive hash mismatch: {model_sha} != {expected_model_archive_sha256}"
        )
    source_map = reconstruct_prompt_map(source_paths)
    registry = verify_registry(source_map, registry_path)
    target_rows = _rows(target_path)
    target = [
        (row.get("case_id") or row.get("record_id") or str(i), row["prompt"])
        for i, row in enumerate(target_rows)
    ]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_dir), device="cpu")
    source_hashes = sorted(source_map)
    source_prompts = [source_map[digest] for digest in source_hashes]
    source_embeddings = model.encode(
        source_prompts,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    target_embeddings = model.encode(
        [prompt for _, prompt in target],
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    scores, indices = max_cosine_scores(
        np.asarray(target_embeddings),
        np.asarray(source_embeddings),
        batch_size=16,
    )
    failures = []
    for i, score in enumerate(scores):
        if float(score) >= threshold:
            failures.append({
                "case_id": target[i][0],
                "cosine": float(score),
                "source_prompt_sha256": source_hashes[int(indices[i])],
            })
    receipt = {
        "schema": "V10_QWEN35_RETENTION_SEMANTIC_SCREEN_V1",
        "status": "PASS" if not failures else "HOLD",
        "threshold": threshold,
        "source_prompt_count": len(source_hashes),
        "target_prompt_count": len(target),
        "target_sha256": target_sha,
        "target_expected_sha256": expected_target_sha256,
        "registry": registry,
        "model_archive_sha256": model_sha,
        "model_archive_expected_sha256": expected_model_archive_sha256,
        "embedding_dimension": int(np.asarray(source_embeddings).shape[1]),
        "max_cosine": float(np.max(scores)) if len(scores) else 0.0,
        "p95_max_cosine": float(np.percentile(scores, 95)) if len(scores) else 0.0,
        "mean_max_cosine": float(np.mean(scores)) if len(scores) else 0.0,
        "failure_count": len(failures),
        "failures": failures,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("sentence-transformers", "scipy", "scikit-learn")
        },
        "claim_ceiling": (
            "RETENTION_ONLY_HEURISTIC_SEMANTIC_SCREEN / "
            "NOT_ZERO_CONTAMINATION_PROOF / FULL_BANK_SCREEN_PENDING"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-archive", type=Path, required=True)
    parser.add_argument("--model-archive-sha256", required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = run_screen(
            target_path=args.target,
            source_paths=args.source,
            registry_path=args.registry,
            model_dir=args.model_dir,
            model_archive=args.model_archive,
            expected_model_archive_sha256=args.model_archive_sha256,
            expected_target_sha256=args.target_sha256,
            threshold=args.threshold,
            output_path=args.output,
        )
    except Exception as exc:
        print(json.dumps({"status": "HOLD", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
