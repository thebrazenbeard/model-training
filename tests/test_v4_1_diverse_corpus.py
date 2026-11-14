from __future__ import annotations

import json
from pathlib import Path

from successor import build_v4_1_diverse_corpus as generator

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "successor" / "corpus" / "v4_1" / "custom"


def test_v41_generator_contract() -> None:
    assert generator.EXPECTED_ROWS == 10_000
    assert generator.ROWS_PER_FAMILY == 1_000
    assert len(generator.DOMAINS) == 100
    assert len(generator.COGNITIVE_LEVELS) == 5
    assert len(generator.REQUEST_FORMS) == 10
    assert len(generator.SURFACE_PREFIXES) == 8
    assert set(generator.COGNITIVE_RESPONSE_FORMS) == {name for name, _ in generator.COGNITIVE_LEVELS}
    assert len(generator.FRAMES) == 12
    assert len(generator.SCENARIO_RESOLUTIONS) == 10
    assert len(generator.FAMILY_BRIDGES) == 10
    assert all(len(forms) == 5 for forms in generator.FAMILY_BRIDGES.values())
    assert len(generator.FAMILY_SCENARIO_SUFFIXES) == 10
    assert all(len(forms) == 5 for forms in generator.FAMILY_SCENARIO_SUFFIXES.values())
    assert all(len(forms) == 5 for forms in generator.SCENARIO_RESOLUTIONS.values())
    assert all(
        "{domain_case}" in form
        for forms in generator.SCENARIO_RESOLUTIONS.values()
        for form in forms
    )
    assert set(generator.STYLE_ARTICLES) == set(generator.STYLE_MODES)
    negative_cores = generator.v4.CONFIG["specs"]["negative_transfer_resistance"]["cores"]
    assert len(negative_cores) == 5
    assert all("408" not in core for core in negative_cores)
    assert all("arithmetic" not in core.lower() for core in negative_cores)



def test_v41_committed_shards_match_generator() -> None:
    total = 0
    all_responses: set[str] = set()
    all_pairs: set[tuple[str, str]] = set()

    for family in generator.v4.CORE_FAMILIES:
        expected = generator.rows_for(family)
        committed = [
            json.loads(line)
            for line in (CORPUS_DIR / f"{family}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert committed == expected, family
        assert len(committed) == 1_000
        assert len({row["prompt"] for row in committed}) == 1_000
        assert len({row["response"] for row in committed}) == 1_000
        assert len({row["domain"] for row in committed}) == 20
        assert len({row["scenario_case"] for row in committed}) == 100
        assert len({row["cognitive_level"] for row in committed}) == 5
        assert all("Please what" not in row["prompt"] for row in committed)
        assert all("a analytical" not in row["prompt"] for row in committed)
        assert all("a operational" not in row["prompt"] for row in committed)
        assert all("a audit" not in row["prompt"] for row in committed)
        assert all("do not proposition" not in row["response"].lower() for row in committed)
        assert all("do not assertion" not in row["response"].lower() for row in committed)
        assert all(row["behavioral_trigger"] for row in committed)
        assert all(row["scenario_resolution"] for row in committed)
        assert all(row["cognitive_level"] in row["response"] for row in committed)

        assert all(row["scenario_case"] in row["scenario_resolution"] for row in committed)

        total += len(committed)
        all_responses.update(row["response"] for row in committed)
        all_pairs.update((row["prompt"], row["response"]) for row in committed)

    assert total == 10_000
    assert len(all_responses) == 10_000
    assert len(all_pairs) == 10_000


def test_v41_manifest_matches_files() -> None:
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == 10_000
    assert manifest["pair_uniqueness"] == 10_000
    assert manifest["response_uniqueness"] == 10_000
    assert len(manifest["families"]) == 10
    for family, meta in manifest["families"].items():
        assert meta["rows"] == 1_000, family
        assert meta["unique_prompts"] == 1_000, family
        assert meta["unique_responses"] == 1_000, family
        assert meta["domains"] == 20, family
        assert meta["scenario_cases"] == 100, family
        assert meta["cognitive_levels"] == 5, family
