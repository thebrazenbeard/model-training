from __future__ import annotations

import argparse
import copy
import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .vera_lab.runner import run_scenario
from .vera_lab.scenario import LabScenario, canonical_json, sha256_json


class FixtureRuntime:
    def __init__(self) -> None:
        self._state: dict[str, Any] = {}

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self._state)

    def apply(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        if kind in {"initialize", "replace_state"}:
            state = event.get("state")
            if not isinstance(state, dict):
                raise ValueError(f"{kind} requires object state")
            self._state = copy.deepcopy(state)
            return
        raise ValueError(f"unsupported runtime event: {kind!r}")


def load_scenarios(path: str | Path) -> list[LabScenario]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}") from exc
        rows.append(LabScenario.from_dict(payload))
    if not rows:
        raise ValueError("scenario file is empty")
    return rows


def scenario_set_digest(scenarios: list[LabScenario]) -> str:
    return sha256_json([scenario.to_dict() for scenario in scenarios])


def load_adapter(entrypoint: str, candidate_id: str, candidate_digest: str):
    if ":" not in entrypoint:
        raise ValueError("adapter entry point must be module:factory")
    module_name, factory_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise ValueError("adapter factory is not callable")
    model = factory(candidate_id=candidate_id, candidate_digest=candidate_digest)
    if getattr(model, "candidate_digest", None) != candidate_digest:
        raise ValueError("adapter candidate_digest does not match requested digest")
    return model


def build_run_manifest(
    scenarios: list[LabScenario],
    candidate_id: str,
    candidate_digest: str,
    adapter_entrypoint: str,
) -> dict[str, Any]:
    model = load_adapter(adapter_entrypoint, candidate_id, candidate_digest)
    transcripts = []
    for scenario in scenarios:
        transcript = run_scenario(scenario, model, FixtureRuntime())
        transcripts.append(transcript.to_dict())
    return {
        "manifest_version": 1,
        "candidate_id": candidate_id,
        "candidate_digest": candidate_digest,
        "scenario_set_digest": scenario_set_digest(scenarios),
        "scenario_count": len(scenarios),
        "transcripts": transcripts,
    }


def _write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _read_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must be an object")
    return payload


def _cmd_validate(args) -> int:
    scenarios = load_scenarios(args.scenarios)
    print(canonical_json({
        "scenario_count": len(scenarios),
        "scenario_set_digest": scenario_set_digest(scenarios),
    }))
    return 0


def _cmd_run(args) -> int:
    if not args.candidate_id or not args.candidate_digest:
        raise ValueError("candidate id and digest must be nonempty")
    scenarios = load_scenarios(args.scenarios)
    payload = build_run_manifest(
        scenarios,
        args.candidate_id,
        args.candidate_digest,
        args.adapter,
    )
    _write_manifest(args.output, payload)
    print(canonical_json({
        "output": str(Path(args.output)),
        "scenario_count": payload["scenario_count"],
        "scenario_set_digest": payload["scenario_set_digest"],
    }))
    return 0


def _cmd_replay(args) -> int:
    expected = _read_manifest(args.manifest)
    scenarios = load_scenarios(args.scenarios)
    actual = build_run_manifest(
        scenarios,
        expected.get("candidate_id", ""),
        expected.get("candidate_digest", ""),
        args.adapter,
    )
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError("replay mismatch")
    print(canonical_json({"match": True, "manifest": str(Path(args.manifest))}))
    return 0


def _cmd_summarize(args) -> int:
    payload = _read_manifest(args.manifest)
    transcripts = payload.get("transcripts")
    if not isinstance(transcripts, list):
        raise ValueError("manifest transcripts must be a list")
    statuses = Counter()
    for transcript in transcripts:
        if not isinstance(transcript, dict):
            raise ValueError("transcript entry must be an object")
        status = transcript.get("completion_status")
        if not isinstance(status, str) or not status:
            raise ValueError("transcript completion_status must be nonempty")
        statuses[status] += 1
    print(canonical_json({
        "candidate_id": payload.get("candidate_id"),
        "candidate_digest": payload.get("candidate_digest"),
        "scenario_count": len(transcripts),
        "completion_statuses": dict(sorted(statuses.items())),
    }))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vera-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("scenarios")
    validate.set_defaults(handler=_cmd_validate)

    run = subparsers.add_parser("run")
    run.add_argument("scenarios")
    run.add_argument("--candidate-id", required=True)
    run.add_argument("--candidate-digest", required=True)
    run.add_argument("--adapter", required=True)
    run.add_argument("--output", required=True)
    run.set_defaults(handler=_cmd_run)

    replay = subparsers.add_parser("replay")
    replay.add_argument("manifest")
    replay.add_argument("--scenarios", required=True)
    replay.add_argument("--adapter", required=True)
    replay.set_defaults(handler=_cmd_replay)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("manifest")
    summarize.set_defaults(handler=_cmd_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
