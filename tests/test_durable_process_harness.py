from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from successor.experiments.run_durable_process import DurableProcessHold, run_durable_process

ROOT = Path(__file__).resolve().parents[1]
R2_SPEC_REL = (
    "successor/experiments/"
    "V10R3R2_CONTINUOUS20_DURABLE_EXECUTION_SPEC_20261004_V1.json"
)
R1_SPEC_REL = (
    "successor/experiments/"
    "V10R3R1_CONTINUOUS20_EXECUTION_SPEC_20261004_V1.json"
)


def _lf_normalized_sha(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _r2_metadata() -> dict[str, str]:
    spec = json.loads((ROOT / R2_SPEC_REL).read_text(encoding="utf-8"))
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    return {
        "repo_head": head,
        "spec_path": R2_SPEC_REL,
        "spec_sha256": _lf_normalized_sha(ROOT / R2_SPEC_REL),
        "runtime_binding_sha256": spec["source_subject"]["runtime_binding_sha256"],
    }


def _child(code: str, spec_rel: str = R2_SPEC_REL) -> list[str]:
    return [
        sys.executable,
        "-c",
        code,
        "--spec",
        spec_rel,
    ]


def _run_args(
    log_dir: Path,
    code: str,
    *,
    metadata: dict[str, str] | None = None,
    spec_rel: str = R2_SPEC_REL,
    watch_seconds: str = "0.04",
) -> list[str]:
    return [
        "--log-dir",
        str(log_dir),
        "--watch-seconds",
        watch_seconds,
        "--expected-optimizer-steps",
        "20",
        "--metadata-json",
        json.dumps(_r2_metadata() if metadata is None else metadata),
        "--cwd",
        str(ROOT),
        "--",
        *_child(code, spec_rel),
    ]


def test_durable_process_persists_subject_step_logs_and_exit(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    code = (
        "import sys,time;"
        "print('  5%|x| 1/20', flush=True);"
        "print('ERR_MARK', file=sys.stderr, flush=True);"
        "time.sleep(0.08);"
        "print(' 10%|x| 2/20', flush=True);"
        "time.sleep(0.08);"
        "sys.exit(7)"
    )

    rc = run_durable_process(_run_args(log_dir, code))

    assert rc == 7
    stdout = (log_dir / "stdout.log").read_text(encoding="utf-8")
    assert "1/20" in stdout
    assert "2/20" in stdout
    assert "ERR_MARK" in (log_dir / "stderr.log").read_text(encoding="utf-8")

    launch = json.loads((log_dir / "launch.json").read_text(encoding="utf-8"))
    final = json.loads((log_dir / "final.json").read_text(encoding="utf-8"))
    watchdog = [
        json.loads(line)
        for line in (log_dir / "watchdog.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert launch["status"] == "STARTED"
    assert launch["pid"] > 0
    assert launch["metadata"] == _r2_metadata()
    assert launch["expected_optimizer_steps"] == 20
    assert final["status"] == "EXITED"
    assert final["exit_code"] == 7
    assert final["signal"] is None
    assert final["latest_optimizer_step"] == 2
    assert final["stdout_bytes"] > 0
    assert final["stderr_bytes"] > 0
    assert watchdog
    assert all(row["pid"] == launch["pid"] for row in watchdog)
    assert max(row["latest_optimizer_step"] or 0 for row in watchdog) == 2


def test_durable_process_reads_optimizer_progress_from_stderr(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    code = (
        "import sys,time;"
        "print(' 15%|x| 3/20', file=sys.stderr, flush=True);"
        "time.sleep(0.08);"
        "print(' 20%|x| 4/20', file=sys.stderr, flush=True);"
    )

    rc = run_durable_process(_run_args(log_dir, code))

    assert rc == 0
    final = json.loads((log_dir / "final.json").read_text(encoding="utf-8"))
    assert final["latest_optimizer_step"] == 4


def test_durable_process_refuses_existing_log_namespace(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    with pytest.raises(DurableProcessHold, match="log namespace already exists"):
        run_durable_process(_run_args(log_dir, "print('x')"))


def test_durable_process_rejects_invalid_metadata_json(tmp_path: Path) -> None:
    with pytest.raises(DurableProcessHold, match="metadata json"):
        run_durable_process(
            [
                "--log-dir",
                str(tmp_path / "logs"),
                "--metadata-json",
                "[]",
                "--cwd",
                str(ROOT),
                "--",
                *_child("print('x')"),
            ]
        )


def test_durable_process_requires_bound_launch_metadata(tmp_path: Path) -> None:
    incomplete = {
        "repo_head": _r2_metadata()["repo_head"],
        "spec_path": R2_SPEC_REL,
    }
    log_dir = tmp_path / "logs"

    with pytest.raises(DurableProcessHold, match="required metadata"):
        run_durable_process(
            _run_args(log_dir, "print('x')", metadata=incomplete)
        )
    assert not log_dir.exists()


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("repo_head", "f" * 40),
        ("spec_path", R1_SPEC_REL),
        ("spec_sha256", "e" * 64),
        ("runtime_binding_sha256", "d" * 64),
    ],
)
def test_durable_process_rejects_well_formed_wrong_subject_metadata(
    tmp_path: Path,
    field: str,
    wrong_value: str,
) -> None:
    metadata = _r2_metadata()
    metadata[field] = wrong_value
    log_dir = tmp_path / field

    with pytest.raises(DurableProcessHold, match=field):
        run_durable_process(
            _run_args(log_dir, "print('SHOULD_NOT_RUN')", metadata=metadata)
        )
    assert not log_dir.exists()


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("repo_head", "not-a-git-sha"),
        ("spec_sha256", "bad"),
        ("runtime_binding_sha256", "bad"),
    ],
)
def test_durable_process_rejects_malformed_subject_hashes(
    tmp_path: Path,
    field: str,
    wrong_value: str,
) -> None:
    metadata = _r2_metadata()
    metadata[field] = wrong_value
    log_dir = tmp_path / ("malformed-" + field)

    with pytest.raises(DurableProcessHold, match=field):
        run_durable_process(
            _run_args(log_dir, "print('SHOULD_NOT_RUN')", metadata=metadata)
        )
    assert not log_dir.exists()


def test_durable_process_rejects_child_spec_path_mismatch(tmp_path: Path) -> None:
    log_dir = tmp_path / "child-spec-mismatch"

    with pytest.raises(DurableProcessHold, match="spec_path"):
        run_durable_process(
            _run_args(
                log_dir,
                "print('SHOULD_NOT_RUN')",
                spec_rel=R1_SPEC_REL,
            )
        )
    assert not log_dir.exists()


def test_durable_process_rejects_non_r2_spec_even_when_metadata_matches(
    tmp_path: Path,
) -> None:
    r1_path = ROOT / R1_SPEC_REL
    r1_spec = json.loads(r1_path.read_text(encoding="utf-8"))
    metadata = {
        "repo_head": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "spec_path": R1_SPEC_REL,
        "spec_sha256": _lf_normalized_sha(r1_path),
        "runtime_binding_sha256": (
            r1_spec["source_subject"]["runtime_binding_sha256"]
        ),
    }
    log_dir = tmp_path / "matched-r1-subject"

    with pytest.raises(
        DurableProcessHold,
        match="protocol-frozen R2 subject",
    ):
        run_durable_process(
            _run_args(
                log_dir,
                "print('SHOULD_NOT_RUN')",
                metadata=metadata,
                spec_rel=R1_SPEC_REL,
            )
        )
    assert not log_dir.exists()
