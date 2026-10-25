import pytest

import bootcamp
import fast_bootcamp
import quality_gate


def test_select_source_segments_is_bounded_and_samples_full_document():
    text = "A" * 5000 + "MIDDLE_MARKER" + "B" * 5000 + "END_MARKER"
    packed = bootcamp.select_source_segments(text, 3000)
    assert len(packed) <= 3100
    assert packed.startswith("A")
    assert "MIDDLE_MARKER" in packed
    assert "END_MARKER" in packed


def test_select_source_segments_returns_short_source_unchanged():
    text = "short source"
    assert bootcamp.select_source_segments(text, 3000) == text


def test_validate_curriculum_accepts_exact_round_count_and_known_capabilities():
    caps = [
        bootcamp.Capability("rls", "RLS", "Row level security", True),
        bootcamp.Capability("queues", "Queues", "Durable queues", False),
    ]
    raw = {
        "rounds": [
            {
                "task": f"task {i}",
                "capability_ids": ["rls" if i % 2 else "queues"],
                "hidden_rubric": ["one check"],
                "adversarial": i % 3 == 0,
                "critical": i % 2 == 1,
            }
            for i in range(1, 7)
        ]
    }
    out = bootcamp.validate_curriculum(raw, 6, caps)
    assert len(out) == 6
    assert out[0]["task"] == "task 1"


def test_validate_curriculum_rejects_unknown_capability():
    caps = [bootcamp.Capability("rls", "RLS", "Row level security", True)]
    raw = {
        "rounds": [
            {
                "task": f"task {i}",
                "capability_ids": ["not_real"],
                "hidden_rubric": ["check"],
                "adversarial": False,
                "critical": False,
            }
            for i in range(6)
        ]
    }
    try:
        bootcamp.validate_curriculum(raw, 6, caps)
    except ValueError as exc:
        assert "unknown capability" in str(exc)
    else:
        raise AssertionError("unknown capability should be rejected")


def test_safe_source_url_rejects_loopback_and_non_https():
    assert not bootcamp.safe_source_url("http://example.com")
    assert not bootcamp.safe_source_url("https://127.0.0.1/private")
    assert not bootcamp.safe_source_url("https://localhost/private")
    assert bootcamp.safe_source_url("https://supabase.com/docs")


def test_fetch_source_rejects_public_hostname_resolving_to_private_ip(monkeypatch):
    """A public-looking hostname must not bypass the network destination gate."""
    from ingest.adapters import http

    monkeypatch.setattr(http, "_resolve_addresses", lambda host: ("127.0.0.1",))

    monkeypatch.setattr(http._PinnedOpener, "open", lambda *a, **kw: pytest.fail("network request attempted"))
    with pytest.raises(http.PolicyRejected, match="private/loopback/link-local"):
        bootcamp.fetch_source("https://public.example/training")


def _stub_source_transport(monkeypatch, handler):
    from ingest.adapters import http

    monkeypatch.setattr(http, "_resolve_addresses", lambda host: ("93.184.216.34",))

    class Opener:
        def __init__(self):
            self.seen = []

        def open(self, request, timeout):
            self.seen.append((request.full_url, timeout))
            return handler(request)

    opener = Opener()
    monkeypatch.setattr(bootcamp, "HttpAdapter", lambda: http.HttpAdapter(opener=opener))
    return opener


def test_fetch_source_does_not_follow_unvalidated_redirect(monkeypatch):
    from urllib.error import HTTPError
    from ingest.adapters.base import PolicyRejected

    def redirect(request):
        raise HTTPError(request.full_url, 302, "Moved", {"Location": "https://127.0.0.1/private"}, None)

    opener = _stub_source_transport(monkeypatch, redirect)
    with pytest.raises(PolicyRejected, match="redirect"):
        bootcamp.fetch_source("https://public.example/source")
    assert opener.seen == [("https://public.example/source", 30.0)]


def test_fetch_source_rejects_oversized_body_instead_of_sampling_it(monkeypatch):
    from ingest.adapters.base import PolicyRejected

    class Response:
        status = 200
        headers = {"Content-Type": "text/plain"}

        def geturl(self):
            return "https://public.example/source"

        def read(self, size):
            return b"x" * size

        def close(self):
            pass

    _stub_source_transport(monkeypatch, lambda request: Response())
    with pytest.raises(PolicyRejected, match="exceeds"):
        bootcamp.fetch_source("https://public.example/source")


def test_fetch_source_rejects_literal_loopback_before_request(monkeypatch):
    monkeypatch.setattr(bootcamp, "HttpAdapter", lambda: pytest.fail("network request attempted"))
    with pytest.raises(ValueError, match="unsafe source URL"):
        bootcamp.fetch_source("https://127.0.0.1/private")


def test_fetch_source_reads_bounded_nonredirect_text(monkeypatch):
    class Response:
        status = 200
        headers = {"Content-Type": "text/plain; charset=utf-8"}
        closed = False

        def geturl(self):
            return "https://public.example/source"

        def read(self, size):
            return b"source"

        def close(self):
            self.closed = True

    response = Response()
    _stub_source_transport(monkeypatch, lambda request: response)
    assert bootcamp.fetch_source("https://public.example/source") == "source"
    assert response.closed


def test_transfer_exams_are_generated_independently(monkeypatch):
    caps = [
        bootcamp.Capability("rls", "RLS", "Row level security", True),
        bootcamp.Capability("queues", "Queues", "Durable queues", False),
    ]
    calls = []

    def fake_json_llm(system, user, max_tokens, temperature=0.1):
        calls.append(user)
        n = len(calls)
        return {
            "rounds": [{
                "task": f"novel transfer {n}",
                "capability_ids": ["rls", "queues"],
                "hidden_rubric": ["check both capabilities"],
                "adversarial": n == 2,
                "critical": True,
            }]
        }

    monkeypatch.setattr(bootcamp, "json_llm", fake_json_llm)
    tasks = fast_bootcamp.robust_build_transfer_tasks(
        {"function": "Supabase Platform Specialist"}, caps, [{"task": "old task"}]
    )
    assert len(tasks) == 2
    assert len(calls) == 2
    assert tasks[0]["task"] != tasks[1]["task"]


def test_single_observation_capability_is_provisional():
    assert quality_gate.capability_status({"score": 4, "observations": 1}) == "PROVISIONAL_SINGLE_OBSERVATION"
    assert quality_gate.capability_status({"score": 4, "observations": 2}) == "PROXY_EVIDENCED"


def test_capsule_never_promotes_practitioner_to_expert_and_preserves_actual_sources():
    spec = {"identity": "SB", "function": "Supabase Platform Specialist", "target": "practitioner"}
    qual = {
        "source_evidence": [{"url": "https://supabase.com/docs/guides/queues", "status": "ok"}],
        "capability_proxy_evidence": [{
            "id": "queues", "name": "Queues", "proxy_score": 4,
            "proxy_observations": 1, "proxy_status": "PROVISIONAL_SINGLE_OBSERVATION", "critical": True,
        }],
    }
    capsule = quality_gate.build_capsule(spec, "Queues are Postgres-native.", [], [], qual)
    assert "Expert" not in capsule
    assert "expert status" in capsule
    assert "https://supabase.com/docs/guides/queues" in capsule
    assert "PROVISIONAL_SINGLE_OBSERVATION" in capsule
    assert "does not establish exhaustive coverage" in capsule
    assert "Native identity qualification has NOT been run" in capsule


def test_external_proxy_can_only_make_package_ready_not_native_identity_pass():
    spec = {"identity": "SB", "function": "Supabase Platform Specialist", "target": "practitioner", "rounds": 2}
    cap = bootcamp.Capability("rls", "RLS", "Row security", True, score=4, observations=2)
    audit = [
        {"round": 1, "exam": {"critical_failure": False}},
        {"round": 2, "exam": {"critical_failure": False}},
    ]
    transfers = [
        {"transfer": 1, "task": {"adversarial": False}, "exam": {"score": 4, "critical_failure": False, "corrections": []}},
        {"transfer": 2, "task": {"adversarial": True}, "exam": {"score": 4, "critical_failure": False, "corrections": []}},
    ]
    qual = quality_gate.conservative_qualification(spec, [cap], audit, transfers, [])
    assert qual["training_package_ready"] is True
    assert qual["native_identity_qualification"] == "NOT_RUN"
    assert "external_bootcamp_pass" not in qual
    assert qual["outcome"] == "CONDITIONAL PASS"


def test_capability_support_filter_rejects_topics_absent_from_source_pack():
    source_pack = "Postgres database. Row Level Security policies. Authentication with JWT. Durable queues."
    supported = bootcamp.Capability("rls", "Row Level Security", "Manage RLS policies for database tables", True)
    unsupported = bootcamp.Capability("deployment", "Deployment & Branching", "Preview environments and branching strategies", True)
    assert quality_gate.capability_supported_by_sources(supported, source_pack)
    assert not quality_gate.capability_supported_by_sources(unsupported, source_pack)


def test_generic_hidden_rubric_is_replaced_with_capability_specific_checks():
    caps = [bootcamp.Capability("rls", "Row Level Security", "Create and reason about table RLS policies", True)]
    task = {
        "task": "Diagnose an RLS failure",
        "capability_ids": ["rls"],
        "hidden_rubric": ["short check"],
        "adversarial": True,
        "critical": True,
    }
    repaired = quality_gate.strengthen_task_rubric(task, caps)
    assert repaired["hidden_rubric"] != ["short check"]
    assert any("RLS" in item or "Row Level Security" in item for item in repaired["hidden_rubric"])
    assert any("unsupported" in item.lower() or "false premise" in item.lower() for item in repaired["hidden_rubric"])


def test_capability_support_filter_rejects_description_smuggling():
    source_pack = "Configure database roles and permissions for users."
    smuggled = bootcamp.Capability(
        "roles_permissions",
        "Roles & Permissions",
        "Configure roles and permissions plus billing quotas and preview deployments",
        True,
    )
    assert not quality_gate.capability_supported_by_sources(smuggled, source_pack)


def test_training_package_not_ready_when_proxy_transfer_or_critical_capability_is_weak():
    spec = {"identity": "SB", "function": "Supabase Platform Specialist", "target": "practitioner", "rounds": 2}
    weak_cap = bootcamp.Capability("rls", "RLS", "Row security", True, score=1, observations=2)
    audit = [
        {"round": 1, "exam": {"score": 1, "passed": False, "critical_failure": False}},
        {"round": 2, "exam": {"score": 1, "passed": False, "critical_failure": False}},
    ]
    transfers = [
        {"transfer": 1, "exam": {"score": 1, "passed": False, "critical_failure": False}},
        {"transfer": 2, "exam": {"score": 1, "passed": False, "critical_failure": False}},
    ]
    qual = quality_gate.conservative_qualification(spec, [weak_cap], audit, transfers, [])
    assert qual["training_package_ready"] is False
    assert qual["outcome"] == "FAIL"


# V3 quality gates discovered by auditing the first real production artifact.

def test_capability_exact_evidence_requires_verbatim_quote_and_source_url():
    source_pack = '<SOURCE url="https://supabase.com/docs/guides/queues">\nSupabase Queues is a Postgres-native durable Message Queue system with guaranteed delivery built on the pgmq database extension.\n</SOURCE>'
    grounded = {
        "name": "Queues",
        "description": "Durable Postgres-native queues with guaranteed delivery",
        "evidence_url": "https://supabase.com/docs/guides/queues",
        "evidence_quote": "Supabase Queues is a Postgres-native durable Message Queue system with guaranteed delivery",
    }
    bad_quote = dict(grounded, evidence_quote="Supabase magically guarantees global exactly-once side effects")
    bad_url = dict(grounded, evidence_url="https://supabase.com/docs/guides/functions")
    assert quality_gate.capability_exact_evidence_supported(grounded, source_pack)
    assert not quality_gate.capability_exact_evidence_supported(bad_quote, source_pack)
    assert not quality_gate.capability_exact_evidence_supported(bad_url, source_pack)


def test_criticality_is_capped_and_curriculum_repairs_repeated_critical_coverage():
    caps = [bootcamp.Capability(f"c{i}", f"Cap {i}", f"Description {i}", True) for i in range(1, 7)]
    normalized = quality_gate.normalize_criticality(caps, max_critical=3)
    assert sum(c.critical for c in normalized) == 3
    curriculum = [
        {"task": f"task {i}", "capability_ids": ["c1"], "hidden_rubric": ["specific check"], "adversarial": i >= 8, "critical": True}
        for i in range(10)
    ]
    repaired = quality_gate.ensure_critical_coverage(curriculum, normalized)
    assert len(repaired) == 10
    for cap in [c for c in normalized if c.critical]:
        assert sum(cap.id in task["capability_ids"] for task in repaired) >= 2


def test_examiner_cannot_award_robust_4_when_it_reports_corrections(monkeypatch):
    monkeypatch.setattr(bootcamp, "json_llm", lambda *a, **k: {
        "score": 4,
        "passed": True,
        "critical_failure": False,
        "capability_updates": {"rls": 4},
        "corrections": ["Material correction required"],
        "durable_lesson": "",
        "regression_case": "",
        "reason": "Mostly correct but correction required",
    })
    exam = bootcamp.examine(
        "Supabase Platform Specialist",
        "source evidence",
        {"task": "RLS task", "capability_ids": ["rls"], "hidden_rubric": ["check"], "critical": True},
        "student answer",
    )
    assert exam["score"] <= 3
    assert exam["capability_updates"]["rls"] <= 3


def test_transfer_generation_retries_duplicate_exam(monkeypatch):
    caps = [
        bootcamp.Capability("rls", "RLS", "Row level security", True),
        bootcamp.Capability("auth", "Auth", "Authentication", True),
    ]
    answers = iter([
        {"rounds": [{"task": "same transfer", "capability_ids": ["rls", "auth"], "hidden_rubric": ["check"], "adversarial": False, "critical": True}]},
        {"rounds": [{"task": "same transfer", "capability_ids": ["rls", "auth"], "hidden_rubric": ["check"], "adversarial": True, "critical": True}]},
        {"rounds": [{"task": "different transfer", "capability_ids": ["rls", "auth"], "hidden_rubric": ["check"], "adversarial": True, "critical": True}]},
    ])
    calls = []
    def fake_json_llm(*args, **kwargs):
        calls.append(1)
        return next(answers)
    monkeypatch.setattr(bootcamp, "json_llm", fake_json_llm)
    tasks = fast_bootcamp.robust_build_transfer_tasks({"function": "Supabase Platform Specialist"}, caps, [])
    assert [t["task"] for t in tasks] == ["same transfer", "different transfer"]
    assert len(calls) == 3


def test_capsule_uses_source_backed_capability_records_not_freeform_knowledge():
    spec = {"identity": "SB", "function": "Supabase Platform Specialist", "target": "practitioner"}
    qual = {
        "source_evidence": [{"url": "https://supabase.com/docs/guides/queues", "status": "ok"}],
        "capability_proxy_evidence": [{
            "id": "queues",
            "name": "Queues",
            "description": "Durable Postgres-native queues",
            "evidence_url": "https://supabase.com/docs/guides/queues",
            "evidence_quote": "Supabase Queues is a Postgres-native durable Message Queue system with guaranteed delivery",
            "proxy_score": 4,
            "proxy_observations": 2,
            "proxy_status": "PROXY_EVIDENCED",
            "critical": True,
        }],
    }
    capsule = quality_gate.build_capsule(spec, "UNSUPPORTED HALLUCINATION ABOUT MAGIC BRANCHING", [], [], qual)
    assert "UNSUPPORTED HALLUCINATION" not in capsule
    assert "Durable Postgres-native queues" in capsule
    assert "Supabase Queues is a Postgres-native durable Message Queue system" in capsule
    assert "https://supabase.com/docs/guides/queues" in capsule
