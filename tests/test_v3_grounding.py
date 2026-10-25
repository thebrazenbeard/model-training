import bootcamp
import fast_bootcamp


def test_grounded_knowledge_pack_is_bounded_source_pack_not_model_synthesis(monkeypatch):
    source_pack = '<SOURCE url="https://supabase.com/docs/guides/queues">\nQueues are durable.\n</SOURCE>'
    evidence = [{"url": "https://supabase.com/docs/guides/queues", "status": "ok"}]
    excerpts = (bootcamp.SourceExcerpt(evidence[0]["url"], "test-sha", "Queues are durable.", ("Queues are durable.",)),)
    monkeypatch.setattr(
        bootcamp,
        "build_source_pack_with_provenance",
        lambda function, sources: (source_pack, evidence, excerpts),
    )

    def forbidden_llm(*args, **kwargs):
        raise AssertionError("knowledge synthesis model call must not occur")

    monkeypatch.setattr(bootcamp, "llm", forbidden_llm)
    knowledge, actual_evidence = fast_bootcamp.grounded_build_knowledge_pack(
        "Supabase Platform Specialist", ["https://supabase.com/docs/guides/queues"]
    )
    assert knowledge == source_pack
    assert actual_evidence == evidence
    assert fast_bootcamp.get_source_pack() == source_pack
