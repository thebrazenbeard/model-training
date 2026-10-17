"""Subprocess-only Mono real source probe: no Pre-Active, Volition or provider.
No production state, model activity, memory admission, or effects.
"""
from __future__ import annotations
import importlib.abc
import json
from pathlib import Path
import sys
import time


class ExcludeSelectedOptional(importlib.abc.MetaPathFinder):
    BLOCKED = ("pre_active", "preactive", "volition")
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in self.BLOCKED):
            raise ImportError("optional runtime package forbidden in minimal probe")
        return None


def main():
    t0 = time.perf_counter_ns()
    # This probe is meant to start in an entirely new CPython process.
    assert not any(n == "pre_active" or n.startswith("pre_active.") or
                   n == "volition" or n.startswith("volition.")
                   for n in sys.modules)
    sys.meta_path.insert(0, ExcludeSelectedOptional())
    arg = json.loads(sys.stdin.read())
    from vera_core import QualifiedVeraRuntime, VeraStateDirectory
    from vera_core.local_conversation import open_message, conversation_history, ConversationError
    state = VeraStateDirectory(Path(arg["root"]), project_id=arg.get("project_id", "veraos"),
                               identity_id=arg.get("identity_id", "VERA"), currentness_subject_id="trial-only")
    rt = QualifiedVeraRuntime.from_state_directory(state)
    assert not rt.provider_execution_transports, "unexpected selected provider"
    result = {}
    try:
        if arg["phase"] == "initial":
            task = open_message(rt.tasks, "console-demo", "message-1", "literal  content")
            result = {"task_id": task["task_id"], "head": task["opened_head"],
                      "source": task["source_digest"], "authenticated": task["origin_authenticated"]}
        elif arg["phase"] == "reopen":
            task = open_message(rt.tasks, "console-demo", "message-1", "literal  content")
            view = conversation_history(rt.tasks, "console-demo")
            result = {"task_id": task["task_id"], "head": task["opened_head"],
                      "source": task["source_digest"], "history": len(view),
                      "literal": view[0]["text"]}
        elif arg["phase"] == "tamper":
            try:
                open_message(rt.tasks, "console-demo", "message-1", arg["changed_text"])
            except ConversationError:
                result["tamper_rejected"] = True
            else:
                raise AssertionError("same ID + different literal accepted")
            task = open_message(rt.tasks, "console-demo", "message-1", "literal  content")
            result.update({"task_id": task["task_id"], "head": task["opened_head"], "source": task["source_digest"]})
        else:
            raise RuntimeError("unknown phase")
        # Forged source/model/route/DONE claims deliberately do not enter any API.
        # Without a verifier they are assertions, not evidence or completion.
        payload = arg.get("untrusted_provider_claim")
        if payload is not None:
            assert isinstance(payload, dict)
            result["forged_claim_ignored"] = True
        task_state = rt.tasks.read(task["task_id"])
        events = rt.tasks.events(task["task_id"])
        result.update({"closed": task_state.closed, "event_types":
                       [event.event_type for event in events],
                       "task_count": len(rt.tasks.events()),
                       "no_optional_imports": not any(
                            n == "pre_active" or n.startswith("pre_active.") or
                            n == "volition" or n.startswith("volition.")
                            for n in sys.modules),
                       "provider_transport_count": len(rt.provider_execution_transports),
                       "elapsed_ns": time.perf_counter_ns() - t0})
    finally:
        # Lifecycle handles belong to host; subprocess exit closes local test resources.
        pass
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
