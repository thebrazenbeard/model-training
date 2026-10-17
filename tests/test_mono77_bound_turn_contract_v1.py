"""Training source-only OS38/Mono77 binding contract unit tests; no model access.

Mock Mono interfaces here. Exact pinned Mono integration remains a separate
optional/full-package test; these unit tests never prove an installed route.
"""
from dataclasses import replace
from types import SimpleNamespace, ModuleType
from unittest import TestCase
from unittest.mock import patch
import sys

from resident_verifier_port_v1 import ResidentModelRequest
from resident_os_composition_gate_v1 import (
    ResidentSourceHold, ScopedResidentTurn, bind_mono77_opened_message,
)
from test_resident_verifier_port_v1 import ResidentVerifierPortTests


class FakeTaskExecutionError(ValueError):
    pass


class FakeConversationError(ValueError):
    pass


class FakeLedger:
    def __init__(self, task_id):
        self.task_id = task_id
        self.read_calls = []
        self.head = "4" * 64
        self.closed = False
        self.mutated = False

    def read(self, task_id):
        self.read_calls.append(task_id)
        if task_id != self.task_id:
            raise FakeTaskExecutionError("missing canonical task")
        return SimpleNamespace(
            closed=self.closed, active_dependencies=(),
            unresolved_correction_ids=(), active_delegations=(),
            journal_head_digest=self.head,
        )


class FakeRuntime:
    def __init__(self, task_id):
        self.tasks = FakeLedger(task_id)


class Mono77BindingUnitTests(TestCase):
    def setUp(self):
        self.fixture = ResidentVerifierPortTests()
        self.fixture.setUp()
        self.intake = {
            "schema": "VERAOS_LOCAL_TASK_INTAKE_V1",
            "conversation_id": "veraos-local-console",
            "message_id": "request:one",
            "task_id": "ui-" + "1" * 64,
            "text": "Answer the specific accepted question.",
            "source_digest": "3" * 64,
            "packet_digest": "2" * 64,
            "opened_head": "4" * 64,
            "origin_authenticated": False,
        }
        self.request = replace(
            self.fixture.request, task_id=self.intake["task_id"],
            task_digest=self.intake["source_digest"],
        )
        self.runtime = FakeRuntime(self.intake["task_id"])
        self.open_calls = []
        self.current = dict(self.intake)

        def open_message(ledger, conversation_id, message_id, text):
            self.open_calls.append((conversation_id, message_id, text))
            if ledger is not self.runtime.tasks:
                raise FakeConversationError("wrong task store")
            return dict(self.current)

        core = ModuleType("vera_core")
        core.QualifiedVeraRuntime = FakeRuntime
        conv = ModuleType("vera_core.local_conversation")
        conv.open_message = open_message
        conv.ConversationError = FakeConversationError
        tasks = ModuleType("vera_core.task_execution")
        tasks.TaskExecutionError = FakeTaskExecutionError
        replacement = {
            "vera_core": core,
            "vera_core.local_conversation": conv,
            "vera_core.task_execution": tasks,
        }
        self.module_patch = patch.dict(sys.modules, replacement)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def bind(self, intake=None, request=None, request_id="request:one"):
        return bind_mono77_opened_message(
            self.runtime, self.intake if intake is None else intake,
            self.request if request is None else request,
            request_id=request_id,
        )

    def test_source_bound_turn_is_not_real_model_admission(self):
        bound = self.bind()
        self.assertIs(type(bound), ScopedResidentTurn)
        self.assertEqual(bound.task_text, self.intake["text"])
        self.assertEqual(bound.expected_task_head, self.intake["opened_head"])
        self.assertEqual(bound.model_request.task_id, self.intake["task_id"])
        self.assertEqual(bound.actions, ("COGNITION_TURN",))
        self.assertFalse(hasattr(bound, "network_dispatch_authorized"))

    def test_substituted_task_and_digest_hold_before_open_message(self):
        for request in (
            replace(self.request, task_id="ui-" + "f" * 64),
            replace(self.request, task_digest="e" * 64),
        ):
            with self.subTest(request=request), self.assertRaises(ResidentSourceHold):
                self.bind(request=request)
        self.assertEqual(self.open_calls, [])

    def test_wrong_conversation_literal_or_origin_hold_on_recheck(self):
        for field, value in (
            ("conversation_id", "someone-else"),
            ("message_id", "other-message"),
            ("text", "do a different task"),
            ("origin_authenticated", True),
            ("opened_head", "a" * 64),
        ):
            with self.subTest(field=field), self.assertRaises(ResidentSourceHold):
                self.bind(intake={**self.intake, field: value})

    def test_same_task_stale_checkpoint_or_closed_task_hold(self):
        self.runtime.tasks.head = "5" * 64
        with self.assertRaisesRegex(ResidentSourceHold, "STALE_OR_UNRESOLVED"):
            self.bind()
        self.runtime.tasks.head = "4" * 64
        self.runtime.tasks.closed = True
        with self.assertRaises(ResidentSourceHold):
            self.bind()

    def test_missing_actual_ledger_task_does_not_create_task(self):
        self.runtime.tasks.task_id = "different:task"
        with self.assertRaisesRegex(ResidentSourceHold, "CANONICAL_MONO_TASK_NOT_VERIFIED"):
            self.bind()
        self.assertEqual(self.open_calls, [])

    def test_readback_conflict_fails_even_with_consistent_forged_descriptor(self):
        forged = {**self.intake, "text": "plausible different task"}
        with self.assertRaisesRegex(ResidentSourceHold, "MONO_INTAKE_DID_NOT_READ_BACK"):
            self.bind(intake=forged)
        with self.assertRaises(ResidentSourceHold):
            self.bind(request_id="other-request")
