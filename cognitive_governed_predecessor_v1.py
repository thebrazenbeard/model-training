"""Read-only Mono task opener recovery for a pruned literal correction.

This helper reads real Mono #77 canonical task history (not test callbacks
or arbitrary "trusted" source dictionaries). It proves only which exact
UNAUTHENTICATED UI text was previously accepted. It does NOT certify
semantic interpretation, factual correctness, actor identity, provider
qualification, memory admission, or model behavior. No new task/effect.
"""
from __future__ import annotations
from dataclasses import replace

from cognitive_continuity_eval_v1 import (
    CognitiveObservation,CognitivePrediction,CognitiveSource,
    guard_prediction,
)

def _hold(prediction: CognitivePrediction) -> CognitivePrediction:
    return CognitivePrediction("HOLD", prediction.task_id, prediction.request_id)

def recover_pruned_predecessor(
    ledger, observation: CognitiveObservation, prediction: CognitivePrediction,
    *, conversation_id: str, predecessor_message_id: str,
) -> CognitivePrediction:
    """Bounded exact predecessor retrieval from Mono TaskExecutionLedger.

    Only a pruned predecessor of a single typed correction can be retrieved.
    Current task ID/head/source digest, current exact raw text, selected
    previous task's same conversation, predecessor event digest and past exact
    accepted literal MUST read back. No readback or changed ledger -> HOLD.
    """
    from vera_core.task_execution import TaskExecutionLedger,TaskExecutionError
    from vera_core.local_conversation import conversation_history,ConversationError

    if (type(ledger) is not TaskExecutionLedger or
        type(observation) is not CognitiveObservation or
        type(prediction) is not CognitivePrediction or
        type(conversation_id) is not str or not conversation_id or
        type(predecessor_message_id) is not str or not predecessor_message_id):
        return _hold(prediction)
    if (len(observation.sources)!=1 or
        observation.sources[0].kind!="CORRECTION" or
        prediction.status!="ANSWER" or
        observation.task_id!=prediction.task_id or
        observation.request_id!=prediction.request_id):
        return _hold(prediction)
    correction=observation.sources[0]
    if (correction.supersedes is None or
        correction.supersedes!=observation.previous_interaction_id or
        prediction.source_event_id!=correction.source_event_id or
        correction.referent!=observation.referent or
        correction.scope!=observation.scope):
        return _hold(prediction)
    try:
        # Pure reads; unlike open_message this path MUST NOT create a task.
        before=ledger.read(observation.task_id)
        nowevents=ledger.events(observation.task_id)
        if (before.closed or before.active_dependencies or
            before.unresolved_correction_ids or before.active_delegations or
            not nowevents or nowevents[0].event_type!="TASK_OPENED" or
            before.journal_head_digest!=observation.task_head or
            before.journal_head_digest!=observation.accepted_task_head or
            before.opened_lifecycle_evidence_digest!=observation.request_source_digest or
            observation.request_source_digest!=observation.accepted_source_digest or
            before.packet.subject!=observation.utterance or
            correction.value!=before.packet.subject or
            correction.source_event_id!="mono-task-open:"+nowevents[0].event_digest):
            return _hold(prediction)

        history=conversation_history(ledger,conversation_id)
        present=[h for h in history if h["task_id"]==observation.task_id and
                 h["message_id"]==observation.request_id and
                 h["text"]==observation.utterance]
        ancestor=[h for h in history if h["message_id"]==predecessor_message_id and
                  h["task_id"]!=observation.task_id]
        if len(present)!=1 or len(ancestor)!=1:
            return _hold(prediction)
        prior=ledger.read(ancestor[0]["task_id"])
        oldevents=ledger.events(ancestor[0]["task_id"])
        if (not oldevents or oldevents[0].event_type!="TASK_OPENED" or
            not prior.packet.subject or
            oldevents[0].payload.get("packet",{}).get("subject")!=prior.packet.subject or
            correction.supersedes!="mono-task-open:"+oldevents[0].event_digest):
            return _hold(prediction)
        # NEVER infer a semantic proposition from an arbitrary old text.
        # The currently supplied referent/scope remain UNAUTHENTICATED.
        predecessor=CognitiveSource(
            source_event_id=correction.supersedes,
            referent=observation.referent,
            scope=observation.scope,
            value=prior.packet.subject,
            kind="USER_STATEMENT",
        )
        # Fail on a concurrent append/replacement before returning a source
        # integrity answer; this is not an external provider trust proof.
        if ledger.read(observation.task_id).journal_head_digest!=before.journal_head_digest:
            return _hold(prediction)
        return guard_prediction(replace(observation,
                            sources=(predecessor,correction)),prediction)
    except (TaskExecutionError, ConversationError, KeyError,
            ValueError, TypeError, AttributeError):
        return _hold(prediction)
