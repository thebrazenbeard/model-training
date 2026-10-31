"""Adversarial SOURCE-only tests; volatile mock is never a security proof."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from trainingday.r140_monotonic_authority import (
    GENESIS_HEAD, Candidate, DemoVolatileWitness, WitnessConflict,
    assess_transition,
)


RUN = "training_run_one"
HOST = "host_principal_one"
DIGEST = "a" * 64


def proposed(*, nonce="nonce_unique_one", step=1, epoch=0, head=GENESIS_HEAD,
             run=RUN, host=HOST, digest=DIGEST):
    return Candidate(
        run_id=run, host_principal=host, step=step,
        checkpoint_sha256=digest, nonce=nonce,
        expected_epoch=epoch, expected_head=head,
    )


def assert_hold(decision):
    assert decision.authorizes_resume is False
    assert decision.authorizes_training is False
    assert decision.physical_custody_verified is False
    assert decision.durable_antirollback_verified is False


def test_first_append_and_successor_are_synthetic_and_never_authorize():
    witness = DemoVolatileWitness()
    first = assess_transition(proposed(), witness)
    assert first.reason == "SYNTHETIC_CAS_MATCH_NO_HOST_AUTHORITY"
    assert first.synthetic_transition_observed
    assert first.witness_epoch == 1
    assert_hold(first)

    second = assess_transition(proposed(
        step=5, nonce="nonce_unique_two", epoch=1, head=first.witness_head,
    ), witness)
    assert second.synthetic_transition_observed
    assert second.witness_epoch == 2
    assert second.witness_head != first.witness_head
    assert_hold(second)


def test_restored_older_sqlite_snapshot_cannot_advance_current_witness():
    witness = DemoVolatileWitness()
    original = proposed()
    first = assess_transition(original, witness)
    second = assess_transition(proposed(
        step=2, nonce="nonce_unique_two", epoch=1, head=first.witness_head,
    ), witness)
    assert second.synthetic_transition_observed

    # Simulate restoring application SQLite to the snapshot before step 1.
    stale = proposed(step=9, nonce="nonce_unique_new", epoch=0, head=GENESIS_HEAD)
    denied = assess_transition(stale, witness)
    assert denied.reason == "STALE_OR_DIVERGENT_SNAPSHOT"
    assert_hold(denied)
    assert witness.demo_read(RUN).head == second.witness_head


def test_same_epoch_divergent_chain_head_rejected():
    witness = DemoVolatileWitness()
    assess_transition(proposed(), witness)
    denied = assess_transition(proposed(
        step=2, nonce="nonce_unique_two", epoch=1, head="f" * 64,
    ), witness)
    assert denied.reason == "STALE_OR_DIVERGENT_SNAPSHOT"
    assert_hold(denied)


def test_replayed_nonce_rejected_even_with_correct_current_epoch():
    witness = DemoVolatileWitness()
    first = assess_transition(proposed(), witness)
    repeated = assess_transition(proposed(
        step=2, epoch=1, head=first.witness_head,
    ), witness)
    assert repeated.reason == "NONCE_REPLAY"
    assert_hold(repeated)


def test_host_principal_cannot_change_mid_run():
    witness = DemoVolatileWitness()
    first = assess_transition(proposed(), witness)
    impersonation = assess_transition(proposed(
        host="different_host_principal", step=2,
        nonce="nonce_unique_two", epoch=1, head=first.witness_head,
    ), witness)
    assert impersonation.reason == "HOST_PRINCIPAL_CHANGED"
    assert_hold(impersonation)


def test_step_must_advance_even_when_epoch_advances():
    witness = DemoVolatileWitness()
    first = assess_transition(proposed(step=4), witness)
    duplicate_step = assess_transition(proposed(
        step=4, nonce="nonce_unique_two", epoch=1, head=first.witness_head,
    ), witness)
    assert duplicate_step.reason == "STEP_NOT_MONOTONIC"
    assert_hold(duplicate_step)


def test_one_atomic_winner_for_concurrent_same_parent_proposals():
    witness = DemoVolatileWitness()
    proposals = [proposed(nonce=f"worker_nonce_{i:08d}") for i in range(32)]
    with ThreadPoolExecutor(max_workers=12) as pool:
        decisions = list(pool.map(lambda item: assess_transition(item, witness), proposals))
    successes = [d for d in decisions if d.synthetic_transition_observed]
    failures = [d for d in decisions if not d.synthetic_transition_observed]
    assert len(successes) == 1
    assert len(failures) == 31
    assert all(d.reason == "STALE_OR_DIVERGENT_SNAPSHOT" for d in failures)
    assert all(not d.authorizes_resume for d in decisions)


def test_run_namespace_isolated_and_unknown_witness_rejected():
    witness = DemoVolatileWitness()
    first = assess_transition(proposed(), witness)
    other = assess_transition(proposed(run="training_run_two"), witness)
    assert first.synthetic_transition_observed
    assert other.synthetic_transition_observed
    no_service = assess_transition(proposed(), None)
    assert no_service.reason == "INDEPENDENT_WITNESS_UNAVAILABLE"
    assert_hold(no_service)


def test_counter_booleans_bad_hash_and_impossible_steps_fail_closed():
    witness = DemoVolatileWitness()
    for corrupted in (
        replace(proposed(), expected_epoch=True),
        replace(proposed(), step=False),
        replace(proposed(), step=2**31),
        replace(proposed(), expected_epoch=-1),
        replace(proposed(), checkpoint_sha256="0x" + DIGEST),
        replace(proposed(), host_principal="../escape"),
    ):
        result = assess_transition(corrupted, witness)
        assert not result.synthetic_transition_observed
        assert_hold(result)
    assert witness.demo_read(RUN) is None


def test_untrusted_transport_forgery_fails_reply_binding():
    request = proposed()

    class FakePort:
        def compare_and_append(self, candidate):
            ledger = DemoVolatileWitness()
            genuine = ledger.compare_and_append(candidate)
            return replace(genuine, checkpoint_sha256="b" * 64)

    denied = assess_transition(request, FakePort())
    assert denied.reason == "WITNESS_REPLY_UNBOUND"
    assert_hold(denied)


def test_timeout_after_possible_commit_is_unknown_and_not_retried():
    witness = DemoVolatileWitness()

    class CommitThenDisconnect:
        def compare_and_append(self, candidate):
            witness.compare_and_append(candidate)
            raise TimeoutError("connection dropped after possible commit")

    decision = assess_transition(proposed(), CommitThenDisconnect())
    assert decision.reason == "WITNESS_OUTCOME_UNKNOWN_HOLD"
    assert_hold(decision)
    assert witness.demo_read(RUN).epoch == 1
    # Caller MUST reconcile through a separately authenticated read path.
    assert assess_transition(proposed(), witness).reason == "NONCE_REPLAY"


def test_witness_can_reject_explicitly_without_raising_transport_error():
    class RejectingPort:
        def compare_and_append(self, candidate):
            raise WitnessConflict("AUTHORITY_DENIED")

    denied = assess_transition(proposed(), RejectingPort())
    assert denied.reason == "AUTHORITY_DENIED"
    assert_hold(denied)
