"""Dependency-free R16 post-save pause gate, not a physical TRAIN grant.

The caller supplies the installed Transformers TrainerCallback base class only
after admission and host preflight. Nothing here imports Torch or deserializes
optimizer/RNG checkpoint files. The callback inventories a *completed* save
before requesting a graceful stop of the current training phase.
"""

from __future__ import annotations

from pathlib import Path

from trainingday.admission import AdmissionError
from trainingday.checkpoint_custody import create_custody


PAUSE_STEP = 8
PLANNED_STEPS = 16


def validate_checkpoint_stop_request(args) -> bool:
    """Reject any accidental use outside a fresh, sixteen-step R16 arm."""
    step = getattr(args, "stop_after_checkpoint_step", None)
    if step is None:
        return False
    if (type(step) is not int or step != PAUSE_STEP
            or getattr(args, "r16_checkpointable", None) is not True
            or getattr(args, "mode", None) not in ("correct", "corrupt")
            or type(getattr(args, "steps", None)) is not int
            or args.steps != PLANNED_STEPS
            or getattr(args, "resume_checkpoint", None) is not None):
        raise AdmissionError("R16 checkpoint pause requires fresh trained 16-step run at step 8")
    return True


def build_checkpoint_stop_callback(callback_base: type, *, trainer_work: Path,
                                   contract_sha256: str, stop_step: int):
    """Return an installed-Trainer-compatible callback; no GPU imports here.

    ``on_save`` is dispatched by Hugging Face *after* it returns from saving a
    Trainer checkpoint. This proves logical save completion only, not fsync or
    independently authenticated custody. A later post-exit verifier must pin
    the receipt outside the run output before optimizer deserialization.
    """
    if (type(callback_base) is not type or type(stop_step) is not int
            or stop_step != PAUSE_STEP or type(contract_sha256) is not str
            or len(contract_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in contract_sha256)):
        raise AdmissionError("invalid R16 checkpoint stopper configuration")
    trainer_work = Path(trainer_work)

    class StopAfterVerifiedCheckpoint(callback_base):
        def __init__(self):
            self.triggered = False
            self.custody = None

        def on_save(self, args, state, control, **kwargs):
            step = getattr(state, "global_step", None)
            if type(step) is not int or step > stop_step or self.triggered:
                raise AdmissionError("R16 checkpoint stop step overshot/invalid/duplicated")
            if step != stop_step:
                return control
            save_strategy = getattr(args, "save_strategy", None)
            save_strategy = getattr(save_strategy, "value", save_strategy)
            if (save_strategy != "steps" or type(getattr(args, "save_steps", None)) is not int
                    or args.save_steps != stop_step
                    or getattr(args, "save_only_model", None) is not False
                    or type(getattr(args, "max_steps", None)) is not int
                    or args.max_steps != PLANNED_STEPS
                    or Path(args.output_dir).resolve() != trainer_work.resolve()
                    or getattr(state, "is_world_process_zero", None) is not True):
                raise AdmissionError("R16 checkpoint stop trainer configuration changed")
            # Hash all checkpoint bytes, inspect JSON state, and require optimizer,
            # scheduler, adapter and RNG files. No pickle or Torch load occurs.
            custody = create_custody(
                trainer_work / f"checkpoint-{stop_step}",
                contract_sha256=contract_sha256,
            )
            if custody["step"] != stop_step:
                raise AdmissionError("R16 checkpoint stop custody step mismatch")
            self.custody = custody
            self.triggered = True
            control.should_training_stop = True
            return control

    return StopAfterVerifiedCheckpoint()


def make_paused_phase_receipt(preflight: dict, *, custody: dict, contract_sha256: str,
                              custody_file_sha256: str) -> dict:
    """Never classify a planned step-eight pause as a completed 16-step run."""
    if (type(preflight) is not dict or type(custody) is not dict
            or custody.get("schema") != "VERA_TRAINER_STEP_CUSTODY_R16_V1"
            or custody.get("step") != PAUSE_STEP
            or custody.get("state") != "CHECKPOINT_BYTES_PINNED_NOT_RESUME_ATTESTED"
            or custody.get("run_contract_sha256") != contract_sha256
            or type(custody_file_sha256) is not str
            or len(custody_file_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in custody_file_sha256)):
        raise AdmissionError("R16 paused phase receipt lacks verified custody")
    return {
        **preflight,
        "status": "R16_STEP8_CHECKPOINT_PAUSED_NOT_COMPLETE",
        "planned_optimizer_steps": PLANNED_STEPS,
        "trainer_global_step_observed": PAUSE_STEP,
        "new_optimizer_steps": None,
        "optimizer_step_count_independently_verified": False,
        "r16_run_contract_sha256": contract_sha256,
        "checkpoint_step": PAUSE_STEP,
        "checkpoint_custody_state": custody["state"],
        "checkpoint_custody_file_sha256": custody_file_sha256,
        "optimizer_resume_equivalence": "UNVERIFIED_REQUIRES_UNINTERRUPTED_MATCH",
        "final_adapter_saved": False,
        "model_promotion_authorized": False,
    }
