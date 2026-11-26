from __future__ import annotations

import hashlib
import json


REQUIRED_FIELDS = {"item_id", "prompt_family", "prompt", "rubric"}
SEALED_COMMITMENT_FILENAME = "V10_SEALED_FINAL_BANK_COMMITMENT_V1.json"
TRAINING_AUTHORITY_FILENAME = "V10_QWEN35_TRAINING_AUTHORITY_V1.json"
TRAINING_RUNTIME_FILENAME = "V10_QWEN35_TRAINING_RUNTIME_BINDING_V1.json"
TRAINING_EXECUTION_FILENAME = "V10_QWEN35_TRAINING_EXECUTION_BINDING_V1.json"
SEALED_DERIVABLE_PRECONDITIONS = {
    "fresh_evaluation_bank_frozen",
    "independent_bank_admission_verified",
    "semantic_contamination_screen_verified",
    "contamination_screen_against_v10_and_consumed_finals_verified",
}


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def prompt_hash(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("heldout prompt must be nonempty text")
    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()


def freeze_heldout_manifest(items, train_prompts, validation_prompts):
    train_hashes = {prompt_hash(p) for p in train_prompts}
    validation_hashes = {prompt_hash(p) for p in validation_prompts}
    seen_prompts = set()
    manifest_items = []
    for item in items:
        missing = REQUIRED_FIELDS - set(item)
        if missing:
            raise ValueError(f"heldout item missing fields: {sorted(missing)}")
        digest = prompt_hash(item["prompt"])
        if digest in seen_prompts:
            raise ValueError("duplicate heldout prompt")
        if digest in train_hashes:
            raise ValueError("heldout prompt has training leakage")
        if digest in validation_hashes:
            raise ValueError("heldout prompt has validation leakage")
        seen_prompts.add(digest)
        item_digest = hashlib.sha256(_canonical(item)).hexdigest()
        manifest_items.append({
            "item_id": item["item_id"],
            "prompt_family": item["prompt_family"],
            "prompt_sha256": digest,
            "item_sha256": item_digest,
        })

    set_digest = hashlib.sha256(_canonical(manifest_items)).hexdigest()
    return {
        "schema_version": 1,
        "item_count": len(manifest_items),
        "items": manifest_items,
        "set_sha256": set_digest,
    }


def _read_json(path):
    from pathlib import Path

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _valid_sha256(value) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _load_sealed_commitment_validator():
    import importlib.util
    from pathlib import Path

    module_path = (
        Path(__file__).resolve().parent
        / "experiments"
        / "sealed_final_bank_commitment.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_v10_sealed_final_bank_commitment",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("sealed commitment validator could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_sealed_final_bank_commitment


def _sealed_commitment_evidence(base, preconditions):
    validate_sealed_final_bank_commitment = (
        _load_sealed_commitment_validator()
    )

    path = base / SEALED_COMMITMENT_FILENAME
    if not path.exists():
        return (
            {
                "status": "ABSENT",
                "bank_id": None,
                "commitment_sha256": None,
            },
            [],
            {},
        )

    commitment = _read_json(path)
    reasons = []
    check = validate_sealed_final_bank_commitment(commitment)
    reasons.extend(
        "sealed_commitment:" + reason
        for reason in check.get("reasons", [])
    )

    claimed_sha = commitment.get("commitment_sha256")
    unsigned = dict(commitment)
    unsigned.pop("commitment_sha256", None)
    expected_sha = hashlib.sha256(_canonical(unsigned)).hexdigest()
    if claimed_sha != expected_sha:
        reasons.append("sealed_commitment_sha256_mismatch")
    if not _valid_sha256(commitment.get("freeze_subject_digest")):
        reasons.append("sealed_commitment_freeze_subject_digest_invalid")

    reasons = sorted(set(reasons))
    if reasons:
        return (
            {
                "status": "INVALID",
                "bank_id": commitment.get("bank_id"),
                "commitment_sha256": claimed_sha,
            },
            reasons,
            {},
        )

    derived = {
        key: True
        for key in sorted(SEALED_DERIVABLE_PRECONDITIONS)
        if key in preconditions
    }
    return (
        {
            "status": "VERIFIED",
            "bank_id": commitment.get("bank_id"),
            "commitment_sha256": claimed_sha,
            "bank_sha256": commitment.get(
                "plaintext_artifacts", {}
            ).get("bank_sha256"),
            "sealed_archive_sha256": commitment.get(
                "plaintext_artifacts", {}
            ).get("sealed_archive_sha256"),
            "plaintext_exposed_to_training_lane": False,
        },
        [],
        derived,
    )


def _training_runtime_evidence(base, contract):
    path = base / TRAINING_RUNTIME_FILENAME
    if not path.exists():
        return (
            {
                "status": "ABSENT",
                "binding_sha256": None,
            },
            [],
            False,
        )

    binding = _read_json(path)
    raw_reasons = []
    if binding.get("schema") != "V10_QWEN35_TRAINING_RUNTIME_BINDING_V1":
        raw_reasons.append("schema_mismatch")
    if binding.get("status") != "FROZEN_TARGET_RUNTIME_SUPPLEMENT":
        raw_reasons.append("status_mismatch")

    recipe = contract.get("training_recipe", {})
    contract_target = recipe.get("runtime_target", {})
    contract_versions = recipe.get("runtime_versions", {})
    target = binding.get("target")
    if not isinstance(target, dict):
        raw_reasons.append("target_missing")
        target = {}

    target_pairs = {
        "python_version": contract_versions.get("python"),
        "base_path": contract_target.get("base_path"),
        "gpu": contract_target.get("gpu"),
        "vram_mib": contract_target.get("vram_mib"),
        "driver": contract_target.get("driver"),
        "cost_class": contract_target.get("cost_class"),
    }
    for field, expected in target_pairs.items():
        if target.get(field) != expected:
            raw_reasons.append(f"target_mismatch:{field}")
    if not isinstance(target.get("python_path"), str) or not target[
        "python_path"
    ].strip():
        raw_reasons.append("target_python_path_missing")
    if not isinstance(target.get("cuda_runtime"), str) or not target[
        "cuda_runtime"
    ].strip():
        raw_reasons.append("target_cuda_runtime_missing")

    packages = binding.get("packages")
    if not isinstance(packages, dict):
        raw_reasons.append("packages_missing")
        packages = {}
    required_packages = (
        "torch",
        "transformers",
        "trl",
        "peft",
        "bitsandbytes",
        "datasets",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "tokenizers",
        "jinja2",
        "numpy",
    )
    for package in required_packages:
        value = packages.get(package)
        if not isinstance(value, str) or not value.strip():
            raw_reasons.append(f"package_version_missing:{package}")
    for package in ("torch", "transformers", "trl", "peft"):
        expected = contract_versions.get(package)
        if packages.get(package) != expected:
            raw_reasons.append(
                f"package_version_mismatch:{package}"
            )

    artifacts = binding.get("base_artifacts")
    if not isinstance(artifacts, dict):
        raw_reasons.append("base_artifacts_missing")
        artifacts = {}
    for filename in (
        "model.safetensors-00001-of-00002.safetensors",
        "model.safetensors-00002-of-00002.safetensors",
        "tokenizer.json",
    ):
        if not _valid_sha256(artifacts.get(filename)):
            raw_reasons.append(f"base_artifact_hash_invalid:{filename}")

    provenance = binding.get("provenance")
    if not isinstance(provenance, dict):
        raw_reasons.append("provenance_missing")
        provenance = {}
    expected_preflight_sha = recipe.get("token_preflight", {}).get(
        "file_sha256"
    )
    if (
        provenance.get("token_preflight_file_sha256")
        != expected_preflight_sha
    ):
        raw_reasons.append("token_preflight_file_sha256_mismatch")
    if not isinstance(
        provenance.get("token_preflight_path"), str
    ) or not provenance["token_preflight_path"].strip():
        raw_reasons.append("token_preflight_path_missing")

    claimed_sha = binding.get("binding_sha256")
    unsigned = dict(binding)
    unsigned.pop("binding_sha256", None)
    expected_sha = hashlib.sha256(_canonical(unsigned)).hexdigest()
    if claimed_sha != expected_sha:
        raw_reasons.append("binding_sha256_mismatch")

    raw_reasons = sorted(set(raw_reasons))
    if raw_reasons:
        return (
            {
                "status": "INVALID",
                "binding_sha256": claimed_sha,
                "python_path": target.get("python_path"),
                "python_version": target.get("python_version"),
                "gpu": target.get("gpu"),
            },
            ["training_runtime:" + reason for reason in raw_reasons],
            False,
        )

    return (
        {
            "status": "VERIFIED",
            "binding_sha256": claimed_sha,
            "python_path": target.get("python_path"),
            "python_version": target.get("python_version"),
            "base_path": target.get("base_path"),
            "gpu": target.get("gpu"),
            "vram_mib": target.get("vram_mib"),
            "driver": target.get("driver"),
            "cuda_runtime": target.get("cuda_runtime"),
            "cost_class": target.get("cost_class"),
            "packages": packages,
            "base_artifacts": artifacts,
        },
        [],
        True,
    )


def _git_blob_sha(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def _working_tree_git_blob_sha(root, path) -> str:
    import subprocess

    if not (root / ".git").exists():
        return _git_blob_sha(path.read_bytes())
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return _git_blob_sha(path.read_bytes())
    result = subprocess.run(
        ["git", "-C", str(root), "hash-object", f"--path={relative}", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    observed = result.stdout.strip().lower()
    if result.returncode == 0 and len(observed) == 40 and all(
        ch in "0123456789abcdef" for ch in observed
    ):
        return observed
    return _git_blob_sha(path.read_bytes())


def _execution_spec_contract_reasons(contract, spec):
    reasons = []
    recipe = contract.get("training_recipe", {})
    trainer = spec.get("trainer", {}) if isinstance(spec, dict) else {}
    quant = spec.get("quantization", {}) if isinstance(spec, dict) else {}
    model_load = spec.get("model_load", {}) if isinstance(spec, dict) else {}
    lora = spec.get("lora", {}) if isinstance(spec, dict) else {}
    if spec.get("schema") != "V10_QWEN35_TRAINING_EXECUTION_SPEC_V1":
        reasons.append("execution_spec_schema_mismatch")
    if spec.get("effect") != "TRAIN_ONE_FRESH_QLORA_ADAPTER":
        reasons.append("execution_spec_effect_mismatch")

    for field in (
        "method",
        "seed",
        "epochs",
        "learning_rate",
        "lr_scheduler_type",
        "warmup_optimizer_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "overflow_policy",
        "optimizer",
        "completion_only_loss",
        "packing",
        "shuffle_dataset",
        "gradient_checkpointing",
        "validation_role",
    ):
        if trainer.get(field) != recipe.get(field):
            reasons.append(f"execution_spec_recipe_mismatch:{field}")

    for field in (
        "load_in_4bit",
        "type",
        "double_quant",
        "compute_dtype",
    ):
        if quant.get(field) != recipe.get("quantization", {}).get(field):
            reasons.append(f"execution_spec_quantization_mismatch:{field}")

    for field in ("r", "alpha", "dropout", "target_modules"):
        if lora.get(field) != recipe.get("lora", {}).get(field):
            reasons.append(f"execution_spec_lora_mismatch:{field}")

    exact_trainer = {
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing_use_reentrant": False,
        "logging_steps": 10,
        "save_strategy": "no",
        "eval_strategy": "no",
        "report_to": "none",
    }
    for field, expected in exact_trainer.items():
        if trainer.get(field) != expected:
            reasons.append(f"execution_spec_exact_trainer_mismatch:{field}")

    expected_model_load = {
        "device_map": {"": 0},
        "dtype": "bfloat16",
        "use_cache": False,
        "prepare_model_for_kbit_training_use_gradient_checkpointing": True,
    }
    if model_load != expected_model_load:
        reasons.append("execution_spec_model_load_mismatch")
    if model_load.get("dtype") != quant.get("compute_dtype"):
        reasons.append("execution_spec_model_quant_dtype_mismatch")
    if lora.get("bias") != "none":
        reasons.append("execution_spec_lora_mismatch:bias")
    if lora.get("task_type") != "CAUSAL_LM":
        reasons.append("execution_spec_lora_mismatch:task_type")

    policy = spec.get("artifact_policy", {})
    base_model = contract.get("base_model", {})
    if policy.get("fresh_adapter_only") is not True:
        reasons.append("execution_spec_fresh_adapter_only_not_true")
    if policy.get("parent_adapter") is not None:
        reasons.append("execution_spec_parent_adapter_not_null")
    if base_model.get("parent_adapter") is not None:
        reasons.append("contract_parent_adapter_not_null")
    if base_model.get("fresh_adapter_required") is not True:
        reasons.append("contract_fresh_adapter_required_not_true")
    if policy.get("validation_checkpoint_selection") is not False:
        reasons.append("execution_spec_validation_selection_not_false")
    return reasons


def _training_execution_evidence(root, base, contract):
    path = base / TRAINING_EXECUTION_FILENAME
    if not path.exists():
        return (
            {
                "status": "ABSENT",
                "binding_sha256": None,
                "runner_git_blob_sha": None,
                "execution_spec_sha256": None,
            },
            ["training_execution_binding_missing"],
            False,
        )

    binding = _read_json(path)
    raw_reasons = []
    if (
        binding.get("schema")
        != "V10_QWEN35_TRAINING_EXECUTION_BINDING_V1"
    ):
        raw_reasons.append("schema_mismatch")
    if (
        binding.get("status")
        != "FROZEN_AUTHORIZABLE_EXECUTION_SUBJECT"
    ):
        raw_reasons.append("status_mismatch")

    runner = binding.get("runner")
    if not isinstance(runner, dict):
        raw_reasons.append("runner_binding_missing")
        runner = {}
    runner_rel = runner.get("path")
    if not isinstance(runner_rel, str) or not runner_rel.strip():
        raw_reasons.append("runner_path_missing")
        runner_path = None
    else:
        runner_path = root / runner_rel
        try:
            runner_path.resolve().relative_to(root.resolve())
        except ValueError:
            raw_reasons.append("runner_path_escapes_repo")
            runner_path = None
    if runner_path is not None:
        if not runner_path.is_file():
            raw_reasons.append("runner_file_missing")
        else:
            observed_blob = _working_tree_git_blob_sha(root, runner_path)
            if observed_blob != runner.get("git_blob_sha"):
                raw_reasons.append("runner_git_blob_sha_mismatch")

    preflight = binding.get("preflight_evaluator")
    if not isinstance(preflight, dict):
        raw_reasons.append("preflight_binding_missing")
        preflight = {}
    preflight_rel = preflight.get("path")
    if not isinstance(preflight_rel, str) or not preflight_rel.strip():
        raw_reasons.append("preflight_path_missing")
        preflight_path = None
    else:
        preflight_path = root / preflight_rel
        try:
            preflight_path.resolve().relative_to(root.resolve())
        except ValueError:
            raw_reasons.append("preflight_path_escapes_repo")
            preflight_path = None
    if preflight_path is not None:
        if not preflight_path.is_file():
            raw_reasons.append("preflight_file_missing")
        else:
            observed_preflight_blob = _working_tree_git_blob_sha(
                root, preflight_path
            )
            if observed_preflight_blob != preflight.get("git_blob_sha"):
                raw_reasons.append("preflight_git_blob_sha_mismatch")

    execution_spec = binding.get("execution_spec")
    if not isinstance(execution_spec, dict):
        raw_reasons.append("execution_spec_binding_missing")
        execution_spec = {}
    spec_rel = execution_spec.get("path")
    if not isinstance(spec_rel, str) or not spec_rel.strip():
        raw_reasons.append("execution_spec_path_missing")
        spec_path = None
    else:
        spec_path = root / spec_rel
        try:
            spec_path.resolve().relative_to(root.resolve())
        except ValueError:
            raw_reasons.append("execution_spec_path_escapes_repo")
            spec_path = None
    observed_spec_sha = None
    if spec_path is not None:
        if not spec_path.is_file():
            raw_reasons.append("execution_spec_file_missing")
        else:
            spec_value = _read_json(spec_path)
            observed_spec_sha = hashlib.sha256(
                _canonical(spec_value)
            ).hexdigest()
            if observed_spec_sha != execution_spec.get("sha256"):
                raw_reasons.append("execution_spec_sha256_mismatch")
            raw_reasons.extend(
                _execution_spec_contract_reasons(contract, spec_value)
            )

    claimed_sha = binding.get("binding_sha256")
    unsigned = dict(binding)
    unsigned.pop("binding_sha256", None)
    expected_sha = hashlib.sha256(_canonical(unsigned)).hexdigest()
    if claimed_sha != expected_sha:
        raw_reasons.append("binding_sha256_mismatch")

    raw_reasons = sorted(set(raw_reasons))
    if raw_reasons:
        return (
            {
                "status": "INVALID",
                "binding_sha256": claimed_sha,
                "runner_git_blob_sha": runner.get("git_blob_sha"),
                "preflight_git_blob_sha": preflight.get("git_blob_sha"),
                "execution_spec_sha256": execution_spec.get("sha256"),
            },
            ["training_execution:" + reason for reason in raw_reasons],
            False,
        )

    return (
        {
            "status": "VERIFIED",
            "binding_sha256": claimed_sha,
            "runner_path": runner.get("path"),
            "runner_git_blob_sha": runner.get("git_blob_sha"),
            "preflight_path": preflight.get("path"),
            "preflight_git_blob_sha": preflight.get("git_blob_sha"),
            "execution_spec_path": execution_spec.get("path"),
            "execution_spec_sha256": observed_spec_sha,
        },
        [],
        True,
    )


def _training_authority_evidence(base, contract, sealed, runtime, execution):
    path = base / TRAINING_AUTHORITY_FILENAME
    if not path.exists():
        return (
            {
                "status": "ABSENT",
                "authority_actor_id": None,
                "receipt_sha256": None,
            },
            [],
            False,
        )

    receipt = _read_json(path)
    raw_reasons = []

    expected_pairs = {
        "schema": "V10_QWEN35_TRAINING_AUTHORITY_V1",
        "status": "AUTHORIZED",
        "authority_actor_id": "PATRICK_USER_AUTHORITY",
        "authorization_source": "EXPLICIT_CURRENT_USER_INSTRUCTION",
        "experiment_id": contract.get("experiment_id"),
        "effect": "TRAIN_ONE_FRESH_QLORA_ADAPTER",
        "base_model_revision": contract.get("base_model", {}).get("revision"),
        "training_corpus_id": contract.get("source_subject", {}).get(
            "training_corpus_id"
        ),
        "train_sha256": contract.get("source_subject", {}).get("train_sha256"),
        "validation_sha256": contract.get("source_subject", {}).get(
            "validation_sha256"
        ),
    }
    for field, expected in expected_pairs.items():
        if receipt.get(field) != expected:
            raw_reasons.append(f"{field}_mismatch")

    expected_recipe_sha = hashlib.sha256(
        _canonical(contract.get("training_recipe"))
    ).hexdigest()
    if receipt.get("training_recipe_sha256") != expected_recipe_sha:
        raw_reasons.append("training_recipe_sha256_mismatch")

    if sealed.get("status") != "VERIFIED":
        raw_reasons.append("sealed_final_bank_not_verified")
    elif (
        receipt.get("sealed_final_bank_commitment_sha256")
        != sealed.get("commitment_sha256")
    ):
        raw_reasons.append("sealed_commitment_sha256_mismatch")

    if runtime.get("status") != "VERIFIED":
        raw_reasons.append("training_runtime_not_verified")
    elif (
        receipt.get("training_runtime_binding_sha256")
        != runtime.get("binding_sha256")
    ):
        raw_reasons.append("training_runtime_binding_sha256_mismatch")

    if execution.get("status") != "VERIFIED":
        raw_reasons.append("training_execution_not_verified")
    elif (
        receipt.get("training_execution_binding_sha256")
        != execution.get("binding_sha256")
    ):
        raw_reasons.append(
            "training_execution_binding_sha256_mismatch"
        )

    if receipt.get("max_training_runs") != 1:
        raw_reasons.append("max_training_runs_must_equal_1")
    output_namespace = receipt.get("output_namespace")
    if not isinstance(output_namespace, str) or not output_namespace.strip():
        raw_reasons.append("output_namespace_missing")
    if receipt.get("paid_compute_authorized") is not False:
        raw_reasons.append("paid_compute_must_remain_false")
    if receipt.get("merge_authorized") is not False:
        raw_reasons.append("merge_authorized_must_remain_false")
    if receipt.get("install_activate_deploy_authorized") is not False:
        raw_reasons.append("install_activate_deploy_must_remain_false")

    claimed_sha = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    expected_sha = hashlib.sha256(_canonical(unsigned)).hexdigest()
    if claimed_sha != expected_sha:
        raw_reasons.append("receipt_sha256_mismatch")

    raw_reasons = sorted(set(raw_reasons))
    if raw_reasons:
        return (
            {
                "status": "INVALID",
                "authority_actor_id": receipt.get("authority_actor_id"),
                "receipt_sha256": claimed_sha,
                "effect": receipt.get("effect"),
                "output_namespace": output_namespace,
            },
            ["training_authority:" + reason for reason in raw_reasons],
            False,
        )

    return (
        {
            "status": "VERIFIED",
            "authority_actor_id": receipt.get("authority_actor_id"),
            "receipt_sha256": claimed_sha,
            "effect": receipt.get("effect"),
            "output_namespace": output_namespace,
            "max_training_runs": 1,
            "paid_compute_authorized": False,
        },
        [],
        True,
    )


def assess_v10_experiment_state(repo_root):
    from pathlib import Path

    root = Path(repo_root)
    base = root / "successor" / "experiments"
    contract_v2 = base / "V10_QWEN35_EXPERIMENT_CONTRACT_V2.json"
    admission_v2 = base / "V10_QWEN35_FINAL_BANK_ADMISSION_V2.json"
    if contract_v2.exists():
        contract = _read_json(contract_v2)
        admission = _read_json(admission_v2)
        expected_contract_schema = "VERA_SUCCESSOR_V10_QWEN35_EXPERIMENT_CONTRACT_V2"
        expected_admission_schema = "V10_QWEN35_FINAL_BANK_ADMISSION_V2"
    else:
        contract = _read_json(base / "V10_QWEN35_EXPERIMENT_CONTRACT_V1.json")
        admission = _read_json(base / "V10_QWEN35_FINAL_BANK_ADMISSION_V1.json")
        expected_contract_schema = "VERA_SUCCESSOR_V10_QWEN35_EXPERIMENT_CONTRACT_V1"
        expected_admission_schema = "V10_QWEN35_FINAL_BANK_ADMISSION_V1"
    exclusion = _read_json(base / "V10_QWEN35_EXCLUSION_REGISTRY_V1.json")

    reasons = []
    if contract.get("schema") != expected_contract_schema:
        reasons.append("invalid_experiment_contract_schema")
    if admission.get("schema") != expected_admission_schema:
        reasons.append("invalid_final_bank_admission_schema")
    if exclusion.get("schema") != "V10_QWEN35_EXCLUSION_REGISTRY_V1":
        reasons.append("invalid_exclusion_registry_schema")
    if exclusion.get("status") != "FROZEN_EXCLUSION_IDENTITIES":
        reasons.append("exclusion_registry_not_frozen")

    preconditions = contract.get("blocking_preconditions")
    if not isinstance(preconditions, dict):
        reasons.append("blocking_preconditions_missing")
        preconditions = {}

    sealed, sealed_reasons, derived_preconditions = (
        _sealed_commitment_evidence(base, preconditions)
    )
    reasons.extend(sealed_reasons)
    sealed_verified = sealed["status"] == "VERIFIED"
    training_runtime, runtime_reasons, runtime_verified = (
        _training_runtime_evidence(base, contract)
    )
    reasons.extend(runtime_reasons)
    training_execution, execution_reasons, execution_verified = (
        _training_execution_evidence(root, base, contract)
    )
    reasons.extend(execution_reasons)
    training_authority, authority_reasons, authority_verified = (
        _training_authority_evidence(
            base,
            contract,
            sealed,
            training_runtime,
            training_execution,
        )
    )
    reasons.extend(authority_reasons)

    if (
        admission.get("status") != "FINAL_BANK_ADMITTED_AND_FROZEN"
        and not sealed_verified
    ):
        reasons.append("final_bank_cases_not_admitted")

    effective_preconditions = {}
    for key, value in sorted(preconditions.items()):
        if key == "patrick_exact_weight_change_authority":
            effective_value = authority_verified
        elif key == "exact_training_runtime_versions_bound":
            effective_value = runtime_verified
        else:
            effective_value = derived_preconditions.get(key, value)
        effective_preconditions[key] = effective_value
        if effective_value is not True:
            reasons.append(key)

    if (
        effective_preconditions.get("fresh_evaluation_bank_frozen") is True
        and not sealed_verified
    ):
        bank = contract.get("evaluation_bank", {})
        required = ("behavioral", "adversarial", "retention")
        if any(
            not _valid_sha256(bank.get(lane, {}).get("sha256"))
            for lane in required
        ):
            reasons.append("fresh_evaluation_bank_hashes_missing")

    reasons = sorted(set(reasons))
    ready = not reasons
    return {
        "schema": "V10_QWEN35_PREFLIGHT_V1",
        "status": "READY_PRECONDITIONS" if ready else "HOLD",
        "training_allowed": ready,
        "reasons": reasons,
        "derived_preconditions": derived_preconditions,
        "sealed_final_bank": sealed,
        "training_runtime": training_runtime,
        "training_execution": training_execution,
        "training_authority": training_authority,
        "effect": "READ_ONLY_PREFLIGHT_NO_WEIGHT_CHANGE",
    }


def _main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--v10-preflight-root")
    args = parser.parse_args(argv)
    if args.v10_preflight_root is None:
        return 0
    try:
        result = assess_v10_experiment_state(args.v10_preflight_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "schema": "V10_QWEN35_PREFLIGHT_V1",
            "status": "HOLD",
            "training_allowed": False,
            "reasons": ["preflight_input_error:" + str(exc)],
            "derived_preconditions": {},
            "sealed_final_bank": {
                "status": "ERROR",
                "bank_id": None,
                "commitment_sha256": None,
            },
            "training_runtime": {
                "status": "ERROR",
                "binding_sha256": None,
            },
            "training_execution": {
                "status": "ERROR",
                "binding_sha256": None,
                "runner_git_blob_sha": None,
                "execution_spec_sha256": None,
            },
            "training_authority": {
                "status": "ERROR",
                "authority_actor_id": None,
                "receipt_sha256": None,
            },
            "effect": "READ_ONLY_PREFLIGHT_NO_WEIGHT_CHANGE",
        }
    print(json.dumps(result, sort_keys=True))
    return 0 if result["training_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(_main())
