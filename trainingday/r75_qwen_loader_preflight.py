"""Read-only Qwen3.5-4B R18 loader declaration preflight.

The official pinned base *declares* the full multimodal configuration and
``Qwen3_5ForConditionalGeneration``. Both R18 train arms currently request
``Qwen3_5ForCausalLM``. This check binds the local config bytes to an
operator-provided independent SHA-256 pin and the source tree to explicit
SHA-256 digests; it DOES NOT import Transformers/torch, load weights, validate
the provider of that independent pin, or infer runtime failure/success.

Source: https://huggingface.co/Qwen/Qwen3.5-4B/blob/
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json
Relevant class/config distinctions: https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5
"""

from __future__ import annotations

import argparse
import ast
import hashlib
from importlib import metadata
import json
import re
from pathlib import Path


SCHEMA = "VERA_TRAININGDAY_R75_QWEN_LOADER_PREFLIGHT_V1"
MODEL_ID = "Qwen/Qwen3.5-4B"
PINNED_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
RUNNER_LOADER = "Qwen3_5ForCausalLM"
OFFICIAL_DECLARATION = "Qwen3_5ForConditionalGeneration"
RUNNERS = ("r18_mezo", "r18_qlora")
MAX_BINDING_BYTES = 131072
MAX_CONFIG_BYTES = 65536
MAX_SOURCE_BYTES = 1000000
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class LoaderPreflightError(ValueError):
    """A bounded local source input failed to establish its claimed binding."""


def _string_mapping(value: object, label: str) -> dict:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise LoaderPreflightError(f"{label}: expected string-key dictionary")
    return value


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise LoaderPreflightError("duplicate JSON field")
        result[key] = value
    return result


def _bad_constant(_value: str) -> None:
    raise LoaderPreflightError("nonfinite JSON numeric constant")


def _regular_bytes(path: Path, size_limit: int) -> bytes:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > size_limit:
            raise LoaderPreflightError("missing, symlinked, or oversized source file")
        data = path.read_bytes()
    except OSError as exc:
        raise LoaderPreflightError("source file cannot be read") from exc
    if len(data) > size_limit:
        raise LoaderPreflightError("source file exceeds size limit")
    return data


def _json_dict(data: bytes, label: str) -> dict:
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_pairs,
                           parse_constant=_bad_constant)
    except (UnicodeError, TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise LoaderPreflightError(f"{label}: invalid UTF-8 JSON") from exc
    return _string_mapping(value, label)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_loader(path: Path) -> tuple[str, str]:
    data = _regular_bytes(path, MAX_SOURCE_BYTES)
    try:
        tree = ast.parse(data, filename=path.name)
    except (SyntaxError, UnicodeError, ValueError, RecursionError) as exc:
        raise LoaderPreflightError("runner source cannot be parsed") from exc

    # This is a deliberately narrow *source declaration* check. AST Name.id
    # alone is not an imported class identity: an alias can hide a different
    # Transformers model, and any later local binding can shadow an import.
    # Require one literal, unaliased import in an unconditional lexical suite.
    parents = {id(child): node for node in ast.walk(tree)
               for child in ast.iter_child_nodes(node)}

    def lexical_scope(node: ast.AST) -> ast.AST:
        parent = parents.get(id(node))
        while parent is not None and not isinstance(
                parent, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parent = parents.get(id(parent))
        return parent if parent is not None else tree

    imports: list[ast.ImportFrom] = []
    transformers_symbols: set[str] = set()
    peft_symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == RUNNER_LOADER and isinstance(
                node.ctx, (ast.Store, ast.Del)):
            raise LoaderPreflightError("runner loader name has a local rebinding")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and node.name == RUNNER_LOADER:
            raise LoaderPreflightError("runner loader name is locally defined")
        if isinstance(node, ast.arg) and node.arg == RUNNER_LOADER:
            raise LoaderPreflightError("runner loader name is a function parameter")
        if isinstance(node, ast.ExceptHandler) and node.name == RUNNER_LOADER:
            raise LoaderPreflightError("runner loader name is an exception binding")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if isinstance(node, ast.ImportFrom) and alias.name == "*":
                    raise LoaderPreflightError("wildcard import cannot preserve loader provenance")
                bound_name = (alias.asname or (alias.name.split(".")[0]
                                             if isinstance(node, ast.Import) else alias.name))
                if isinstance(node, ast.ImportFrom) and node.module == "transformers":
                    if node.level == 0:
                        transformers_symbols.add(bound_name)
                    if (alias.name.startswith(("Qwen3_5For", "AutoModel"))
                            and (alias.name != RUNNER_LOADER or alias.asname is not None
                                 or node.level != 0)):
                        raise LoaderPreflightError("unexpected source loader transformers model import")
                if (isinstance(node, ast.ImportFrom) and node.module == "peft"
                        and node.level == 0 and alias.name == "PeftModel"
                        and alias.asname is None):
                    peft_symbols.add("PeftModel")
                if bound_name != RUNNER_LOADER:
                    continue
                if (not isinstance(node, ast.ImportFrom)
                        or node.module != "transformers"
                        or node.level != 0
                        or alias.name != RUNNER_LOADER
                        or alias.asname is not None):
                    raise LoaderPreflightError("runner loader import is disguised or foreign")
                scope = lexical_scope(node)
                if (not isinstance(scope, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef))
                        or node not in scope.body):
                    raise LoaderPreflightError("runner loader import is conditional or nested")
                imports.append(node)

    if len(imports) != 1:
        raise LoaderPreflightError("source loader requires one unaliased transformers import")

    model_loads = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_pretrained"):
            continue
        if not isinstance(node.func.value, ast.Name):
            raise LoaderPreflightError("unexpected source loader nested pretrained call")
        name = node.func.value.id
        if name == "AutoTokenizer":
            local_flags = [kw.value for kw in node.keywords if kw.arg == "local_files_only"]
            if (name not in transformers_symbols or len(local_flags) != 1
                    or not isinstance(local_flags[0], ast.Constant)
                    or local_flags[0].value is not True):
                raise LoaderPreflightError("tokenizer pretrained load is not statically offline")
            continue
        if name == "PeftModel" and name in peft_symbols:
            # An adapter load is not a second base-model load. Its bytes and
            # path require separate custody; this AST check cannot attest them.
            continue
        if name != RUNNER_LOADER:
            raise LoaderPreflightError("unexpected model source loader")
        call_scope = lexical_scope(node)
        declaration_scope = lexical_scope(imports[0])
        direct_stmt = parents.get(id(node))
        if (not isinstance(call_scope, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef))
                or not isinstance(direct_stmt, (ast.Assign, ast.AnnAssign, ast.Return, ast.Expr))
                or getattr(direct_stmt, "value", None) is not node
                or direct_stmt not in call_scope.body
                or declaration_scope not in (call_scope, tree)
                or imports[0].lineno >= node.lineno):
            raise LoaderPreflightError("runner loader call is conditional, nested, or unbound")
        local_flags = [kw.value for kw in node.keywords if kw.arg == "local_files_only"]
        if len(local_flags) != 1 or not isinstance(local_flags[0], ast.Constant) or local_flags[0].value is not True:
            raise LoaderPreflightError("model loader does not statically require local_files_only=True")
        model_loads.append(name)

    if model_loads != [RUNNER_LOADER]:
        raise LoaderPreflightError("R18 source loader is missing, changed, or ambiguous")
    return model_loads[0], _sha256(data)


def audit(base_binding: Path, *, expected_config_sha256: str, source_root: Path) -> dict:
    """Audit local snapshot/source declarations, never touch a host's GPU.

    The independent hash comes from the CALLER, not the producer's binding.
    The caller must authenticate that hash separately; a passing comparison
    cannot authenticate the pin's provenance.
    """
    if (type(expected_config_sha256) is not str
            or SHA256_RE.fullmatch(expected_config_sha256) is None):
        raise LoaderPreflightError("independent config digest pin must be 64 lower hex")

    binding_data = _regular_bytes(Path(base_binding), MAX_BINDING_BYTES)
    binding = _json_dict(binding_data, "base binding")
    for key, required in (
        ("schema", "TRAININGDAY_PINNED_BASE_V1"),
        ("model_id", MODEL_ID),
        ("revision", PINNED_REVISION),
        ("loader_class", RUNNER_LOADER),
    ):
        if type(binding.get(key)) is not str or binding[key] != required:
            raise LoaderPreflightError(f"base binding {key} is not the pinned R18 value")
    declared_files = _string_mapping(binding.get("files"), "base binding files")
    declared_sha = declared_files.get("config.json")
    if type(declared_sha) is not str or SHA256_RE.fullmatch(declared_sha) is None:
        raise LoaderPreflightError("base binding lacks valid config.json hash")
    snapshot_text = binding.get("snapshot_path")
    if type(snapshot_text) is not str or not snapshot_text:
        raise LoaderPreflightError("base snapshot path is missing")
    snapshot = Path(snapshot_text)
    if not snapshot.is_absolute() or snapshot.is_symlink() or not snapshot.is_dir():
        raise LoaderPreflightError("base snapshot must be an existing nonsymlink absolute directory")
    config_data = _regular_bytes(snapshot / "config.json", MAX_CONFIG_BYTES)
    actual_sha = _sha256(config_data)
    if actual_sha != declared_sha:
        raise LoaderPreflightError("base binding config digest mismatches the local bytes")
    if actual_sha != expected_config_sha256:
        raise LoaderPreflightError("independent config digest mismatches the local bytes")
    config = _json_dict(config_data, "pinned config")
    architecture = config.get("architectures")
    if (type(architecture) is not list or architecture != [OFFICIAL_DECLARATION]
            or type(config.get("model_type")) is not str or config["model_type"] != "qwen3_5"):
        raise LoaderPreflightError("pinned official architecture/model_type was not observed")
    text_config = _string_mapping(config.get("text_config"), "text_config")
    vision_config = _string_mapping(config.get("vision_config"), "vision_config")
    if (type(text_config.get("model_type")) is not str
            or text_config["model_type"] != "qwen3_5_text"
            or type(vision_config.get("model_type")) is not str
            or vision_config["model_type"] != "qwen3_5"):
        raise LoaderPreflightError("pinned nested text/vision configuration was not observed")

    root = Path(source_root)
    if not root.is_dir() or root.is_symlink():
        raise LoaderPreflightError("runner source root missing or symlinked")
    loaders: dict[str, str] = {}
    sources: dict[str, str] = {}
    for name in RUNNERS:
        loaders[name], sources[name] = _source_loader(root / "trainingday" / f"{name}.py")

    try:
        installed_version = metadata.version("transformers")
    except metadata.PackageNotFoundError:
        installed_version = None
    return {
        "schema": SCHEMA,
        "status": "DECLARED_MISMATCH_UNKNOWN_RUNTIME",
        "audit_scope": "SOURCE_INSPECTION_AND_LOCAL_SHA256_ONLY",
        "model_id": MODEL_ID,
        "base_revision": PINNED_REVISION,
        "declared_architecture": OFFICIAL_DECLARATION,
        "declared_model_type": "qwen3_5",
        "declared_text_model_type": "qwen3_5_text",
        "consumer_expected_config_family": "Qwen3_5TextConfig",
        "declared_config_family": "Qwen3_5Config",
        "runner_loader_classes": loaders,
        "source_sha256": sources,
        "binding_sha256": _sha256(binding_data),
        "operator_config_sha256": expected_config_sha256,
        "transformers_distribution_version": installed_version,
        "transformers_classes_imported": False,
        "runtime_loader_verified": False,
        "weights_loaded": False,
        "training_authorized": False,
        "external_hash_authority_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Qwen3.5-4B R18 loader/source audit")
    parser.add_argument("--base-binding", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True,
                        help="separately authenticated pinned official config SHA256")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    try:
        result = audit(args.base_binding, expected_config_sha256=args.expected_config_sha256,
                       source_root=args.source_root)
    except (LoaderPreflightError, OSError) as exc:
        parser.exit(2, f"HOLD: {exc}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
