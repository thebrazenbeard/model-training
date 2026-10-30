"""Train-only, source-bound supervision for a *restricted* Python program subset.

No source module, function, dynamic code, imports, or subprocess is executed.
Two distinct AST traversal/evaluation paths agree on each emitted integer
answer. Their arithmetic primitives are shared, so agreement is not an
independent oracle for those primitives. This is not verification of full
CPython behavior or coding ability.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import itertools
import json
import re
from pathlib import Path

from trainingday.admission import AdmissionError, _hash_file, _require


SCHEMA = "vera.trainingday.program_supervision.v1"
ORACLE = "RESTRICTED_PYTHON_AST_CROSS_PATH_CHECK_V1"
SOURCE_SCHEMA = "vera.trainingday.corpus.v1"
MAX_INT = 1_000_000
MAX_FUNCTION_CHARS = 1600
ALLOWED_OPS = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
ALLOWED_CMPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
INPUTS = (-7, -3, -1, 0, 1, 3, 7)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bounded(value: int) -> int:
    if type(value) is not int or abs(value) > MAX_INT:
        raise ValueError("non-integer or unbounded computation")
    return value


def _operand(left: int, right: int, op: ast.operator) -> int:
    if isinstance(op, ast.Add):
        return _bounded(left + right)
    if isinstance(op, ast.Sub):
        return _bounded(left - right)
    if isinstance(op, ast.Mult):
        return _bounded(left * right)
    if isinstance(op, ast.FloorDiv):
        return _bounded(left // right)
    if isinstance(op, ast.Mod):
        return _bounded(left % right)
    raise ValueError("unsupported operator")


def _compare(left: int, right: int, op: ast.cmpop) -> bool:
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.Gt):
        return left > right
    if isinstance(op, ast.GtE):
        return left >= right
    raise ValueError("unsupported comparison")


def _validate_expression(node: ast.expr, names: set[str], depth: int = 0) -> None:
    if depth > 10:
        raise ValueError("expression too deep")
    if isinstance(node, ast.Constant) and type(node.value) is int:
        _bounded(node.value)
    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in names:
        return
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        _validate_expression(node.operand, names, depth + 1)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ALLOWED_OPS):
        _validate_expression(node.left, names, depth + 1)
        _validate_expression(node.right, names, depth + 1)
    elif isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1 \
            and isinstance(node.ops[0], ALLOWED_CMPS):
        _validate_expression(node.left, names, depth + 1)
        _validate_expression(node.comparators[0], names, depth + 1)
    else:
        raise ValueError("expression not in admitted pure-integer grammar")


def _validate_statements(body: list[ast.stmt], names: set[str], depth: int = 0) -> None:
    if depth > 3 or not body or len(body) > 8:
        raise ValueError("unsupported block size/depth")
    local_names = names.copy()
    for node in body:
        if isinstance(node, ast.Return) and node.value is not None:
            _validate_expression(node.value, local_names)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            if target.startswith("_") or not target.isidentifier():
                raise ValueError("unsupported variable")
            _validate_expression(node.value, local_names)
            local_names.add(target)
        elif isinstance(node, ast.If) and node.orelse:
            _validate_expression(node.test, local_names)
            _validate_statements(node.body, local_names, depth + 1)
            _validate_statements(node.orelse, local_names, depth + 1)
        else:
            raise ValueError("statement not in admitted pure-integer grammar")


def _direct_expr(node: ast.expr, env: dict[str, int]) -> int | bool:
    if isinstance(node, ast.Constant):
        return _bounded(node.value)
    if isinstance(node, ast.Name):
        return _bounded(env[node.id])
    if isinstance(node, ast.UnaryOp):
        value = _direct_expr(node.operand, env)
        return _bounded(value if isinstance(node.op, ast.UAdd) else -value)
    if isinstance(node, ast.BinOp):
        return _operand(_direct_expr(node.left, env), _direct_expr(node.right, env), node.op)
    if isinstance(node, ast.Compare):
        return _compare(_direct_expr(node.left, env), _direct_expr(node.comparators[0], env), node.ops[0])
    raise ValueError("unreachable expression")


def _stack_expr(node: ast.expr, env: dict[str, int]) -> int | bool:
    """Independent nonrecursive postfix-machine expression implementation."""
    instructions = []
    todo = [(node, False)]
    while todo:
        item, visited = todo.pop()
        if not visited:
            if isinstance(item, ast.Constant):
                instructions.append(("const", item.value))
            elif isinstance(item, ast.Name):
                instructions.append(("name", item.id))
            elif isinstance(item, ast.UnaryOp):
                todo.append((item, True))
                todo.append((item.operand, False))
            elif isinstance(item, (ast.BinOp, ast.Compare)):
                todo.append((item, True))
                rhs = item.right if isinstance(item, ast.BinOp) else item.comparators[0]
                todo.append((rhs, False))
                todo.append((item.left, False))
            else:
                raise ValueError("unreachable stack expression")
        else:
            if isinstance(item, ast.UnaryOp):
                instructions.append(("unary", type(item.op)))
            elif isinstance(item, ast.BinOp):
                instructions.append(("binary", type(item.op)))
            else:
                instructions.append(("compare", type(item.ops[0])))
    stack = []
    for kind, value in instructions:
        if kind == "const":
            stack.append(_bounded(value))
        elif kind == "name":
            stack.append(_bounded(env[value]))
        elif kind == "unary":
            val = stack.pop()
            stack.append(_bounded(val if value is ast.UAdd else -val))
        else:
            right, left = stack.pop(), stack.pop()
            stack.append(_operand(left, right, value()) if kind == "binary" else _compare(left, right, value()))
    if len(stack) != 1:
        raise ValueError("postfix stack invalid")
    return stack[0]


def _run_statements(body: list[ast.stmt], env: dict[str, int], expr) -> int:
    for node in body:
        if isinstance(node, ast.Return):
            result = expr(node.value, env)
            if type(result) is not int:
                raise ValueError("non-integer return")
            return _bounded(result)
        if isinstance(node, ast.Assign):
            env[node.targets[0].id] = _bounded(expr(node.value, env))
            continue
        if isinstance(node, ast.If):
            condition = expr(node.test, env)
            return _run_statements(node.body if condition else node.orelse, env, expr)
    raise ValueError("program path does not return")


def _candidate_functions(text: str):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef) or fn.decorator_list or getattr(fn, "type_params", []) \
                or fn.args.defaults or fn.args.kw_defaults or fn.args.vararg or fn.args.kwarg \
                or fn.args.kwonlyargs or fn.args.posonlyargs or not (1 <= len(fn.args.args) <= 2):
            continue
        names = [a.arg for a in fn.args.args]
        if len(set(names)) != len(names) or any(n.startswith("_") for n in names) \
                or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,50}", fn.name) is None:
            continue
        body = fn.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            body = body[1:]
        try:
            _validate_statements(body, set(names))
        except ValueError:
            continue
        # Canonicalize to an annotation/docstring/comment-free function. All
        # admitted semantics are retained, without source comment instructions.
        safe = copy.copy(fn)
        safe.args = copy.copy(fn.args)
        safe.args.args = [ast.arg(arg=name, annotation=None) for name in names]
        safe.body = body
        safe.returns = None
        safe.type_comment = None
        segment = ast.unparse(safe)
        if segment and len(segment) <= MAX_FUNCTION_CHARS:
            yield fn.name, names, body, segment


def derive_cases(row: dict, *, per_file: int = 3) -> list[dict]:
    """Derive restricted-semantics program examples from one MIT-marked code row.

    The original program is never imported or executed. Two AST execution
    paths share arithmetic primitives; this is not independent math-oracle
    verification or independent verification of the source Git blob.
    """
    _require(type(per_file) is int and 1 <= per_file <= 16, "invalid per-file cap")
    _require(row.get("kind") == "code" and row.get("verification") == "SOURCE_ONLY_NOT_FUNCTIONALLY_TESTED"
             and row.get("license") == "MIT", "not a licensed source-only code row")
    path = row.get("source_path")
    text = row.get("text")
    _require(isinstance(path, str) and path.endswith(".py") and isinstance(text, str),
             "only Python source rows admitted")
    digest = _sha(text.encode("utf-8"))
    _require(row.get("content_sha256") == digest and row.get("id") == "code-" + digest,
             "code source hash/id mismatch")
    source_repo = row.get("source_repo")
    revision = row.get("source_revision")
    license_sha = row.get("license_sha256")
    _require(all(isinstance(v, str) for v in (source_repo, revision, license_sha))
             and re.fullmatch(r"[0-9a-f]{40}", revision) is not None
             and re.fullmatch(r"[0-9a-f]{64}", license_sha) is not None,
             "incomplete code provenance")
    cases = []
    for name, params, statements, snippet in _candidate_functions(text):
        pairs = itertools.product(INPUTS, repeat=len(params))
        for args in pairs:
            env = dict(zip(params, args))
            try:
                a = _run_statements(statements, env.copy(), _direct_expr)
                b = _run_statements(statements, env.copy(), _stack_expr)
            except (ValueError, ArithmeticError, KeyError, IndexError, OverflowError):
                continue
            if a != b:
                raise AdmissionError("two restricted AST interpreters disagree")
            call = f"{name}({', '.join(map(str, args))})"
            prompt = f"For the restricted integer Python function below, give only the exact integer returned by {call}.\n```python\n{snippet}\n```"
            payload = f"{digest}\n{call}\n{a}\n{ORACLE}".encode()
            cases.append({"id": "program-" + _sha(payload), "family": "python_pure_integer_subset",
                          "prompt": prompt, "answer": str(a), "verification": ORACLE,
                          "license": "MIT", "license_sha256": license_sha,
                          "source_repo": source_repo, "source_revision": revision,
                          "source_path": path, "source_content_sha256": digest,
                          "function": name, "call": call})
            if len(cases) >= per_file:
                return cases
    return cases


def _data_file(root: Path, relative: str, split: str) -> Path:
    _require(isinstance(relative, str) and re.fullmatch(
        rf"{re.escape(split)}/[A-Za-z0-9_.-]+\.jsonl", relative) is not None,
        "unconfined corpus shard path")
    target = root / relative
    _require(not root.is_symlink() and not target.parent.is_symlink()
             and target.is_file() and not target.is_symlink(), "missing/symlinked shard")
    return target


def build_verified_code_corpus(manifest_path: Path, output_dir: Path, *,
                               expected_manifest_sha256: str, producer_git_head: str,
                               max_cases: int = 1024, per_file: int = 3) -> dict:
    """Create a new JSONL derivative with copied MIT notices and exact custody.

    All source train+eval shard hashes checked before train code is interpreted;
    evaluation JSONL *contents* are never opened or parsed by this routine.
    """
    manifest_path = Path(manifest_path)
    output = Path(output_dir)
    _require(manifest_path.is_file() and not manifest_path.is_symlink(), "manifest absent")
    _require(re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is not None
             and _hash_file(manifest_path) == expected_manifest_sha256,
             "manifest digest does not match the pinned expectation")
    _require(re.fullmatch(r"[0-9a-f]{40}", producer_git_head) is not None,
             "producer head must be an exact commit")
    _require(type(max_cases) is int and 1 <= max_cases <= 100_000, "invalid case cap")
    _require(not output.exists() and not output.is_symlink() and output.parent.is_dir(),
             "output must be a new path with existing parent")
    stage = output.with_name(output.name + ".building")
    _require(not stage.exists() and not stage.is_symlink(), "staging path already exists")
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == SOURCE_SCHEMA
             and manifest.get("state") == "CORPUS_BUILT_NO_MODEL_TRAINING", "source manifest invalid")
    source_map = {}
    for source in manifest.get("sources", []):
        _require(source.get("license") == "MIT" and source.get("repo") not in source_map
                 and re.fullmatch(r"[0-9a-f]{40}", source.get("revision", "")) is not None,
                 "source identity invalid")
        source_map[source["repo"]] = source
    heldout = manifest.get("heldout_repo")
    _require(heldout in source_map and len(source_map) >= 2, "independent heldout repo absent")
    notices = []
    for item in manifest.get("licenses", []):
        repo = item.get("repo")
        _require(repo in source_map and item.get("license") == "MIT"
                 and item.get("path") == "licenses/" + repo.replace("/", "__") + ".txt",
                 "unbound MIT notice")
        license_path = root / item["path"]
        _require(license_path.is_file() and not license_path.is_symlink()
                 and not license_path.parent.is_symlink()
                 and _hash_file(license_path) == item["sha256"]
                 and item["sha256"] == source_map[repo].get("license_sha256"),
                 "MIT license notice missing/tampered")
        notices.append(item)
    _require({x["repo"] for x in notices} == set(source_map), "MIT notices incomplete")
    shards = manifest.get("shards")
    _require(isinstance(shards, list) and shards, "missing source shards")
    train_shards = []
    counts = {"train": 0, "eval": 0}
    paths = set()
    for shard in shards:
        split = shard.get("split")
        relative = shard.get("path")
        _require(split in counts and relative not in paths
                 and type(shard.get("count")) is int and shard["count"] > 0
                 and type(shard.get("bytes")) is int and shard["bytes"] > 0
                 and re.fullmatch(r"[0-9a-f]{64}", shard.get("sha256", "")) is not None,
                 "invalid or repeated shard metadata")
        paths.add(relative)
        path = _data_file(root, relative, split)
        _require(path.stat().st_size == shard["bytes"] and _hash_file(path) == shard["sha256"],
                 "source shard bytes/hash mismatch")
        counts[split] += shard["count"]
        if split == "train":
            train_shards.append(shard)
    _require(counts == manifest.get("counts") and all(counts.values()), "source count mismatch")
    _require({p.relative_to(root).as_posix() for p in (root / "train").glob("*.jsonl")}
             == {x["path"] for x in train_shards}, "undeclared training shard")

    stage.mkdir()
    (stage / "train").mkdir()
    (stage / "licenses").mkdir()
    for item in notices:
        (stage / item["path"]).write_bytes((root / item["path"]).read_bytes())
    target = stage / "train" / "000000.jsonl"
    sha = hashlib.sha256()
    count = files = 0
    with target.open("wb") as output_stream:
        for shard in sorted(train_shards, key=lambda x: x["path"]):
            file = _data_file(root, shard["path"], "train")
            checksum = hashlib.sha256()
            observed = 0
            with file.open("rb") as stream:
                for raw in stream:
                    checksum.update(raw)
                    observed += 1
                    row = json.loads(raw.decode("utf-8"))
                    if row.get("kind") != "code":
                        continue
                    repo = row.get("source_repo")
                    _require(repo in source_map and repo != heldout
                             and row.get("license_sha256") == source_map[repo]["license_sha256"]
                             and row.get("source_revision") == source_map[repo]["revision"],
                             "heldout/unbound code in training source")
                    files += 1
                    if count >= max_cases:
                        continue
                    for case in derive_cases(row, per_file=per_file):
                        data = (json.dumps(case, sort_keys=True, separators=(",", ":")) + "\n").encode()
                        output_stream.write(data)
                        sha.update(data)
                        count += 1
                        if count == max_cases:
                            break
            _require(observed == shard["count"] and checksum.hexdigest() == shard["sha256"],
                     "source train shard changed during ingestion")
    _require(count > 0, "no safe supported Python functions in this source")
    report = {"schema": SCHEMA, "state": "DERIVED_TRAIN_ONLY_NO_WEIGHT_UPDATE",
              "oracle": ORACLE, "source_manifest_sha256": expected_manifest_sha256,
              "source_producer_git_head": producer_git_head,
              "source_eval_rows_never_parsed": counts["eval"],
              "source_train_code_files_seen": files, "derived_examples": count,
              "license_notices": notices,
              "shard": {"path": "train/000000.jsonl", "bytes": target.stat().st_size,
                        "sha256": sha.hexdigest(), "count": count},
              "scope": "RESTRICTED_PYTHON_STATIC_SEMANTICS_ONLY_NOT_CPYTHON_EXECUTION",
              "qualification_hold": "SOURCE_GIT_BLOB_AND_INDEPENDENT_ARITHMETIC_ORACLE_NOT_VERIFIED"}
    (stage / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
    stage.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description="Derive source-licensed restricted Python program labels")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--producer-git-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=1024)
    parser.add_argument("--per-file", type=int, default=3)
    args = parser.parse_args()
    report = build_verified_code_corpus(args.manifest, args.output,
                                        expected_manifest_sha256=args.manifest_sha256,
                                        producer_git_head=args.producer_git_head,
                                        max_cases=args.max_cases, per_file=args.per_file)
    print(json.dumps({"state": report["state"], "derived_examples": report["derived_examples"],
                      "source_code_files": report["source_train_code_files_seen"]}))


if __name__ == "__main__":
    main()
