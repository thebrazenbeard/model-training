"""Public synthetic tests. No actual model or owner system prompt is used."""

import json
import os
from pathlib import Path
import socket
import tempfile
import time
from threading import Thread
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest
from unittest.mock import patch

from trainingday.q5_conversation_probe import (
    CASES, LOCAL_Q5_FILE_SHA256, SEEDS, _client_api_key_file, _endpoint, _reply,
    execute, local_chat, main, plan, private_write,
)


def completed(text="Mock answer"):
    return {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": text}}]}


class Q5ProbeContractTests(unittest.TestCase):
    def test_windows_private_key_and_output_fail_closed_before_access(self):
        # Linux-hosted negative simulation, not Windows NTFS qualification.
        from trainingday import q5_conversation_probe as subject
        with tempfile.TemporaryDirectory() as tmp:
            key = Path(tmp) / "synthetic-key.txt"
            key.write_text("SYNTHETIC_NOT_AN_OWNER_KEY")
            folder = Path(tmp)
            with patch.object(subject.os, "name", "nt"):
                with self.assertRaisesRegex(ValueError, "Windows private-file"):
                    _client_api_key_file(key)
                with self.assertRaisesRegex(ValueError, "Windows private-file"):
                    private_write(folder, {"review_id": "synthetic"}, {})
            self.assertFalse(list(folder.glob("review_*.json")))

    def test_preflight_output_custody_before_model_or_key_access(self):
        from trainingday import q5_conversation_probe as subject
        from contextlib import redirect_stderr
        from io import StringIO
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "synthetic-system.txt"
            prompt.write_text("SYNTHETIC_PRIVATE_PROMPT")
            token = root / "synthetic-token.txt"
            token.write_text("SYNTHETIC_KEY")
            args = ["run", "--case", CASES[0][0], "--seed", str(SEEDS[0]),
                    "--endpoint", "http://127.0.0.1:11435/v1/chat/completions",
                    "--system-prompt-file", str(prompt),
                    "--client-api-key-file", str(token),
                    "--private-output-dir", str(root),
                    "--consent-to-local-inference"]
            with patch.object(subject, "_validate_private_output_dir",
                              side_effect=ValueError("PRIVATE_OUTPUT_TEST_BLOCK")):
                with patch.object(subject, "_client_api_key_file",
                                  side_effect=AssertionError("key must not be touched")):
                    with patch.object(subject, "execute",
                                      side_effect=AssertionError("model must not be called")):
                        with redirect_stderr(StringIO()) as stderr:
                            self.assertEqual(main(args), 2)
                        self.assertNotIn("SYNTHETIC_PRIVATE_PROMPT", stderr.getvalue())
                        self.assertNotIn("SYNTHETIC_KEY", stderr.getvalue())

    def test_windows_cli_run_blocks_before_key_prompt_and_model(self):
        from trainingday import q5_conversation_probe as subject
        from contextlib import redirect_stderr
        from io import StringIO
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "synthetic-system.txt"
            prompt.write_text("SYNTHETIC_SECRET_PROMPT")
            key = root / "synthetic-key.txt"
            key.write_text("SYNTHETIC_SECRET_BEARER")
            args = ["run", "--case", CASES[0][0], "--seed", str(SEEDS[0]),
                    "--endpoint", "http://127.0.0.1:11435/v1/chat/completions",
                    "--system-prompt-file", str(prompt),
                    "--client-api-key-file", str(key),
                    "--private-output-dir", str(root),
                    "--consent-to-local-inference"]
            native_path_class = type(root)
            stderr = StringIO()
            # Keep native Path construction valid while simulating Windows on
            # the Linux test host; no native Windows DACL claim is made.
            with patch.object(subject, "Path", native_path_class):
                with patch.object(subject.os, "name", "nt"):
                    with patch.object(subject, "_client_api_key_file",
                                      side_effect=AssertionError("key accessed")):
                        with patch.object(subject, "execute",
                                          side_effect=AssertionError("model called")):
                            with redirect_stderr(stderr):
                                self.assertEqual(main(args), 2)
            self.assertNotIn("SYNTHETIC_SECRET_BEARER", stderr.getvalue())
            self.assertNotIn("SYNTHETIC_SECRET_PROMPT", stderr.getvalue())
            self.assertFalse(list(root.glob("review_*.json")))

    def test_public_plan_is_12_by_3_and_not_inference(self):
        value = plan()
        self.assertEqual(value["case_count"], 12)
        self.assertEqual(value["planned_sessions"], 36)
        self.assertEqual(len(SEEDS), 3)
        self.assertEqual(len({x[0] for x in CASES}), 12)
        self.assertFalse(value["model_inference_performed"])
        self.assertFalse(value["independent_holdout"])
        self.assertEqual(len(value["manifest_sha256"]), 64)
        self.assertEqual(plan(), value)
        self.assertTrue(all(len(turns) >= 2 for _, turns in CASES))

    def test_normal_multi_turn_only_forwards_completed_assistant_text(self):
        sent = []
        def fake_transport(messages, seed, timeout):
            sent.append(json.loads(json.dumps(messages)))
            self.assertEqual(seed, SEEDS[0])
            self.assertGreater(timeout, 0)
            self.assertLessEqual(timeout, 30)
            return completed("Mock assistant turn " + str(len(sent)))
        review, custody = execute(CASES[0][0], SEEDS[0], "mock prompt", fake_transport)
        self.assertEqual(len(review["turns"]), 3)
        self.assertEqual(custody["complete_turn_count"], 3)
        self.assertEqual(sent[0][0], {"role": "system", "content": "mock prompt"})
        self.assertEqual([x["role"] for x in sent[0]], ["system", "user"])
        self.assertEqual([x["role"] for x in sent[1]], ["system", "user", "assistant", "user"])
        self.assertEqual([x["role"] for x in sent[2]],
                         ["system", "user", "assistant", "user", "assistant", "user"])
        self.assertFalse(custody["runtime_process_attested"])
        self.assertFalse(custody["loaded_model_attested"])
        self.assertFalse(custody["human_review_performed"])
        self.assertFalse(custody["semantic_persona_pass"])
        self.assertNotIn("seed", review)
        self.assertNotIn("model", review)
        self.assertEqual(custody["declared_model_file_sha256"], LOCAL_Q5_FILE_SHA256)

    def test_length_limited_never_replays_or_writes_partial(self):
        seen = []
        def fake(messages, seed, timeout):
            seen.append(messages)
            return {"choices": [{"finish_reason": "length", "message": {
                "role": "assistant", "content": "SECRET_PARTIAL_CANARY"}}]}
        review, receipt = execute(CASES[0][0], SEEDS[0], "mock", fake)
        self.assertEqual(len(seen), 1)
        self.assertEqual(review["turns"][0]["outcome"], "truncated")
        self.assertIsNone(review["turns"][0]["assistant"])
        self.assertEqual(receipt["complete_turn_count"], 0)
        self.assertNotIn("SECRET_PARTIAL_CANARY", json.dumps(review))

    def test_reject_forged_roles_tool_calls_missing_finish(self):
        variants = (
            {"choices": [{"finish_reason": "stop", "message": {"role": "system", "content": "fake"}}]},
            {"choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": "fake", "tool_calls": []}}]},
            {"choices": [{"message": {"role": "assistant", "content": "fake"}}]},
            {"choices": [{"finish_reason": "content_filter",
                          "message": {"role": "assistant", "content": "fake"}}]},
            {"choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": ""}}]},
            {"choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": "x" * 8001}}]},
        )
        for bad in variants:
            with self.subTest(bad=bad):
                self.assertEqual(_reply(bad), ("upstream_error", None))

    @unittest.skipIf(os.name == "nt", "Windows private I/O explicitly blocked")
    def test_private_output_is_no_overwrite_and_blinded(self):
        review, receipt = execute(CASES[0][0], SEEDS[0], "PRIVATE_TEST_PROMPT",
                                  lambda m, s, t: completed())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.chmod(root, 0o700)
            review_path, custody_path = private_write(root, review, receipt)
            self.assertEqual(json.loads(review_path.read_text())["review_id"],
                             json.loads(custody_path.read_text())["review_id"])
            self.assertNotIn("PRIVATE_TEST_PROMPT", review_path.read_text())
            self.assertNotIn("PRIVATE_TEST_PROMPT", custody_path.read_text())
            self.assertNotIn("seed", json.loads(review_path.read_text()))
            self.assertIn('"seed":17', custody_path.read_text())
            if os.name != "nt":
                self.assertEqual(review_path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(custody_path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                private_write(root, review, receipt)

    def test_refuse_public_repo_output_and_world_readable_parent(self):
        from trainingday import q5_conversation_probe as subject
        review, receipt = execute(CASES[0][0], SEEDS[0], "mock",
                                  lambda m, s, t: completed())
        # On Windows the global private-file refusal intentionally precedes
        # source-repository output validation; neither permits a write.
        reason = ("Windows private-file DACL enforcement unavailable"
                  if os.name == "nt" else "source repository")
        with self.assertRaisesRegex(ValueError, reason):
            private_write(Path(subject.__file__).resolve().parents[1], review, receipt)
        if os.name != "nt":
            with tempfile.TemporaryDirectory() as tmp:
                os.chmod(tmp, 0o755)
                with self.assertRaisesRegex(ValueError, "owner-only"):
                    private_write(Path(tmp), review, receipt)

    def test_endpoint_blocks_network_redirects_and_remote_addresses(self):
        bad = (
            "https://127.0.0.1:11435/v1/chat/completions",
            "http://localhost:11435/v1/chat/completions",
            "http://127.0.0.2:11435/v1/chat/completions",
            "http://192.168.1.2:11435/v1/chat/completions",
            "http://127.0.0.1:11435/another",
            "http://127.0.0.1:11435/v1/chat/completions?secret=1",
            "http://user:pass@127.0.0.1:11435/v1/chat/completions",
            "http://127.0.0.1:10/v1/chat/completions",
        )
        for url in bad:
            with self.subTest(url=url), self.assertRaises(ValueError):
                _endpoint(url)
        self.assertEqual(_endpoint("http://127.0.0.1:11435/v1/chat/completions"),
                         ("127.0.0.1", 11435, "/v1/chat/completions"))

    def test_cli_requires_explicit_model_opt_in_and_does_not_echo_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "private.txt"
            prompt.write_text("CANARY_SYSTEM_PROMPT_NOT_TO_LOG", encoding="utf-8")
            from contextlib import redirect_stderr
            from io import StringIO
            buf = StringIO()
            with redirect_stderr(buf):
                result = main(["run", "--case", CASES[0][0], "--seed", str(SEEDS[0]),
                               "--endpoint", "http://127.0.0.1:11435/v1/chat/completions",
                               "--system-prompt-file", str(prompt),
                               "--private-output-dir", str(root)])
            self.assertEqual(result, 2)
            self.assertNotIn("CANARY_SYSTEM_PROMPT_NOT_TO_LOG", buf.getvalue())
            self.assertFalse(list(root.glob("review_*.json")))

    def test_bounded_prompt_and_invalid_seed_fail(self):
        with self.assertRaises(ValueError):
            execute(CASES[0][0], 999, "mock", lambda m, s, t: completed())
        with self.assertRaises(ValueError):
            execute(CASES[0][0], SEEDS[0], "x" * 12001, lambda m, s, t: completed())
        with self.assertRaises(ValueError):
            execute("not_a_case", SEEDS[0], "mock", lambda m, s, t: completed())

    def test_unresponsive_transport_is_censored(self):
        def failed(messages, seed, timeout):
            raise socket.timeout("host state private")
        review, receipt = execute(CASES[1][0], SEEDS[0], "mock", failed)
        self.assertEqual(review["turns"][0]["outcome"], "timeout")
        self.assertIsNone(review["turns"][0]["assistant"])
        self.assertEqual(receipt["complete_turn_count"], 0)

    def test_actual_local_http_request_never_contains_scoring_intent(self):
        observed = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                observed.append((self.path, json.loads(self.rfile.read(length))))
                data = json.dumps(completed()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def log_message(self, *args):
                pass
        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                # Ephemeral ports are >= 1024 on normal OS allocation.
                response = local_chat(
                    "http://127.0.0.1:" + str(port) + "/v1/chat/completions",
                    [{"role": "system", "content": "SYNTHETIC"},
                     {"role": "user", "content": "public test"}], SEEDS[0],
                    api_key="SYNTHETIC_KEY")
            finally:
                server.shutdown()
                thread.join(timeout=2)
        self.assertEqual(_reply(response), ("complete", "Mock answer"))
        self.assertEqual(observed[0][0], "/v1/chat/completions")
        self.assertEqual(observed[0][1]["seed"], SEEDS[0])
        self.assertEqual(set(observed[0][1]),
                         {"messages", "temperature", "max_tokens", "seed", "stream"})
        # This is a raw llama.cpp probe, not the guarded web #30 schema:
        # web #30 requires a model selector and refuses temperature/seed.
        self.assertNotIn("model", observed[0][1])
        self.assertEqual(observed[0][1]["messages"][-1]["content"], "public test")
        self.assertNotIn("rubric", json.dumps(observed[0]))
        self.assertNotIn("expected", json.dumps(observed[0]))

    def test_local_chat_rejects_bearer_header_injection(self):
        for token in ("", "line\r\nX-Injected: yes", " leading-space", "x" * 513):
            with self.subTest(token=token[:16]), self.assertRaises(ValueError):
                local_chat("http://127.0.0.1:11435/v1/chat/completions",
                           [{"role": "user", "content": "public"}],
                           SEEDS[0], api_key=token)

    def test_dripping_http_body_is_bounded_by_absolute_wall_deadline(self):
        class Drip(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "60")
                self.end_headers()
                for _ in range(60):
                    try:
                        self.wfile.write(b"x")
                        self.wfile.flush()
                    except OSError:
                        break
                    time.sleep(0.05)
            def log_message(self, *args):
                pass
        with ThreadingHTTPServer(("127.0.0.1", 0), Drip) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            started = time.monotonic()
            try:
                port = server.server_address[1]
                with self.assertRaises(TimeoutError):
                    local_chat("http://127.0.0.1:" + str(port)
                               + "/v1/chat/completions",
                               [{"role": "user", "content": "public mock"}],
                               SEEDS[0], timeout=0.22, api_key="SYNTHETIC_KEY")
            finally:
                server.shutdown()
                thread.join(timeout=2)
            self.assertLess(time.monotonic() - started, 1.5)

    def test_remaining_session_budget_is_passed_down_to_transport(self):
        # Simulate a slow prior turn so the final request cannot spend a full
        # fresh 30-second socket budget right before the global deadline.
        clock = iter([0, 0, 99.5, 119.75])
        budgets = []
        def fake(messages, seed, timeout):
            budgets.append(timeout)
            return completed()
        execute(CASES[0][0], SEEDS[0], "mock", fake,
                monotonic=lambda: next(clock))
        self.assertEqual(budgets, [30, 20.5, 0.25])

    @unittest.skipIf(os.name == "nt", "Windows private I/O explicitly blocked")
    def test_owner_opt_in_cli_collects_one_three_turn_http_smoke_only(self):
        from contextlib import redirect_stdout
        from io import StringIO
        calls = []
        bearer_headers = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                bearer_headers.append(self.headers.get("Authorization"))
                calls.append(json.loads(self.rfile.read(
                    int(self.headers["Content-Length"]))))
                encoded = json.dumps(completed("MOCK_PUBLIC_RESPONSE")).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            def log_message(self, *args):
                pass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "private"
            out.mkdir(mode=0o700)
            prompt = root / "system.txt"
            prompt.write_text("SYNTHETIC_MODEL_SYSTEM", encoding="utf-8")
            token = "SYNTHETIC_OWNER_BEARER_123"
            key_file = root / "client-api-key.txt"
            key_file.write_text(token + "\n", encoding="utf-8")
            os.chmod(key_file, 0o600)
            with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
                thread = Thread(target=server.serve_forever, daemon=True)
                thread.start()
                stdout = StringIO()
                try:
                    args = [
                        "run", "--case", CASES[0][0], "--seed", str(SEEDS[0]),
                        "--endpoint", "http://127.0.0.1:"
                                      + str(server.server_address[1])
                                      + "/v1/chat/completions",
                        "--system-prompt-file", str(prompt),
                        "--client-api-key-file", str(key_file),
                        "--private-output-dir", str(out),
                        "--consent-to-local-inference",
                    ]
                    with redirect_stdout(stdout):
                        self.assertEqual(main(args), 0)
                finally:
                    server.shutdown()
                    thread.join(timeout=2)
            self.assertEqual(len(calls), 3)
            self.assertEqual(bearer_headers, ["Bearer " + token] * 3)
            self.assertEqual([len(x["messages"]) for x in calls], [2, 4, 6])
            review = list(out.glob("review_*.json"))
            custody = list(out.glob("custody_*.json"))
            self.assertEqual(len(review), 1)
            self.assertEqual(len(custody), 1)
            self.assertNotIn("MOCK_PUBLIC_RESPONSE", stdout.getvalue())
            self.assertEqual(json.loads(custody[0].read_text())["complete_turn_count"], 3)
            self.assertFalse(json.loads(stdout.getvalue())["semantic_persona_pass"])
            self.assertNotIn(token, stdout.getvalue())
            self.assertNotIn(token, review[0].read_text())
            self.assertNotIn(token, custody[0].read_text())

    @unittest.skipIf(os.name == "nt", "Windows private I/O explicitly blocked")
    def test_owner_key_file_format_and_posix_permissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "key.txt"
            for value in (b"", b"#COMMENT\n", b"TOKEN1\nTOKEN2\n",
                          b"TOKEN\r\nX-Forged: yes\n"):
                with self.subTest(value=value):
                    path.write_bytes(value)
                    os.chmod(path, 0o600)
                    with self.assertRaises(ValueError):
                        _client_api_key_file(path)
            path.write_bytes(b"SYNTHETIC_OWNER_KEY\n")
            os.chmod(path, 0o600)
            self.assertEqual(_client_api_key_file(path), "SYNTHETIC_OWNER_KEY")
            path.write_bytes(b"SYNTHETIC_OWNER_KEY\r\n")
            self.assertEqual(_client_api_key_file(path), "SYNTHETIC_OWNER_KEY")
            if os.name != "nt":
                symlink = Path(tmp) / "alias-key.txt"
                symlink.symlink_to(path)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    _client_api_key_file(symlink)
                os.chmod(tmp, 0o755)
                with self.assertRaisesRegex(ValueError, "parent"):
                    _client_api_key_file(path)
                os.chmod(tmp, 0o700)
                os.chmod(path, 0o644)
                with self.assertRaisesRegex(ValueError, "owner-only"):
                    _client_api_key_file(path)

    def test_unauthorized_raw_endpoint_401_cannot_return_assistant(self):
        captured = []
        class Unauthorized(BaseHTTPRequestHandler):
            def do_POST(self):
                captured.append(self.headers.get("Authorization"))
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(401)
                self.send_header("Content-Length", "0")
                self.end_headers()
            def log_message(self, *args):
                pass
        with ThreadingHTTPServer(("127.0.0.1", 0), Unauthorized) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                address = "http://127.0.0.1:" + str(server.server_address[1])
                with self.assertRaises(OSError) as err:
                    local_chat(address + "/v1/chat/completions",
                               [{"role": "user", "content": "public mock"}],
                               SEEDS[0], api_key="SECRET_SYNTHETIC_CANARY")
            finally:
                server.shutdown()
                thread.join(timeout=2)
        self.assertEqual(captured, ["Bearer SECRET_SYNTHETIC_CANARY"])
        self.assertNotIn("SECRET_SYNTHETIC_CANARY", str(err.exception))

    def test_owner_opt_in_cli_requires_private_key_file(self):
        from contextlib import redirect_stderr
        from io import StringIO
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "system.txt"
            prompt.write_text("SYNTHETIC", encoding="utf-8")
            stderr = StringIO()
            with redirect_stderr(stderr):
                result = main([
                    "run", "--case", CASES[0][0], "--seed", str(SEEDS[0]),
                    "--endpoint", "http://127.0.0.1:11435/v1/chat/completions",
                    "--system-prompt-file", str(prompt),
                    "--private-output-dir", str(root),
                    "--consent-to-local-inference",
                ])
            self.assertEqual(result, 2)
            self.assertFalse(list(root.glob("review_*.json")))

    @unittest.skipIf(os.name == "nt", "Windows private I/O explicitly blocked")
    def test_cli_unauthorized_http_returns_nonzero_without_token_in_receipts(self):
        from contextlib import redirect_stdout, redirect_stderr
        from io import StringIO
        class Unauthorized(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(401)
                self.send_header("Content-Length", "0")
                self.end_headers()
            def log_message(self, *args):
                pass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "private"
            out.mkdir(mode=0o700)
            token = "SYNTHETIC_BAD_BEARER_123"
            key_file = root / "key.txt"
            key_file.write_text(token, encoding="ascii")
            os.chmod(key_file, 0o600)
            prompt = root / "system.txt"
            prompt.write_text("SYNTHETIC", encoding="utf-8")
            with ThreadingHTTPServer(("127.0.0.1", 0), Unauthorized) as server:
                thread = Thread(target=server.serve_forever, daemon=True)
                thread.start()
                stdout, stderr = StringIO(), StringIO()
                try:
                    args = [
                        "run", "--case", CASES[0][0], "--seed", str(SEEDS[0]),
                        "--endpoint", "http://127.0.0.1:"
                                      + str(server.server_address[1])
                                      + "/v1/chat/completions",
                        "--system-prompt-file", str(prompt),
                        "--client-api-key-file", str(key_file),
                        "--private-output-dir", str(out),
                        "--consent-to-local-inference",
                    ]
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        status = main(args)
                finally:
                    server.shutdown()
                    thread.join(timeout=2)
            self.assertEqual(status, 3)
            review = list(out.glob("review_*.json"))
            receipt = list(out.glob("custody_*.json"))
            self.assertEqual(len(review), 1)
            self.assertEqual(len(receipt), 1)
            self.assertEqual(json.loads(review[0].read_text())["turns"][0]["outcome"],
                             "upstream_error")
            for output in (stdout.getvalue(), stderr.getvalue(),
                           review[0].read_text(), receipt[0].read_text()):
                self.assertNotIn(token, output)


if __name__ == "__main__":
    unittest.main()
