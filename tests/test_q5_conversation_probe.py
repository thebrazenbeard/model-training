"""Public synthetic tests. No actual model or owner system prompt is used."""

import json
import os
from pathlib import Path
import socket
import tempfile
from threading import Thread
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest

from trainingday.q5_conversation_probe import (
    CASES, LOCAL_Q5_FILE_SHA256, SEEDS, _endpoint, _reply,
    execute, local_chat, main, plan, private_write,
)


def completed(text="Mock answer"):
    return {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": text}}]}


class Q5ProbeContractTests(unittest.TestCase):
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
        def fake_transport(messages, seed):
            sent.append(json.loads(json.dumps(messages)))
            self.assertEqual(seed, SEEDS[0])
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
        def fake(messages, seed):
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

    def test_private_output_is_no_overwrite_and_blinded(self):
        review, receipt = execute(CASES[0][0], SEEDS[0], "PRIVATE_TEST_PROMPT",
                                  lambda m, s: completed())
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
                                  lambda m, s: completed())
        with self.assertRaisesRegex(ValueError, "source repository"):
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
            execute(CASES[0][0], 999, "mock", lambda m, s: completed())
        with self.assertRaises(ValueError):
            execute(CASES[0][0], SEEDS[0], "x" * 12001, lambda m, s: completed())
        with self.assertRaises(ValueError):
            execute("not_a_case", SEEDS[0], "mock", lambda m, s: completed())

    def test_unresponsive_transport_is_censored(self):
        def failed(messages, seed):
            raise socket.timeout("host state private")
        review, receipt = execute(CASES[1][0], SEEDS[0], "mock", failed)
        self.assertEqual(review["turns"][0]["outcome"], "upstream_error")
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
                     {"role": "user", "content": "public test"}], SEEDS[0])
            finally:
                server.shutdown()
                thread.join(timeout=2)
        self.assertEqual(_reply(response), ("complete", "Mock answer"))
        self.assertEqual(observed[0][0], "/v1/chat/completions")
        self.assertEqual(observed[0][1]["seed"], SEEDS[0])
        self.assertEqual(observed[0][1]["messages"][-1]["content"], "public test")
        self.assertNotIn("rubric", json.dumps(observed[0]))
        self.assertNotIn("expected", json.dumps(observed[0]))


if __name__ == "__main__":
    unittest.main()
