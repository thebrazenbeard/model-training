"""Compare actual HF and llama.cpp next-token probabilities for a tiny Qwen3.5.

The model is a randomly initialized 32-layer synthetic fixture from
``qwen35_tiny_gguf_e2e_cpu``, not official pretrained model weights. This
checks the model's *outputs* across formats, beyond tensor representation.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROMPT_IDS = (5, 6, 7, 8, 9)
SAMPLES = 12
# F16 GGUF from F32 HF can move close logits. These bounds are deliberately
# loose for an early smoke, but reject substantial cross-runtime divergence.
MAX_TOKEN_LOGPROB_DELTA = 0.25
MEAN_TOKEN_LOGPROB_DELTA = 0.05


def assert_sampled_logprobs(
    hf_logprobs, reported: list[dict], *,
    max_delta: float = MAX_TOKEN_LOGPROB_DELTA,
    mean_delta: float = MEAN_TOKEN_LOGPROB_DELTA,
) -> dict:
    """Check GGUF's returned full-softmax log-probs against independent HF.

    ``reported`` is the actual llama-server ``top_logprobs`` collection, not
    probabilities reconstructed from the same HF logits. A corruption of one
    returned token or probability must fail.
    """
    if not isinstance(reported, list) or len(reported) < SAMPLES:
        raise AssertionError("llama.cpp response missing required top_logprobs")
    if len(hf_logprobs) < 128:
        raise AssertionError("unexpected tiny HF vocabulary")

    seen: set[int] = set()
    deltas: list[float] = []
    for entry in reported[:SAMPLES]:
        if not isinstance(entry, dict):
            raise AssertionError("invalid llama.cpp top_logprobs item")
        token_id = entry.get("id")
        observed = entry.get("logprob")
        if (type(token_id) is not int or token_id < 0 or
                token_id >= len(hf_logprobs) or token_id in seen):
            raise AssertionError("invalid, duplicated or out-of-range token ID")
        seen.add(token_id)
        if type(observed) not in (int, float) or not math.isfinite(observed):
            raise AssertionError("invalid llama.cpp reported log probability")
        expected = float(hf_logprobs[token_id])
        if not math.isfinite(expected):
            raise AssertionError("nonfinite HF log probability")
        deltas.append(abs(float(observed) - expected))

    maximum = max(deltas)
    average = sum(deltas) / len(deltas)
    if maximum > max_delta or average > mean_delta:
        raise AssertionError(
            "HF↔GGUF next-token logprob divergence: "
            f"max={maximum:.6f} mean={average:.6f} n={len(deltas)}"
        )
    return {"compared_tokens": len(deltas), "max_abs_logprob_delta": maximum,
            "mean_abs_logprob_delta": average}


def _request_json(url: str, body: dict | None = None, *, timeout: float = 10) -> dict:
    req = Request(
        url,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise AssertionError("llama.cpp returned a non-object JSON response")
    return payload


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def llama_logprobs(gguf: Path, server_bin: Path) -> tuple[list[dict], dict]:
    if not gguf.is_file() or not server_bin.is_file():
        raise AssertionError("actual GGUF and pinned llama-server binary required")
    port = _free_loopback_port()
    command = [str(server_bin), "--model", str(gguf), "--host", "127.0.0.1",
               "--port", str(port), "--ctx-size", "128", "--parallel", "1",
               "--n-gpu-layers", "0"]
    # This subprocess runs only inside the isolated public GitHub CPU runner.
    # A real file avoids pipe-buffer deadlock during verbose GGUF model loading.
    server_log = tempfile.TemporaryFile(mode="w+t")
    server = subprocess.Popen(command, stdout=server_log,
                              stderr=subprocess.STDOUT, text=True)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 100
        while True:
            if server.poll() is not None:
                raise AssertionError(f"llama-server quit before readiness: {server.returncode}")
            try:
                health = _request_json(url + "/health", timeout=2)
                if health.get("status") == "ok":
                    break
            except (URLError, HTTPError, TimeoutError, OSError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise AssertionError("pinned llama-server readiness timeout")
            time.sleep(0.5)

        response = _request_json(url + "/completion", {
            "prompt": list(PROMPT_IDS), "n_predict": 1,
            "temperature": -1, "top_k": 0, "top_p": 1,
            "min_p": 0, "repeat_penalty": 1,
            "cache_prompt": False, "n_probs": SAMPLES,
            "return_tokens": True, "stream": False,
        }, timeout=100)
        entries = response.get("probs")
        if not isinstance(entries, list) or len(entries) != 1:
            raise AssertionError("llama-server missing one completion probability record")
        top = entries[0].get("top_logprobs")
        if not isinstance(top, list):
            raise AssertionError("llama-server did not return top_logprobs")
        return top, {"generated_ids": response.get("tokens"),
                     "reported_topk": len(top), "server_exit_before_stop": server.poll()}
    except Exception as exc:
        # Preserve actionable loader/server diagnostics in the failing CI step.
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        server_log.seek(0)
        detail = server_log.read()[-3000:]
        raise AssertionError(f"pinned llama-server comparison failed: {exc}; "
                             f"server log tail: {detail}") from exc
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        server_log.close()


def hf_logprobs(merged_dir: Path):
    # Must use the *saved and merged* HF model produced by the earlier step,
    # not a fresh reinitialized model or an unstaged LoRA adapter.
    import torch
    from transformers import Qwen3_5ForCausalLM

    torch.set_num_threads(2)
    model = Qwen3_5ForCausalLM.from_pretrained(
        str(merged_dir), local_files_only=True,
    ).eval().cpu()
    with torch.inference_mode():
        ids = torch.tensor([list(PROMPT_IDS)], dtype=torch.long)
        logits = model(input_ids=ids, use_cache=False).logits[0, -1, :]
        result = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
    if torch.cuda.is_initialized():
        raise AssertionError("HF output probe initialized GPU")
    return result


def verify(merged_dir: Path, gguf: Path, server_bin: Path) -> dict:
    hf = hf_logprobs(merged_dir)
    top, server_receipt = llama_logprobs(gguf, server_bin)
    metrics = assert_sampled_logprobs(hf, top)
    return {
        "schema": "TRAININGDAY_QWEN35_SYNTHETIC_GGUF_LOGPROB_PARITY_V1",
        "scope": "random tiny 32-layer Qwen3.5 HF versus pinned llama.cpp GGUF CPU next-token top-k",
        "prompt_ids": list(PROMPT_IDS),
        "limits_nats": {"max": MAX_TOKEN_LOGPROB_DELTA,
                        "mean": MEAN_TOKEN_LOGPROB_DELTA},
        **metrics, **server_receipt,
        "official_pretrained_weights_exercised": False,
        "trained_model_quality": "NOT_TESTED",
        "production_logit_parity": "NOT_TESTED",
        "gpu_fit": "UNKNOWN",
        "protected_training": False,
    }


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: python -m trainingday.qwen35_gguf_logprobs_cpu "
                         "MERGED_HF_DIR GGUF_FILE PINNED_LLAMA_SERVER")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(json.dumps(verify(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])),
                     sort_keys=True))


if __name__ == "__main__":
    main()
