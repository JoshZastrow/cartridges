"""vLLM server for the OCR model, on Modal (OpenAI-compatible HTTP API).

Modal gives the GPU containers, the autoscaler, and the HTTPS URL. vLLM runs
the model and batches concurrent requests on the GPU (continuous batching).
One container serves up to MAX_NUM_SEQS pages at the same time.

The deploy step bakes the model SHA into the image environment. Thus:
- a new SHA makes a new image, and Modal replaces the old containers;
- --served-model-name is the SHA, so a client can read the served SHA from
  /v1/models and compare it with its cache key.

Each new container has its own cold start (load weights, capture CUDA
graphs). A warm container stays up for SCALEDOWN_WINDOW after its last
request. To pre-warm containers, set min_containers with
Function.update_autoscaler (cartridges serve_warm).

Deploy with:
uv run cartridges serve_deploy --revision prod
"""

from __future__ import annotations

import json
import os
import subprocess

import modal

MINUTES = 60
PORT = 8000

# Set by `cartridges serve_deploy` before it imports this module. In the
# container, the image environment gives the same values.
MODEL_SHA = os.environ.get("CARTRIDGES_OCR_MODEL_SHA", "base")
MAX_NUM_SEQS = int(os.environ.get("CARTRIDGES_OCR_MAX_NUM_SEQS", "32"))
MAX_CONTAINERS = int(os.environ.get("CARTRIDGES_OCR_MAX_CONTAINERS", "8"))
SCALEDOWN_WINDOW = 10 * MINUTES  # Modal maximum is 20 minutes

BASE_MODEL_DIR = "/cache/deepseek_ocr"
MODELS_DIR = "/cache/models"
# Official DeepSeek-OCR-2 logits processor (see the vLLM multimodal example
# run_deepseek_ocr2). Per-request settings come in through vllm_xargs.
LOGITS_PROCESSOR = "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.29.0")
    .env(
        {
            "CARTRIDGES_OCR_MODEL_SHA": MODEL_SHA,
            "CARTRIDGES_OCR_MAX_NUM_SEQS": str(MAX_NUM_SEQS),
            "CARTRIDGES_OCR_MAX_CONTAINERS": str(MAX_CONTAINERS),
        }
    )
)

app = modal.App("cartridges-ocr-vllm")
model_cache = modal.Volume.from_name("cartridges-training-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("cartridges-vllm-cache", create_if_missing=True)
# Create once with: modal secret create cartridges-ocr-api-key VLLM_API_KEY=<key>
# vLLM reads VLLM_API_KEY and then requires "Authorization: Bearer <key>".
api_key_secret = modal.Secret.from_name("cartridges-ocr-api-key")


def model_dir(sha: str) -> str:
    return BASE_MODEL_DIR if sha == "base" else f"{MODELS_DIR}/{sha}"


def vllm_command(sha: str, max_num_seqs: int) -> list[str]:
    return [
        "vllm", "serve", model_dir(sha),
        "--served-model-name", sha,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--trust-remote-code",
        "--hf-overrides", json.dumps({"architectures": ["DeepseekOCR2ForCausalLM"]}),
        "--logits-processors", LOGITS_PROCESSOR,
        # The official recipe turns these off: page images do not repeat, so
        # the caches only use memory.
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0",
        "--limit-mm-per-prompt", json.dumps({"image": 1}),
        "--max-model-len", "8192",
        "--max-num-seqs", str(max_num_seqs),
        "--gpu-memory-utilization", "0.90",
    ]


@app.function(
    image=vllm_image,
    # Decoding is memory-bandwidth-bound (see training/finetune.py); A100 it is.
    gpu="A100",
    scaledown_window=SCALEDOWN_WINDOW,
    max_containers=MAX_CONTAINERS,
    # Per request. A looped page at max_tokens is the slow case.
    timeout=15 * MINUTES,
    volumes={"/cache": model_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[api_key_secret],
)
@modal.concurrent(max_inputs=MAX_NUM_SEQS)
@modal.web_server(port=PORT, startup_timeout=20 * MINUTES)
def serve() -> None:
    command = vllm_command(MODEL_SHA, MAX_NUM_SEQS)
    print(*command)
    subprocess.Popen(command)
