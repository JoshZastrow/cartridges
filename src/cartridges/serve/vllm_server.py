"""The packaged OCR service on Modal: vLLM plus the OCR recipe, in one container.

Modal gives the GPU containers, the autoscaler, and the HTTPS URL. vLLM runs
the model and batches concurrent requests on the GPU (continuous batching).
A small FastAPI app in front of vLLM owns the recipe (serve/recipe.py):

    POST /ocr     {"image_base64": ...} -> {"text", "model_id", ...}
    GET  /info    model id + full recipe
    GET  /health  no auth

A client sends an image only. Prompt, decoding, and postprocessing come from
the recipe; preprocessing is pinned with vLLM --mm-processor-kwargs. vLLM
listens on an internal port only.

The deploy step bakes the recipe into the image environment. Thus a new
recipe (new weights or any new setting) makes a new image, and Modal replaces
the old containers. The model id is the recipe hash.

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
import time

import modal
from pydantic import BaseModel, Field

from cartridges.serve.recipe import OcrRecipe

MINUTES = 60
VLLM_PORT = 8001  # internal; the FastAPI app is the only public surface

# Set by `cartridges serve_deploy` before it imports this module. In the
# container, the image environment gives the same values.
RECIPE = OcrRecipe.model_validate_json(
    os.environ.get("CARTRIDGES_OCR_RECIPE") or OcrRecipe(model_sha="base").model_dump_json()
)
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
            "CARTRIDGES_OCR_RECIPE": RECIPE.model_dump_json(),
            "CARTRIDGES_OCR_MAX_NUM_SEQS": str(MAX_NUM_SEQS),
            "CARTRIDGES_OCR_MAX_CONTAINERS": str(MAX_CONTAINERS),
        }
    )
    .add_local_python_source("cartridges")
)

app = modal.App("cartridges-ocr-vllm")
model_cache = modal.Volume.from_name("cartridges-training-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("cartridges-vllm-cache", create_if_missing=True)
# Create once with: modal secret create cartridges-ocr-api-key VLLM_API_KEY=<key>
# The FastAPI app checks "Authorization: Bearer <key>"; vLLM reads the same
# variable for its internal port.
api_key_secret = modal.Secret.from_name("cartridges-ocr-api-key")


def model_dir(sha: str) -> str:
    return BASE_MODEL_DIR if sha == "base" else f"{MODELS_DIR}/{sha}"


def vllm_command(recipe: OcrRecipe, max_num_seqs: int) -> list[str]:
    return [
        "vllm", "serve", model_dir(recipe.model_sha),
        "--served-model-name", recipe.model_id,
        "--host", "127.0.0.1",
        "--port", str(VLLM_PORT),
        "--trust-remote-code",
        "--hf-overrides", json.dumps({"architectures": ["DeepseekOCR2ForCausalLM"]}),
        "--mm-processor-kwargs", json.dumps(recipe.mm_processor_kwargs()),
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


class OcrRequest(BaseModel):
    image_base64: str
    top_logprobs: int = Field(default=0, ge=0, le=20)  # parity checks only


class OcrResponse(BaseModel):
    text: str
    model_id: str
    finish_reason: str | None
    completion_tokens: int | None
    # [(token_id, logprob)] for the first output token, if top_logprobs > 0.
    first_token_top_logprobs: list[tuple[int, float]] | None = None


def build_app(recipe: OcrRecipe, api_key: str, vllm):
    """The public API. `vllm` is an httpx.AsyncClient for the internal vLLM
    server (a mock transport in tests)."""
    from fastapi import Depends, FastAPI, Header, HTTPException

    def check_key(authorization: str = Header(default="")) -> None:
        if authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="invalid API key")

    web = FastAPI(title="cartridges OCR")

    @web.get("/health")
    async def health() -> dict:
        return {"ok": True}

    @web.get("/info", dependencies=[Depends(check_key)])
    async def info() -> dict:
        return {"model_id": recipe.model_id, "recipe": recipe.model_dump(mode="json")}

    @web.post("/ocr", dependencies=[Depends(check_key)])
    async def ocr(request: OcrRequest) -> OcrResponse:
        body = recipe.chat_request(request.image_base64, request.top_logprobs)
        response = await vllm.post("/v1/chat/completions", json=body)
        if response.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"vLLM {response.status_code}: {response.text[:500]}")
        body = response.json()
        choice = body["choices"][0]
        first_top = None
        if request.top_logprobs and (choice.get("logprobs") or {}).get("content"):
            # return_tokens_as_token_ids gives tokens as "token_id:<id>".
            first_top = [
                (int(t["token"].removeprefix("token_id:")), t["logprob"])
                for t in choice["logprobs"]["content"][0]["top_logprobs"]
            ]
        return OcrResponse(
            text=recipe.postprocess(choice["message"]["content"] or ""),
            model_id=recipe.model_id,
            finish_reason=choice.get("finish_reason"),
            completion_tokens=(body.get("usage") or {}).get("completion_tokens"),
            first_token_top_logprobs=first_top,
        )

    return web


def _wait_ready(process: subprocess.Popen, url: str, timeout_s: float) -> None:
    import httpx

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {process.returncode}")
        try:
            if httpx.get(f"{url}/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM not ready after {timeout_s:.0f}s")


@app.function(
    image=vllm_image,
    # Decoding is memory-bandwidth-bound (see training/finetune.py); A100 it is.
    gpu="A100",
    scaledown_window=SCALEDOWN_WINDOW,
    max_containers=MAX_CONTAINERS,
    # Per request. A looped page at max_tokens is the slow case.
    timeout=15 * MINUTES,
    # Container start: load weights, capture CUDA graphs.
    startup_timeout=20 * MINUTES,
    volumes={"/cache": model_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[api_key_secret],
)
@modal.concurrent(max_inputs=MAX_NUM_SEQS)
@modal.asgi_app()
def serve():
    import httpx

    command = vllm_command(RECIPE, MAX_NUM_SEQS)
    print(*command)
    url = f"http://127.0.0.1:{VLLM_PORT}"
    _wait_ready(subprocess.Popen(command), url, timeout_s=19 * MINUTES)
    api_key = os.environ["VLLM_API_KEY"]
    client = httpx.AsyncClient(
        base_url=url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(15 * MINUTES),
        limits=httpx.Limits(max_connections=MAX_NUM_SEQS * 2),
    )
    return build_app(RECIPE, api_key, client)
