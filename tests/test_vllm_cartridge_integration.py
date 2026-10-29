"""The vLLM cartridge path end to end, against the deployed CartridgeEngine
(4B smoke cartridge, L4) and the real gateway. Costs GPU time; set
CARTRIDGES_MODAL_TESTS=1:

    uv run modal deploy -m cartridges.serve.vllm_cartridge
    uv run cartridges cartridge_export_vllm --cartridge /vol/cartridges/exp-20261013-smoke/grpo
    CARTRIDGES_MODAL_TESTS=1 uv run pytest tests/test_vllm_cartridge_integration.py -s

Stop the engine afterwards (`modal container list`, `modal container stop <id>`).
"""

import json
import os
import socket
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("CARTRIDGES_MODAL_TESTS") != "1",
                                reason="needs Modal and a GPU; set CARTRIDGES_MODAL_TESTS=1")

MODEL_ID = "Qwen/Qwen3.5-4B"
CARTRIDGE = "/vol/cartridges/exp-20261013-smoke/grpo"
MESSAGES = [{"role": "user", "content": "What topics come up most often?"}]


def test_deployed_engine_streams_tokens():
    from cartridges.serve.vllm_cartridge import engine_cls

    events = list(engine_cls(MODEL_ID).generate_stream.remote_gen(CARTRIDGE, MESSAGES, max_tokens=64))
    *deltas, final = events
    assert len(deltas) > 1 and all(set(e) == {"text", "tokens"} for e in deltas)
    assert "".join(e["text"] for e in deltas).strip()
    assert sum(e["tokens"] for e in deltas) == final["completion_tokens"] > 0
    assert final["finish_reason"] in {"stop", "length"} and final["prompt_tokens"] > 0
    print(f"{len(deltas)} deltas, {final}")


def test_deployed_engine_refuses_an_unexported_cartridge():
    from cartridges.serve.vllm_cartridge import engine_cls

    with pytest.raises(Exception, match="cartridge_export_vllm"):
        engine_cls(MODEL_ID).generate.remote("/vol/cartridges/exp-20261013-smoke/distill", MESSAGES, max_tokens=8)


@pytest.fixture(scope="module")
def gateway():
    """`cartridges gateway_serve` on a free port, with the real backends."""
    import httpx

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cli = os.path.join(os.path.dirname(sys.executable), "cartridges")
    proc = subprocess.Popen([cli, "gateway_serve", "--port", str(port)],
                            env={**os.environ, "GATEWAY_API_KEY": "test-key"})
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if httpx.get(f"{url}/v1/models", headers={"Authorization": "Bearer test-key"}).status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.5)
        else:
            raise RuntimeError("gateway did not start")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def test_gateway_streams_the_vllm_cartridge_incrementally(gateway):
    import httpx

    body = {"model": "journal-cartridge-4b-smoke-vllm", "messages": MESSAGES, "stream": True, "max_tokens": 64}
    arrivals = []
    with httpx.stream("POST", f"{gateway}/v1/chat/completions", json=body, timeout=1800,
                      headers={"Authorization": "Bearer test-key"}) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line:
                arrivals.append((time.perf_counter(), line))
    assert arrivals[-1][1] == "data: [DONE]"
    chunks = [(t, json.loads(line.removeprefix("data: "))) for t, line in arrivals[:-1]]
    assert all("error" not in c for _, c in chunks), chunks[-1]
    content = [(t, c) for t, c in chunks if c["choices"][0]["delta"].get("content")]
    assert len(content) > 1
    final = chunks[-1][1]
    assert final["choices"][0]["finish_reason"] in {"stop", "length"}
    assert sum(c["tokens"] for _, c in content) == final["usage"]["completion_tokens"]
    # Incremental: the deltas arrive over time, not as one buffered burst at the end.
    spread = content[-1][0] - content[0][0]
    assert spread > 0.05, f"all {len(content)} deltas arrived within {spread:.3f}s"
    print(f"{len(content)} deltas over {spread:.2f}s, usage {final['usage']}")


def test_parity_with_hf():
    """FR1 on the 4B smoke cartridge (and a p=512 one in the same batch), vs HF Cartridge."""
    from vllm_cartridge_modal import SMOKE_CARTRIDGES, app, parity

    from cartridges.cartridge.tracking import group_name

    with app.run():
        out = parity.remote(MODEL_ID, SMOKE_CARTRIDGES.split(","), 256, group_name("vllm-parity"))
    print(json.dumps(out["summary"], indent=2))
    s = out["summary"]
    trained = SMOKE_CARTRIDGES.split(",")[0]
    assert s[f"{trained}/greedy32_match"] == 1.0 and s[f"{trained}/top1_agree_min"] >= 0.99
    # The rest within HF-vs-vLLM bf16 noise: on the base model alone, greedy runs fork
    # after 20+ tokens on 2-3 of 8 prompts and the top-1 agreement dips to ~97-98%
    # (it varies run to run); a wrong layout gives far lower agreement.
    for c in SMOKE_CARTRIDGES.split(","):
        assert s[f"{c}/greedy32_match"] >= s["base/greedy32_match"] - 1 / 8
        assert s[f"{c}/top1_agree_min"] >= 0.95
