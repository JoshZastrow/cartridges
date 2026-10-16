# LibreChat on the Teacher + Journal model: design and contract

Status: in progress, 2026-10-15. Three agents (frontend, model/gateway, backend)
build against this contract. Lead verifies end to end with Playwright.

## Goal

LibreChat hosted locally at http://localhost:3080. A model picker offers our
journal models; a multi-turn chat with questions sampled from the Task Store
gets grounded, cited answers from the 27B Teacher running on Modal via vLLM.

## Environment facts (checked)

- No Docker on this Mac. LibreChat runs from source (`npm`), MongoDB from
  `mongodb-memory-server` or the `mongodb/brew` tap. Meilisearch/RAG off.
- Node v24.13.0, Playwright 1.64 + browsers installed. `uv run modal` works
  (client 1.5.6). `.env.local` holds DATABASE_URL, HF_TOKEN, WANDB_API_KEY.
  ANTHROPIC_API_KEY is NOT set anywhere (commented out in ~/.zshrc).
- Modal app `cartridges-teacher` is deployed: class `Teacher` (vLLM AsyncLLM,
  journal context prefilled). `teacher.teacher_cls("Teacher", model_id)` gives
  a handle; `.generate.remote(messages, max_tokens=..., temperature=...)`
  returns `list[Completion]` (text, finish_reason, ttft_s, latency_s). No
  streaming. 4B runs on L4 ($0.80/h, ~5-10 min cold start); 27B on
  H200/H100 ($4.54/h, ~15 min cold start). Default pool scaledown 5 min.
- Cartridge artifact on volume `cartridges-cartridge-cache`:
  `/vol/cartridges/exp-20261013-smoke/grpo` (base Qwen/Qwen3.5-4B). `ask.py`
  shows base+cartridge generation with `Cartridge.load` + `cart.generate`.
- Sample tasks: `set -a; source .env.local; set +a; uv run cartridges
  tasks_sample --split eval --n 3 --seed 1` (JSON with `tasks[].text`).

## Components and owners

```
LibreChat (3080) --OpenAI custom endpoint--> Gateway (8008, /v1) --Modal--> Teacher (vLLM, 27B/4B)
                                                                    \--Modal--> CartridgeServer (HF, base+cartridge)
```

### Frontend (LibreChat) — owner: frontend agent

- Clone LibreChat to `~/Github/LibreChat` (sibling repo, not inside this one).
- Repo-tracked config in `librechat/`: `librechat.yaml`, `.env.example`,
  `start.sh` (starts Mongo, backend, frontend). Secrets stay out of git.
- `librechat.yaml` custom endpoint named **Journal**:
  `baseURL: http://127.0.0.1:8008/v1`, `apiKey: cartridges-local`,
  `models.fetch: true` (fallback default list below), `titleConvo: false`,
  `dropParams: [frequency_penalty, presence_penalty, user]`, streaming on.
- Registration enabled (`ALLOW_REGISTRATION=true`) so Playwright can create
  a user. `SEARCH=false`, no Meilisearch.
- Playwright: `playwright.config.ts` baseURL `http://localhost:3080`, chromium
  only, `e2e/librechat.spec.ts`: register/login, pick Journal endpoint and a
  model, send 3 turns (questions from the Task Store sample, in a JSON
  fixture `e2e/fixtures/tasks.json`), assert each assistant reply is
  non-empty and the conversation has 3 user + 3 assistant messages.
- Model for the UI smoke test: `mock-echo` (gateway, no GPU). Anthropic key
  is not available; skip Anthropic unless `ANTHROPIC_API_KEY` appears.

### Gateway (routing, retries, schemas) — owner: model agent

`src/cartridges/serve/gateway.py`, run with
`uv run cartridges gateway_serve --port 8008` (add the CLI command). FastAPI,
OpenAI-compatible:

- `GET /v1/models` → `{object: "list", data: [{id, object: "model", owned_by}]}`
  from the registry.
- `POST /v1/chat/completions` with `model`, `messages`, `stream`,
  `max_tokens`, `temperature`. Pydantic request/response schemas. Bearer key
  check (`GATEWAY_API_KEY`, default `cartridges-local`).
- `stream: true` → SSE `chat.completion.chunk` events: the role chunk first,
  then the text (in a few chunks), then `finish_reason`, then `data: [DONE]`.
  Modal does not stream, so the gateway chunks the finished text.
- Retries: 2 retries with backoff on Modal transport/`RemoteError`-class
  failures only, never on 4xx. Per-request timeout 25 min (cold start) via
  `.spawn()` + `.get(timeout=...)`. Errors come back as OpenAI-style
  `{error: {message, type, code}}` with 502/504.
- Thinking is off; the Teacher strips `<think>` already (enable_thinking=False).
  Also strip a leftover `<think>...</think>` block defensively (`evals.strip_think`).
- System prompt: pass the client's messages through as they are; the Teacher
  adds the journal context block itself. Drop empty system turns.
- Model registry (`serve/chat_models.py`, a dict of frozen pydantic
  `ChatModel` rows): `id`, `backend` (`teacher` | `cartridge` | `mock`),
  `model_id`, `with_context`, `cartridge` (volume path). Rows:

  | id | backend | notes |
  |---|---|---|
  | `journal-teacher-27b` | teacher Qwen/Qwen3.8-27B, with_context=True | the deliverable |
  | `journal-teacher-4b` | teacher Qwen/Qwen3.5-4B, with_context=True | dev/smoke |
  | `qwen3.5-4b-base` | teacher 4B, with_context=False | baseline |
  | `journal-cartridge-4b-smoke` | cartridge `/vol/cartridges/exp-20261013-smoke/grpo` | base+cartridge |
  | `mock-echo` | mock | UI tests; echoes the last user turn |

  Hot-swap = picking a model id in LibreChat; adding a cartridge = one row.
- Tests `tests/test_gateway.py` with a fake Modal handle (no GPU): models
  list, non-stream, stream framing, retry on transport error, 401, 404 model.

### Backend (Modal inference) — owner: backend agent

- Teacher: keep `cartridges-teacher` as deployed; confirm `check_deployed`
  passes for 4B (`uv run python -m cartridges.cartridge.teacher`). Confirm the
  27B weights are on the volume (`download_weights` if not) so the 27B cold
  start does not pay for a 54 GB download on the GPU. Do NOT start a 27B
  container during development; the lead triggers the one 27B run.
- CartridgeServer: new Modal class in `src/cartridges/cartridge/serve_cartridge.py`
  (app `cartridges-cartridge-serve`, A100-80GB or L4 for 4B, scaledown 5 min,
  `modal.parameter` `cartridge` = volume path): `@enter` loads base model
  from `cartridge.json` and the cartridge; `generate(messages, max_tokens,
  temperature) -> dict(text, finish_reason, prompt_tokens, completion_tokens)`.
  Prompt = `teacher.render(tokenizer, messages)[1]` (student prompt_ids; the
  cartridge stands in for the context block). Reuse `Cartridge.generate`.
  Deploy with `uv run modal deploy -m cartridges.cartridge.serve_cartridge`.
  Smoke it with the 4B smoke cartridge. Expose a small client helper
  `cartridge_cls(cartridge_path)` like `teacher_cls`.
- Backend functions the gateway calls, in `src/cartridges/serve/backends.py`:
  `teacher_generate(model_id: str, with_context: bool, messages: list[dict],
  max_tokens: int, temperature: float | None) -> Generation` and
  `cartridge_generate(cartridge: str, messages, max_tokens, temperature) ->
  Generation`, where `Generation` is a frozen pydantic model with `text`,
  `finish_reason`, `prompt_tokens`, `completion_tokens`. Both block until
  done (the gateway runs them in a thread). The backend agent owns this file;
  the gateway imports it and tests against a fake.

## Ports and keys

| What | Value |
|---|---|
| LibreChat | http://localhost:3080 |
| Gateway | http://127.0.0.1:8008/v1, key `cartridges-local` |
| MongoDB | mongodb://127.0.0.1:27017/LibreChat |

## Lifecycle

`make chat-up` (gateway + Mongo + LibreChat), `make chat-test [MODEL=...]`,
`make chat-status`, `make chat-down` (also stops idle Modal GPU containers),
`make chat-clean` (chat-down + delete Mongo data, logs, Playwright output).

## Done when

1. `npx playwright test` starts the gateway and LibreChat itself (Playwright
   `webServer`, reusing running ones); Modal containers cold-start on the first
   request. Mongo is left running between runs (persistent users).
2. `npx playwright test e2e/librechat.spec.ts` passes with `mock-echo`.
3. The same test passes with `journal-teacher-27b` (lead runs it once; the
   Teacher's 27B container is stopped afterwards with `teacher.stop()`).

Verified 2026-10-16: mock-echo (24 s from cold), journal-teacher-27b (6 min,
cited answers), journal-cartridge-4b-smoke (5 min; plumbing works, but the
smoke cartridge's answers are a repeated sentence: a model-quality issue).
