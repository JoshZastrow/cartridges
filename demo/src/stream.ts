// One pane's stream: the gateway's OpenAI-style SSE (`chat.completion.chunk`
// events, each with a non-standard `tokens` count) reduced to text and timings.

export type Phase = "idle" | "waiting" | "streaming" | "done" | "error";

export interface PaneState {
  phase: Phase;
  text: string;
  tokens: number;
  startedAt?: number; // ms (performance.now) when the request was sent
  firstTokenAt?: number; // ms when the first token arrived
  firstTokens: number; // tokens in that first chunk (not counted in the decode rate)
  endedAt?: number;
  finishReason?: string;
  error?: string;
}

export type StreamEvent =
  | { kind: "start"; at: number }
  | { kind: "delta"; text: string; tokens: number; at: number }
  | { kind: "end"; finishReason: string; completionTokens?: number; at: number }
  | { kind: "error"; message: string; at: number };

export const initialPane = (): PaneState => ({ phase: "idle", text: "", tokens: 0, firstTokens: 0 });

export function reduce(state: PaneState, event: StreamEvent): PaneState {
  switch (event.kind) {
    case "start":
      return { ...initialPane(), phase: "waiting", startedAt: event.at };
    case "delta": {
      const first = state.firstTokenAt === undefined && event.tokens > 0;
      return {
        ...state,
        phase: "streaming",
        text: state.text + event.text,
        tokens: state.tokens + event.tokens,
        firstTokenAt: first ? event.at : state.firstTokenAt,
        firstTokens: first ? event.tokens : state.firstTokens,
      };
    }
    case "end":
      return {
        ...state,
        phase: "done",
        endedAt: event.at,
        finishReason: event.finishReason,
        tokens: event.completionTokens ?? state.tokens,
      };
    case "error":
      return { ...state, phase: "error", endedAt: event.at, error: event.message };
  }
}

export interface Stats {
  ttftS?: number; // request sent -> first token (includes a cold start)
  tokPerS?: number; // decode rate: tokens after the first chunk / time since it
  elapsedS?: number; // request sent -> last token (live while streaming)
}

export function stats(state: PaneState, now: number): Stats {
  if (state.startedAt === undefined) return {};
  const end = state.endedAt ?? now;
  const out: Stats = { elapsedS: (end - state.startedAt) / 1000 };
  if (state.firstTokenAt !== undefined) {
    out.ttftS = (state.firstTokenAt - state.startedAt) / 1000;
    const decodeS = (end - state.firstTokenAt) / 1000;
    const decoded = state.tokens - state.firstTokens;
    if (decodeS > 0 && decoded > 0) out.tokPerS = decoded / decodeS;
  }
  return out;
}

/** The JSON payload of each `data:` line, up to `data: [DONE]`. */
export async function* sseData(body: AsyncIterable<Uint8Array>): AsyncGenerator<any> {
  const decoder = new TextDecoder();
  let buffer = "";
  for await (const bytes of body) {
    buffer += decoder.decode(bytes, { stream: true });
    let newline: number;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline).trimEnd();
      buffer = buffer.slice(newline + 1);
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") return;
      yield JSON.parse(data);
    }
  }
}

export interface ChatOptions {
  url: string;
  apiKey: string;
  model: string;
  prompt: string;
  maxTokens: number;
  temperature: number;
  signal?: AbortSignal;
}

/** POST one single-turn chat with stream: true and emit its events. */
export async function streamChat(opts: ChatOptions, emit: (event: StreamEvent) => void): Promise<void> {
  const now = () => performance.now();
  emit({ kind: "start", at: now() });
  try {
    const response = await fetch(`${opts.url}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${opts.apiKey}` },
      body: JSON.stringify({
        model: opts.model,
        messages: [{ role: "user", content: opts.prompt }],
        stream: true,
        max_tokens: opts.maxTokens,
        temperature: opts.temperature,
      }),
      signal: opts.signal,
    });
    if (!response.ok || !response.body) {
      const body = await response.text();
      let message = body;
      try {
        message = JSON.parse(body).error?.message ?? body;
      } catch {}
      emit({ kind: "error", message: `HTTP ${response.status}: ${message}`, at: now() });
      return;
    }
    for await (const chunk of sseData(response.body)) {
      if (chunk.error) {
        emit({ kind: "error", message: chunk.error.message, at: now() });
        return;
      }
      const choice = chunk.choices?.[0];
      const content: string | undefined = choice?.delta?.content;
      if (chunk.tokens !== undefined || content) {
        emit({ kind: "delta", text: content ?? "", tokens: chunk.tokens ?? 0, at: now() });
      }
      if (choice?.finish_reason) {
        emit({ kind: "end", finishReason: choice.finish_reason, completionTokens: chunk.usage?.completion_tokens, at: now() });
      }
    }
  } catch (e: any) {
    emit({ kind: "error", message: e?.name === "AbortError" ? "stopped" : String(e?.message ?? e), at: now() });
  }
}
