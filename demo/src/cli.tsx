// Teacher vs cartridge, side by side: one prompt goes to both through the
// gateway (`uv run cartridges gateway_serve`), each pane streams its tokens and
// shows time to first token, decode tokens/s and total time.
//
//   npm start -- [--url http://127.0.0.1:8008] [--max-tokens 512] [--prompt "..."] [--no-warmup]

import React, { useCallback, useEffect, useRef, useState } from "react";
import { Box, Text, render, useApp, useInput, useStdout } from "ink";
import TextInput from "ink-text-input";
import wrapAnsi from "wrap-ansi";
import { type PaneState, type StreamEvent, initialPane, reduce, stats, streamChat } from "./stream.js";

interface Pane {
  model: string;
  name: string;
  detail: string;
  color: string;
}

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] !== undefined ? process.argv[i + 1] : fallback;
}

const config = {
  url: arg("url", "http://127.0.0.1:8008"),
  apiKey: arg("key", process.env.GATEWAY_API_KEY ?? "cartridges-local"),
  maxTokens: Number(arg("max-tokens", "512")),
  temperature: Number(arg("temperature", "0")), // greedy on both sides
  prompt: arg("prompt", ""),
  warmup: !process.argv.includes("--no-warmup"),
};

const PANES: Pane[] = [
  {
    model: arg("teacher", "journal-teacher-27b"),
    name: "Teacher",
    detail: "Qwen3.8-27B + 205k-token journal in context · vLLM · H200",
    color: "cyan",
  },
  {
    model: arg("cartridge", "journal-cartridge-27b-am"),
    name: "Cartridge",
    detail: "Qwen3.8-27B + 2,048-position cartridge · HF decode · H200",
    color: "magenta",
  },
];

const fmt = (x: number | undefined, digits: number, unit: string) => (x === undefined ? "–" : `${x.toFixed(digits)}${unit}`);

function statusOf(state: PaneState, warming: boolean): string {
  if (state.phase === "error") return `error: ${state.error}`;
  if (warming) return state.phase === "done" ? "ready" : "warming up (a cold start can take several minutes)";
  if (state.phase === "waiting") return "waiting for the first token";
  if (state.phase === "done") return state.finishReason === "length" ? "done (max tokens)" : "done";
  return state.phase === "streaming" ? "streaming" : "";
}

function PaneView(props: {
  pane: Pane;
  state: PaneState;
  warming: boolean;
  now: number;
  width: number;
  bodyHeight: number;
  scroll: number;
}) {
  const { pane, state, warming, now, width, bodyHeight, scroll } = props;
  const s = stats(state, now);
  const lines = wrapAnsi(state.text, Math.max(10, width - 4), { hard: true, trim: false }).split("\n");
  const start = Math.max(0, lines.length - bodyHeight - scroll);
  const body = lines.slice(start, start + bodyHeight);
  return (
    <Box flexDirection="column" width={width} borderStyle="round" borderColor={pane.color} paddingX={1}>
      <Text>
        <Text bold color={pane.color}>
          {pane.name}
        </Text>
        <Text dimColor> {pane.model}</Text>
      </Text>
      <Text dimColor wrap="truncate-end">
        {pane.detail}
      </Text>
      <Text wrap="truncate-end">
        <Text bold>{fmt(s.tokPerS, 1, " tok/s")}</Text>
        {"  ·  total "}
        <Text bold>{fmt(s.elapsedS, 1, " s")}</Text>
        {"  ·  first token "}
        {fmt(s.ttftS, 2, " s")}
        {"  ·  "}
        {state.tokens} tok
      </Text>
      <Text dimColor wrap="truncate-end">
        {statusOf(state, warming)}
        {start > 0 ? `  ·  ${start} lines above` : ""}
      </Text>
      <Box flexDirection="column" height={bodyHeight}>
        {warming ? null : body.map((line, i) => <Text key={i}>{line || " "}</Text>)}
      </Box>
    </Box>
  );
}

function App() {
  const { exit } = useApp();
  const { stdout } = useStdout();
  const [size, setSize] = useState({ columns: stdout.columns, rows: stdout.rows });
  const [states, setStates] = useState<PaneState[]>(PANES.map(initialPane));
  const [warming, setWarming] = useState(config.warmup);
  const [input, setInput] = useState("");
  const [prompt, setPrompt] = useState("");
  const [scroll, setScroll] = useState(0);
  const [now, setNow] = useState(performance.now());
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    const onResize = () => setSize({ columns: stdout.columns, rows: stdout.rows });
    stdout.on("resize", onResize);
    return () => void stdout.off("resize", onResize);
  }, [stdout]);

  const busy = states.some((s) => s.phase === "waiting" || s.phase === "streaming");
  useEffect(() => {
    if (!busy) return;
    const timer = setInterval(() => setNow(performance.now()), 100);
    return () => clearInterval(timer);
  }, [busy]);

  const run = useCallback((text: string, maxTokens: number) => {
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;
    setScroll(0);
    return Promise.all(
      PANES.map((pane, i) =>
        streamChat(
          { url: config.url, apiKey: config.apiKey, model: pane.model, prompt: text, maxTokens,
            temperature: config.temperature, signal: controller.signal },
          (event: StreamEvent) => {
            setStates((prev) => prev.map((s, j) => (j === i ? reduce(s, event) : s)));
            setNow(performance.now());
          },
        ),
      ),
    );
  }, []);

  const submit = useCallback(
    (text: string) => {
      if (!text.trim() || warming) return;
      setPrompt(text.trim());
      setInput("");
      void run(text.trim(), config.maxTokens);
    },
    [run, warming],
  );

  useEffect(() => {
    // Warm both containers with a one-token request, then run --prompt if given.
    const start = config.warmup ? run("Say OK.", 1) : Promise.resolve();
    void start.then(() => {
      setWarming(false);
      if (config.prompt) {
        setPrompt(config.prompt);
        void run(config.prompt, config.maxTokens);
      }
    });
  }, [run]);

  const paneWidth = Math.floor(size.columns / PANES.length);
  const bodyHeight = Math.max(3, size.rows - 12);
  useInput((_, key) => {
    if (key.escape) abort.current?.abort();
    if (key.upArrow) setScroll((s) => s + 1);
    if (key.downArrow) setScroll((s) => Math.max(0, s - 1));
    if (key.pageUp) setScroll((s) => s + bodyHeight);
    if (key.pageDown) setScroll((s) => Math.max(0, s - bodyHeight));
    if (key.ctrl && _ === "c") {
      abort.current?.abort();
      exit();
    }
  });

  return (
    <Box flexDirection="column" width={size.columns}>
      <Text>
        <Text bold>Journal memory: Teacher vs cartridge</Text>
        <Text dimColor>
          {"  "}
          {config.url} · temperature {config.temperature} · max {config.maxTokens} tokens
        </Text>
      </Text>
      <Text wrap="truncate-end">
        <Text dimColor>prompt: </Text>
        {prompt || "–"}
      </Text>
      <Box>
        {PANES.map((pane, i) => (
          <PaneView key={pane.model} pane={pane} state={states[i]} warming={warming} now={now}
                    width={paneWidth} bodyHeight={bodyHeight} scroll={scroll} />
        ))}
      </Box>
      <Box>
        <Text color="green">{"› "}</Text>
        <TextInput value={input} onChange={setInput} onSubmit={submit}
                   placeholder={warming ? "warming up…" : "ask both models a question"} />
      </Box>
      <Text dimColor>enter: send to both · ↑/↓ PgUp/PgDn: scroll · esc: stop · ctrl+c: quit</Text>
    </Box>
  );
}

render(<App />, { exitOnCtrlC: false });
