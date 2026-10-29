import assert from "node:assert/strict";
import { test } from "node:test";
import { initialPane, reduce, sseData, stats } from "./stream.js";

test("stats: first token, decode rate after the first chunk, total time", () => {
  let s = reduce(initialPane(), { kind: "start", at: 1000 });
  s = reduce(s, { kind: "delta", text: "", tokens: 0, at: 1100 }); // role chunk: not a token
  assert.equal(s.firstTokenAt, undefined);
  s = reduce(s, { kind: "delta", text: "Hel", tokens: 1, at: 1500 });
  s = reduce(s, { kind: "delta", text: "lo", tokens: 4, at: 2500 });
  assert.deepEqual(stats(s, 2500), { elapsedS: 1.5, ttftS: 0.5, tokPerS: 4 });
  s = reduce(s, { kind: "end", finishReason: "stop", completionTokens: 5, at: 3000 });
  assert.equal(s.text, "Hello");
  assert.deepEqual(stats(s, 9999), { elapsedS: 2, ttftS: 0.5, tokPerS: 4 / 1.5 });
});

test("sseData: payloads split across reads, stops at [DONE]", async () => {
  const enc = new TextEncoder();
  async function* body() {
    yield enc.encode('data: {"a":1}\n\ndata: {"b"');
    yield enc.encode(':2}\n\ndata: [DONE]\n\ndata: {"c":3}\n\n');
  }
  const out = [];
  for await (const x of sseData(body())) out.push(x);
  assert.deepEqual(out, [{ a: 1 }, { b: 2 }]);
});
