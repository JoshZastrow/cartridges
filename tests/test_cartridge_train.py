import json
import re

import pytest

torch = pytest.importorskip("torch")
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402

from cartridges.cartridge.cache import Cartridge  # noqa: E402
from cartridges.cartridge.train_distill import (  # noqa: E402
    Example,
    distill_loss,
    prepare_frozen,
    student_logits,
    to_tensors,
    train,
)

V = 97
K = 8


def tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    config = Qwen3_5TextConfig(
        vocab_size=V,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        initializer_range=0.2,
    )
    return Qwen3_5ForCausalLM(config).eval()


def rand_ids(n: int, seed: int) -> list[int]:
    return torch.randint(0, V, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


@torch.no_grad()
def teacher_examples(model, context: list[int], n: int) -> list[Example]:
    """Conversations scored by the model with `context` in front: the
    distribution the cartridge has to learn to stand in for."""
    out = []
    for i in range(n):
        prompt, completion = rand_ids(6, 100 + i), rand_ids(10, 200 + i)
        ids = torch.tensor([context + prompt + completion])
        logits = model(input_ids=ids).logits[0, -len(completion) - 1 : -1]
        lp, top = torch.log_softmax(logits.float(), -1).topk(K, dim=-1)
        out.append(Example(id=str(i), prompt_ids=prompt, completion_ids=completion,
                           topk_ids=top.tolist(), topk_logprobs=lp.tolist()))
    return out


def setup(checkpointing: bool):
    model = tiny_model()
    context = rand_ids(120, 1)
    examples = teacher_examples(model, context, 8)
    # Initialized from other text, so the cartridge starts off wrong.
    cart = Cartridge.from_prefill(model, torch.tensor([rand_ids(16, 2)]))
    prepare_frozen(model, gradient_checkpointing=checkpointing)
    return model, cart, examples


def eval_kl(model, cart, examples) -> float:
    with torch.no_grad():
        total = 0.0
        for ex in examples:
            ids, top, lp = to_tensors(ex, 2048)
            total += distill_loss(student_logits(model, cart, ids, top.shape[0]), top, lp)[1].item()
    return total / len(examples)


def test_distill_loss_is_ce_against_topk():
    logits = torch.randn(3, V)
    top = torch.stack([torch.randperm(V)[:K] for _ in range(3)])
    teacher = torch.log_softmax(torch.randn(3, K), -1)
    loss, kl = distill_loss(logits, top, teacher)
    student = torch.log_softmax(logits, -1).gather(-1, top)
    assert loss.item() == pytest.approx(-(teacher.exp() * student).sum(-1).mean().item())
    # student == teacher on the top-k support: KL is 0
    exact = torch.full((3, V), -1e4).scatter(-1, top, teacher)
    assert distill_loss(exact, top, teacher)[1].item() == pytest.approx(0.0, abs=1e-4)


def test_only_cartridge_gets_grads():
    model, cart, examples = setup(checkpointing=True)
    ids, top, lp = to_tensors(examples[0], 2048)
    distill_loss(student_logits(model, cart, ids, top.shape[0]), top, lp)[0].backward()
    assert [n for n, p in model.named_parameters() if p.grad is not None] == []
    trainable = [(n, p) for n, p in cart.named_parameters() if p.requires_grad]
    assert trainable and all(p.grad is not None and p.grad.abs().sum() > 0 for _, p in trainable), [
        n for n, p in trainable if p.grad is None or p.grad.abs().sum() == 0
    ]


def test_loss_decreases():
    model, cart, examples = setup(checkpointing=True)
    before = eval_kl(model, cart, examples)
    logged = []
    train(model, cart, examples, lr=2e-2, batch_size=4, epochs=25, log=lambda m, step: logged.append(m))
    after = eval_kl(model, cart, examples)
    assert after < 0.5 * before, (before, after)
    assert logged[-1]["train/loss"] < logged[0]["train/loss"]


def test_gradient_checkpointing_matches_plain_backward():
    grads = []
    for checkpointing in (False, True):
        model, cart, examples = setup(checkpointing)
        ids, top, lp = to_tensors(examples[0], 2048)
        loss, _ = distill_loss(student_logits(model, cart, ids, top.shape[0]), top, lp)
        loss.backward()
        grads.append([p.grad.clone() for p in cart.parameters() if p.requires_grad])
    for plain, ckpt in zip(*grads, strict=True):
        torch.testing.assert_close(ckpt, plain, rtol=1e-4, atol=1e-5)


def test_long_example_is_cut_to_max_seq_len():
    ex = Example(id="x", prompt_ids=[1] * 5, completion_ids=[2] * 10,
                 topk_ids=[[2] * K] * 10, topk_logprobs=[[0.0] * K] * 10)
    ids, top, lp = to_tensors(ex, 12)
    assert ids.shape == (1, 12) and top.shape == (7, K) and lp.shape == (7, K)


# ---------------------------------------------------------------- data.py


def _transcripts(tmp_path):
    from cartridges.batch.outputs import collate

    for journal, n_pages in (("A", 12), ("B", 8)):
        d = tmp_path / journal
        d.mkdir()
        pages = [f"## p. {n} - date: 2021-11-{n:02d}\n\nI walked {n} miles with Sam." for n in range(1, n_pages + 1)]
        (d / "transcript.md").write_text("\n\n---\n".join(pages) + "\n")
    collate(tmp_path)
    return tmp_path / "timeline.md"


class FakeTeacher:
    """Bot A echoes a question; bot B answers with fake top-k. Records calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, batch, **kw):
        from cartridges.cartridge.teacher import Completion

        self.calls.append((batch, kw))
        out = []
        for i, messages in enumerate(batch):
            text = "\n".join(m["content"] for m in messages)
            if '"pairs"' in text:  # a QA prompt
                page = re.search(r"\[(\w+:p\.\d+)\]", text).group(1)
                n = page.split(".")[-1]
                pair = {"question": f"How far did I walk on page {page}?", "answer": f"{n} miles",
                        "answer_type": "short_text", "citations": [page], "evidence": f"I walked {n} miles with Sam."}
                out.append(Completion(text="```json\n" + json.dumps({"pairs": [pair]}) + "\n```",
                                      prompt_ids=[1], completion_ids=[5]))
            elif "<info>" in text:  # bot A: JSON with the question and its reference
                page = re.search(r"\[(\w+:p\.\d+)\]", text).group(1)
                ref = {"question": "How far did I walk on 2021-11-02?", "citations": [page],
                       "evidence": [f"I walked {page.split('.')[-1]} miles with Sam."]}
                out.append(Completion(text=json.dumps(ref), prompt_ids=[1], completion_ids=[5]))
            elif i == 0:
                out.append(None)  # a failed call is dropped
            else:
                out.append(Completion(text="2 miles", prompt_ids=[1, 2, 3], completion_ids=[7, 8],
                                      topk_ids=[[7, 9], [8, 9]], topk_logprobs=[[-0.1, -2.0], [-0.2, -1.5]],
                                      finish_reason="stop"))
        return out


def test_build_selfstudy_rows_are_examples(tmp_path):
    from cartridges.cartridge.chunks import sample_chunks
    from cartridges.cartridge.data import build_selfstudy

    chunks = sample_chunks(_transcripts(tmp_path), 3, min_tokens=10, max_tokens=60)
    teacher = FakeTeacher()
    rows = build_selfstudy(teacher, chunks, kinds=("question", "creative"))

    (a_batch, a_kw), (b_batch, b_kw) = teacher.calls
    assert len(a_batch) == 6 and a_kw["with_context"] is False and a_kw["temperature"] == 0.6
    assert all(m[0]["role"] == "system" and "<info>" in m[0]["content"] and m[1]["role"] == "user" for m in a_batch)
    # bot B: the question alone (parsed from bot A's JSON), timeline in context, greedy, top-k logprobs
    assert b_batch[0] == [{"role": "user", "content": "How far did I walk on 2021-11-02?"}]
    assert b_kw["with_context"] is True and b_kw["temperature"] == 0.0 and b_kw["logprobs"] == 20
    assert len(rows) == 5
    examples = [Example.model_validate(r) for r in rows]
    ids, top, lp = to_tensors(examples[0], 2048)
    assert ids.tolist() == [[1, 2, 3, 7, 8]] and top.shape == (2, 2)
    assert {r["kind"] for r in rows} == {"question", "creative"}
    assert all(r["grounded"] and len(r["citations"]) == 1 and r["evidence"] for r in rows)


def test_build_qa_keeps_verified_pairs(tmp_path):
    from cartridges.cartridge.chunks import sample_chunks
    from cartridges.cartridge.data import build_qa

    chunks = sample_chunks(_transcripts(tmp_path), 4, min_tokens=10, max_tokens=60)
    rows, counts = build_qa(FakeTeacher(), chunks)
    assert rows and counts["qa/written"] == counts["qa/verified"] == 4
    assert counts["qa/duplicates"] == 4 - len(rows)
    assert len({r["question"] for r in rows}) == len(rows)
    assert set(rows[0]) >= {"question", "answer", "answer_type", "citations", "evidence", "chunk_id"}


def test_quality_stats_and_heldout_assertion(tmp_path):
    from cartridges.cartridge.chunks import sample_chunks
    from cartridges.cartridge.data import build_qa, build_selfstudy, quality_stats
    from cartridges.cartridge.evals import load_pages

    timeline = _transcripts(tmp_path)
    chunks = sample_chunks(timeline, 3, min_tokens=10, max_tokens=60)
    selfstudy = build_selfstudy(FakeTeacher(), chunks, kinds=("question",))
    qa, counts = build_qa(FakeTeacher(), chunks)
    pages = load_pages(tmp_path)
    stats = quality_stats(selfstudy, chunks, qa, counts, pages, heldout=set())
    assert stats["qa/verify_pass_rate"] == 1.0 and stats["selfstudy/truncation_rate"] == 0.0
    assert stats["selfstudy/grounded_rate"] == 1.0 and stats["selfstudy/teacher_cites_reference"] == 0.0
    assert sum(stats["selfstudy/page_coverage"].values()) == sum(len(c.citations) for c in chunks)
    with pytest.raises(AssertionError, match="held-out"):
        quality_stats(selfstudy, chunks, qa, counts, pages, heldout={qa[0]["citations"][0]})


def test_labeled_targets_feed_training(tmp_path):
    """Teacher's Target Store rows (read_labeled) are Examples that to_tensors and the loss take."""
    import uuid

    from cartridges.cartridge.tasks import targets
    from cartridges.cartridge.tasks.types import Label, TargetRow

    model = tiny_model()
    ex = teacher_examples(model, rand_ids(40, 1), 2)
    labels, rows = [], []
    for i, e in enumerate(ex):
        label_id, task_id = uuid.uuid4(), uuid.uuid4()
        rows.append(TargetRow(label_id=str(label_id), task_id=str(task_id), prompt_ids=e.prompt_ids,
                              completion_ids=e.completion_ids, topk_ids=e.topk_ids, topk_logprobs=e.topk_logprobs))
        labels.append(Label(label_id=label_id, task_id=task_id, run_id="r", teacher_model="m", corpus_id="c",
                            instructions_hash="h", answer="a", finish_reason="stop", n_prompt_tokens=6,
                            n_completion_tokens=10, status="validated", targets_uri=str(tmp_path / "part-00000.parquet"),
                            targets_row=i))
    targets.write_shard(rows, tmp_path / "part-00000.parquet")
    examples = list(targets.read_labeled(labels))
    assert [e.id for e in examples] == [str(lab.label_id) for lab in labels]
    ids, top, lp = to_tensors(examples[0], 2048)
    torch.testing.assert_close(lp, to_tensors(ex[0], 2048)[2], rtol=0, atol=1e-6)  # float32 round trip
    cart = Cartridge.from_prefill(model, torch.tensor([rand_ids(16, 2)]))
    prepare_frozen(model)
    logged = []
    train(model, cart, examples, batch_size=1, epochs=3, relative_lr=True, log=lambda m, s: logged.append(m["train/kl"]))
    assert logged[-1] < logged[0]


# ---------------------------------------------------------------- origin-weighted sampling


def test_weighted_order_follows_mix_and_cycles_evenly():
    from collections import Counter

    from cartridges.cartridge.train_distill import effective_epochs, weighted_order

    # the real label counts: chat 316, synthetic 133, psych_test 2486
    groups = ["chat"] * 316 + ["synthetic"] * 133 + ["psych_test"] * 2486
    mix = {"chat": 0.40, "synthetic": 0.35, "psych_test": 0.25}
    order = weighted_order(groups, mix, n=9944, seed=0)  # one psych_test epoch's worth of draws
    share = {g: c / len(order) for g, c in Counter(groups[i] for i in order).items()}
    assert all(abs(share[g] - w) < 0.02 for g, w in mix.items()), share
    epochs = effective_epochs(groups, order)
    assert abs(epochs["chat"] - 12.6) < 0.5 and abs(epochs["synthetic"] - 26.2) < 1.2
    assert abs(epochs["psych_test"] - 1.0) < 0.1
    # within a group, examples repeat evenly (reshuffled cycles), not at random
    synth = Counter(i for i in order if groups[i] == "synthetic")
    assert max(synth.values()) - min(synth.values()) <= 1
    assert weighted_order(groups, mix, n=100, seed=1) == weighted_order(groups, mix, n=100, seed=1)
    with pytest.raises(ValueError, match="no examples"):
        weighted_order(["chat"], mix, n=5)


def test_label_origins_tags_labels_and_rejects_eval_tasks():
    import uuid
    from types import SimpleNamespace as NS

    from cartridges.cartridge.train_distill import DistillConfig, label_origins

    train = [NS(task_id=uuid.uuid4(), origin=o) for o in ("chat", "psych_test")]
    held = NS(task_id=uuid.uuid4(), origin="human")
    labels = [NS(label_id=uuid.uuid4(), task_id=t.task_id) for t in train]
    tasks = {"train": train, "eval": [held], "exemplar": []}
    assert label_origins(labels, tasks) == ["chat", "psych_test"]
    with pytest.raises(AssertionError, match="another split"):
        label_origins(labels + [NS(label_id=uuid.uuid4(), task_id=held.task_id)], tasks)
    with pytest.raises(ValueError, match="not a validated train task"):
        label_origins([NS(label_id=uuid.uuid4(), task_id=uuid.uuid4())], tasks)
    with pytest.raises(ValueError, match="samples > 0"):
        DistillConfig(origin_mix={"chat": 1.0}, label_runs=["r"])
    # validation labels: eval tasks only, and a train task is rejected
    held_label = NS(label_id=uuid.uuid4(), task_id=held.task_id)
    assert label_origins([held_label], tasks, split="eval") == ["human"]
    with pytest.raises(AssertionError, match="eval labels on tasks of another split"):
        label_origins([labels[0]], tasks, split="eval")


def test_train_follows_explicit_order():
    model, cart, examples = setup(checkpointing=False)
    seen = []
    import cartridges.cartridge.train_distill as td

    real = td.to_tensors
    td.to_tensors = lambda ex, n: (seen.append(ex.id), real(ex, n))[1]
    try:
        train(model, cart, examples, batch_size=2, order=[3, 3, 0, 5, 1])
    finally:
        td.to_tensors = real
    assert seen == ["3", "3", "0", "5", "1"]


# ---------------------------------------------------------------- task answers + judge


def test_answer_messages_reports_stop_and_length():
    from test_cartridge_cache import CharTokenizer

    from cartridges.cartridge.train_distill import answer_messages

    class Tok(CharTokenizer):
        def decode(self, ids, skip_special_tokens=True):
            return " ".join(map(str, ids))

    model = tiny_model()
    cart = Cartridge.from_prefill(model, torch.tensor([rand_ids(16, 2)]))
    convs = [[{"role": "user", "content": "hi"}], [{"role": "user", "content": "how are you"}]]
    free = answer_messages(model, Tok(), cart, convs, max_new_tokens=6, eos_ids=[V + 5])  # unreachable id
    assert [r["finish_reason"] for r in free] == ["length", "length"] and [r["n_tokens"] for r in free] == [6, 6]
    first = int(free[0]["text"].split()[0])  # greedy: make the first token the stop id
    stopped = answer_messages(model, Tok(), cart, convs[:1], max_new_tokens=6, eos_ids=[first])
    assert stopped == [{"text": "", "finish_reason": "stop", "n_tokens": 1}]


def test_task_judge_pairs_cartridge_as_x_and_teacher_as_y():
    import uuid
    from types import SimpleNamespace as NS

    from cartridges.cartridge.tasks.judge import load_rubric
    from cartridges.cartridge.train_distill import judge_scalars, task_judge

    tasks = [NS(task_id=uuid.uuid4(), origin=o, kind="self_synthesis", text=f"task {i}")
             for i, o in enumerate(("chat", "synthetic", "psych_test"))]
    asked = []

    def answer(convs):
        asked.extend(convs)
        return [{"text": f"CART {c[-1]['content']}", "finish_reason": "stop", "n_tokens": 3} for c in convs]

    def judge(batch, **kw):  # prefers the teacher in both orders
        assert kw["with_context"] is True and kw["temperature"] == 0.0
        out = []
        for m in batch:
            text = m[0]["content"]
            winner = "A" if text.index("TEACHER") < text.index("CART") else "B"
            out.append(NS(text=json.dumps({"winner": winner, "factual_errors": "none"})))
        return out

    outcomes, summary, answers = task_judge(answer, tasks, [f"TEACHER {t.text}" for t in tasks], judge, load_rubric("v0"))
    from cartridges.cartridge.tasks.label import answer_messages

    assert asked == [answer_messages(t.text) for t in tasks]  # the teacher's task format
    assert [o.winner for o in outcomes] == ["y", "y", "y"] and all(o.consistent for o in outcomes)
    scalars = judge_scalars(summary)
    assert scalars["judge/all/loss"] == 1.0 and scalars["judge/origin:chat/n"] == 1
    assert answers[0]["text"] == "CART task 0"


def test_stratified_subset_quotas_match_the_eval_split():
    import uuid
    from collections import Counter
    from types import SimpleNamespace as NS

    from cartridges.cartridge.train_distill import stratified_subset

    tasks = [NS(task_id=uuid.uuid4(), origin=o) for o, k in (("chat", 34), ("psych_test", 189), ("human", 8))
             for _ in range(k)]
    final = stratified_subset(tasks, 60, seed=0)
    assert Counter(t.origin for t in final) == {"human": 8, "chat": 26, "psych_test": 26}
    ckpt = stratified_subset(final, 24, seed=0)
    assert Counter(t.origin for t in ckpt) == {"human": 8, "chat": 8, "psych_test": 8}
    assert {t.task_id for t in ckpt} <= {t.task_id for t in final}
    assert stratified_subset(list(reversed(tasks)), 60, seed=0) == final  # input order does not matter


def test_val_kl_is_teacher_forced_and_per_origin():
    from cartridges.cartridge.train_distill import LabeledSet, val_kl

    model, cart, examples = setup(checkpointing=True)
    val = LabeledSet(examples=examples[:4], origins=["chat", "chat", "psych_test", "psych_test"],
                     task_ids=["a", "b", "c", "d"])
    before = val_kl(model, cart, val, 2048)
    assert set(before) == {"val/kl", "val/kl/chat", "val/kl/psych_test"}
    assert all(p.grad is None for p in cart.parameters())  # no gradients
    train(model, cart, examples, batch_size=4, epochs=25, relative_lr=False)
    assert val_kl(model, cart, val, 2048)["val/kl"] < before["val/kl"]


def test_warmup_scales_lr_linearly():
    model, cart, examples = setup(checkpointing=False)
    lrs = []
    train(model, cart, examples, lr=0.01, batch_size=2, max_steps=4, warmup_steps=4,
          log=lambda m, s: lrs.append(round(m["train/lr"], 6)))
    assert lrs == [0.0025, 0.005, 0.0075, 0.01]


def test_chunked_student_loss_matches_full_logits():
    from cartridges.cartridge.train_distill import student_loss

    model, cart, examples = setup(checkpointing=True)
    ids, top, lp = to_tensors(examples[0], 4096)
    full_loss, full_kl = distill_loss(student_logits(model, cart, ids, top.shape[0]), top, lp)
    full_loss.backward()
    full_grads = [p.grad.clone() for p in cart.parameters() if p.requires_grad]
    for p in cart.parameters():
        p.grad = None
    loss, kl = student_loss(model, cart, ids, top, lp, chunk=3)  # T=10: chunks of 3, 3, 3, 1
    loss.backward()
    torch.testing.assert_close(loss, full_loss)
    torch.testing.assert_close(kl, full_kl)
    for a, b in zip(full_grads, [p.grad for p in cart.parameters() if p.requires_grad], strict=True):
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-6)


def test_current_labels_keeps_one_instructions_hash():
    from types import SimpleNamespace as NS

    from cartridges.cartridge.train_distill import current_labels

    labs = [NS(status="validated", targets_uri="u", instructions_hash=h) for h in ("new", "old", "new")]
    labs.append(NS(status="rejected", targets_uri="u", instructions_hash="new"))
    assert len(current_labels(labs, "new")) == 2
    with pytest.raises(ValueError, match="no labels with instructions_hash"):
        current_labels(labs, "other")


def test_task_prompts_match_the_teacher_format():
    """Student prompts for tasks come from label.answer_messages, the format the teacher labels."""
    from cartridges.cartridge.tasks.label import answer_messages, task_messages

    task = type("T", (), {"text": "What did I write about running?"})()
    assert task_messages(task) == answer_messages(task.text)


def test_label_prompts_and_student_answers_render_identically():
    """A label row's prompt_ids (label.to_prompt -> teacher.render(task_messages)) equal what the
    student sees when answering (answer_messages via train_distill._prompt_ids), reminder included;
    check_prompts rejects a label rendered as a bare user turn."""
    from test_cartridge_cache import CharTokenizer

    from cartridges.cartridge.tasks.label import CITATION_REMINDER, answer_messages, to_prompt
    from cartridges.cartridge.teacher import render
    from cartridges.cartridge.train_distill import _prompt_ids, check_prompts

    tok, text = CharTokenizer(), "What did I write about running?"
    task = type("T", (), {"text": text})()
    label_prompt = list(to_prompt(task, tok))
    assert label_prompt == _prompt_ids(tok, answer_messages(text)) == render(tok, answer_messages(text))[1]
    assert tok.encode(CITATION_REMINDER) == label_prompt[len(tok.encode("<|im_start|>system\n")):][: len(tok.encode(CITATION_REMINDER))]

    good = Example(id="a", prompt_ids=label_prompt, completion_ids=[5], topk_ids=[[5]], topk_logprobs=[[0.0]])
    check_prompts([good], [text], tok)
    bare = good.model_copy(update={"id": "b", "prompt_ids": render(tok, [{"role": "user", "content": text}])[1]})
    with pytest.raises(ValueError, match="differ"):
        check_prompts([good, bare], [text, text], tok)


def test_early_stopping_keeps_best_and_stops_after_patience():
    from cartridges.cartridge.train_distill import EarlyStopping

    saved = []
    es = EarlyStopping(3, save=lambda: saved.append(es.step))
    stops = [es.update(kl, step) for step, kl in [(0, 1.0), (10, 0.8), (20, 0.85), (30, 0.7), (40, 0.75), (50, 0.71), (60, 0.9)]]
    assert stops == [False, False, False, False, False, False, True]
    assert (es.best, es.step) == (0.7, 30) and saved == [0, 10, 30]
    never = EarlyStopping(0, save=lambda: saved.append("x"))
    assert not any(never.update(kl, i) for i, kl in enumerate([1, 2, 3, 4, 5]))
    assert "x" not in saved


def test_train_stops_when_on_step_says_so_and_rejects_nan():
    model, cart, examples = setup(checkpointing=False)
    steps = []
    train(model, cart, examples, batch_size=1, epochs=1, log=lambda m, s: steps.append(s), on_step=lambda s: s == 3)
    assert steps == [1, 2, 3]
    bad = examples[0].model_copy(update={"topk_logprobs": [[float("nan")] * K] * len(examples[0].completion_ids)})
    with pytest.raises(FloatingPointError):
        train(model, cart, [bad], batch_size=1)


def test_subset_labeled_is_stratified():
    from collections import Counter

    from cartridges.cartridge.train_distill import LabeledSet, subset_labeled

    origins = ["chat"] * 34 + ["psych_test"] * 189
    ls = LabeledSet(examples=[Example(id=str(i), prompt_ids=[1], completion_ids=[2], topk_ids=[[2]], topk_logprobs=[[0.0]])
                              for i in range(len(origins))], origins=origins, task_ids=[f"t{i}" for i in range(len(origins))])
    sub = subset_labeled(ls, 40, seed=0)
    assert Counter(sub.origins) == {"chat": 20, "psych_test": 20}
    assert [e.id for e in sub.examples] == sorted((e.id for e in sub.examples), key=int)


def test_score_examples_matches_loss_and_teacher_topk():
    from cartridges.cartridge.train_distill import score_examples, student_loss

    model, cart, examples = setup(checkpointing=False)
    rows = score_examples(model, cart, examples[:2], task_ids=["t0", "t1"], chunk=4)
    ex = examples[0]
    ids, top, lp = to_tensors(ex, 4096)
    _, kl = student_loss(model, cart, ids, top, lp)
    assert rows[0]["task_id"] == "t0" and rows[0]["id"] == ex.id
    assert rows[0]["n_tokens"] == len(ex.completion_ids) == len(rows[0]["logprobs"])
    assert rows[0]["kl_topk"] == pytest.approx(kl.item(), rel=1e-4)
    logq = torch.log_softmax(student_logits(model, cart, ids, top.shape[0]).float(), -1)
    actual = logq.gather(-1, torch.tensor(ex.completion_ids)[:, None])[:, 0]
    torch.testing.assert_close(torch.tensor(rows[0]["logprobs"]), actual, rtol=1e-4, atol=1e-5)
    assert rows[0]["argmax"] == logq.argmax(-1).tolist()
    for t, tok in enumerate(ex.completion_ids):  # teacher logprob when the token is in its top-k
        expect = ex.topk_logprobs[t][ex.topk_ids[t].index(tok)] if tok in ex.topk_ids[t] else None
        assert rows[0]["teacher_logprobs"][t] == (pytest.approx(expect) if expect is not None else None)
    base = score_examples(model, None, examples[:1])
    plain = torch.log_softmax(model(input_ids=ids).logits[0, -top.shape[0] - 1 : -1].float(), -1)
    torch.testing.assert_close(torch.tensor(base[0]["logprobs"]),
                               plain.gather(-1, torch.tensor(ex.completion_ids)[:, None])[:, 0], rtol=1e-4, atol=1e-5)
    cut = score_examples(model, cart, [ex], max_seq_len=len(ex.prompt_ids) + 3)
    assert cut[0]["n_tokens"] == 3 == len(cut[0]["logprobs"])


def test_score_systems_base_only_without_cartridge():
    from cartridges.cartridge.train_distill import score_examples, score_systems

    model, cart, examples = setup(checkpointing=False)
    both = score_systems(model, cart, examples[:2], ["t0", "t1"])
    base_only = score_systems(model, None, examples[:2], ["t0", "t1"])
    assert set(both) == {"cartridge", "base"} and set(base_only) == {"base"}
    assert base_only["base"] == both["base"] == score_examples(model, None, examples[:2], ["t0", "t1"])
    assert both["cartridge"][0]["logprobs"] != both["base"][0]["logprobs"]  # the cartridge changes the scores
    assert set(score_systems(model, cart, examples[:1], ["t0"], with_base=False)) == {"cartridge"}


def test_teacher_written_shard_reads_under_our_mount(tmp_path):
    """Labels store the teacher's /cache/... path; read_labeled(root=<our mount>) resolves it."""
    import uuid

    from cartridges.cartridge.tasks import targets
    from cartridges.cartridge.tasks.types import Label, TargetRow

    label_id, task_id = uuid.uuid4(), uuid.uuid4()
    targets.write_shard([TargetRow(label_id=str(label_id), task_id=str(task_id), prompt_ids=[1], completion_ids=[2],
                                   topk_ids=[[2]], topk_logprobs=[[0.0]])], tmp_path / "targets/r/part-0.parquet")
    lab = Label(label_id=label_id, task_id=task_id, run_id="r", teacher_model="m", corpus_id="c", instructions_hash="h",
                answer="a", finish_reason="stop", n_prompt_tokens=1, n_completion_tokens=1, status="validated",
                targets_uri="/cache/targets/r/part-0.parquet", targets_row=0)
    with pytest.raises(FileNotFoundError):
        list(targets.read_labeled([lab]))
    assert [e.id for e in targets.read_labeled([lab], root=str(tmp_path))] == [str(label_id)]
    relative = lab.model_copy(update={"targets_uri": "targets/r/part-0.parquet"})
    assert [e.id for e in targets.read_labeled([relative], root=str(tmp_path))] == [str(label_id)]


def test_missing_lineage_artifact_is_a_warning_not_a_crash():
    from cartridges.cartridge.train_distill import use_input

    class Run:
        def __init__(self):
            self.summary = {}

        def use_artifact(self, ref, type=None, download=True):
            if ref.startswith("targets-missing"):
                raise RuntimeError("artifact membership not found")

    run = Run()
    assert use_input(run, "targets-ok", "dataset") is True
    assert use_input(run, "targets-missing1", "dataset") is False
    assert use_input(run, "targets-missing2", "dataset") is False
    assert run.summary["lineage/missing"] == ["targets-missing1", "targets-missing2"]


def test_pick_val_by_ids_is_stable_when_the_pool_changes():
    from cartridges.cartridge.train_distill import LabeledSet, pick_val

    def pool(ids, origins):
        return LabeledSet(examples=[Example(id=i, prompt_ids=[1], completion_ids=[2], topk_ids=[[2]],
                                            topk_logprobs=[[0.0]]) for i in ids], origins=origins, task_ids=ids)

    before = pool(["a", "b", "c", "d"], ["chat", "chat", "psych_test", "psych_test"])
    chosen = [e.id for e in pick_val(before, [], 2, 0).examples]
    after = pool(["e", "d", "c", "b", "a"], ["psych_test", "psych_test", "psych_test", "chat", "chat"])  # grew
    assert [e.id for e in pick_val(after, chosen, 2, 0).examples] == chosen
    with pytest.raises(ValueError, match="not found"):
        pick_val(pool(["a"], ["chat"]), ["a", "zz"], 0, 0)
    assert len(pick_val(before, [], 0, 0).examples) == 4


def test_early_stopping_seeded_from_a_resumed_checkpoint():
    from cartridges.cartridge.train_distill import EarlyStopping

    saves = []
    es = EarlyStopping(3, save=lambda: saves.append(es.step), best=0.3856, step=154)
    assert [es.update(kl, s) for s, kl in [(160, 0.39), (170, 0.3857), (180, 0.38), (190, 0.381), (200, 0.382), (210, 0.383)]] \
        == [False, False, False, False, False, True]
    assert saves == [180] and es.saved and (es.best, es.step) == (0.38, 180)
    never = EarlyStopping(2, save=lambda: saves.append("x"), best=0.3)
    assert [never.update(0.4, 1), never.update(0.4, 2)] == [False, True] and not never.saved


def test_resume_order_continues_the_original_draws():
    from cartridges.cartridge.train_distill import weighted_order

    origins = ["chat"] * 30 + ["synthetic"] * 40 + ["psych_test"] * 200
    mix = {"chat": 0.4, "synthetic": 0.35, "psych_test": 0.25}
    full = weighted_order(origins, mix, 4800, 0)
    resumed = weighted_order(origins, mix, 4800, 0)[154 * 16 :]
    assert resumed == full[2464:] and len(resumed) == 146 * 16


def test_stream_answers_persists_each_row_and_resumes(tmp_path):
    from cartridges.cartridge.train_distill import stream_answers

    tasks = [{"task_id": f"t{i}", "text": f"q{i}"} for i in range(7)]
    decoded, commits, logs = [], [], []

    def answer_one(text):
        decoded.append(text)
        if text == "q5":
            raise RuntimeError("client gone")  # interruption mid-run
        return {"text": text.upper(), "finish_reason": "stop", "n_tokens": 3}

    out = tmp_path / "answers" / "run.jsonl"
    rows = []
    with pytest.raises(RuntimeError):
        for row in stream_answers(answer_one, tasks, out, commit=lambda: commits.append(1), every=2, log=logs.append):
            rows.append(row)
            assert json.loads(out.read_text().splitlines()[-1]) == row  # persisted before it is yielded
    assert [r["task_id"] for r in rows] == ["t0", "t1", "t2", "t3", "t4"] and len(commits) == 2
    assert any("2/7 new" in line and "tok/s" in line and "ETA" in line for line in logs)

    decoded.clear()
    resumed = list(stream_answers(lambda t: {"text": t.upper(), "finish_reason": "length", "n_tokens": 4},
                                  tasks, out, every=2, log=logs.append))
    assert [r["task_id"] for r in resumed] == [f"t{i}" for i in range(7)]  # all, in order: 5 from file, 2 new
    assert decoded == [] and resumed[0]["finish_reason"] == "stop" and resumed[6]["finish_reason"] == "length"
    assert len(out.read_text().splitlines()) == 7
