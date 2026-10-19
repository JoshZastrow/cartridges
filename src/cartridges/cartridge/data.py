"""Training data for the cartridge: self-study conversations and QA pairs.

Self-study (Cartridges, arXiv 2506.06266, section 4.1), one conversation per
(chunk, seed prompt), chunks from the whole corpus (chunks.sample_chunks), as
in the paper: the cartridge has to stand in for all of it.

1. Bot A writes the user turn. It sees the chunk (system turn) and a seed
   prompt (selfstudy.seed_prompts), but not the timeline (with_context=False):
   the chunk is what it should ask about. Temperature 0.6, as in the paper.
2. Bot B answers that user turn with the full timeline in context, greedily
   (temperature 0), and the teacher returns its raw top-k logprobs at every
   generated token: the distillation targets. Its `prompt_ids` are the
   conversation without the context block, which is what the student sees
   after the cartridge.

Each row is a train_distill.Example plus its chunk, seed kind, and texts.

QA pairs (GRPO prompts and references): the teacher writes question/answer
pairs about a chunk (chunks.qa_messages) that avoids eval's held-out pages
(data/cartridge/eval/heldout_pages.json), and chunks.grounded_qa keeps the
ones whose citations and evidence check out. eval builds qa_heldout.jsonl.

`generate(batch, **kw)` is any callable that answers a list of conversations
with one teacher.Completion each (None on failure): `modal_generate(teacher)`
for the deployed Teacher, a fake in tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from cartridges.cartridge.chunks import Chunk, grounded_qa, qa_messages, sample_chunks
from cartridges.cartridge.selfstudy import KINDS, parse_seed_reply, seed_prompts

logger = logging.getLogger(__name__)

Generate = Callable[..., list]  # (list[messages], **kw) -> list[Completion | None]

LOCAL_DIR = Path("data/cartridge")
VOLUME_DIR = "data"  # on the cartridges-cartridge-cache volume (/vol/data in train_distill)
TEMPERATURE_A = 0.6
TEMPERATURE_B = 0.0
TOP_K = 20
CORPUS = "data/journals/output/2026_10_13/teacher_context.md"
CORPUS_META = {"corpus": "teacher_context.md", "corpus_tokens": 204512, "corpus_chars": 812699}


def build_selfstudy(
    generate: Generate,
    chunks: list[Chunk],
    kinds: tuple[str, ...] = KINDS,
    seed: int = 0,
    max_question_tokens: int = 512,  # JSON: question + citations + evidence quotes
    max_answer_tokens: int = 1024,
    top_k: int = TOP_K,
) -> list[dict]:
    """One scored conversation per (chunk, kind). Each row keeps Bot A's
    citations and evidence quotes and whether they check out against the chunk
    (`grounded`), the reference for grounding checks on answers."""
    by_id = {c.id: c for c in chunks}
    seeds = [s for c in chunks for s in seed_prompts(c, kinds, seed)]
    replies = generate(
        [s.messages for s in seeds],
        max_tokens=max_question_tokens, temperature=TEMPERATURE_A, with_context=False, seed=seed,
    )
    asked = [(s, parse_seed_reply(by_id[s.chunk_id], r.text)) for s, r in zip(seeds, replies, strict=True)
             if r is not None]
    asked = [(s, ref) for s, ref in asked if ref]
    if seeds and not asked:
        raise RuntimeError("no teacher reply to any seed prompt; see the warnings above")
    answers = generate(
        [[{"role": "user", "content": ref["question"]}] for _, ref in asked],
        max_tokens=max_answer_tokens, temperature=TEMPERATURE_B, logprobs=top_k, with_context=True,
    )
    rows = []
    for (s, ref), a in zip(asked, answers, strict=True):
        if a is None or not a.completion_ids or len(a.topk_ids) != len(a.completion_ids):
            continue
        rows.append({
            "id": f"{s.chunk_id}#{s.kind}",
            "chunk_id": s.chunk_id,
            "kind": s.kind,
            "question": ref["question"],
            "citations": ref["citations"],
            "evidence": ref["evidence"],
            "grounded": ref["grounded"],
            "answer": a.text,
            "finish_reason": a.finish_reason,
            "prompt_ids": a.prompt_ids,
            "completion_ids": a.completion_ids,
            "topk_ids": a.topk_ids,
            "topk_logprobs": a.topk_logprobs,
        })
    logger.info("self-study: %d seeds, %d questions, %d scored conversations", len(seeds), len(asked), len(rows))
    return rows


def build_qa(
    generate: Generate, chunks: list[Chunk], pairs_per_chunk: int = 4, max_tokens: int = 2048
) -> tuple[list[dict], dict]:
    """Verified QA pairs about train chunks, deduplicated by question, and
    counts: written (pairs in the replies), verified, duplicates dropped."""
    replies = generate(
        [qa_messages(c, pairs_per_chunk) for c in chunks],
        max_tokens=max_tokens, temperature=0.7, with_context=False,
    )
    rows, seen = [], set()
    written = verified = 0
    for chunk, reply in zip(chunks, replies, strict=True):
        if reply is None:
            continue
        written += reply.text.count('"question"')
        pairs = grounded_qa(chunk, reply.text)
        verified += len(pairs)
        for pair in pairs:
            key = " ".join(pair["question"].lower().split())
            if key not in seen:
                seen.add(key)
                rows.append(pair)
    logger.info("qa: %d chunks, %d written, %d verified, %d kept", len(chunks), written, verified, len(rows))
    return rows, {"qa/written": written, "qa/verified": verified, "qa/duplicates": verified - len(rows)}


def quality_stats(selfstudy: list[dict], chunks: list[Chunk], qa: list[dict], qa_counts: dict,
                  pages: dict[str, str], heldout: set[str]) -> dict:
    """Datagen quality for the run summary. Raises if a QA pair cites a held-out page."""
    from cartridges.cartridge.evals import parse_citations

    leaked = sorted({c for pair in qa for c in pair["citations"] if c in heldout})
    if leaked:
        raise AssertionError(f"qa_train cites held-out pages: {leaked[:5]}")
    lengths = sorted(len(r["completion_ids"]) for r in selfstudy) or [0]
    cited = [parse_citations(r["answer"]) for r in selfstudy]
    all_cited = [c for cs in cited for c in cs]
    covered = Counter()
    for chunk in chunks:
        for citation, (_, label, _) in zip(chunk.citations, chunk.pages):
            month = label[:7] if label[:4].isdigit() else "undated"
            covered[f"{citation.split(':p.')[0]}/{month}"] += 1
    return {
        **qa_counts,
        "qa/pairs": len(qa),
        "qa/verify_pass_rate": qa_counts["qa/verified"] / max(qa_counts["qa/written"], 1),
        "qa/answer_type": dict(Counter(pair["answer_type"] for pair in qa)),
        "qa/heldout_citations": 0,
        "selfstudy/conversations": len(selfstudy),
        "selfstudy/kind": dict(Counter(r["kind"] for r in selfstudy)),
        "selfstudy/completion_tokens": sum(lengths),
        "selfstudy/len_p50": lengths[len(lengths) // 2],
        "selfstudy/len_p95": lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))],
        "selfstudy/truncation_rate": sum(r["finish_reason"] == "length" for r in selfstudy) / max(len(selfstudy), 1),
        "selfstudy/cite_rate": sum(bool(cs) for cs in cited) / max(len(selfstudy), 1),
        "selfstudy/citation_valid": sum(c in pages for c in all_cited) / max(len(all_cited), 1),
        # Bot A's reference (citations + quotes) checks out against its chunk
        "selfstudy/grounded_rate": sum(r["grounded"] for r in selfstudy) / max(len(selfstudy), 1),
        # the teacher's answer cites a page of that reference: the teacher-side grounding agreement
        "selfstudy/teacher_cites_reference": sum(
            bool(set(cs) & set(r["citations"])) for r, cs in zip(selfstudy, cited) if r["grounded"]
        ) / max(sum(r["grounded"] for r in selfstudy), 1),
        "selfstudy/page_coverage": dict(sorted(covered.items())),
    }


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def modal_generate(teacher) -> Generate:
    """`generate` over the deployed Teacher: one spawned call per conversation
    (the Teacher batches concurrent calls in vLLM), all finished before it
    returns. teacher.generate_all cancels every unfinished call on an
    interrupt, so nothing stays queued to restart a container."""
    from cartridges.cartridge.teacher import generate_all

    def generate(batch: list[list[dict]], **kw) -> list:
        return [out[0] if out else None for out in generate_all(teacher, batch, **kw)]

    return generate


def build_datasets(
    model_id: str,
    group: str,
    timeline_path: str = CORPUS,
    n_chunks: int = 200,
    n_qa_chunks: int = 100,
    kinds: tuple[str, ...] = KINDS,
    min_tokens: int = 512,
    max_tokens: int = 4096,
    max_answer_tokens: int = 1024,
    seed: int = 0,
    out_dir: str | Path = LOCAL_DIR,
) -> dict:
    """Build selfstudy/train.jsonl and grpo/qa_train.jsonl with the deployed
    teacher, upload them to the volume under /data, and log them as W&B
    dataset artifacts in `group`."""
    from cartridges.cartridge import tracking
    from cartridges.cartridge.evals import heldout_pages, load_pages
    from cartridges.cartridge.teacher import GPU_FOR_MODEL, teacher_cls

    out_dir = Path(out_dir)
    config = {k: v for k, v in locals().items() if isinstance(v, (str, int, tuple))}
    with tracking.init_run("datagen", model_id, group, config=config, gpu="CPU") as run:
        generate = modal_generate(teacher_cls("Teacher", model_id))
        chunks = sample_chunks(Path(timeline_path), n_chunks, min_tokens, max_tokens, seed=seed)
        with run.timer("selfstudy"):
            selfstudy = build_selfstudy(generate, chunks, kinds, seed, max_answer_tokens=max_answer_tokens)
        heldout = heldout_pages(LOCAL_DIR / "eval" / "heldout_pages.json")
        qa_chunks = sample_chunks(Path(timeline_path), n_qa_chunks, min_tokens, 2048, seed=seed + 1, exclude_pages=heldout)
        with run.timer("qa"):
            qa, qa_counts = build_qa(generate, qa_chunks)

        files = {
            "selfstudy/train.jsonl": write_jsonl(out_dir / "selfstudy" / "train.jsonl", selfstudy),
            "grpo/qa_train.jsonl": write_jsonl(out_dir / "grpo" / "qa_train.jsonl", qa),
        }
        upload(files)

        pages = load_pages(Path(timeline_path).parent)
        summary = {
            **quality_stats(selfstudy, chunks, qa, qa_counts, pages, heldout),
            "selfstudy/bytes": files["selfstudy/train.jsonl"].stat().st_size,
            "teacher_gpu": GPU_FOR_MODEL[model_id],
        }
        run.log({k: v for k, v in summary.items() if not isinstance(v, dict)})
        run.summary.update(summary)
        import wandb

        columns = ["question", "answer", "answer_type", "citations", "evidence", "chunk_id"]
        run.run.log({"qa/pairs_table": wandb.Table(columns=columns, data=[[str(p[c]) for c in columns] for p in qa])})
        meta = {**CORPUS_META, "tokenizer": model_id, "teacher": model_id}
        sha = hashlib.sha256(Path(CORPUS).read_bytes()).hexdigest()[:12]
        run.log_artifact(CORPUS, type="dataset", name="corpus-teacher_context", metadata=CORPUS_META, aliases=[sha])
        # Pinned refs for lineage: the pipeline's runs use_artifact these (manifest.json on the volume).
        manifest = {"corpus": f"corpus-teacher_context:{sha}"}
        table_columns = {  # browsable in W&B; the token-id columns stay in the jsonl only
            "selfstudy": ["id", "kind", "chunk_id", "question", "citations", "evidence", "grounded", "answer",
                          "finish_reason"],
            "qa-train": columns,
        }
        for key, kind in (("selfstudy/train.jsonl", "selfstudy"), ("grpo/qa_train.jsonl", "qa-train")):
            manifest[kind] = run.log_artifact(
                files[key], type="dataset", name=f"{kind}-{tracking.config_hash(config)}", metadata=meta,
                table_columns=table_columns[kind],
            )
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        upload({"manifest.json": manifest_path})
        summary["artifacts"] = manifest
    return summary


def upload(files: dict[str, Path]) -> None:
    """Copy local files to /data/<key> on the cartridges-cartridge-cache volume."""
    import modal

    volume = modal.Volume.from_name("cartridges-cartridge-cache", create_if_missing=True)
    with volume.batch_upload(force=True) as batch:
        for remote, local in files.items():
            batch.put_file(str(local), f"/{VOLUME_DIR}/{remote}")


def upload_eval_sets(eval_dir: Path = LOCAL_DIR / "eval") -> list[str]:
    """eval's qa_heldout*.jsonl, for the pipeline's evals (/data/eval/ on the volume)."""
    files = {f"eval/{p.name}": p for p in sorted(eval_dir.glob("qa_heldout*.jsonl"))}
    upload(files)
    return list(files)
