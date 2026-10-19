"""Command-line interface for cartridges."""

import logging
from contextlib import contextmanager
from pathlib import Path

import fire

from cartridges.logging.config import configure_logging

logger = logging.getLogger(__name__)

TASKS_CORPUS = "data/journals/output/2026_10_13/teacher_context.md"
TASKS_HELDOUT = "data/cartridge/eval/heldout_pages.json"
TASKS_MODEL = "Qwen/Qwen3.5-4B"
TASKS_QUESTIONS = "data/cartridge/user_questions.jsonl"
# Label priority (2026-10-16): the most valuable labels first, in case the cost guard stops the run.
LABEL_PHASES = "train:chat,train:synthetic,eval:chat,train:psych_test,eval:psych_test,eval:human"


class Cli:
    """cartridges command-line interface."""

    def prepare_data(
        self,
        input_path: str,
        output_path: str,
        model: str = "",
        use_agent: bool = False,
        engine: str = "model",
        dpi: int = 400,
    ) -> None:
        """Convert a source document into a golden-format markdown transcription.

        By default this runs the local Prefect flow. Pass use_agent=True to run
        the Anthropic agent workflow instead.

        model: optional model name to use (e.g. 'prithivMLmods/Qwen2-VL-OCR-2B-Instruct').
        For the non-agent Prefect path, engine='model' uses the cached Modal
        model OCR engine; engine='paddleocr' uses local PaddleOCR.
        """
        from cartridges.datagen.flows import prepare_data_flow

        prepare_data_flow(
            input_path=Path(input_path),
            output_path=Path(output_path),
            engine=engine,
            model=model,
            use_agent=use_agent,
            dpi=dpi,
        )

    def pdf_to_image(
        self, pdf_path: str, page_number: int, output_path: str, dpi: int = 400
    ) -> None:
        """Rasterize a single PDF page to a PNG, without running OCR/transcription."""
        from cartridges.processors.ocr import pdf_to_images

        pages = pdf_to_images(Path(pdf_path), dpi=dpi)
        if not (1 <= page_number <= len(pages)):
            raise ValueError(f"page_number must be 1..{len(pages)}")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        pages[page_number - 1].save(output)
        logger.info("wrote %s", output)

    def ocr(self, image_path: str, page_number: int = 1, revision: str = "", dpi: int = 250) -> None:
        """Transcribe one image (or one page of a PDF) with the deployed OCR
        service: the same /ocr endpoint and recipe that notes_process uses.

        image_path: a .png/.jpg file, or a .pdf (page_number selects which page,
        1-indexed; rendered at `dpi`, like notes_process). revision: if given,
        check that the deployed model is that revision (e.g. "prod").
        Prints the transcribed text to stdout. Needs VLLM_API_KEY.
        """
        import asyncio
        import io

        from cartridges.batch.pipeline import render_png

        path = Path(image_path)
        if path.suffix.lower() == ".pdf":
            png = render_png(str(path), page_number, dpi)
        else:
            from PIL import Image

            buf = io.BytesIO()
            Image.open(path).save(buf, format="PNG")
            png = buf.getvalue()
        record = _require_deployed(revision)
        backend = _service_backend(concurrency=1)  # URL lookup is a blocking Modal call

        async def run():
            try:
                served = await backend.model_id()
                if served != record["model_id"]:
                    raise SystemExit(f"service serves {served}, deploy record says {record['model_id']}")
                return await backend.transcribe(png)
            finally:
                await backend.aclose()

        print(asyncio.run(run()))
        logger.info("ocr done: model_id=%s revision=%s", record["model_id"], record["revision"])

    # --- model registry and eval (see docs/plans/batch-ocr-pipeline.md) -------

    def model_eval(
        self,
        revision: str = "",
        split: str = "test",
        dataset_dir: str = "data/train/hf_dataset",
        out: str = "eval_report.json",
        concurrency: int = 32,
    ) -> None:
        """Evaluate the deployed OCR service on a labeled split.

        Sends each page to the same /ocr endpoint that production uses, then
        scores corpus CER, median CER, and loop rate. revision: if given, the
        command checks that the deployed model is that revision. Writes the
        full report to `out` and prints the summary. Needs VLLM_API_KEY.
        """
        report = _eval_served(revision, split, dataset_dir, concurrency)
        Path(out).write_text(report.model_dump_json(indent=2) + "\n")
        _print_json({"report": out, **report.model_dump(exclude={"predictions"})})

    def eval_review(
        self,
        results_dir: str = "data/train/loop_eval/results",
        out: str = "data/train/loop_eval/review.html",
        runs: str = "",
    ) -> None:
        """Write a local HTML page to review eval predictions against labels.

        Reads every <run>.json in results_dir (or only the comma-separated
        `runs`, in that order; the last one is shown first, compared with the
        first). Filters: journal, split, loop/control set, looped, data-quality
        flags, heading labels, min CER, label text. Open `out` in a browser.
        """
        from cartridges.training.review import build_review

        run_list = [r.strip() for r in runs.split(",") if r.strip()] if runs else None
        _print_json(build_review(Path(results_dir), Path(out), run_list))

    def model_promote(
        self,
        revision: str,
        max_cer: float = 0.20,  # served E2 run-8eqhat68: 0.168 (2026-10-13); vLLM noise ~±0.013
        max_loop_rate: float = 0.02,
        split: str = "test",
        dataset_dir: str = "data/train/hf_dataset",
        concurrency: int = 32,
        dry_run: bool = False,
    ) -> None:
        """Evaluate a deployed adapter revision through the OCR service. If
        it passes the thresholds (corpus CER, loop rate, no errors), tag it
        verified-<date>-<sha7> and move the tag prod to it.

        Deploy the revision first (serve_deploy). Prints a JSON result. Exit
        code 2 if the revision does not pass. dry_run=True evaluates and
        checks, but does not change the tags.
        """
        from cartridges.serve.registry import BASE, check_promotion, tag_verified

        report = _eval_served(revision, split, dataset_dir, concurrency)
        result = check_promotion(
            n=report.n, n_errors=report.n_errors, corpus_cer=report.corpus_cer,
            loop_rate=report.loop_rate, max_cer=max_cer, max_loop_rate=max_loop_rate,
        )
        promote = result.passed and not dry_run and report.model_sha != BASE
        tags = tag_verified(report.model_sha) if promote else []
        _print_json(
            {
                "revision": revision, "model_sha": report.model_sha, "model_id": report.model_id,
                "corpus_cer": report.corpus_cer, "loop_rate": report.loop_rate,
                "n_errors": report.n_errors, "tags": tags, **result.model_dump(),
            }
        )
        if not result.passed:
            raise SystemExit(2)

    def model_merge(self, revision: str = "prod", overwrite: bool = False) -> None:
        """Merge the adapter at `revision` into the base weights, on the volume."""
        from cartridges.serve.registry import resolve_revision

        sha = resolve_revision(revision)
        _print_json(_merge(sha, overwrite=overwrite))

    # --- serving ---------------------------------------------------------------

    def serve_deploy(
        self,
        revision: str = "prod",
        max_containers: int = 8,
        max_num_seqs: int = 32,
        ngram_size: int = 10,
        repetition_penalty: float = 1.05,
    ) -> None:
        """Deploy the OCR service for `revision` ("base" for the base model).

        Merges the adapter first if necessary. The service runs the default
        OcrRecipe for those weights, with the given n-gram size and
        repetition penalty; the model id is the recipe hash. The
        deploy does not start a GPU container; the first request (or
        serve_warm) does.
        """
        import os

        from cartridges.serve.recipe import OcrRecipe
        from cartridges.serve.registry import BASE, record_deploy, resolve_revision

        sha = resolve_revision(revision)
        if sha != BASE:
            _merge(sha)
        recipe = OcrRecipe(model_sha=sha, ngram_size=ngram_size, repetition_penalty=repetition_penalty)
        os.environ["CARTRIDGES_OCR_RECIPE"] = recipe.model_dump_json()
        os.environ["CARTRIDGES_OCR_MAX_CONTAINERS"] = str(max_containers)
        os.environ["CARTRIDGES_OCR_MAX_NUM_SEQS"] = str(max_num_seqs)
        import modal

        from cartridges.serve.vllm_server import app  # reads the environment above

        with modal.enable_output():
            app.deploy()
        _print_json({**record_deploy(revision, recipe), "url": _serve_function().get_web_url()})

    def serve_status(self) -> None:
        """Print the deployed model, the URL, and the container counts. This
        does not send a request to the server, so it does not start a container."""
        from cartridges.serve.registry import deployed

        fn = _serve_function()
        stats = fn.get_current_stats()
        _print_json(
            {
                "deployed": deployed(),
                "url": fn.get_web_url(),
                "backlog": stats.backlog,
                "num_total_runners": stats.num_total_runners,
            }
        )

    def serve_warm(self, containers: int = 1) -> None:
        """Keep `containers` GPU containers warm until serve_cool. Billed while idle."""
        _serve_function().update_autoscaler(min_containers=containers)
        _print_json({"min_containers": containers})

    def serve_cool(self) -> None:
        """Let all idle containers scale down (min_containers=0)."""
        _serve_function().update_autoscaler(min_containers=0)
        _print_json({"min_containers": 0})

    def gateway_serve(self, port: int = 8008, host: str = "127.0.0.1") -> None:
        """Run the OpenAI-compatible chat gateway for LibreChat (serve/gateway.py).
        Bearer key from GATEWAY_API_KEY (default cartridges-local). Backend calls
        go to the deployed Modal Teacher and CartridgeServer."""
        import os

        import uvicorn

        from cartridges.serve import backends
        from cartridges.serve.gateway import DEFAULT_API_KEY, build_app

        web = build_app(
            backends.teacher_generate,
            backends.cartridge_generate,
            api_key=os.environ.get("GATEWAY_API_KEY", DEFAULT_API_KEY),
        )
        uvicorn.run(web, host=host, port=port)

    # --- batch notes pipeline --------------------------------------------------

    def notes_process(
        self,
        *pdfs: str,
        out: str = "notes_out",
        concurrency: int = 64,
        containers: int = 0,
        dpi: int = 250,
        retry_flagged: bool = False,
        dry_run: bool = False,
    ) -> None:
        """OCR all pages of the PDFs (or of all *.pdf in a directory) with the
        deployed model. Write the outputs to `out`.

        Pages already in the cache for this file and this model id are not
        sent again. containers=K pre-warms K GPU containers for the run and
        sets the minimum back to 0 at the end. Needs DATABASE_URL, and
        VLLM_API_KEY unless dry_run.

        Prints a JSON report to stdout (logs go to stderr). Exit code 0: all
        pages OK. 2: one or more pages failed or are looped. 1: fatal error.
        """
        import asyncio

        from cartridges.batch.pipeline import process
        from cartridges.storage.cache import PostgresPageStore

        paths = _pdf_paths(pdfs)
        record = _require_deployed("")
        fn = _serve_function()

        async def run():
            store = await PostgresPageStore.connect()
            backend = _service_backend(concurrency, require_key=not dry_run)
            try:
                return await process(
                    paths, backend=backend, store=store,
                    model_id=record["model_id"], model_sha=record["model_sha"],
                    dpi=dpi, concurrency=concurrency, retry_flagged=retry_flagged,
                    dry_run=dry_run, out_dir=Path(out),
                )
            finally:
                await backend.aclose()
                await store.close()

        warm = containers > 0 and not dry_run
        if warm:
            fn.update_autoscaler(min_containers=containers)
        try:
            report = asyncio.run(run())
        finally:
            if warm:
                fn.update_autoscaler(min_containers=0)
        _print_json({"ok": report.ok, "out": out, **report.model_dump()})
        if not report.ok:
            raise SystemExit(2)

    def notes_timeline(self, out: str = "notes_out") -> None:
        """Merge all journals in `out` into one timeline: out/timeline.md
        (year > month > dated entry, across journals) and its PageIndex tree,
        out/timeline_pageindex_tree.json. Reads the transcripts that
        notes_process wrote; no model, no database."""
        from cartridges.batch.timeline import build_timeline

        _print_json(build_timeline(Path(out)))

    def notes_collate(self, out: str = "notes_out", budget: int = 1_000_000) -> None:
        """Join all transcripts in `out` into out/teacher_context.md, in
        timeline order (see notes_timeline), one [journal:p.N] anchor per page. Exit code
        2 if the token estimate is over `budget`."""
        from cartridges.batch.outputs import collate

        summary = collate(Path(out), budget_tokens=budget)
        _print_json(summary)
        if not summary["fits"]:
            raise SystemExit(2)

    def cartridge_train(
        self,
        eval_set: str,
        name: str,
        model: str = "Qwen/Qwen3.5-4B",
        p: int = 2048,
        build_data: bool = True,
        n_chunks: int = 200,
        n_qa_chunks: int = 100,
        distill_steps: int = 0,
        grpo_steps: int = 50,
        eval_max_items: int = 0,
        judge: bool = True,
        group: str = "",
        init_cartridge: str = "",
        label_runs: str = "",
        val_label_runs: str = "",
        samples: int = 0,
        save_every: int = 0,
        task_eval: bool = False,
        warmup_steps: int = 0,
        lr: float = 0.02,
        batch_size: int = 32,
        origin_mix: str = "",
        val_every: int = 10,
        val_max: int = 0,
        patience: int = 0,
        qa_eval: bool = True,
        gpu: str = "",
        estimate_usd: float = 0.0,
        dry_run: bool = False,
        val_ids_file: str = "",
        resume_from: str = "",
        skip_samples: int = 0,
        best_val_kl: float = 0.0,
        out_dir: str = "",
        detach: bool = False,
    ) -> None:
        """Train a cartridge end to end in one W&B group exp-<date>-<name>. `name`
        is required and should say what the experiment is for (e.g.
        reflective-questions-v1): it prefixes every run name in the group
        (or `group`, to continue an existing experiment).

        Build the self-study and QA data with the deployed teacher (unless
        build_data=False reuses what is on the volume), then on one Modal GPU:
        distill (stage 1) + a quick eval on eval_max_items, GRPO (stage 2,
        skipped if grpo_steps=0), and the final cartridge eval on all of
        data/cartridge/eval/<eval_set> (required, e.g. qa_heldout.jsonl),
        graded by the teacher if judge. distill_steps=0 trains one epoch.
        Prints each stage's summary.

        label_runs (comma-separated Teacher label runs) trains on the Task
        Store targets instead: `samples` conversations drawn by task origin
        (taxonomy.LABEL_ORIGIN_MIX). task_eval judges the cartridge against the
        teacher on split='eval' tasks at every save_every checkpoint and at the
        end. val_label_runs (teacher labels of split='eval' tasks) gives the
        validation-KL curve every val_every steps (val_max: a stratified
        subset); patience > 0 stops after that many evals without a new best
        and keeps the best checkpoint. origin_mix "chat=0.5,synthetic=0.5"
        overrides the 40/35/25 mix. init_cartridge (a /vol/... cartridge dir)
        skips stage 1. gpu overrides A100-80GB (e.g. H100); estimate_usd
        overrides the budget guard's estimate. dry_run runs every step before
        the model is needed on a CPU container (train_distill.dry_run) and
        prints what the GPU run would train on. val_ids_file: a JSON list of
        label_ids that fixes the validation set exactly (e.g.
        data/cartridge/val_ids_27b.json), so val KL is comparable across runs.
        resume_from (a /vol/... cartridge) continues stage 1 from it: skip_samples
        draws of the origin order are not replayed, best_val_kl seeds early
        stopping, out_dir should be new. detach spawns the GPU job in a detached
        app and returns its call id: it keeps running if this client exits."""
        from cartridges.cartridge import data, tracking, train_distill as td
        from cartridges.cartridge.tasks.taxonomy import LABEL_ORIGIN_MIX

        def strs(x):
            return [v for v in x.split(",") if v] if isinstance(x, str) else list(x)

        runs = strs(label_runs)
        group = group or tracking.group_name(name)
        # Budget guard before any GPU starts: training on one A100 (~2 h for 27B,
        # ~1 h for smaller) + teacher containers (datagen ~2 h, judge ~0.5 h).
        from cartridges.cartridge.teacher import GPU_FOR_MODEL

        rates = tracking.GPU_RATES_USD_PER_HOUR
        teacher_hours = (2.0 if build_data else 0.0) + (0.5 if judge else 0.0) + (0.5 if task_eval else 0.0)
        estimate = rates[td.GPU] * (2.0 if "27B" in model else 1.0) + rates[GPU_FOR_MODEL[model]] * teacher_hours
        estimate = estimate_usd or estimate
        tracking.check_budget(estimate)
        results = {"group": group, "estimate_usd": estimate}
        if build_data:
            results["data"] = data.build_datasets(model, group, n_chunks=n_chunks, n_qa_chunks=n_qa_chunks)
        results["eval_sets"] = data.upload_eval_sets()
        config = td.PipelineConfig(
            distill=td.DistillConfig(
                model_id=model, p=p, max_steps=distill_steps, group=group, save_every=save_every,
                label_runs=runs, origin_mix=(_mix(origin_mix) or dict(LABEL_ORIGIN_MIX)) if runs else {},
                samples=samples, val_label_runs=strs(val_label_runs), warmup_steps=warmup_steps, lr=lr,
                batch_size=batch_size, val_every=val_every, val_max=val_max, patience=patience,
                val_label_ids=_json_id_list(val_ids_file), resume_from=resume_from, skip_samples=skip_samples,
                best_val_kl=best_val_kl or None, out_dir=out_dir or f"{td.VOLUME_PATH}/cartridges/{group}",
            ),
            grpo={"max_steps": grpo_steps} if grpo_steps else None,
            eval_path=f"{td.VOLUME_PATH}/data/eval/{eval_set}",
            eval_max_items=eval_max_items,
            judge=judge,
            init_cartridge=init_cartridge,
            task_eval=task_eval,
            qa_eval=qa_eval,
        )
        if dry_run:
            with td.cpu_app.run():
                results["dry_run"] = td.dry_run.remote(config)
            _print_json(results)
            return
        fn = td.pipeline.with_options(gpu=gpu) if gpu else td.pipeline
        if detach:
            with td.app.run(detach=True):
                call = fn.spawn(config)
            results["detached_call_id"] = call.object_id
            _print_json(results)
            return
        with td.app.run():
            results.update(fn.remote(config))
        _print_json(results)

    def cartridge_sweep(
        self,
        name: str,
        label_runs: str,
        val_label_runs: str,
        model: str = "Qwen/Qwen3.5-4B",
        lrs: str = "0.01,0.02,0.05",
        ps: str = "2048",
        steps: int = 30,
        warmup_steps: int = 0,
        group: str = "",
        batch_size: int = 32,
        origin_mix: str = "",
        val_every: int = 10,
        val_max: int = 0,
        gpu: str = "",
        estimate_usd: float = 0.0,
        val_ids_file: str = "",
    ) -> None:
        """Short stage-1 sweep on one Modal GPU: relative lr x p, `steps` steps
        each on origin-weighted batches, scored by validation KL on the eval
        label runs (no judge). One distill run per setting in the group."""
        from cartridges.cartridge import tracking, train_distill as td
        from cartridges.cartridge.tasks.taxonomy import LABEL_ORIGIN_MIX

        def floats(x):
            if isinstance(x, (int, float)):  # Fire parses "2048" as a number
                return [float(x)]
            return [float(v) for v in (x.split(",") if isinstance(x, str) else x)]

        def strs(x):
            return [v for v in x.split(",") if v] if isinstance(x, str) else list(x)

        lr_list, p_list = floats(lrs), [int(v) for v in floats(ps)]
        group = group or tracking.group_name(name)
        base = td.DistillConfig(
            model_id=model, group=group, label_runs=strs(label_runs), val_label_runs=strs(val_label_runs),
            origin_mix=_mix(origin_mix) or dict(LABEL_ORIGIN_MIX), samples=steps * batch_size,
            warmup_steps=warmup_steps, batch_size=batch_size, val_every=val_every, val_max=val_max,
            val_label_ids=_json_id_list(val_ids_file),
        )
        runs = len(lr_list) * len(p_list)
        # ~8 min per setting on 4B (30 steps x 32 x ~750 tok at ~2k tok/s + 4 val passes), ~6x on 27B
        tracking.check_budget(estimate_usd or tracking.GPU_RATES_USD_PER_HOUR[td.GPU] * (0.1 + runs * (0.8 if "27B" in model else 0.14)))
        with td.app.run():
            fn = td.sweep.with_options(gpu=gpu) if gpu else td.sweep
            results = fn.remote(td.SweepConfig(base=base, lrs=lr_list, ps=p_list, steps=steps))
        _print_json({"group": group, **{k: v[-1] for k, v in results.items()}})

    def cartridge_curves(self, group: str, out: str = "docs/report/figures/distill_curves.png") -> None:
        """Plot train KL, val KL and grad norm per step for every distill run
        in `group` (one line per run) to a PNG for the report. Needs matplotlib:
        `uv run --with matplotlib cartridges cartridge_curves <group>`."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        from cartridges.cartridge import tracking

        api = wandb.Api()
        runs = api.runs(f"{tracking.ENTITY}/{tracking.PROJECT}", filters={"group": group, "jobType": "distill"})
        panels = [("train/kl", "train KL (top-20)"), ("val/kl", "validation KL"), ("train/grad_norm", "grad norm")]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for run in runs:
            if run.state == "running" or "dryrun" in run.name:
                continue
            history = list(run.scan_history(keys=None))
            for ax, (key, _) in zip(axes, panels):
                points = [(r["_step"], r[key]) for r in history if r.get(key) is not None]
                if points:
                    c = run.config
                    label = f"lr {c.get('lr')} p {c.get('p')}" + (f" warmup {c['warmup_steps']}" if c.get("warmup_steps") else "")
                    label += " (resume)" if c.get("resume_from") else ""
                    ax.plot(*zip(*points), label=label, marker="." if key == "val/kl" else None)
        for ax, (key, title) in zip(axes, panels):
            ax.set_title(title)
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
        axes[2].set_yscale("log")
        axes[0].legend(fontsize=7)
        fig.suptitle(group)
        fig.tight_layout()
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(out)

    def cartridge_ask(self, question: str, cartridge: str, max_new_tokens: int = 256) -> None:
        """Answer a question with base model + cartridge (no timeline in context)
        on a Modal GPU; prints the answer and its [journal:p.N] citations.
        cartridge: a local directory, a path on the cartridge volume
        (/vol/cartridges/...), or a W&B artifact (cartridge-<run_id>[:alias])."""
        from cartridges.cartridge.ask import run

        _print_json(run(question, cartridge, max_new_tokens))

    def tasks_init(self, corpus: str = TASKS_CORPUS, heldout: str = TASKS_HELDOUT) -> None:
        """Create the Task Store schema `cartridge` in DATABASE_URL (idempotent)
        and load the corpus pages, marking eval's held-out pages. See
        docs/plans/task-db.md."""
        from cartridges.cartridge.tasks import seed, store

        corpus_row, pages = seed.load_corpus(Path(corpus), _heldout(heldout))
        with store.connect() as conn:
            store.init(conn)
            store.insert_corpus(conn, corpus_row, pages)
        _print_json({"corpus_id": corpus_row.corpus_id, "pages": len(pages), "heldout": sum(p.heldout for p in pages)})

    def tasks_seed(
        self, file: str, model: str = TASKS_MODEL, n_eval: int = 8, seed: int = 0, group: str = "",
        replace: bool = False, no_templates: bool = False,
    ) -> None:
        """Insert the user's questions (JSONL: question, kind or kind_v2, optional
        split) as human eval/exemplar tasks, then a placeholder template of each
        exemplar written by the deployed teacher `model`. replace=True first
        deletes all human tasks and templates, to re-seed an edited file.
        no_templates=True inserts the questions only, with no teacher call."""
        _print_json(_seed(file, model, group, n_eval=n_eval, seed=seed, replace=replace, no_templates=no_templates))

    def tasks_synthesize(
        self, n: int, seed: int = 0, model: str = TASKS_MODEL, corpus: str = TASKS_CORPUS, group: str = ""
    ) -> None:
        """Write n synthetic train tasks with the deployed teacher `model` (full
        journal in context), validate them and insert them (each batch as it
        arrives), rejects included."""
        _print_json(_synthesize(n, seed, model, corpus, group))

    def tasks_qgen_eval(
        self, group: str, model: str = TASKS_MODEL, file: str = TASKS_QUESTIONS, n: int = 150, topup_n: int = 50,
        pairs_n: int = 100, queue: str = "qgen-eval-v1", corpus: str = TASKS_CORPUS, run_id: str = "",
        no_seed: bool = False, classify_chats: bool = False, split: str = "0.9,0.1,0",
        max_usd_qgen: float = 1.0, max_usd_classify: float = 1.0,
    ) -> None:
        """The question-generator eval in one process, on one Teacher pool:
        probe the Teacher (waits up to 35 min for capacity and cold start, logging
        each minute), seed (templates), synthesize n tasks, top up kinds short of
        their pairs_n quota with topup_n more, answer pairs_n of them with and
        without the journal for Weave annotation (tasks.pairs), and always end
        the session (_tasks_stop: own calls cancelled, no bare stop). Each step logs its report as it finishes.
        run_id: earlier synthesis runs (comma-separated) whose tasks count toward
        the quota (n=0 then only tops up); no_seed skips seeding;
        classify_chats first classifies the uncached chat prompts (no context)
        and promotes the relevant ones to tasks (split: train,eval,exemplar).
        max_usd_qgen / max_usd_classify cap the estimated cost of each part
        (pass what is left of an approval)."""
        from functools import partial

        from cartridges.cartridge import tracking
        from cartridges.cartridge.tasks import pairs, qgen, sampler, store
        from cartridges.cartridge.teacher import GPU_FOR_MODEL

        # Hours of one Teacher container, measured on the 4B/L4 session of 2026-10-15:
        # ~0.15 h start, ~0.15 h to classify the chats, 64 drafts per ~0.14 h, ~0.4 h of pairs.
        rate = tracking.GPU_RATES_USD_PER_HOUR[GPU_FOR_MODEL[model]]
        classify_usd = rate * 0.15 if classify_chats else 0.0
        qgen_usd = rate * (0.15 + (n + topup_n) / 64 * 0.14 + 0.4 * pairs_n / 100)
        if classify_usd > max_usd_classify or qgen_usd > max_usd_qgen:
            raise SystemExit(f"estimate classify ${classify_usd:.2f} (cap ${max_usd_classify:.2f}), "
                             f"qgen ${qgen_usd:.2f} (cap ${max_usd_qgen:.2f})")
        tracking.check_budget(classify_usd + qgen_usd)

        def sample(query):
            with store.connect() as conn:  # a connection per step: Neon drops idle ones
                return sampler.sample_tasks(partial(store.fetch_tasks, conn), query)

        report = qgen.qgen_eval(
            spawn_probe=partial(_tasks_probe, model),
            classify=(lambda: _classify_chats(model, group) | {"promote": _promote_chats(model, _split_ratios(split))})
            if classify_chats else None,
            seed=None if no_seed else partial(_seed, file, model, group),
            run_ids=_id_list(run_id),
            synthesize=lambda k, seed, mix: _synthesize(k, seed, model, corpus, group, kind_mix=mix),
            shortfall=lambda run_ids: sample(pairs.run_query(run_ids, pairs_n, 0)).shortfall,
            pairs=lambda run_ids: pairs.run(sample, run_ids, _tasks_generate(model), n=pairs_n, queue=queue),
            stop=_tasks_stop,
            n=n,
            topup_n=topup_n,
        )
        _print_json(report)

    def tasks_import_chats(
        self, dir: str = "", model: str = TASKS_MODEL, limit: int = 0, group: str = "exp-20261015-chats",
        dry_run: bool = False, extract_only: bool = False, max_usd: float = 1.0, split: str = "0.9,0.1,0",
        seed: int = 0,
    ) -> None:
        """Import a ChatGPT export (DIR/conversations-*.json) in three stages.
        1 (CPU): the first user message of each conversation, deduplicated,
        upserted into cartridge.chat_prompts. 2 (GPU): every stored text without
        a label for this `model` and chats.prompt_hash() (the cache key in
        cartridge.chat_labels) is classified by the deployed Teacher (no
        journal, greedy) and its label cached; unparsed replies are not cached.
        3 (CPU): prompts labelled relevant with a task kind become origin
        "chat" tasks, split train,eval,exemplar by `split` per kind (seeded);
        texts already in the store are skipped.
        dry_run=True only extracts and validates (no database, no GPU);
        extract_only=True stops after stage 1; without DIR only stage 2 runs,
        from the database. Prints the report and, after
        stage 2, 10 random relevant and 10 random discarded prompts."""
        from functools import partial

        from cartridges.cartridge import tracking
        from cartridges.cartridge.tasks import chats, qgen, store
        from cartridges.cartridge.teacher import GPU_FOR_MODEL

        report: dict = {}
        if dir:
            prompts, empty = chats.extract(chats.load_conversations(Path(dir)))
            kept, dups = chats.validate(prompts)
            lengths = sorted(len(p.text) for p in kept)
            report = {
                "prompts": len(prompts), "no_user_text": empty, "duplicates": dups, "kept": len(kept),
                "chars": {f"p{q}": lengths[int(q / 100 * (len(lengths) - 1))] for q in (10, 50, 90, 99)}
                | {"max": lengths[-1]},
                "truncated": sum(n > chats.MAX_CHARS for n in lengths),
                "est_input_tokens": sum(min(n, chats.MAX_CHARS) for n in lengths) // 4,
            }
            if dry_run:
                _print_json(report)
                return

        prompt_hash = chats.prompt_hash()
        with store.connect() as conn:
            store.init(conn)
            if dir:
                report["stored"] = store.upsert_chat_prompts(conn, kept)
            todo, hits = store.fetch_unlabeled_chats(conn, model, prompt_hash)
        todo = todo[: limit or None]
        report |= {"prompt_hash": prompt_hash[:12], "cache_hits": hits, "cache_misses": len(todo)}
        ratios = _split_ratios(split)
        if extract_only or not todo:
            if not extract_only:
                report["promote"] = _promote_chats(model, ratios, seed)
            _print_json(report)
            return

        # One container: start (with the journal prefill) + the batch, ~30 min.
        estimate = 0.5 * tracking.GPU_RATES_USD_PER_HOUR[GPU_FOR_MODEL[model]]
        if estimate > max_usd:
            raise SystemExit(f"estimate ${estimate:.2f} is over max_usd ${max_usd:.2f}")
        tracking.check_budget(estimate)

        try:
            qgen.probe(partial(_tasks_probe, model), qgen.PROBE_TIMEOUT_S, qgen.PROBE_LOG_EVERY_S)
            report |= _classify_chats(model, group, todo, hits)
        finally:
            _tasks_stop()
        report["promote"] = _promote_chats(model, ratios, seed)
        _print_json(report)

    def tasks_ksweep(
        self, group: str = "exp-20261015-qgen-ksweep", model: str = TASKS_MODEL, ks: str = "0,2,4,8,16",
        n: int = 100, seed: int = 0, corpus: str = TASKS_CORPUS, max_usd: float = 1.0,
    ) -> None:
        """Sweep the number of real questions shown to the Synthesizer: for each
        k, the same n recipes with k examples drawn per prompt from half A of
        the real non-eval questions (tasks.ksweep). One synthesis run and W&B
        run per k in `group`, each logging artifact qgen-k<K> (a table of every
        draft); plus qgen-real-reference (half B and the eval questions). All
        calls are with-context. Chat tasks in the exemplar split move to train
        first."""
        import json
        from functools import partial

        from cartridges.cartridge import tracking
        from cartridges.cartridge.tasks import ksweep, neighbors, qgen, seed as seed_mod, store
        from cartridges.cartridge.tasks.synthesize import plan_recipes, render_prompt, synth_batches, synth_report
        from cartridges.cartridge.tasks.taxonomy import PROMPT_VERSION, STYLE_MIX
        from cartridges.cartridge.tasks.types import SynthConfig
        from cartridges.cartridge.tasks.validate import word_set
        from cartridges.cartridge.teacher import GPU_FOR_MODEL

        k_values = [int(k) for k in str(ks).split(",")]
        # 0.15 h start + 64 drafts per ~0.05 h (4B/L4 with V1 + cascade: ~167-280 tok/s
        # with context, ~274 tokens per draft; measured 2026-10-16, padded for preemption).
        estimate = tracking.GPU_RATES_USD_PER_HOUR[GPU_FOR_MODEL[model]] * (0.15 + len(k_values) * n / 64 * 0.05)
        if estimate > max_usd:
            raise SystemExit(f"estimate ${estimate:.2f} is over max_usd ${max_usd:.2f}")
        tracking.check_budget(estimate)

        corpus_id = seed_mod.corpus_id(Path(corpus).read_text())
        with store.connect() as conn:
            store.init(conn)
            moved = store.resplit_chat_exemplars(conn)
            tasks = store.list_tasks(conn)
            pages = {p.citation: p for p in store.fetch_pages(conn, corpus_id)}
            seen = store.fetch_hashes(conn)
        pool = ksweep.real_pool(tasks)
        half_a, half_b = ksweep.split_halves(pool, seed)
        evals = [t for t in tasks if t.split == "eval" and t.origin in ksweep.REAL_ORIGINS and t.status == "validated"]
        mix = ksweep.kind_mix([t for t in tasks if t.origin == "chat"])
        by_id = {t.task_id: t for t in pool}
        eval_words = [word_set(t.text) for t in tasks if t.split == "eval"]
        real_words = [word_set(t.text) for t in pool]  # near_exemplar against every real question
        recipes = plan_recipes(n, mix, STYLE_MIX, list(pages.values()), {}, seed, exemplars_per_prompt=0)
        config = {"ks": k_values, "n": n, "seed": seed, "kind_mix": mix, "half_a": len(half_a), "half_b": len(half_b),
                  "exemplar_chars": ksweep.EXEMPLAR_CHARS, "prompt_version": PROMPT_VERSION}
        report = {"chat_exemplars_moved_to_train": moved, "estimate_usd": estimate, **config, "runs": {}}
        out_dir = Path("data/cartridge/ksweep")
        out_dir.mkdir(parents=True, exist_ok=True)

        def jsonl(name: str, rows: list[dict]) -> Path:
            path = out_dir / f"{name}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            return path

        reference = ksweep.reference_rows(half_b, evals)
        embed = neighbors.local_embedder()  # once, for every k
        with tracking.init_run("datagen", model, group, config=config, gpu="CPU", role="reference") as run:
            run.log_artifact(jsonl("qgen-real-reference", reference), type="dataset",
                             name="qgen-real-reference", table_columns=ksweep.REFERENCE_COLUMNS)
        try:
            qgen.probe(partial(_tasks_probe, model), qgen.PROBE_TIMEOUT_S, qgen.PROBE_LOG_EVERY_S)
            generate = _tasks_generate(model)
            for k in k_values:
                rec_k = ksweep.with_exemplars(recipes, half_a, k, seed)
                prompts = [render_prompt(r, [pages[c] for c in r.citations], ksweep.exemplar_texts(r, by_id))
                           for r in rec_k]
                meta = [{"k": k, "exemplar_ids": [str(i) for i in r.exemplar_ids]} for r in rec_k]
                with store.connect() as conn, _tasks_run(conn, "synthesize", model, group, config | {"k": k}) as (run, _):
                    synth = SynthConfig(n=n, seed=seed, model=model, run_id=run.id, corpus_id=corpus_id, kind_mix=mix,
                                        style_mix=STYLE_MIX, prompt_version=PROMPT_VERSION)

                    def insert(batch):
                        with store.connect() as c:  # per batch: Neon drops idle connections
                            return store.insert_tasks(c, batch)

                    made, drafts, inserted = synth_batches(synth, rec_k, prompts, pages, seen, eval_words, real_words,
                                                           generate, insert, metadata=meta)
                    result = synth_report(made, drafts, inserted)
                    run.log({f"ksweep/{key}": v for key, v in result.model_dump().items() if isinstance(v, (int, float))})
                    rows = neighbors.add_neighbor_columns(ksweep.table_rows(made, rec_k, by_id, k), reference, embed=embed)
                    valid = [r for r in rows if r["status"] == "validated"]  # rejects would skew the metrics
                    run.summary.update(neighbors.summarize_k(valid, reference, embed=embed))
                    run.log_artifact(jsonl(f"qgen-k{k}", rows), type="dataset", name=f"qgen-k{k}",
                                     table_columns=ksweep.TABLE_COLUMNS)
                    report["runs"][k] = {"run_id": run.id, **result.model_dump()}
        finally:
            _tasks_stop()
        _print_json(report)

    def tasks_load_psych(self, dir: str = "data/corpus/psych_tests", eval_frac: float = 0.1, seed: int = 0) -> None:
        """Load the scraped psychology test items as origin "psych_test" tasks
        (tasks.psych): whole instruments, eval_frac of them stratified by
        category, held out as eval. Idempotent. Also moves chat exemplars to
        train (the 90/10 chat split). Prints task counts per origin and split."""
        from cartridges.cartridge.tasks import psych, store

        run, tasks, held = psych.load(Path(dir), eval_frac, seed)
        with store.connect() as conn:
            store.init(conn)
            store.insert_run(conn, run)
            inserted = store.insert_tasks(conn, tasks)
            moved = store.resplit_chat_exemplars(conn)
            counts = conn.execute(
                "SELECT origin, split, status, count(*) FROM cartridge.tasks GROUP BY 1, 2, 3 ORDER BY 1, 2, 3"
            ).fetchall()
        _print_json({
            "psych_tasks": len(tasks), "inserted": inserted, "heldout_instruments": sorted(held),
            "chat_exemplars_moved_to_train": moved,
            "counts": [dict(zip(("origin", "split", "status", "n"), r)) for r in counts],
        })

    def tasks_revalidate(self, run_id: str) -> None:
        """Re-check the stored synthetic drafts of synthesis run(s) `run_id`
        (comma-separated) with the current parse and validation, in place, with
        no GPU. Prints reject reasons before and after and the validated count
        per kind against the 100-task quota."""
        from collections import Counter

        from cartridges.cartridge.tasks import sampler, store
        from cartridges.cartridge.tasks.synthesize import revalidate
        from cartridges.cartridge.tasks.taxonomy import KIND_MIX
        from cartridges.cartridge.tasks.validate import word_set

        run_ids = _id_list(run_id)
        with store.connect() as conn:
            stored = store.list_tasks(conn, origin="synthetic", run_ids=run_ids)
            if not stored:
                raise SystemExit(f"no synthetic tasks for run(s) {run_ids}")
            pages = {p.citation: p for p in store.fetch_pages(conn, stored[0].corpus_id)}
            others = store.fetch_hashes(conn) - {t.text_hash for t in stored}
            eval_words = [word_set(t.text) for t in store.list_tasks(conn, split="eval")]
            exemplar_words = [word_set(t.text) for t in store.list_tasks(conn, split="exemplar")]
            new = revalidate(stored, pages, others, eval_words, exemplar_words)
            changed = [n for n, o in zip(new, stored, strict=True) if n != o]
            store.update_validation(conn, changed)
            valid = Counter(t.kind for t in store.list_tasks(conn, origin="synthetic", status="validated"))
        _print_json({
            "run_ids": run_ids, "drafts": len(stored), "changed": len(changed),
            "before": dict(Counter(t.reject_reason or "validated" for t in stored).most_common()),
            "after": dict(Counter(t.reject_reason or "validated" for t in new).most_common()),
            "validated_vs_quota": {k: f"{valid[k]}/{q}" for k, q in sampler.quotas(100, KIND_MIX).items()},
        })

    def tasks_export_md(self, out: str = "", split: str = "", origin: str = "", run_id: str = "",
                        status: str = "") -> None:
        """Write tasks (filtered by split, origin, comma-separated run ids and
        status) to Markdown, grouped by kind. Default:
        data/cartridge/exports/tasks-<filters>.md (gitignored)."""
        from cartridges.cartridge.tasks import store
        from cartridges.cartridge.tasks.export import render_md

        run_ids = _id_list(run_id)
        name = "-".join(f for f in (split, origin, status, *run_ids) if f) or "all"
        path = Path(out or f"data/cartridge/exports/tasks-{name}.md")
        with store.connect() as conn:
            tasks = store.list_tasks(conn, split=split or None, origin=origin or None, run_ids=run_ids,
                                     status=status or None)
            parents = {t.task_id: t for t in store.list_tasks(conn, split="exemplar")}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_md(tasks, parents, f"Tasks: {name}"))
        print(path)

    def tasks_label(
        self, teacher: str = TASKS_MODEL, gpu: str = "H100", group: str = "exp-20261016-label-4b",
        max_usd: float = 3.0, spent_usd: float = 0.0, seed: int = 0, corpus: str = TASKS_CORPUS,
        targets_root: str = "/cache/targets", pilot: int = 0, max_inputs: int = 0, phases: str = LABEL_PHASES,
        probe_min: float = 35, drain_min: float = 3,
    ) -> None:
        """Label unlabeled tasks with the deployed `teacher` on one pool (gpu
        and a window of calls in flight, also the handle's max_inputs:
        label.window_for the session's origins unless max_inputs is given;
        journal in context, greedy, top-20 logprobs), phase by phase in priority
        order: `phases` is a comma-separated list of split:origin, each phase its
        own label run, streamed by label.label_stream (no batch barrier; shards
        and labels rows flushed in the background). A running cost guard: the
        spend so far (spent_usd, earlier work under the same approval, plus this
        session's wall clock from the first call at the GPU's rate) plus
        drain_min of drain must stay within max_usd, else no new calls start and
        the window drains; flushed answers stay saved and a rerun resumes
        (unlabeled_by = the labels' key). Only this session's own container is
        stopped, by id. Prints the run ids per split (trainer reads them) and per
        phase: counts, reject reasons, cite rate, truncation; logs a
        targets-<run_id> artifact per phase. pilot=N instead labels N train
        tasks split by taxonomy.LABEL_ORIGIN_MIX and prints sample answers, the
        output tok/s and label.project_cost for every unlabeled task in
        `phases`. probe_min: how long to wait for a container (GPU capacity
        queues plus the cold start)."""
        import time
        from functools import partial

        from cartridges.cartridge import teacher as teacher_mod, tracking
        from cartridges.cartridge.tasks import label, qgen, sampler, seed as seed_mod, store
        from cartridges.cartridge.tasks.taxonomy import LABEL_ORIGIN_MIX
        from cartridges.cartridge.tasks.types import TaskQuery

        rate = tracking.GPU_RATES_USD_PER_HOUR[gpu]
        tracking.check_budget(max_usd - spent_usd)
        corpus_id = seed_mod.corpus_id(Path(corpus).read_text())
        key = (teacher, corpus_id, label.instructions_hash())
        # (split, origins or None for the pilot's mix, tasks to label)
        plan = [("train", None, pilot)] if pilot else [
            (split, frozenset({origin}), 0) for split, origin in (p.split(":") for p in phases.split(","))
        ]
        with store.connect() as conn:
            pages = store.fetch_pages(conn, corpus_id)
            phase_origins = [(sp, o) for sp, o in (p.split(":") for p in phases.split(","))]
            n_all = sum(len([t for t in store.fetch_tasks(conn, TaskQuery(split=sp, n=1, seed=0, unlabeled_by=key))
                             if t.origin == o]) for sp, o in phase_origins)
            for i, (split, origins, n) in enumerate(plan):
                left = len([t for t in store.fetch_tasks(conn, TaskQuery(split=split, n=1, seed=0, unlabeled_by=key))
                            if (t.origin != "human" or split == "eval")  # human tasks: eval references only
                            and (origins is None or t.origin in origins)])
                plan[i] = (split, origins, min(n, left) if n else left)
        report: dict = {"label_key": key, "max_usd": max_usd, "spent_usd_before": spent_usd,
                        "plan": [f"{s}:{','.join(sorted(o)) if o else 'mix'}={n}" for s, o, n in plan], "phases": []}

        def sample(query):
            with store.connect() as conn:  # a connection per call: Neon drops idle ones
                return sampler.sample_tasks(partial(store.fetch_tasks, conn), query)

        def insert_labels(labels):
            with store.connect() as conn:
                return store.insert_labels(conn, labels)

        # One session, one pool: the window (calls in flight) is also the handle's max_inputs.
        window = max_inputs or label.window_for(
            {o for _, origins, _ in plan for o in (origins or LABEL_ORIGIN_MIX)})
        report["window"] = window
        handle = teacher_mod.teacher_cls("Teacher", teacher, **_pool(gpu, window))

        def spawn(messages, **kw):
            return handle.generate.spawn(messages, **kw)

        start = time.monotonic()

        def spend() -> float:
            return spent_usd + (time.monotonic() - start) / 3600 * rate

        stopped = [False]

        def should_stop() -> bool:
            """The $ guard, checked before every new call: stop early enough
            that the in-flight window can drain (~drain_min) under max_usd."""
            stopped[0] = stopped[0] or spend() + drain_min / 60 * rate > max_usd
            return stopped[0]

        def label_phase(run, split: str, origins, left: int) -> str:
            """Stream the phase (label.label_stream: `window` calls in flight,
            background flushes) until done or the $ guard stops it."""
            t0 = time.monotonic()

            def log(stats: dict) -> None:  # after every flush, while the container is warm
                run.log(stats | {"label/spend_usd": spend()})
                if (c := _session_container(teacher, gpu, window)) and c not in containers:
                    containers.append(c)  # a preemption swaps the container: remember every id

            result = label.label_stream(
                label.LabelConfig(run_id=run.id, corpus_id=corpus_id, n=left, seed=seed, teacher_model=teacher,
                                  split=split, origins=origins, origin_mix=LABEL_ORIGIN_MIX if pilot else None,
                                  kind_mix=None, targets_root=targets_root),
                pages, sample=sample, spawn=spawn, write_shard=_tasks_write_shard(), insert_labels=insert_labels,
                window=window, log=log, should_stop=should_stop,
            )
            if pilot and result.overall.n_tasks:
                wall = time.monotonic() - t0
                report["projection"] = label.project_cost(
                    result.overall, wall, n_total=n_all, usd_per_hour=rate,
                    overhead_s=report.get("cold_start_s", 0) + 300) | {"n_total": n_all, "batch_s": wall}
            if stopped[0]:
                return f"stopped: ${spend():.2f} spent; the guard keeps ~{drain_min} min of drain under ${max_usd:.2f}"
            return "done"

        containers: list[str] = []
        try:
            qgen.probe(partial(_tasks_probe, teacher, gpu, window), probe_min * 60, qgen.PROBE_LOG_EVERY_S)
            # Only while a container is warm (right after the probe or a batch): on a cold pool,
            # info.remote() would start a container just to answer.
            containers += [c for c in [_session_container(teacher, gpu, window)] if c]
            report["cold_start_s"] = time.monotonic() - start
            for split, origins, n in plan:
                if not n:
                    continue
                name = f"{split}:{','.join(sorted(origins)) if origins else 'mix'}"
                config = {"split": split, "origins": sorted(origins) if origins else None, "phase": name}
                with store.connect() as conn, _tasks_run(conn, "label", teacher, group, config) as (run, group):
                    phase = {"phase": name, "split": split, "run_id": run.id, "planned": n}
                    report["phases"].append(phase)
                    phase["status"] = label_phase(run, split, origins, n)
                    _log_targets(run, run.id)
                if phase["status"] != "done":
                    break
        finally:
            _tasks_stop(containers)
        with store.connect() as conn:
            for phase in report["phases"]:
                phase["by_origin"] = store.label_summary(conn, [phase["run_id"]])
            if pilot and report["phases"]:
                report["samples"] = store.label_samples(conn, report["phases"][0]["run_id"], per_origin=1)
                if "projection" in report:  # all answers' tokens, rejected ones too, over the batch's wall time
                    tokens = sum(o["completion_tokens"] for o in report["phases"][0]["by_origin"].values())
                    report["projection"]["tok_per_s"] = tokens / report["projection"]["batch_s"]
        report["run_ids"] = {s: [p["run_id"] for p in report["phases"] if p["split"] == s] for s in ("train", "eval")}
        _print_json(report | {"spend_usd": spend(), "session_usd": spend() - spent_usd})

    def tasks_export_qa(self, train_runs: str, eval_runs: str = "", name: str = "qa-27b",
                        model: str = "Qwen/Qwen3.8-27B", group: str = "exp-20261016-label-27b") -> None:
        """Log the teacher's validated answers as W&B dataset artifacts
        <name>-train and <name>-eval (label runs comma-separated), each a
        browsable table: question, answer, origin, kind, citations, n_tokens.
        Reads the Task Store only; the top-k targets stay in Parquet
        (tasks.qa.load_qa reads them where the Volume is mounted)."""
        import json

        from cartridges.cartridge import tracking
        from cartridges.cartridge.tasks import qa, store

        report = {}
        with tracking.init_run("datagen", model, group, config={"train_runs": train_runs, "eval_runs": eval_runs},
                               gpu="CPU", role="qa-export") as run:
            for split, runs in (("train", train_runs), ("eval", eval_runs)):
                run_ids = _id_list(runs)
                if not run_ids:
                    continue
                with store.connect() as conn:
                    labels = store.fetch_labels(conn, run_ids, split=split)
                    tasks = store.fetch_tasks_by_id(conn, {lab.task_id for lab in labels})
                rows = qa.table_rows(labels, tasks)
                path = Path("data/cartridge/qa") / f"{name}-{split}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                ref = run.log_artifact(path, type="dataset", name=f"{name}-{split}", table_columns=qa.TABLE_COLUMNS,
                                       metadata={"label_runs": run_ids, "split": split})
                report[split] = {"rows": len(rows), "artifact": ref, "path": str(path)}
        _print_json(report)

    def tasks_log_targets(self, run_ids: str, model: str = "Qwen/Qwen3.8-27B",
                          group: str = "exp-20261016-label-27b") -> None:
        """Log a targets-<run_id> dataset artifact (a manifest of the label run's
        Parquet shards and label counts) for each label run (comma-separated)
        that has none, so trainer's distill runs can link to their label sets.
        New label sessions log it themselves at the end of each phase."""
        from cartridges.cartridge import tracking

        refs = {}
        with tracking.init_run("datagen", model, group, config={"run_ids": run_ids}, gpu="CPU",
                               role="targets-manifest") as run:
            for run_id in _id_list(run_ids):
                refs[run_id] = _log_targets(run, run_id)
        _print_json(refs)

    def tasks_sample(self, split: str, n: int, seed: int = 0) -> None:
        """Print a deterministic sample of validated tasks of one split."""
        from functools import partial

        from cartridges.cartridge.tasks import sampler, store
        from cartridges.cartridge.tasks.types import TaskQuery

        with store.connect() as conn:
            s = sampler.sample_tasks(partial(store.fetch_tasks, conn), TaskQuery(split=split, n=n, seed=seed))
        _print_json({
            "sample_hash": s.sample_hash,
            "tasks": [{"task_id": t.task_id, "kind": t.kind, "origin": t.origin, "text": t.text} for t in s.tasks],
        })

    def tasks_stats(self) -> None:
        """Task counts by split, origin, kind and status; reject reasons; labels."""
        from cartridges.cartridge.tasks import store

        with store.connect() as conn:
            _print_json(store.stats(conn))


    def tasks_pairs(self, run_id: str, n: int = 100, model: str = TASKS_MODEL, seed: int = 0, group: str = "",
                    queue: str = "qgen-eval-v1") -> None:
        """Question-generator eval: answer n validated tasks (default kind mix) of
        synthesis run `run_id` (comma-separated for top-up runs) with and without
        the journal (deployed teacher `model`, greedy, 1536 tokens), log each
        blinded pair to Weave and put them in the annotation queue `queue`. The
        A/B mapping goes to data/cartridge/annotations/<queue>.pairs.jsonl."""
        from functools import partial

        from cartridges.cartridge.tasks import pairs, sampler, store

        from cartridges.cartridge import tracking
        from cartridges.cartridge.teacher import GPU_FOR_MODEL

        run_ids = _id_list(run_id)
        # ~45 min of one Teacher container (cold start, 2 x n answers, 5 min scaledown)
        tracking.check_budget(0.75 * tracking.GPU_RATES_USD_PER_HOUR[GPU_FOR_MODEL[model]])
        with store.connect() as conn:
            report = pairs.run(partial(sampler.sample_tasks, partial(store.fetch_tasks, conn)), run_ids,
                               _tasks_generate(model), n=n, seed=seed, queue=queue)
        _print_json(report)

    def tasks_export_annotations(self, queue: str = "qgen-eval-v1") -> None:
        """Read the queue's human annotations from Weave, un-blind them and write
        data/cartridge/annotations/<queue>.jsonl."""
        from cartridges.cartridge.tasks import pairs

        print(pairs.export_annotations(queue))

    def eval_cgc(self, cartridge: str, label_runs: str, model: str = "Qwen/Qwen3.8-27B", group: str = "",
                 cache_dir: str = "data/cartridge/targets_cache", out: str = "data/cartridge/eval/cgc-report.md",
                 estimate_usd: float = 1.5) -> None:
        """Context gap closed on the labeled eval split: the cartridge and base
        terms from train_distill.score_tasks (A100), then the teacher term from the
        deployed TeacherScorer (H200, with_context only), one after the other.
        label_runs: comma-separated label run ids. Logs one eval run
        (system=cartridge) with the cgc-report artifact in `group`."""
        import modal

        from cartridges.cartridge import tracking, train_distill
        from cartridges.cartridge.tasks import cgc, store, targets

        tracking.check_budget(estimate_usd)
        run_ids = _id_list(label_runs)  # fire passes "a,b" as a tuple
        with store.connect() as conn:
            labels = store.fetch_labels(conn, run_ids, split="eval")
            tasks = list(store.fetch_tasks_by_id(conn, [lab.task_id for lab in labels]).values())
        if not labels:  # before any GPU starts
            raise SystemExit(f"no validated eval labels in runs {run_ids}")
        root = _download_shards([lab.targets_uri for lab in labels if lab.targets_uri], Path(cache_dir))

        def score_cart(cart: str, label_dicts: list[dict], model_id: str) -> dict:
            with modal.enable_output(), train_distill.app.run(detach=True):  # survives a client disconnect
                return train_distill.score_tasks.remote(cart, label_dicts, model_id)

        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model)  # CPU: decodes answer tokens for the token-type split
        group = group or tracking.group_name("eval-27b")
        scored, segments, check = cgc.final_cgc(
            cartridge, model, labels, tasks, read_labeled=lambda valid: targets.read_labeled(valid, root=root),
            score_cart=score_cart, session=cgc.scorer_session, pieces=lambda ids: [tok.decode([i]) for i in ids],
        )
        meta = {"cartridge": cartridge, "model": model, "label_runs": ",".join(run_ids), "tasks": len(scored),
                "crosscheck_in_label_topk": check["in_label_topk"], "crosscheck_mean_abs_diff": check["mean_abs_diff"]}
        config = {"system": "cartridge", "metric": "cgc", **meta}
        with tracking.init_run("eval", model, group, config, role="cgc") as run:
            if not cartridge.startswith("/"):  # a W&B artifact ref (a /vol path has no lineage to record)
                run.use_artifact(cartridge, type="cartridge")
            path = cgc.log_report(run, scored, segments, meta, Path(out))
        _print_json({"group": group, "report": str(path), "cgc": segments["all"]["cgc"],
                     "cgc_ci": segments["all"]["cgc_ci"], "tasks": len(scored), "crosscheck": check})

    def eval_judge(self, cartridge: str, label_runs: str, model: str = "Qwen/Qwen3.8-27B", group: str = "",
                   n: int = 60, seed: int = 0, rubric: str = "v0.1", max_new_tokens: int = 3072,
                   out_dir: str = "data/cartridge/eval", journals: str = "data/journals/output/2026_10_13",
                   estimate_usd: float = 3.0) -> None:
        """Side-by-side judge on a stratified subset of n eval tasks. Three answers per
        task, all to the same prompt (label.answer_messages): TEACHER = the eval label
        (journal in context, no GPU); CARTRIDGE = train_distill.answer_tasks (A100,
        greedy); BASE = the Teacher pool with no journal (vLLM, greedy). All are saved
        to <out_dir>/judge-<group>.jsonl before judging; a rerun resumes from it. Then
        the 27B judge (same Teacher container, journal in context, both orders, one
        batch): cartridge vs teacher (headline) and base vs teacher (floor). Logs one
        eval run (system=cartridge, role=judge) with the judge metrics, citation
        validity for all three answers, and the artifact judge-side-by-side-27b."""
        import json

        import modal
        import wandb

        from cartridges.cartridge import teacher as teacher_mod
        from cartridges.cartridge import tracking, train_distill
        from cartridges.cartridge.data import modal_generate
        from cartridges.cartridge.evals import load_pages
        from cartridges.cartridge.tasks import judge, store
        from cartridges.cartridge.tasks.label import answer_messages

        tracking.check_budget(estimate_usd)
        run_ids = _id_list(label_runs)  # fire passes "a,b" as a tuple
        with store.connect() as conn:
            labels = {str(lab.task_id): lab for lab in store.fetch_labels(conn, run_ids, split="eval")}
            tasks = [t for t in store.fetch_tasks_by_id(conn, list(labels)).values()]
        subset = judge.stratified_subset(tasks, n, seed)
        if not subset:  # before any GPU starts
            raise SystemExit(f"no validated eval labels in runs {run_ids}")
        ids = [str(t.task_id) for t in subset]
        group = group or tracking.group_name("eval-27b")
        answers_path = Path(out_dir) / f"judge-{group}.jsonl"
        saved = {r["task_id"]: r for r in map(json.loads, answers_path.read_text().splitlines())} \
            if answers_path.exists() else {}
        if not set(saved) <= set(ids):  # rows land one at a time, so a partial file is fine
            raise SystemExit(f"{answers_path} is for another subset; move it away or pass another --group")

        def write() -> None:
            answers_path.parent.mkdir(parents=True, exist_ok=True)
            answers_path.write_text("".join(json.dumps(saved[tid]) + "\n" for tid in ids if tid in saved))

        def save(system: str, outs: list[dict]) -> None:
            for tid, o in zip(ids, outs, strict=True):
                saved.setdefault(tid, {"task_id": tid})[system] = o
            write()

        todo = _unanswered(ids, subset, saved, "cartridge")
        if todo:  # streamed: each answer is saved as it is decoded (the container also keeps /vol/answers/<group>.jsonl)
            with modal.enable_output(), train_distill.app.run(detach=True):  # survives a client disconnect
                _collect(train_distill.answer_tasks.remote_gen(cartridge, todo, group, model, max_new_tokens),
                         saved, "cartridge", write)
        if not all("cartridge" in saved.get(tid, {}) for tid in ids):
            raise SystemExit(f"cartridge answers missing for some tasks; rerun to resume ({answers_path})")
        rub = judge.load_rubric(rubric)
        handle = teacher_mod.teacher_cls("Teacher", model)
        container_id = handle.info.remote()["container_id"]
        try:
            generate = modal_generate(handle)
            if not all("base" in saved.get(tid, {}) for tid in ids):  # a no-journal batch, before the judge batch
                outs = generate([answer_messages(t.text) for t in subset], max_tokens=max_new_tokens,
                                temperature=0.0, with_context=False)
                save("base", [{"text": o.text if o else "", "finish_reason": o.finish_reason if o else "error",
                               "n_tokens": len(o.completion_ids) if o else 0} for o in outs])
            teacher_ans = {tid: {"text": labels[tid].answer, "finish_reason": labels[tid].finish_reason,
                                 "n_tokens": labels[tid].n_completion_tokens} for tid in ids}
            items = {system: [judge.Item(task_id=tid, origin=t.origin, kind=t.kind, question=t.text,
                                         x=saved[tid][system]["text"], y=teacher_ans[tid]["text"])
                              for tid, t in zip(ids, subset, strict=True)] for system in ("cartridge", "base")}
            outcomes = judge.judge_pairs(items["cartridge"] + items["base"], generate, rub)  # one with-context batch
        finally:
            teacher_mod.stop(container_ids=[container_id])
        by_system = {"cartridge": outcomes[: len(ids)], "base": outcomes[len(ids):]}
        summaries = {s: judge.summarize(o, seed) for s, o in by_system.items()}
        pages = set(load_pages(Path(journals)))
        answers = {"teacher": teacher_ans, "cartridge": {tid: saved[tid]["cartridge"] for tid in ids},
                   "base": {tid: saved[tid]["base"] for tid in ids}}
        cites = {s: judge.citation_summary([answers[s][tid]["text"] for tid in ids], [t.origin for t in subset], pages)
                 for s in answers}
        rows = judge.side_by_side_rows(subset, answers, {s: {o.task_id: o for o in by_system[s]} for s in by_system},
                                       pages)
        rows_path = Path(out_dir) / f"judge-side-by-side-{group}.jsonl"
        rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        config = {"system": "cartridge", "metric": "judge", "cartridge": cartridge, "n": len(ids), "seed": seed,
                  "max_new_tokens": max_new_tokens, "label_runs": ",".join(run_ids),
                  **judge.judge_config(rub, model, "cartridge|base", "teacher")}
        with tracking.init_run("eval", model, group, config, role="judge") as run:
            if not cartridge.startswith("/"):
                run.use_artifact(cartridge, type="cartridge")
            for s in by_system:
                judge.log_comparison(run, by_system[s], summaries[s], prefix=f"judge/{s}_vs_teacher")
            scalars = {(f"eval/{s}/" if k == "all" else f"eval/{s}/{k.removeprefix('origin:')}/") + m: v
                       for s, cs in cites.items() for k, st in cs.items() for m, v in st.items()
                       if v is not None and m != "n"}
            scalars |= {f"eval/{s}/truncated": sum(bool(answers[s][tid]["finish_reason"] == "length") for tid in ids)
                        / len(ids) for s in answers}
            table = wandb.Table(columns=list(rows[0]), data=[list(r.values()) for r in rows])
            run.log(scalars | {"judge/side_by_side": table})
            run.run.summary.update(scalars)
            art = wandb.Artifact("judge-side-by-side-27b", type="dataset",
                                 metadata={"group": group, "rubric": rub.version, "rubric_hash": rub.hash,
                                           "cartridge": cartridge, "n": len(ids)})
            art.add(table, "side_by_side")
            art.add_file(str(rows_path))
            run.run.log_artifact(art)
            url = run.url
        head = {s: {k: summaries[s]["all"][k] for k in ("non_loss", "non_loss_ci", "win", "tie", "loss",
                                                        "inconsistency", "x_error_rate", "y_error_rate")}
                for s in summaries}
        _print_json({"group": group, "run": url, "vs_teacher": head,
                     "citation_validity": {s: cites[s]["all"] for s in cites}, "answers": str(answers_path),
                     "side_by_side": str(rows_path)})


def _unanswered(ids: list[str], tasks: list, saved: dict, system: str) -> list[dict]:
    """[{"task_id", "text"}] of the tasks with no saved answer from `system`, in order."""
    return [{"task_id": tid, "text": t.text} for tid, t in zip(ids, tasks, strict=True)
            if system not in saved.get(tid, {})]


def _collect(stream, saved: dict, system: str, write) -> None:
    """Store each streamed answer row under saved[task_id][system] and write()
    right away, so a failure part way keeps every answer decoded so far."""
    for row in stream:
        saved.setdefault(row["task_id"], {"task_id": row["task_id"]})[system] = {
            k: row[k] for k in ("text", "finish_reason", "n_tokens")}
        write()


def _download_shards(uris: list[str], cache_dir: Path) -> Path:
    """Copy the Target Store shards these uris point to from the
    cartridges-cartridge-cache Volume into cache_dir (Volume-relative layout);
    returns cache_dir, the root for targets.read_labeled."""
    import modal

    from cartridges.cartridge.tasks import targets

    volume = modal.Volume.from_name("cartridges-cartridge-cache")
    for rel in dict.fromkeys(r for u in uris if (r := targets.volume_relative(u))):
        local = cache_dir / rel
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            with open(local, "wb") as f:
                for chunk in volume.read_file(rel):
                    f.write(chunk)
    return cache_dir


def _seed(file: str, model: str, group: str, n_eval: int = 8, seed: int = 0, replace: bool = False,
          no_templates: bool = False) -> dict:
    from functools import partial

    from cartridges.cartridge.tasks import seed as seed_mod, store

    rows = seed_mod.read_jsonl(Path(file))
    config = {"file": file, "n_eval": n_eval, "seed": seed, "replace": replace, "no_templates": no_templates}
    with store.connect() as conn, _tasks_run(conn, "seed", model, group, config) as (run, group):
        deleted = store.delete_human_tasks(conn) if replace else 0
        generate = None if no_templates else _tasks_generate(model)
        report = seed_mod.seed(
            rows, run_id=run.id, generate=generate, insert=partial(store.insert_tasks, conn), n_eval=n_eval, seed=seed,
        )
        run.log({"datagen/inserted_human": report["inserted_human"],
                 "datagen/inserted_templates": report["inserted_templates"]})
    return {"run_id": run.id, "deleted": deleted, **report}


def _synthesize(n: int, seed: int, model: str, corpus: str, group: str, kind_mix: dict | None = None) -> dict:
    """One synthesis run; kind_mix defaults to taxonomy.KIND_MIX."""
    from functools import partial

    from cartridges.cartridge.tasks import sampler, seed as seed_mod, store
    from cartridges.cartridge.tasks.synthesize import synthesize
    from cartridges.cartridge.tasks.taxonomy import KIND_MIX, PROMPT_VERSION, STYLE_MIX
    from cartridges.cartridge.tasks.types import SynthConfig

    kind_mix = kind_mix or KIND_MIX
    corpus_id = seed_mod.corpus_id(Path(corpus).read_text())
    config = {"n": n, "seed": seed, "corpus_id": corpus_id, "prompt_version": PROMPT_VERSION, "kind_mix": kind_mix}
    with store.connect() as conn, _tasks_run(conn, "synthesize", model, group, config) as (run, group):
        report = synthesize(
            SynthConfig(n=n, seed=seed, model=model, run_id=run.id, corpus_id=corpus_id, kind_mix=kind_mix,
                        style_mix=STYLE_MIX, prompt_version=PROMPT_VERSION),
            fetch_pages=partial(store.fetch_pages, conn),
            fetch_hashes=partial(store.fetch_hashes, conn),
            sample=partial(sampler.sample_tasks, partial(store.fetch_tasks, conn)),
            generate=_tasks_generate(model),
            insert=partial(store.insert_tasks, conn),
        )
        run.log({f"datagen/{k}": v for k, v in report.model_dump().items() if isinstance(v, (int, float))})
    return {"run_id": run.id, **report.model_dump()}


def _classify_chats(model: str, group: str, todo: list | None = None, hits: int | None = None) -> dict:
    """Classify stored chat prompts with no cached label for (model,
    chats.prompt_hash()) on a running Teacher, caching each batch as it
    returns. The caller probes and ends the session. Returns the stats and 10
    random relevant and 10 random discarded prompts."""
    import random
    import time

    from cartridges.cartridge.tasks import chats, store
    from cartridges.cartridge.tasks.types import ChatRow

    prompt_hash = chats.prompt_hash()
    if todo is None:
        with store.connect() as conn:
            todo, hits = store.fetch_unlabeled_chats(conn, model, prompt_hash)
    report: dict = {"prompt_hash": prompt_hash[:12], "cache_hits": hits, "cache_misses": len(todo)}
    if not todo:
        return report
    config = {"prompt_version": chats.PROMPT_VERSION, "prompt_hash": prompt_hash, "cache_hits": hits,
              "cache_misses": len(todo)}
    rows: list[ChatRow] = []
    with store.connect() as conn, _tasks_run(conn, "classify", model, group, config) as (run, group):

        def write(batch):
            new = [ChatRow(prompt=p, label=lab, classifier_model=model, prompt_hash=prompt_hash, run_id=run.id)
                   for p, lab in batch]
            with store.connect() as conn:  # per batch: Neon drops idle connections
                store.insert_chat_labels(conn, new)
            rows.extend(new)

        start = time.monotonic()
        labels = chats.classify(todo, _tasks_generate(model), write)
        wall = time.monotonic() - start
        stats = chats.label_stats(labels)
        report |= {"run_id": run.id, "cached": sum(r.label.category != "unparsed" for r in rows), "classify_s": wall,
                   "prompts_per_s": len(todo) / wall, **stats}
        run.log({"chats/relevant_rate": stats["relevant_rate"], "chats/unparsed_rate": stats["unparsed_rate"],
                 "chats/prompts_per_s": len(todo) / wall, "chats/cache_hits": hits, "chats/cache_misses": len(todo),
                 **{f"chats/category/{k}": v for k, v in stats["categories"].items()}})
    rng = random.Random(0)
    for name, pool in (("relevant", [r for r in rows if r.label.relevant is True]),
                       ("discarded", [r for r in rows if r.label.relevant is not True])):
        report[f"sample_{name}"] = [
            {"category": r.label.category, "task_kind": r.label.task_kind, "reason": r.label.reason,
             "text": r.prompt.text[:300]}
            for r in rng.sample(pool, min(10, len(pool)))
        ]
    return report


def _split_ratios(split: str) -> dict[str, float]:
    """"0.8,0.1,0.1" -> {"train": 0.8, "eval": 0.1, "exemplar": 0.1}."""
    values = [float(v) for v in str(split).split(",")]
    if len(values) != 3 or min(values) < 0 or not sum(values):
        raise SystemExit(f"split must be three non-negative ratios train,eval,exemplar, got {split!r}")
    return dict(zip(("train", "eval", "exemplar"), values))


def _promote_chats(model: str, ratios: dict[str, float], seed: int = 0) -> dict:
    """Relevant chat prompts with a task kind (labels of `model` and the current
    prompt) into cartridge.tasks as origin "chat"; idempotent."""
    from collections import Counter

    from cartridges.cartridge.tasks import chats, store

    with store.connect() as conn:
        labeled = store.fetch_promotable_chats(conn, model, chats.prompt_hash())
        tasks, dups = chats.promote(labeled, store.fetch_hashes(conn), ratios, seed)
        inserted = store.insert_tasks(conn, tasks)
    return {
        "relevant_with_kind": len(labeled), "new": len(tasks), "inserted": inserted, "already_in_store": dups,
        "splits": dict(Counter(t.split for t in tasks if t.status == "validated")),
        "kinds": dict(Counter(t.kind for t in tasks if t.status == "validated")),
        "rejected": dict(Counter(t.reject_reason for t in tasks if t.reject_reason)),
    }


def _log_targets(run, run_id: str) -> str | None:
    """Log targets-<run_id>: a manifest JSON of the label run's shards (on the
    cartridges-cartridge-cache Volume) and label counts."""
    import json

    from cartridges.cartridge.tasks import store

    with store.connect() as conn:
        manifest = store.targets_manifest(conn, run_id)
    path = Path("data/cartridge/targets") / f"targets-{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return run.log_artifact(path, type="dataset", name=f"targets-{run_id}", metadata=manifest)

def _tasks_probe(model: str, gpu: str | None = None, max_inputs: int | None = None):
    """Spawn one short request with the journal in context, on the same Teacher
    handle as the batch; returns the FunctionCall (its result is [Completion])."""
    from cartridges.cartridge.teacher import teacher_cls

    t = teacher_cls("Teacher", model, **_pool(gpu, max_inputs))
    return t.generate.spawn([{"role": "user", "content": "Say OK."}], max_tokens=8, temperature=0.0,
                            with_context=True)


def _tasks_stop(container_ids: list[str] | None = None) -> None:
    """End of a GPU session: stop only this session's own containers, by id.
    Never the bare teacher.stop(): it stops every cartridges-teacher container,
    other sessions' too (the user's serve gateway on 2026-10-16). Without ids
    the session's calls are already done or cancelled (generate_all and the
    probe cancel on any interrupt), so its container scales down after the
    idle window."""
    from cartridges.cartridge import teacher

    if container_ids:
        logger.info("stopped this session's teacher containers: %s", teacher.stop(container_ids=container_ids))
    else:
        logger.info("teacher left to scale down after its idle window (no container id known)")


def _session_container(model: str, gpu: str | None = None, max_inputs: int | None = None) -> str | None:
    """The id of the container serving this handle's pool (Teacher.info's
    container_id), so the session can stop exactly it; None if unknown."""
    from cartridges.cartridge.teacher import teacher_cls

    return teacher_cls("Teacher", model, **_pool(gpu, max_inputs)).info.remote().get("container_id")


def _id_list(value) -> list[str]:
    """Comma-separated ids as fire passes them: a str, or a tuple/list when
    the value has a comma."""
    parts = value if isinstance(value, (list, tuple)) else str(value).split(",")
    return [str(p).strip() for p in parts if str(p).strip()]


def _heldout(path: str) -> set[str]:
    import json

    return set(json.loads(Path(path).read_text())["pages"])


def _pool(gpu: str | None, max_inputs: int | None) -> dict:
    """teacher_cls options; gpu and max_inputs together define the container
    pool, so a session passes the same pair to every call."""
    return {"gpu": gpu} | ({"max_inputs": max_inputs} if max_inputs else {})


def _tasks_generate(model: str, gpu: str | None = None, max_inputs: int | None = None):
    """`generate(conversations, **kw) -> [Completion | None]` over the deployed
    Teacher for `model`: one spawned call each (teacher.generate_all), so an
    interrupt cancels every unfinished call and nothing restarts a container
    after stop()."""
    from cartridges.cartridge import teacher

    handle = teacher.teacher_cls("Teacher", model, **_pool(gpu, max_inputs))  # the same pool for every call
    return lambda convs, **kw: [r[0] if r else None for r in teacher.generate_all(handle, convs, **kw)]


def _tasks_write_shard():
    """Shards are written locally and uploaded to the cartridges-cartridge-cache
    Volume; labels store the /cache/... path that containers see."""
    import modal

    from cartridges.cartridge.tasks import targets

    return targets.volume_writer(modal.Volume.from_name("cartridges-cartridge-cache", create_if_missing=True))


@contextmanager
def _tasks_run(conn, component: str, model: str, group: str, config: dict):
    """(W&B run, its group) with the run's gen_runs row (run_id = the W&B run id)."""
    from cartridges.cartridge import tracking
    from cartridges.cartridge.tasks import store
    from cartridges.cartridge.tasks.taxonomy import PROMPT_VERSION
    from cartridges.cartridge.tasks.types import GenRun

    group = group or tracking.group_name("tasks")
    job_type = "teacher" if component == "label" else "datagen"
    with tracking.init_run(job_type, model, group, config={"component": component, **config}, gpu="CPU") as run:
        store.insert_run(conn, GenRun(run_id=run.id, component=component, model=model,
                                      prompt_version=PROMPT_VERSION if component == "synthesize" else None,
                                      config=config))
        yield run, group


def _json_id_list(path: str) -> list[str]:
    import json

    return json.loads(Path(path).read_text()) if path else []


def _mix(text) -> dict[str, float]:
    """"chat=0.5,synthetic=0.5" -> {"chat": 0.5, "synthetic": 0.5}; "" -> {}."""
    items = text.split(",") if isinstance(text, str) else list(text)
    return {k.strip(): float(v) for k, v in (item.split("=") for item in items if item)}


def _print_json(data: dict) -> None:
    import json

    print(json.dumps(data, indent=2, default=str))


def _pdf_paths(items: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        paths.extend(sorted(path.glob("*.pdf")) if path.is_dir() else [path])
    if not paths:
        raise SystemExit("no PDF files given")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"not found: {missing}")
    return paths


def _serve_function():
    import modal

    return modal.Function.from_name("cartridges-ocr-vllm", "serve")


def _require_deployed(revision: str) -> dict:
    """The deploy record. If `revision` is given, check that it is deployed."""
    from cartridges.serve.registry import deployed, resolve_revision

    record = deployed()
    if record is None or "model_id" not in record:
        raise SystemExit("no deployed OCR service; run `cartridges serve_deploy` first")
    if revision and resolve_revision(revision) != record["model_sha"]:
        raise SystemExit(
            f"deployed revision is {record['revision']!r} ({record['model_sha'][:8]}); "
            f"run `cartridges serve_deploy --revision {revision}` first"
        )
    return record


def _load_split(dataset_dir: str, split: str) -> tuple[list[str], list[bytes]]:
    """(references, PNG bytes) of a labeled split. Same bytes as eval's
    example["image"].save(<path>.png)."""
    import io

    from datasets import load_from_disk

    references, images = [], []
    for example in load_from_disk(dataset_dir)[split]:
        buf = io.BytesIO()
        example["image"].save(buf, format="PNG")
        references.append(example["text"])
        images.append(buf.getvalue())
    return references, images


def _service_backend(concurrency: int, require_key: bool = True):
    import os

    from cartridges.batch.backends import OcrServiceBackend

    api_key = os.environ["VLLM_API_KEY"] if require_key else os.environ.get("VLLM_API_KEY", "")
    return OcrServiceBackend(
        _serve_function().get_web_url(), api_key=api_key, max_connections=concurrency
    )


def _eval_served(revision: str, split: str, dataset_dir: str, concurrency: int):
    """Evaluate the deployed OCR service on a labeled split (serve.evaluate)."""
    import asyncio
    import time

    from cartridges.serve.evaluate import EvalReport, score, transcribe_all

    record = _require_deployed(revision)
    references, images = _load_split(dataset_dir, split)

    async def run():
        backend = _service_backend(concurrency)
        try:
            # /info first: it waits for a cold start, so the timing below
            # measures transcription only.
            served = await backend.model_id()
            if served != record["model_id"]:
                raise SystemExit(f"service serves {served}, deploy record says {record['model_id']}")
            logger.info("eval: model_id=%s split=%s n=%d", served, split, len(images))
            start = time.monotonic()
            hypotheses = await transcribe_all(images, backend, concurrency)
            return hypotheses, time.monotonic() - start
        finally:
            await backend.aclose()

    hypotheses, elapsed = asyncio.run(run())
    return EvalReport(
        model_id=record["model_id"], model_sha=record["model_sha"], revision=record["revision"],
        split=split, elapsed_s=elapsed, pages_per_s=len(images) / elapsed,
        **score(references, hypotheses),
    )


def _merge(sha: str, overwrite: bool = False) -> dict:
    from cartridges.serve.merge import app, merge_adapter

    with app.run():
        return merge_adapter.remote(sha, overwrite=overwrite)


def main() -> None:
    configure_logging()
    fire.Fire(Cli)
