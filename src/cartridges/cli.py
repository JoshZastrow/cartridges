"""Command-line interface for cartridges."""

import logging
from pathlib import Path

import fire

from cartridges.logging.config import configure_logging

logger = logging.getLogger(__name__)


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
