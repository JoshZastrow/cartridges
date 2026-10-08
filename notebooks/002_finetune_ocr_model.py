import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(f"""
    # Fine-tune the OCR model on hand-labeled pages

    This orchestrates a LoRA fine-tune of `unsloth/DeepSeek-OCR-2` on the
    dataset built by `001_build_ocr_dataset.py`. The actual training/serving
    runs on Modal (this Mac has no CUDA) — everything here is a thin
    `.remote()` call into `cartridges.training.finetune` (training + eval)
    or `cartridges.serve.inference` (one-shot generation).

    **Note:** each `.remote()` call launches real (billed) Modal GPU time.
    Review the config below, especially `max_steps`, before running the
    training cell.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Download the base model

    Snapshot-downloads `unsloth/DeepSeek-OCR-2` into a Modal volume
    (`cartridges-training-cache`), so it's cached across runs instead of
    re-downloading every time.
    """)
    return


@app.cell
def _():
    from cartridges.training.finetune import app as training_app
    from cartridges.training.finetune import (
        download_model,
        evaluate,
        finetune,
        preview_collator_example,
    )
    from cartridges.training.config import DownloadConfig, EvalConfig, ModelSource, TrainConfig

    return (
        DownloadConfig,
        EvalConfig,
        ModelSource,
        TrainConfig,
        download_model,
        evaluate,
        finetune,
        preview_collator_example,
        training_app,
    )


@app.cell
async def _(DownloadConfig, download_model, training_app):
    async with training_app.run():
        base_model_dir = await download_model.remote.aio(DownloadConfig())
    base_model_dir
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Pick an example and visualize the collator

    The raw pair straight from the dataset — free, local, no Modal call.
    """)
    return


@app.cell
def _():
    from cartridges.training.dataset import load_training_dataset

    example_index = 0
    example = load_training_dataset()["train"][example_index]
    return example, example_index


@app.cell
def _(example, mo):
    mo.vstack([mo.image(example["image"], width=400), mo.md(example["text"])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then the same example run through `DeepSeekOCRDataCollator` on Modal
    (tokenizer + image tiling only, no GPU/model weights, so it's cheap) to
    see what actually reaches the trainer: total token count, how many are
    image-placeholder tokens, how many are masked out of the loss, and the
    decoded text of the *supervised* region (should match the raw
    transcription, modulo the EOS token).
    """)
    return


@app.cell
async def _(TrainConfig, example_index, preview_collator_example, training_app):
    async with training_app.run():
        collator_preview = await preview_collator_example.remote.aio(TrainConfig(), example_index)
    collator_preview
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Prove inference actually works, on that same example

    Before trusting anything downstream, run one real `.remote()` call
    through `cartridges.serve.inference.generate` (the standalone serving
    module, separate from `training/` — plain `transformers` load, no
    unsloth) with the exact image + text from above, and look at the raw
    model output plus a per-stage timing breakdown (model load, image
    decode, inference). This is also the smoke test for the base model that
    produced that suspicious "present perfect tense" baseline output
    earlier: if the model's answer here has nothing to do with the image,
    the bug is upstream of anything eval-specific.
    """)
    return


@app.cell
def _():
    from cartridges.serve.inference import app as serve_app
    from cartridges.serve.inference import generate as serve_generate
    from cartridges.serve.schema import ChatMessage, GenerateRequest, ImageContent, ServeConfig, TextContent

    return (
        ChatMessage,
        GenerateRequest,
        ImageContent,
        ServeConfig,
        TextContent,
        serve_app,
        serve_generate,
    )


@app.cell
def _(ChatMessage, GenerateRequest, ImageContent, ServeConfig, TextContent, example):
    import base64
    import io

    _buf = io.BytesIO()
    example["image"].save(_buf, format="PNG")
    _image_base64 = base64.b64encode(_buf.getvalue()).decode("ascii")

    serve_request = GenerateRequest(
        serve_config=ServeConfig(),
        messages=[
            ChatMessage(
                content=[
                    TextContent(text="<image>\nFree OCR. "),
                    ImageContent(image_base64=_image_base64),
                ]
            )
        ],
    )
    return (serve_request,)


@app.cell
async def _(serve_app, serve_generate, serve_request):
    async with serve_app.run():
        serve_response = await serve_generate.remote.aio(serve_request)
    serve_response
    return (serve_response,)


@app.cell
def _(example, mo, serve_response):
    mo.vstack(
        [
            mo.md(f"**Reference:**\n\n{example['text']}"),
            mo.md(f"**Model output:**\n\n{serve_response.text}"),
            mo.md(
                "**Timings (s):** "
                + ", ".join(f"{k}={v:.1f}" for k, v in serve_response.stage_timings_s.items())
                + f", total={serve_response.total_s:.1f}"
            ),
        ]
    )
    return


@app.cell
def _():
    import jiwer

    def character_error_rate(reference: str, hypothesis: str) -> float:
        """Ad-hoc scoring, e.g. for one prediction by hand. `evaluate()` already
        computes per-example CER server-side (see each prediction's "cer")."""
        return jiwer.cer(reference, hypothesis)

    def render_eval_table(results: list[dict]) -> str:
        """Markdown table over `evaluate()`'s return shape, any model_source."""

        def _fmt(value, spec="{}"):
            return "—" if value is None else spec.format(value)

        def _run_link(fold: dict) -> str:
            if not fold.get("wandb_run_url"):
                return "—"
            return f"[{fold['wandb_run_name']}]({fold['wandb_run_url']})"

        def _row(fold: dict) -> str:
            scored = [p["cer"] for p in fold["predictions"] if p["cer"] is not None]
            n_errors = len(fold["predictions"]) - len(scored)
            mean_cer = sum(scored) / len(scored) if scored else None
            return (
                f"| {fold['model_id']} | {fold['model_source']} | {_fmt(fold['hf_revision'])} "
                f"| {_fmt(fold['fold'])} | {_fmt(fold['train_size'])} | {fold['eval_size']} "
                f"| {_fmt(mean_cer, '{:.2%}')} | {n_errors} "
                f"| {_fmt(fold['total_tokens'])} | {_fmt(fold['avg_tokens_per_example'], '{:.0f}')} "
                f"| {_run_link(fold)} |"
            )

        header = (
            "| Model | Model source | Artifact | Fold | Train size | Eval size | Mean CER | Errors "
            "| Total tokens | Avg tokens/example | Source run |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        )
        return header + "\n".join(_row(fold) for fold in results)

    return character_error_rate, render_eval_table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Baseline: score the un-fine-tuned model

    Before spending any time/GPU on training, see how the plain base model
    already does on the held-out `test` split — this is the number
    fine-tuning actually needs to beat. `evaluate()` with the default
    `ModelSource.BASE` runs no LoRA, no training, just inference.
    """)
    return


@app.cell
def _(EvalConfig, mo):
    baseline_config = EvalConfig(split="test")
    mo.md(
        f"Evaluating **{baseline_config.train_config.model_id.value}** "
        f"(`{baseline_config.model_source.value}`) on the `{baseline_config.split}` split."
    )
    return (baseline_config,)


@app.cell
async def _(baseline_config, evaluate, training_app):
    async with training_app.run():
        baseline_results = await evaluate.remote.aio(baseline_config)
    return (baseline_results,)


@app.cell
def _(baseline_results, mo, render_eval_table):
    mo.md(render_eval_table(baseline_results))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Review the training config

    LoRA rank/alpha, batch size, and `max_steps` mirror the reference
    Unsloth notebook's quick demo run on the same model. Bump `max_steps` up (and drop
    `learning_rate`/warmup accordingly) once a short run shows the loss
    actually moving.
    """)
    return


@app.cell
def _(TrainConfig):
    train_config = TrainConfig(max_steps=1)
    train_config.dict()
    return (train_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Run the fine-tune

    `finetune()` (in `cartridges/training/finetune.py`) loads the base model,
    wraps it with a LoRA adapter, builds the `DeepSeekOCRDataCollator` over
    the dataset shipped into the image from `data/train/hf_dataset/`, trains,
    and saves the adapter back to the cache volume.
    """)
    return


@app.cell
async def _(finetune, train_config, training_app):
    async with training_app.run():
        adapter_dir = await finetune.remote.aio(train_config)
    adapter_dir
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. K-fold evaluation

    Cross-validate over the training split (never touches the true
    held-out `test` split from `001_build_ocr_dataset.py`): for each of
    `k` folds, LoRA fine-tune on the other folds and run inference on the
    held-out one. Each fold is a full training run on Modal — this is the
    most expensive cell in the notebook. Start with a small `k` and the
    smoke-test `max_steps` from `train_config` above before scaling up.

    CER is computed server-side (see `evaluate()`'s logs for per-example
    progress); `render_eval_table` just formats what comes back.
    """)
    return


@app.cell
def _(EvalConfig, ModelSource, mo, train_config):
    kfold_config = EvalConfig(train_config=train_config, model_source=ModelSource.KFOLD_RETRAIN, k=3)
    mo.md(
        f"Evaluating **{kfold_config.train_config.model_id.value}** "
        f"(`{kfold_config.model_source.value}`, k={kfold_config.k}) over the `train` split."
    )
    return (kfold_config,)


@app.cell
async def _(evaluate, kfold_config, training_app):
    async with training_app.run():
        kfold_results = await evaluate.remote.aio(kfold_config)
    kfold_results
    return (kfold_results,)


@app.cell
def _(kfold_results, mo, render_eval_table):
    mo.md(render_eval_table(kfold_results))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Final held-out eval

    Once k-fold CER looks reasonable, fine-tune once on the full `train`
    split (section 6 above) — note the `ocr-finetuned:vN` artifact version
    it logs to W&B — then score *that specific version* against the true
    held-out `test` split. This is the number that actually matters; k-fold
    is for iterating on config, not a substitute for this.
    """)
    return


@app.cell
def _(mo):
    hf_revision_input = mo.ui.text(
        value="run-liw4qzol", label="HF Hub revision (tag/branch/commit)"
    )
    hf_revision_input
    return (hf_revision_input,)


@app.cell
def _(EvalConfig, ModelSource, hf_revision_input, mo):
    final_config = EvalConfig(
        model_source=ModelSource.ARTIFACT,
        hf_revision=hf_revision_input.value,
        split="test",
    )
    mo.md(
        f"Evaluating **{final_config.train_config.model_id.value}** "
        f"+ adapter **{final_config.train_config.hf_repo_id}@{final_config.hf_revision}** "
        f"on the `{final_config.split}` split."
    )
    return (final_config,)


@app.cell
async def _(evaluate, final_config, training_app):
    async with training_app.run():
        final_results = await evaluate.remote.aio(final_config)
    final_results
    return (final_results,)


@app.cell
def _(final_results, mo, render_eval_table):
    mo.md(render_eval_table(final_results))
    return


if __name__ == "__main__":
    app.run()
