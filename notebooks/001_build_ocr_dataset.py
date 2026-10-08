import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Build a Hugging Face image-to-text dataset from hand-labeled pages

    Source of truth is `data/journals/to_process_manual/<journal>/`: each
    journal folder has one `page_NNNN.md` per hand-transcribed page (YAML
    frontmatter + verbatim transcription) and a `_images/page-NNN.png` for
    every rasterized page in the source PDF (labeled and unlabeled alike).

    The staging/loading/splitting logic lives in
    `cartridges.training.dataset` — this notebook just calls it and inspects
    the result.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Stage the labeled pages into an ImageFolder layout, load, and split

    `build_dataset()`:

    1. Parses every `page_*.md`'s frontmatter + verbatim text and matches it
       to its rasterized image (`_images/page-{page:03d}.png`).
    2. Copies matched images into `data/train/images/` (journal-prefixed, so
       pages from different journals never collide) alongside a
       `metadata.jsonl` — the layout `datasets`' `ImageFolder` builder expects
       (https://huggingface.co/docs/datasets/en/image_dataset#imagefolder-with-metadatajsonl).
    3. Loads it with `load_dataset("imagefolder", ...)`, which auto-decodes
       `file_name` into an `Image` feature and attaches every other
       `metadata.jsonl` column as-is.
    4. Splits 80/20 train/test (same idea as the golden set: never train on
       what you evaluate against) and saves to `data/train/hf_dataset/`.
    """)
    return


@app.cell
def _():
    from cartridges.training.dataset import build_dataset

    split_dataset = build_dataset()
    split_dataset
    return (split_dataset,)


@app.cell
def _(mo):
    mo.md(r"""## 2. Inspect a sample""")
    return


@app.cell
def _(split_dataset):
    split_dataset["train"][0]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Reload from disk

    Later steps (e.g. the fine-tuning script) can call this directly instead
    of re-running the staging step.
    """)
    return


@app.cell
def _():
    from cartridges.training.dataset import load_training_dataset

    reloaded_dataset = load_training_dataset()
    reloaded_dataset
    return (reloaded_dataset,)


@app.cell
def _(reloaded_dataset):
    reloaded_dataset["test"][0]
    return


if __name__ == "__main__":
    app.run()
