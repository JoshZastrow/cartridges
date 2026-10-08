"""Build and load the hand-labeled image-to-text fine-tuning dataset.

Source of truth is `data/journals/to_process_manual/<journal>/`: each journal
folder has one `page_NNNN.md` per hand-transcribed page (YAML frontmatter +
verbatim transcription) and a `_images/page-NNN.png` for every rasterized
page in the source PDF (labeled and unlabeled alike).

Building the dataset stages the labeled subset into an ImageFolder layout
(images + `metadata.jsonl`) and loads it with `datasets.load_dataset`. See
https://huggingface.co/docs/datasets/en/image_dataset#imagefolder-with-metadatajsonl
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import yaml
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

DEFAULT_SOURCE_ROOT = Path("data/journals/to_process_manual")
DEFAULT_DATASET_DIR = Path("data/train/images")
DEFAULT_SAVE_DIR = Path("data/train/hf_dataset")

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n\n(.*)\Z", re.DOTALL)


def parse_labeled_page(md_path: Path) -> dict:
    """Return {journal, page, date, source_file, text, image_path} for one labeled page."""
    match = _FRONTMATTER_RE.match(md_path.read_text())
    if not match:
        raise ValueError(f"{md_path} does not look like a frontmatter'd page")
    frontmatter = yaml.safe_load(match.group(1))
    text = match.group(2).strip()

    page = frontmatter["page"]
    image_path = md_path.parent / "_images" / f"page-{page:03d}.png"
    if not image_path.exists():
        raise FileNotFoundError(f"no matching image for {md_path}: {image_path}")

    return {
        "journal": md_path.parent.name,
        "page": page,
        "date": frontmatter.get("date"),
        "source_file": frontmatter.get("source_file"),
        "text": text,
        "image_path": image_path,
    }


def stage_dataset(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> list[dict]:
    """Copy every labeled page's image into `dataset_dir` and write `metadata.jsonl`.

    Returns the rows written to `metadata.jsonl`.
    """
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for journal_dir in sorted(source_root.iterdir()):
        if not journal_dir.is_dir():
            continue
        for md_path in sorted(journal_dir.glob("page_*.md")):
            example = parse_labeled_page(md_path)
            file_name = f"{example['journal']}_page-{example['page']:03d}.png"
            shutil.copyfile(example["image_path"], dataset_dir / file_name)
            rows.append(
                {
                    "file_name": file_name,
                    "text": example["text"],
                    "journal": example["journal"],
                    "page": example["page"],
                    "date": example["date"],
                    "source_file": example["source_file"],
                }
            )

    with (dataset_dir / "metadata.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return rows


def build_dataset(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    save_dir: Path | None = DEFAULT_SAVE_DIR,
    test_size: float = 0.2,
    seed: int = 3407,
) -> DatasetDict:
    """Stage labeled pages, load them as a Dataset, split, and (optionally) save to disk."""
    stage_dataset(source_root, dataset_dir)
    dataset: Dataset = load_dataset("imagefolder", data_dir=str(dataset_dir), split="train")
    split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)

    if save_dir is not None:
        split_dataset.save_to_disk(str(save_dir))

    return split_dataset


def load_training_dataset(save_dir: Path = DEFAULT_SAVE_DIR) -> DatasetDict:
    """Load a dataset previously written by `build_dataset`."""
    return load_from_disk(str(save_dir))
