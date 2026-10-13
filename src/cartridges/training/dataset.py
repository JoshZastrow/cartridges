"""Build and load the hand-labeled image-to-text fine-tuning dataset.

Source of truth is `data/journals/to_process_manual/<journal>/`: each journal
folder has one `page_NNNN.md` per hand-transcribed page (YAML frontmatter +
verbatim transcription) and a `_images/page-NNN.png` for every rasterized
page in the source PDF (labeled and unlabeled alike).

Building the dataset stages the labeled subset into an ImageFolder layout
(images + `metadata.jsonl`) and loads it with `datasets.load_dataset`. See
https://huggingface.co/docs/datasets/en/image_dataset#imagefolder-with-metadatajsonl

The train/test split is frozen in `configs/dataset/split.json` (membership and
row order, keyed by [journal, page]), so adding, removing or renumbering pages
never reshuffles the test set. Without that file, the first build makes a
random split and writes it.
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
DEFAULT_SPLIT_FILE = Path("configs/dataset/split.json")

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
    # Clear the previous staging: a removed or renumbered page would otherwise
    # leave its old copy behind for imagefolder to pick up.
    for stale in dataset_dir.glob("*.png"):
        stale.unlink()

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


def ordered_split(keys: list[tuple[str, int]], spec: dict) -> tuple[list[int], list[int]]:
    """Row indices for (train, test) from a frozen split spec.

    `keys` are the staged rows' (journal, page); `spec` is the split file's
    {"train": [[journal, page], ...], "test": [...]}. Each split keeps the
    spec's order; staged pages the spec doesn't list are appended to train in
    staging order. A listed page that isn't staged, or one listed in both
    splits, is an error.
    """
    position = {key: i for i, key in enumerate(keys)}
    train_keys = [tuple(k) for k in spec["train"]]
    test_keys = [tuple(k) for k in spec["test"]]
    overlap = set(train_keys) & set(test_keys)
    if overlap:
        raise ValueError(f"pages listed in both splits: {sorted(overlap)}")
    missing = [k for k in train_keys + test_keys if k not in position]
    if missing:
        raise ValueError(f"split file lists pages not in the staged data: {missing}")
    listed = set(train_keys) | set(test_keys)
    train = [position[k] for k in train_keys] + [i for i, k in enumerate(keys) if k not in listed]
    test = [position[k] for k in test_keys]
    return train, test


def build_dataset(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    save_dir: Path | None = DEFAULT_SAVE_DIR,
    test_size: float = 0.2,
    seed: int = 3407,
    split_file: Path | None = DEFAULT_SPLIT_FILE,
) -> DatasetDict:
    """Stage labeled pages, load them as a Dataset, split, and (optionally) save to disk.

    With `split_file` present the split comes from it (see `ordered_split`);
    otherwise a random split (`test_size`, `seed`) is made and written there.
    """
    stage_dataset(source_root, dataset_dir)
    dataset: Dataset = load_dataset("imagefolder", data_dir=str(dataset_dir), split="train")

    if split_file is not None and split_file.exists():
        keys = list(zip(dataset["journal"], dataset["page"]))
        train, test = ordered_split(keys, json.loads(split_file.read_text()))
        split_dataset = DatasetDict({"train": dataset.select(train), "test": dataset.select(test)})
    else:
        split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)
        if split_file is not None:
            split_file.parent.mkdir(parents=True, exist_ok=True)
            spec = {
                name: [[j, p] for j, p in zip(split_dataset[name]["journal"], split_dataset[name]["page"])]
                for name in ("train", "test")
            }
            split_file.write_text(json.dumps(spec, indent=1) + "\n")

    if save_dir is not None:
        split_dataset.save_to_disk(str(save_dir))

    return split_dataset


def load_training_dataset(save_dir: Path = DEFAULT_SAVE_DIR) -> DatasetDict:
    """Load a dataset previously written by `build_dataset`."""
    return load_from_disk(str(save_dir))
