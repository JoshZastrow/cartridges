"""A torch IterableDataset of verified question/answer pairs (see datagen.qa).

Each step samples a passage at random, asks the teacher for pairs, and yields
the pairs that pass verify(). It never ends: take what you need with
itertools.islice or a step count. Use DataLoader(..., batch_size=None) so the
pairs come out one dict at a time; num_workers > 0 runs teacher calls in
parallel, each worker with its own seed.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator

from torch.utils.data import IterableDataset, get_worker_info

from cartridges.datagen.qa import Passage, verify


class QAPairs(IterableDataset):
    def __init__(self, passages: list[Passage], teacher: Callable[[Passage], list[dict]], seed: int = 0):
        if not passages:
            raise ValueError("no passages")
        self.passages, self.teacher, self.seed = passages, teacher, seed

    def __iter__(self) -> Iterator[dict]:
        worker = get_worker_info()
        rng = random.Random(self.seed + (worker.id if worker else 0))
        while True:
            passage = rng.choice(self.passages)
            for pair in self.teacher(passage):
                if verify(pair, passage):
                    yield {
                        **{k: pair[k] for k in ("question", "answer", "answer_type", "citations", "evidence")},
                        "passage_id": passage.id,
                        "date": passage.pages[0][1],
                    }
