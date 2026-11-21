"""MRCR full-history dataset adapter for AIPerf."""

from __future__ import annotations

import json
from itertools import islice
from typing import Any

from aiperf.common.models import Conversation, Turn
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader
from datasets import load_dataset


class MRCRLoader(BaseHFDatasetLoader):
    """Load full MRCR histories without replaying or regenerating earlier turns."""

    def __init__(self, *, filters: dict[str, str] | None = None, **kwargs):
        options = filters or {}
        self.needles = int(options.get("needles", "8"))
        self.max_samples = int(options.get("max_samples", "100"))
        self.hf_revision = options.get("revision")
        if self.needles not in (2, 4, 8) or self.max_samples < 1:
            raise ValueError("MRCR requires needles=2,4,8 and max_samples >= 1")
        super().__init__(**kwargs)

    def _load_hf_dataset(self) -> Any:
        return load_dataset(
            self.hf_dataset_name,
            data_files=f"{self.needles}needle/*.parquet",
            split=self.hf_split,
            revision=self.hf_revision,
            streaming=True,
        )

    async def convert_to_conversations(
        self, dataset: dict[str, Any]
    ) -> list[Conversation]:
        conversations = []
        for index, row in enumerate(islice(dataset["dataset"], self.max_samples)):
            messages = row["prompt"]
            if isinstance(messages, str):
                messages = json.loads(messages)
            conversations.append(
                Conversation(
                    session_id=f"mrcr-{self.needles}-{index}",
                    turns=[Turn(raw_messages=messages)],
                )
            )
        return conversations
