"""JSONL storage layer rooted at a directory."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class JsonlStorage:
    """Append-only JSONL storage rooted at a single directory."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, relpath: str) -> Path:
        path = (self.root / relpath).resolve()
        root = self.root.resolve()
        if root not in path.parents and path != root:
            raise ValueError(f"Path {relpath} escapes storage root {root}")
        return path

    def append(self, relpath: str, item: BaseModel) -> None:
        path = self._resolve(relpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(item.model_dump_json() + "\n")

    def read(self, relpath: str, cls: type[T]) -> Iterator[T]:
        path = self._resolve(relpath)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield cls.model_validate(json.loads(line))
