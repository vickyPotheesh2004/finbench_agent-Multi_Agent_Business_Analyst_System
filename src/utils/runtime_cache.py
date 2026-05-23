from __future__ import annotations

import gc
import hashlib
import pickle
from pathlib import Path
from typing import Any

CACHE_ROOT = Path("cache")

CACHE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


class RuntimeCache:

    @staticmethod
    def _path(namespace: str, key: str) -> Path:

        digest = hashlib.sha256(
            key.encode("utf-8")
        ).hexdigest()

        folder = CACHE_ROOT / namespace

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        return folder / f"{digest}.pkl"

    @classmethod
    def exists(
        cls,
        namespace: str,
        key: str,
    ) -> bool:

        return cls._path(
            namespace,
            key,
        ).exists()

    @classmethod
    def save(
        cls,
        namespace: str,
        key: str,
        obj: Any,
    ) -> None:

        path = cls._path(
            namespace,
            key,
        )

        with open(path, "wb") as f:
            pickle.dump(
                obj,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(
        cls,
        namespace: str,
        key: str,
    ):

        path = cls._path(
            namespace,
            key,
        )

        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def cleanup() -> None:
        gc.collect()