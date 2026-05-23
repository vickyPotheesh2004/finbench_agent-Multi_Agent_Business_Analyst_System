from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor


class ParallelExecutor:

    def __init__(
        self,
        max_workers: int = 2,
    ):
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers
        )

    def run(
        self,
        bm25_fn,
        bge_fn,
    ):

        bm25_future = self.executor.submit(
            bm25_fn
        )

        bge_future = self.executor.submit(
            bge_fn
        )

        return (
            bm25_future.result(),
            bge_future.result(),
        )