from __future__ import annotations


def batched(items, batch_size=128):

    for i in range(
        0,
        len(items),
        batch_size,
    ):
        yield items[i:i + batch_size]