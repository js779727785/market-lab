from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class BootstrapResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    estimate: float
    lower_bound: float
    upper_bound: float
    p_value: float = Field(ge=0, le=1)
    confidence_level: float = Field(gt=0, lt=1)
    samples: int = Field(gt=0)
    block_size: int = Field(gt=0)


def block_bootstrap_difference(
    champion: Sequence[float],
    challenger: Sequence[float],
    *,
    block_size: int = 20,
    samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    if len(champion) != len(challenger) or not champion:
        raise ValueError("champion and challenger samples must be non-empty and aligned")
    if block_size <= 0 or samples <= 0:
        raise ValueError("block_size and samples must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    difference = np.asarray(challenger, dtype=float) - np.asarray(champion, dtype=float)
    count = len(difference)
    effective_block_size = min(block_size, count)
    random = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for sample_index in range(samples):
        selected: list[int] = []
        while len(selected) < count:
            start = int(random.integers(0, count))
            selected.extend((start + offset) % count for offset in range(effective_block_size))
        estimates[sample_index] = float(np.mean(difference[selected[:count]]))
    tail = (1.0 - confidence_level) / 2.0
    return BootstrapResult(
        estimate=float(np.mean(difference)),
        lower_bound=float(np.quantile(estimates, tail)),
        upper_bound=float(np.quantile(estimates, 1.0 - tail)),
        p_value=float((np.count_nonzero(estimates <= 0) + 1) / (samples + 1)),
        confidence_level=confidence_level,
        samples=samples,
        block_size=effective_block_size,
    )


def benjamini_hochberg(p_values: Mapping[str, float]) -> dict[str, float]:
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for reverse_index, (key, value) in enumerate(reversed(ordered), start=1):
        rank = count - reverse_index + 1
        running = min(running, value * count / rank)
        adjusted[key] = min(1.0, max(0.0, running))
    return adjusted
