from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class SplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_train_periods: int = Field(ge=1)
    test_periods: int = Field(ge=1)
    step_periods: int = Field(ge=1)
    embargo_periods: int = Field(default=0, ge=0)


class Fold(BaseModel):
    model_config = ConfigDict(frozen=True)

    fold_id: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


class PurgedWalkForward:
    def __init__(self, config: SplitConfig) -> None:
        self.config = config

    def split(
        self,
        index: Sequence[date],
        label_end: Sequence[date],
    ) -> Iterator[Fold]:
        if len(index) != len(label_end):
            raise ValueError("index and label_end must have equal length")
        if list(index) != sorted(index):
            raise ValueError("index must be sorted")
        fold_id = 0
        for test_start in range(
            self.config.min_train_periods,
            len(index),
            self.config.step_periods,
        ):
            test_stop = min(test_start + self.config.test_periods, len(index))
            if test_stop <= test_start:
                break
            train_stop = max(0, test_start - self.config.embargo_periods)
            cutoff = index[test_start]
            train = tuple(
                position for position in range(train_stop) if label_end[position] < cutoff
            )
            test = tuple(range(test_start, test_stop))
            if not train or not test:
                continue
            yield Fold(fold_id=fold_id, train_indices=train, test_indices=test)
            fold_id += 1
