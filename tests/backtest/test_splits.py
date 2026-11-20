from datetime import date, timedelta

from market_lab.backtest.splits import PurgedWalkForward, SplitConfig


def test_purged_walk_forward_removes_overlapping_labels_and_embargoes() -> None:
    index = [date(2026, 1, 1) + timedelta(days=day) for day in range(12)]
    label_end = [value + timedelta(days=3) for value in index]
    splitter = PurgedWalkForward(
        SplitConfig(min_train_periods=5, test_periods=2, step_periods=2, embargo_periods=1)
    )

    folds = list(splitter.split(index, label_end))

    assert folds
    for fold in folds:
        test_start = index[fold.test_indices[0]]
        assert all(label_end[position] < test_start for position in fold.train_indices)
        assert max(fold.train_indices, default=-1) < fold.test_indices[0] - 1
