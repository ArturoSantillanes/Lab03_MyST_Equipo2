"""Prueba 1 de Milca: la partición es cronológica, sin traslape y sin look-ahead (data/split.py)."""
import pandas as pd

from data.split import PERIODS, assign_period, compute_splits, slice_period

DATES = pd.bdate_range("2011-01-03", periods=3772, name="date")


def test_split_is_chronological_contiguous_and_60_20_20():
    splits = compute_splits(DATES)
    assert sum(splits[p]["n_obs"] for p in PERIODS) == len(DATES)
    for share, period in zip((0.6, 0.2, 0.2), PERIODS):
        assert abs(splits[period]["n_obs"] - share * len(DATES)) <= 1
    for earlier, later in zip(PERIODS, PERIODS[1:]):
        end, start = pd.Timestamp(splits[earlier]["end"]), pd.Timestamp(splits[later]["start"])
        assert end < start
        assert DATES.get_loc(start) == DATES.get_loc(end) + 1


def test_every_date_has_exactly_one_period_in_order():
    labels = assign_period(DATES, compute_splits(DATES))
    assert labels.notna().all()
    assert list(dict.fromkeys(labels)) == list(PERIODS)


def test_train_slice_never_reaches_test_dates():
    splits = compute_splits(DATES)
    train = slice_period(pd.DataFrame({"x": range(len(DATES))}, index=DATES), splits, "train")
    assert train.index.max() < pd.Timestamp(splits["test"]["start"])
    assert len(train) == splits["train"]["n_obs"]
