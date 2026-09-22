"""AnnualExpandingSplit：年份级扩展划分+行级gap，训练集绝不含测试年及跨年标签行。"""

import numpy as np
import pandas as pd
import pytest

from .test_factor_algo import quotes


def test_gap_purges_rows_whose_label_crosses_the_test_year():
    """gap=skip+horizon=6：年前最后6行标签跨年须被清除，第7行保留；训练起点不变、逐折扩展。"""
    from src.factor_algo import forward_label
    from src.splitter import AnnualExpandingSplit

    q = quotes(1500)
    dates = q.index[q.index <= "2021-12-31"]
    _, end = forward_label(q.close, 5, skip=1)
    end = end.reindex(dates)
    cv = AnnualExpandingSplit(n_test_years=2, gap=6)
    folds = list(cv.split(np.zeros(len(dates)), groups=dates.year.to_numpy()))
    assert len(folds) == cv.get_n_splits() == 2
    assert [dates[te[0]].year for _, te in folds] == [2020, 2021]
    assert len(folds[0][0]) < len(folds[1][0])
    assert folds[0][0][0] == folds[1][0][0] == 0
    for train, test in folds:
        train_dates, test_dates = dates[train], dates[test]
        cutoff = pd.Timestamp(f"{test_dates[0].year}-01-01")
        assert (end.loc[train_dates] < cutoff).all()
        assert train_dates.max() < test_dates.min()
        assert (test_dates.year == test_dates[0].year).all()
        before = dates[dates < cutoff]
        assert not set(before[-6:]) & set(train_dates)
        assert before[-7] in train_dates


def test_never_trains_on_future_and_expands():
    from src.splitter import AnnualExpandingSplit

    dates = pd.bdate_range("2015-01-01", "2020-12-31")
    folds = list(AnnualExpandingSplit(3, gap=6).split(np.zeros(len(dates)), groups=dates.year.to_numpy()))
    assert [dates[te[0]].year for _, te in folds] == [2018, 2019, 2020]
    for tr, te in folds:
        assert tr.max() < te.min() - 6
    assert len(folds[0][0]) < len(folds[1][0]) < len(folds[2][0])
    assert folds[0][0][0] == folds[2][0][0] == 0


def test_max_train_years_rolls_window():
    from src.splitter import AnnualExpandingSplit

    dates = pd.bdate_range("2015-01-01", "2020-12-31")
    tr, te = next(
        AnnualExpandingSplit(1, gap=0, max_train_years=2).split(np.zeros(len(dates)), groups=dates.year.to_numpy())
    )
    assert set(dates[tr].year) == {2018, 2019} and set(dates[te].year) == {2020}


def test_single_test_year_with_two_years():
    from src.splitter import AnnualExpandingSplit

    tr, te = next(
        AnnualExpandingSplit(1, gap=0).split(np.zeros(4), groups=np.array([2020, 2020, 2021, 2021]))
    )
    np.testing.assert_array_equal(tr, [0, 1])
    np.testing.assert_array_equal(te, [2, 3])


def test_single_test_year_max_train_years_on_three_years():
    from src.splitter import AnnualExpandingSplit

    groups = np.array([2018, 2018, 2019, 2019, 2020, 2020])
    tr, te = next(
        AnnualExpandingSplit(1, gap=0, max_train_years=1).split(np.zeros(6), groups=groups)
    )
    np.testing.assert_array_equal(tr, [2, 3])
    np.testing.assert_array_equal(te, [4, 5])


def test_requires_sorted_groups():
    from src.splitter import AnnualExpandingSplit

    with pytest.raises(ValueError):
        list(AnnualExpandingSplit(1).split(np.zeros(4)))
    with pytest.raises(ValueError):
        list(AnnualExpandingSplit(1).split(np.zeros(4), groups=np.array([2020, 2019, 2020, 2021])))


def test_works_with_sklearn_grid_search():
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV

    from src.splitter import AnnualExpandingSplit

    dates = pd.bdate_range("2016-01-01", "2020-12-31")
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(len(dates), 3)), rng.normal(size=len(dates))
    gs = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=AnnualExpandingSplit(2, gap=6))
    gs.fit(x, y, groups=dates.year.to_numpy())
    assert gs.best_params_["alpha"] in (0.1, 1.0)
