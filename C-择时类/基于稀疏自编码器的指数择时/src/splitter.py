"""自然年扩展划分：在年份数组上复用sklearn的TimeSeriesSplit，再映射回行并清除越界标签。"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit

from ._validation import require_positive_int


class AnnualExpandingSplit(BaseCrossValidator):
    """最后n_test_years个自然年逐年作测试，训练集为此前全部（或最近若干）年份。

    BaseCrossValidator默认把训练集定义为测试集的补集（含未来），故必须重写split。

    :param n_test_years: 测试年数，取groups中最后这么多个年份。
    :param gap: 测试年首行之前清除的行数，应等于标签跨度 ``label_skip + horizon``；
        行连续时与"标签实现日早于测试年首日"严格等价。
    :param max_train_years: 训练最多回看的年数，None为扩展窗口。
    """

    def __init__(
        self, n_test_years: int, gap: int = 0, max_train_years: int | None = None
    ) -> None:
        self.n_test_years = n_test_years
        self.gap = gap
        self.max_train_years = max_train_years

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """产出(训练行下标, 测试行下标)。

        :param X: 样本，仅用其长度。
        :param y: 忽略。
        :param groups: 每行所属年份，须与X等长且非降序。
        :returns: 逐折下标对的迭代器。
        :raises ValueError: groups缺失、长度不符、乱序，或年份不足。
        """
        require_positive_int(self.n_test_years, "n_test_years")
        if groups is None or len(groups) != len(X):
            raise ValueError("groups须为与X等长的年份数组")
        groups = np.asarray(groups)
        if (np.diff(groups) < 0).any():
            raise ValueError("groups须按时间非降序")
        if self.gap < 0:
            raise ValueError("gap不能为负")
        years = np.unique(groups)
        if len(years) <= self.n_test_years:
            raise ValueError("年份数须多于测试年数")
        if self.n_test_years == 1:
            # sklearn的TimeSeriesSplit要求样本数(年份数)>n_splits+1，n_test_years=1时
            # 年份数恰为2就会不满足（need n_folds=2<=samples），故单测试年直接手写：
            # 测试年=最后一年，训练年=此前全部（或最近max_train_years个）。
            splits = [(np.arange(len(years) - 1), np.array([len(years) - 1]))]
        else:
            # sklearn的TimeSeriesSplit要求n_splits>=2；n_test_years=1时上面已单独处理，
            # 此分支n_test_years>=2，直接按其取n_splits即可。
            inner = TimeSeriesSplit(
                n_splits=self.n_test_years, test_size=1, max_train_size=self.max_train_years
            )
            splits = list(inner.split(years))
        for train_years, test_year in splits:
            if self.n_test_years == 1 and self.max_train_years is not None:
                train_years = train_years[-self.max_train_years :]
            test = np.flatnonzero(groups == years[test_year[0]])
            train = np.flatnonzero(np.isin(groups, years[train_years]))
            # 清除标签实现日跨进测试年的训练行
            yield train[train < test[0] - self.gap], test

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """:returns: 折数，即测试年数。"""
        return self.n_test_years
