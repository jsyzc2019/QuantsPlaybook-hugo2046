"""共享输入校验工具的契约测试。"""

import numpy as np
import pandas as pd
import pytest


def _frame(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame({"close": np.arange(len(index), dtype=float)}, index=index)


def test_valid_datetime_index_passes():
    from src._validation import validate_datetime_index

    idx = pd.bdate_range("2020-01-01", periods=5)
    validate_datetime_index(_frame(idx), name="行情")
    validate_datetime_index(pd.Series(np.arange(5, dtype=float), index=idx), name="序列")


def test_empty_rejected_unless_allowed():
    from src._validation import validate_datetime_index

    empty = _frame(pd.DatetimeIndex([]))
    with pytest.raises(ValueError, match="不能为空"):
        validate_datetime_index(empty)
    validate_datetime_index(empty, allow_empty=True)


def test_non_datetime_index_rejected():
    from src._validation import validate_datetime_index

    with pytest.raises(ValueError, match="DatetimeIndex"):
        validate_datetime_index(pd.DataFrame({"close": [1.0, 2.0]}))


def test_duplicate_and_unsorted_rejected():
    from src._validation import validate_datetime_index

    dup = pd.DatetimeIndex(["2020-01-01", "2020-01-01"])
    with pytest.raises(ValueError, match="唯一"):
        validate_datetime_index(_frame(dup))
    desc = pd.bdate_range("2020-01-01", periods=3)[::-1]
    with pytest.raises(ValueError, match="升序"):
        validate_datetime_index(_frame(desc))


def test_nat_rejected():
    from src._validation import validate_datetime_index

    with pytest.raises(ValueError, match="DatetimeIndex"):
        validate_datetime_index(_frame(pd.DatetimeIndex(["2020-01-01", pd.NaT])))


def test_multiindex_datetime_level_passes():
    from src._validation import validate_datetime_index

    idx = pd.bdate_range("2020-01-01", periods=3)
    mi = pd.MultiIndex.from_product([["000905.SH"], idx], names=["code", "datetime"])
    validate_datetime_index(_frame(mi))


def test_require_positive_int_contract():
    from src._validation import require_positive_int

    require_positive_int(1, "窗口")
    require_positive_int(256, "窗口")
    for bad in (0, -3, True, "5", 2.0):
        with pytest.raises(ValueError, match="必须为正整数"):
            require_positive_int(bad, "窗口")
