"""算法与回测层共享的输入校验工具。"""

from __future__ import annotations

import pandas as pd


def validate_datetime_index(
    data: pd.Series | pd.DataFrame,
    *,
    name: str = "数据",
    allow_empty: bool = False,
) -> None:
    """校验数据索引为唯一、非空且升序的 DatetimeIndex。

    :param data: 待校验索引的 Series 或 DataFrame；MultiIndex 取 datetime 层级。
    :param name: 错误信息中的对象名。
    :param allow_empty: 是否允许空数据，默认不允许。
    :raises ValueError: 索引非 DatetimeIndex、含空、重复、乱序，或（不允许时）为空。
    """
    idx = (
        data.index.get_level_values("datetime")
        if isinstance(data.index, pd.MultiIndex)
        else data.index
    )
    if (
        not isinstance(idx, pd.DatetimeIndex)
        or idx.hasnans
        or not idx.is_unique
        or not idx.is_monotonic_increasing
    ):
        raise ValueError(f"{name}日期必须为唯一、非空且升序的DatetimeIndex")
    if not allow_empty and len(data) == 0:
        raise ValueError(f"{name}不能为空")


def require_positive_int(value: int, name: str) -> None:
    """校验参数为正整数，排除 bool 等非严格整数。

    :param value: 待校验值。
    :param name: 错误信息中的参数名。
    :raises ValueError: 非正整数。
    """
    if type(value) is not int or value < 1:
        raise ValueError(f"{name}必须为正整数")
