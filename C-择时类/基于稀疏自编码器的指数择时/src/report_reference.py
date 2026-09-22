"""研报倒数空头口径的纯回测与绩效计算，不包含交易费用。"""

import numpy as np
import pandas as pd

from ._validation import require_positive_int, validate_datetime_index


def _validate_positions(positions: pd.Series) -> None:
    if not positions.isin([-1, 0, 1]).all():
        raise ValueError("信号或持仓只能为 -1、0、1，且不能缺失")


def backtest(close: pd.Series, signal: pd.Series, *, mode: str = "long_short") -> pd.DataFrame:
    """按实现收益日期生成两交易日滞后回测。

    T 收盘信号于 T+1 收盘成交，T+2 确认首笔收益。空头采用价格
    倒数收益，为研报研究口径。首日无前价，收益为 NaN，净值为 1。
    输入必须覆盖同一完整交易日历，本函数无法识别双方共同遗漏的交易日。

    :param close: 严格升序、正值且无缺失的日收盘价。
    :param signal: 与价格索引完全一致的 -1/0/1 信号。
    :param mode: long_short、long_only 或 short_only。
    :return: signal、收益持仓position、成交持仓execution_position、两类收益及净值。
    :raises ValueError: 输入索引、价格、信号或交易模式不合法。
    """
    validate_datetime_index(close, name="close")
    validate_datetime_index(signal, name="signal")
    if not close.index.equals(signal.index):
        raise ValueError("价格与信号必须严格对齐到同一完整交易日索引")
    if not np.isfinite(close.to_numpy(dtype=float)).all() or close.le(0).any():
        raise ValueError("价格必须是有限正数，禁止前填缺价")
    _validate_positions(signal)
    if mode not in {"long_short", "long_only", "short_only"}:
        raise ValueError("mode 必须为 long_short、long_only 或 short_only")
    execution_position = signal.shift(1, fill_value=0).astype(int)
    position = signal.shift(2, fill_value=0).astype(int)
    if mode == "long_only":
        position = position.clip(lower=0)
        execution_position = execution_position.clip(lower=0)
    elif mode == "short_only":
        position = position.clip(upper=0)
        execution_position = execution_position.clip(upper=0)
    # 不使用 pct_change 默认填充；空头必须使用倒数，不能简单取负。
    benchmark_return = close / close.shift(1) - 1
    short_return = close.shift(1) / close - 1
    strategy_return = pd.Series(
        np.select([position.eq(1), position.eq(-1)], [benchmark_return, short_return], default=0.0),
        index=close.index,
    )
    strategy_return.iloc[0] = np.nan
    return pd.DataFrame(
        {
            "signal": signal,
            "position": position,
            "execution_position": execution_position,
            "benchmark_return": benchmark_return,
            "strategy_return": strategy_return,
            "nav": (1 + strategy_return.fillna(0)).cumprod(),
            "benchmark_nav": (1 + benchmark_return.fillna(0)).cumprod(),
        }
    )


def performance(
    returns: pd.Series,
    positions: pd.Series,
    annual_days: int = 252,
) -> dict[str, float | int | pd.Timestamp | None]:
    """计算几何年化、风险与交易次数，并纳入初始资金的回撤。

    波动率采用样本标准差（ddof=1），Sharpe=几何年化/年化波动，
    无风险收益为零；Calmar=几何年化/最大回撤绝对值。胜率分母包含
    所有有效收益日（包括现金日），只允许首日无前价的一个 NaN。
    交易次数按从现金开仓或多空翻转计一次，平仓不计；单独传入某年
    切片时视为此前现金，跨年统计请使用 annual_performance。
    最大回撤从初始资金 1 开始；初始资金对应样本前时点，起点以 NaT
    表示，避免把首笔亏损日期误报为峰值。无回撤时起止均为 NaT。
    波动率或回撤为零时，相应比率为 NaN。

    :param returns: 严格升序的日收益率，所有非空值须有限且大于 -1。
    :param positions: 与收益严格对齐的 -1/0/1 实际持仓。
    :param annual_days: 年化交易日数，正整数。
    :return: 收益、风险、胜率、交易次数及最大回撤起止日期字典。
    :raises ValueError: 收益、持仓、索引或年化日数无效。
    """
    validate_datetime_index(returns, name="returns")
    validate_datetime_index(positions, name="positions")
    if not returns.index.equals(positions.index):
        raise ValueError("收益与持仓必须严格对齐")
    _validate_positions(positions)
    require_positive_int(annual_days, "annual_days")
    valid = returns.iloc[1:] if pd.isna(returns.iloc[0]) else returns
    if valid.empty or not np.isfinite(valid.to_numpy(dtype=float)).all() or valid.le(-1).any():
        raise ValueError("有效收益须非空、有限且大于 -1，仅允许首日缺失")
    annual_return = float(np.expm1(np.log1p(valid).mean() * annual_days))
    volatility = float(valid.std(ddof=1) * np.sqrt(annual_days))
    # 显式补初始资金，防止第一笔亏损被累计净值的第一个峰值吞掉。
    wealth = np.r_[1.0, (1 + valid).cumprod().to_numpy()]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    trough = int(np.argmin(drawdown))
    max_drawdown = float(drawdown[trough])
    start, end = pd.NaT, pd.NaT
    if max_drawdown < 0:
        peak = int(np.flatnonzero(wealth[: trough + 1] == wealth[: trough + 1].max())[-1])
        start = valid.index[peak - 1] if peak > 0 else pd.NaT
        end = valid.index[trough - 1]
    openings = positions.ne(0) & positions.ne(positions.shift(1, fill_value=0))
    return {
        "annual_return": annual_return,
        "arithmetic_return": float(valid.mean() * annual_days),
        "volatility": volatility,
        "sharpe": annual_return / volatility if volatility > 0 else float("nan"),
        "max_drawdown": max_drawdown,
        "calmar": annual_return / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
        "win_rate": float(valid.gt(0).mean()),
        "trades": int(openings.sum()),
        "drawdown_start": start,
        "drawdown_end": end,
    }


def annual_performance(bt: pd.DataFrame, annual_days: int = 252) -> pd.DataFrame:
    """按收益实现年份和全期间汇总绩效，持仓变化在全年序列计算。

    各年从资金 1 计算年度回撤，但保留跨年持仓和首个交易日收益。
    年度交易次数按开立/翻转发生日归属年份，延续仓位不重复开仓。

    :param bt: backtest 输出，至少包含 strategy_return 与 position。
    :param annual_days: 年化交易日数。
    :return: 年份整数及 overall 为索引的绩效表。
    :raises ValueError: 回测数据或年化参数不符合 performance 契约。
    """
    positions = bt["execution_position"] if "execution_position" in bt else bt["position"]
    overall = performance(bt["strategy_return"], positions, annual_days)
    openings = positions.ne(0) & positions.ne(positions.shift(1, fill_value=0))
    result = {}
    # 按年只有少量分组；逐日收益和持仓运算仍保持向量化。
    for year, group in bt.groupby(bt.index.year):
        if group["strategy_return"].notna().any():
            row = performance(group["strategy_return"], group["position"], annual_days)
            row["trades"] = int(openings.loc[group.index].sum())
            result[int(year)] = row
    result["overall"] = overall
    table = pd.DataFrame.from_dict(result, orient="index")
    table.index.name = "period"
    return table
