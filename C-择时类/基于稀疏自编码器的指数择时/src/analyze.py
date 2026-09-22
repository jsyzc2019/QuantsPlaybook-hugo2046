"""vectorbt原生账户回测：原始close成交/估值，零手续费和滑点。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._validation import require_positive_int, validate_datetime_index

if TYPE_CHECKING:
    from vectorbt import Portfolio

INITIAL_CASH: float = 1_000_000.0


def backtest(close: pd.Series, signal: pd.Series, *, mode: str = "long_short") -> Portfolio:
    """把算法信号延后一日，交给vectorbt生成真实持仓和盈亏。

    相同信号持续持仓，反向信号平仓并反向开仓；开仓使用框架可用
    资金最大仓位，不逐日重设名义敞口。T信号于T+1 close成交，
    T+2开始产生价格盈亏。空头由数量与账户权益估值，不用倒数收益。

    :param close: 完整交易日历上的单指数原始收盘价，有限正数。
    :param signal: 同索引-1/0/1信号；不含缺失、日期唯一升序。
    :param mode: long_short、long_only或short_only。
    :returns: vectorbt.Portfolio，可直接调用value/returns/orders/trades/stats。
    :raises ValueError: 日期、价格、信号或模式不合法。
    """
    # vectorbt 1.0 + 当前Numba在只读site-packages中无法写包内缓存；
    # 默认把缓存放进当前worktree，调用方仍可显式覆盖该环境变量。
    cache_dir = Path(os.environ.get("NUMBA_CACHE_DIR", Path.cwd() / ".numba_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
    import vectorbt as vbt

    validate_datetime_index(close, name="close")
    if not close.index.equals(signal.index) or not signal.isin([-1, 0, 1]).all():
        raise ValueError("信号须与行情严格对齐且仅包含-1/0/1")
    if not np.isfinite(close.to_numpy(dtype=float)).all() or close.le(0).any():
        raise ValueError("close必须为有限正数，禁止填补缺失行情")
    if mode not in {"long_short", "long_only", "short_only"}:
        raise ValueError("未知回测mode")
    state = signal.shift(1, fill_value=0)
    if mode == "long_only":
        state = state.clip(lower=0)
    elif mode == "short_only":
        state = state.clip(upper=0)
    return vbt.Portfolio.from_signals(
        close=close,
        price=close,
        entries=state.eq(1),
        exits=state.ne(1),
        short_entries=state.eq(-1),
        short_exits=state.ne(-1),
        upon_opposite_entry="reverse",
        accumulate=False,
        size=np.inf,
        size_type="amount",
        init_cash=INITIAL_CASH,
        fees=0.0,
        fixed_fees=0.0,
        slippage=0.0,
        reject_prob=0.0,
        raise_reject=True,
        freq="1D",
    )


def backtest_frame(portfolio: Portfolio, signal: pd.Series) -> pd.DataFrame:
    """从框架结果提取兼容绘图和CSV导出的逐日表，不重算收益。

    :param portfolio: backtest产生的单指数Portfolio。
    :param signal: 原始算法信号，与组合时间索引相同。
    :returns: 信号、实际成交/收益持仓方向、框架收益及归一化净值。
    :raises ValueError: 信号日期未对齐。
    """
    if not portfolio.wrapper.index.equals(signal.index):
        raise ValueError("信号与Portfolio日期未对齐")
    execution = np.sign(portfolio.assets()).astype("int8")
    return pd.DataFrame(
        {
            "signal": signal,
            "position": execution.shift(1, fill_value=0),
            "execution_position": execution,
            "benchmark_return": portfolio.benchmark_returns(),
            "strategy_return": portfolio.returns(),
            "nav": portfolio.value() / portfolio.init_cash,
            "benchmark_nav": portfolio.benchmark_value() / portfolio.init_cash,
        }
    )


def _return_metrics(returns: pd.Series, annual_days: int) -> dict[str, float]:
    require_positive_int(annual_days, "annual_days")
    # 1D是交易日步长；year_freq=252D显式覆盖框架默认365D。
    acc = returns.vbt.returns(freq="1D", year_freq=f"{annual_days}D")
    return {
        "annual_return": float(acc.annualized()),
        "total_return": float(acc.total()),
        "volatility": float(acc.annualized_volatility(ddof=1)),
        "sharpe": float(acc.sharpe_ratio(risk_free=0.0, ddof=1)),
        "max_drawdown": float(acc.max_drawdown()),
        "calmar": float(acc.calmar_ratio()),
        # 研报图5是日度胜率；另输出框架平仓交易胜率，避免混淆。
        "win_rate": float(returns.gt(0).vbt.signals.rate()),
    }


def performance(portfolio: Portfolio, annual_days: int = 252) -> dict[str, float | int]:
    """使用vectorbt收益访问器及交易记录生成汇总绩效。

    首日0收益保留；Sharpe使用框架算术超额均值/标准差，risk_free=0。
    trades为框架entry_trades数（含未平仓），不是买卖订单总数。

    :param portfolio: 单指数Portfolio。
    :param annual_days: 年化交易日数，默认252。
    :returns: 框架收益/风险、日胜率、交易胜率、开仓数及订单数。
    """
    return {
        **_return_metrics(portfolio.returns(), annual_days),
        "trades": int(portfolio.entry_trades.count()),
        "order_count": int(portfolio.orders.count()),
        "trade_win_rate": float(portfolio.trades.closed.win_rate()),
    }


def annual_performance(portfolio: Portfolio, annual_days: int = 252) -> pd.DataFrame:
    """在完整账户结果上按年统计，保留跨年持仓，不重新回测年度切片。

    行索引为收益实现年份（整数）外加 ``overall``。各年从资金 1 起算年度回撤，
    但持仓不在年界重置；年度交易按实际开仓日归年，跨年持仓不重复计开仓。

    :param portfolio: :func:`backtest` 产出的单指数 Portfolio。
    :param annual_days: 年化交易日数，正整数，默认 252。
    :returns: 以 ``period``（年份整数或 ``"overall"``）为索引的绩效表，列：

        - ``annual_return`` / ``total_return`` / ``volatility`` / ``sharpe`` /
          ``max_drawdown`` / ``calmar``：与 :func:`performance` 同定义。
        - ``win_rate``：**日度**胜率（收益大于 0 的交易日占比）。
        - ``trades``：该年开仓交易数（含未平仓）；``order_count``：订单总数。
        - ``trade_win_rate``：**已平仓交易**胜率，仅 ``overall`` 行有值，分年行为 NaN。
    :raises ValueError: ``annual_days`` 非正整数（经 :func:`require_positive_int`）。
    """
    returns = portfolio.returns()
    entries = portfolio.entry_trades.records_readable["Entry Timestamp"]
    orders = portfolio.orders.records_readable["Timestamp"]
    result = {}
    for year, group in returns.groupby(returns.index.year):
        result[int(year)] = {
            **_return_metrics(group, annual_days),
            "trades": int(entries.dt.year.eq(year).sum()),
            "order_count": int(orders.dt.year.eq(year).sum()),
        }
    result["overall"] = performance(portfolio, annual_days)
    return pd.DataFrame.from_dict(result, orient="index").rename_axis("period")


def threshold_performance(
    close: pd.Series,
    score: pd.Series,
    *,
    thresholds: tuple[float, ...] = (0.0, 0.002, 0.004),
    annual_days: int = 252,
    mode: str = "long_short",
) -> pd.DataFrame:
    """逐阈值生成迟滞信号并回测，返回每阈值一行的原生账户绩效表。

    对标 :func:`annual_performance` 的「一张表」形态：行是阈值，列是
    :func:`performance` 的指标，供绘图与落库复用。信号由
    ``factor_algo.threshold_signal`` 生成，回测与本模块 ``backtest``/``performance``
    同口径（vectorbt 原生账户），不是研报倒数口径。

    :param close: 同一完整交易日历上的单指数原始收盘价；严格升序、日期唯一、
        无缺失、有限正数（与 :func:`backtest` 要求一致）。
    :param score: 与 ``close`` 严格对齐（同一索引）的日度预测值；无缺失、全部有限。
    :param thresholds: 待评估阈值序列（收益小数，非负有限）；输出行序与它一致。
    :param annual_days: 年化交易日数，正整数，默认 252。
    :param mode: ``long_short``、``long_only`` 或 ``short_only``。
    :returns: 每阈值一行、默认 RangeIndex 的 DataFrame，列：

        - ``threshold``：该行的迟滞信号阈值（收益小数）。
        - ``annual_return`` / ``total_return`` / ``volatility`` / ``sharpe`` /
          ``max_drawdown`` / ``calmar``：与 :func:`performance` 同定义。
        - ``win_rate``：**日度**胜率（收益大于 0 的交易日占比，含空仓日）；
          研报图 5 的「胜率」即此列。
        - ``trade_win_rate``：**已平仓交易**的盈利比例，口径与 ``win_rate`` 不同。
        - ``trades``：框架 ``entry_trades`` 数（含未平仓）；``order_count``：订单总数。
    :raises ValueError: 阈值非负/有限、``score`` 有缺失或非有限、``close`` 与
        ``score`` 索引不一致、价格非有限正数，或 ``mode`` 非法。
    """
    from .factor_algo import threshold_signal

    rows = []
    for threshold in thresholds:
        signal = threshold_signal(score, threshold)
        rows.append(
            {
                "threshold": threshold,
                **performance(backtest(close, signal, mode=mode), annual_days),
            }
        )
    return pd.DataFrame(rows)
