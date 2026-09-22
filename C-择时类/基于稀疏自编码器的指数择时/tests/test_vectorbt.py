"""原生vectorbt账户计算，不能使用倒数价格或手工拼接收益。"""

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt


def test_native_short_uses_original_close_and_fixed_units():
    from src.analyze import backtest

    dates = pd.bdate_range("2020", periods=4)
    close = pd.Series([110.0, 110.0, 100.0, 90.0], index=dates)
    pf = backtest(close, pd.Series(-1, index=dates))
    assert isinstance(pf, vbt.Portfolio)
    pd.testing.assert_series_equal(pf.close, close)
    np.testing.assert_allclose(pf.value() / 1_000_000, [1.0, 1.0, 12 / 11, 13 / 11])
    np.testing.assert_allclose(pf.returns(), [0.0, 0.0, 1 / 11, 1 / 12], atol=1e-12)
    orders = pf.orders.records_readable
    assert len(orders) == 1
    assert orders.iloc[0]["Timestamp"] == dates[1]
    assert orders.iloc[0]["Price"] == 110.0
    assert orders.iloc[0]["Fees"] == 0.0
    assert orders.iloc[0]["Side"] == "Sell"


def test_reverse_at_next_close_and_frame_comes_from_portfolio():
    from src.analyze import backtest, backtest_frame

    dates = pd.bdate_range("2020", periods=4)
    close = pd.Series([100.0, 110.0, 121.0, 100.0], index=dates)
    signal = pd.Series([1, -1, 0, 1], index=dates)
    pf = backtest(close, signal)
    frame = backtest_frame(pf, signal)
    np.testing.assert_allclose(frame.nav, [1.0, 1.0, 1.1, 1.1 * (1 + 21 / 121)])
    np.testing.assert_allclose(frame.nav, pf.value() / pf.init_cash)
    np.testing.assert_allclose(frame.strategy_return, pf.returns())
    np.testing.assert_allclose(frame.benchmark_return, pf.benchmark_returns())
    assert frame.execution_position.tolist() == [0, 1, -1, 0]
    assert frame.position.tolist() == [0, 0, 1, -1]
    orders = pf.orders.records_readable
    assert orders.Timestamp.tolist() == dates[1:].tolist()
    np.testing.assert_allclose(orders.Price, [110.0, 121.0, 100.0])
    assert orders.Fees.eq(0).all()


@pytest.mark.parametrize("mode", ["long_short", "long_only", "short_only"])
def test_future_prices_cannot_change_past_portfolio(mode):
    from src.analyze import backtest

    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020", periods=100)
    close = pd.Series(100 * np.exp(rng.normal(0, 0.01, 100).cumsum()), index=dates)
    signal = pd.Series(rng.choice([-1, 0, 1], 100), index=dates)
    original = backtest(close, signal, mode=mode)
    altered = close.copy()
    altered.iloc[60:] *= 1.5
    later = backtest(altered, signal, mode=mode)
    pd.testing.assert_series_equal(original.value().iloc[:60], later.value().iloc[:60])
    if mode == "long_only":
        assert original.assets().ge(0).all()
    if mode == "short_only":
        assert original.assets().le(0).all()


def test_native_metrics_and_year_boundary():
    from src.analyze import annual_performance, backtest, performance

    dates = pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05"])
    pf = backtest(
        pd.Series([100.0, 110.0, 121.0, 110.0], index=dates),
        pd.Series([1, -1, -1, -1], index=dates),
    )
    result = performance(pf)
    acc = pf.returns_acc(year_freq="252D")
    assert result["annual_return"] == pytest.approx(acc.annualized())
    assert result["volatility"] == pytest.approx(acc.annualized_volatility(ddof=1))
    assert result["sharpe"] == pytest.approx(acc.sharpe_ratio(risk_free=0.0, ddof=1))
    assert result["max_drawdown"] == pytest.approx(acc.max_drawdown())
    table = annual_performance(pf)
    assert table.loc[2020, "trades"] == 1
    assert table.loc[2021, "trades"] == 1
    assert table.loc["overall", "trades"] == 2
    assert table.loc[2021, "annual_return"] == pytest.approx(
        pf.returns().loc["2021"].vbt.returns(freq="1D", year_freq="252D").annualized()
    )


def test_all_cash_and_invalid_input():
    from src.analyze import backtest

    dates = pd.bdate_range("2020", periods=3)
    close = pd.Series([100.0, 110.0, 90.0], index=dates)
    pf = backtest(close, pd.Series(0, index=dates))
    assert pf.orders.count() == 0
    np.testing.assert_array_equal(pf.value(), 1_000_000.0)
    with pytest.raises(ValueError):
        backtest(close, pd.Series([1, 2, 0], index=dates))
    with pytest.raises(ValueError):
        backtest(close.iloc[:2], pd.Series(0, index=dates))
    with pytest.raises(ValueError):
        backtest(close * 0, pd.Series(0, index=dates))
    with pytest.raises(ValueError):
        backtest(close, pd.Series(0, index=dates), mode="bad")


def test_threshold_performance_matches_single_threshold_backtest():
    from src.analyze import (
        backtest,
        performance,
        threshold_performance,
    )
    from src.factor_algo import threshold_signal

    dates = pd.bdate_range("2020", periods=30)
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(rng.normal(0, 0.01, 30).cumsum()), index=dates)
    score = pd.Series(rng.normal(0, 0.01, 30), index=dates)
    table = threshold_performance(close, score, thresholds=(0.0, 0.004))
    assert list(table["threshold"]) == [0.0, 0.004]
    single = performance(backtest(close, threshold_signal(score, 0.0)), 252)
    for metric in ("annual_return", "trades", "win_rate", "max_drawdown"):
        assert table.loc[0, metric] == pytest.approx(single[metric])
    with pytest.raises(ValueError):
        threshold_performance(close, score, thresholds=(-0.001,))
