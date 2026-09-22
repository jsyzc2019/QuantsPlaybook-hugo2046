"""研报回测口径与绩效边界测试。"""

import numpy as np
import pandas as pd
import pytest

from src.report_reference import (
    annual_performance,
    backtest,
    performance,
)


def test_two_day_delay_and_reciprocal_short_return() -> None:
    dates = pd.bdate_range("2020-01-01", periods=4)
    close = pd.Series([100.0, 110.0, 121.0, 100.0], index=dates)
    signal = pd.Series([1, -1, 0, 1], index=dates)
    bt = backtest(close, signal)
    assert bt.position.tolist() == [0, 0, 1, -1]
    assert pd.isna(bt.strategy_return.iloc[0])
    assert pd.isna(bt.benchmark_return.iloc[0])
    np.testing.assert_allclose(bt.strategy_return.iloc[1:], [0.0, 0.1, 0.21])
    np.testing.assert_allclose(bt.nav, [1.0, 1.0, 1.1, 1.331])
    np.testing.assert_allclose(bt.benchmark_nav, [1.0, 1.1, 1.21, 1.0])
    assert performance(bt.strategy_return, bt.position)["win_rate"] == 2 / 3


@pytest.mark.parametrize(
    "mode,expected", [("long_only", [0.0, 0.1, 0.0]), ("short_only", [0.0, 0.0, 0.21])]
)
def test_direction_filters(mode: str, expected: list[float]) -> None:
    dates = pd.bdate_range("2020-01-01", periods=4)
    bt = backtest(
        pd.Series([100.0, 110.0, 121.0, 100.0], index=dates),
        pd.Series([1, -1, 0, 1], index=dates),
        mode=mode,
    )
    np.testing.assert_allclose(bt.strategy_return.iloc[1:], expected)


@pytest.mark.parametrize(
    "prices",
    [[100.0, np.nan, 110.0], [100.0, 0.0, 110.0], [100.0, -1.0, 110.0], [100.0, np.inf, 110.0]],
)
def test_bad_prices_rejected(prices: list[float]) -> None:
    dates = pd.bdate_range("2020-01-01", periods=3)
    with pytest.raises(ValueError):
        backtest(pd.Series(prices, index=dates), pd.Series(1, index=dates))


@pytest.mark.parametrize(
    "index",
    [pd.to_datetime(["2020-01-01", "2020-01-01"]), pd.to_datetime(["2020-01-02", "2020-01-01"])],
)
def test_invalid_chronology_rejected(index: pd.DatetimeIndex) -> None:
    with pytest.raises(ValueError):
        backtest(pd.Series([100.0, 110.0], index=index), pd.Series(1, index=index))


def test_signal_must_match_exact_dates_and_be_valid() -> None:
    dates = pd.bdate_range("2020-01-01", periods=3)
    close = pd.Series([100.0, 101.0, 102.0], index=dates)
    with pytest.raises(ValueError):
        backtest(close, pd.Series(1, index=dates[1:]))
    with pytest.raises(ValueError):
        backtest(close, pd.Series([1, np.nan, 1], index=dates))
    with pytest.raises(ValueError):
        backtest(close, pd.Series([1, 2, 1], index=dates))
    with pytest.raises(ValueError):
        backtest(close, pd.Series(1, index=dates), mode="invalid")


def test_first_loss_drawdown_includes_initial_capital() -> None:
    dates = pd.bdate_range("2020-01-01", periods=3)
    stats = performance(
        pd.Series([-0.2, 0.0, 0.1], index=dates), pd.Series([1, 0, 1], index=dates), annual_days=3
    )
    assert stats["max_drawdown"] == pytest.approx(-0.2)
    assert pd.isna(stats["drawdown_start"])
    assert stats["drawdown_end"] == dates[0]
    assert stats["annual_return"] == pytest.approx(-0.12)
    assert stats["arithmetic_return"] == pytest.approx(-0.1)
    assert stats["volatility"] == pytest.approx(np.std([-0.2, 0.0, 0.1], ddof=1) * np.sqrt(3))
    assert stats["sharpe"] == pytest.approx(stats["annual_return"] / stats["volatility"])
    assert stats["calmar"] == pytest.approx(-0.6)
    assert stats["win_rate"] == pytest.approx(1 / 3)
    assert stats["trades"] == 2  # 平仓不计数，重新开仓计一次。


def test_known_peak_date_and_position_flips() -> None:
    dates = pd.bdate_range("2020-01-01", periods=4)
    stats = performance(
        pd.Series([0.1, -0.2, 0.0, 0.1], index=dates), pd.Series([1, -1, 0, -1], index=dates)
    )
    assert stats["drawdown_start"] == dates[0]
    assert stats["drawdown_end"] == dates[1]
    assert stats["trades"] == 3


def test_annual_performance_carries_positions_and_counts_actual_changes() -> None:
    dates = pd.to_datetime(["2020-12-29", "2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05"])
    bt = backtest(
        pd.Series([100.0, 100.0, 110.0, 121.0, 121.0], index=dates), pd.Series(1, index=dates)
    )
    result = annual_performance(bt, annual_days=2)
    assert list(result.index) == [2020, 2021, "overall"]
    assert result.loc[2020, "trades"] == 1
    assert result.loc[2021, "trades"] == 0
    assert result.loc["overall", "trades"] == 1
    assert result.loc[2021, "annual_return"] == pytest.approx(0.1)
    assert result.loc["overall", "annual_return"] == pytest.approx(0.1)


def test_performance_rejects_internal_missing_and_bad_alignment() -> None:
    dates = pd.bdate_range("2020-01-01", periods=3)
    with pytest.raises(ValueError):
        performance(pd.Series([0.0, np.nan, 0.1], index=dates), pd.Series(1, index=dates))
    with pytest.raises(ValueError):
        performance(pd.Series([0.0, 0.1, 0.1], index=dates), pd.Series(1, index=dates[::-1]))
    with pytest.raises(ValueError):
        performance(
            pd.Series([0.0, 0.1, 0.1], index=dates), pd.Series(1, index=dates), annual_days=0
        )


def test_trades_count_execution_day_not_return_day():
    import pandas as pd

    from src.report_reference import annual_performance, backtest

    dates = pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05"])
    bt = backtest(
        pd.Series([100.0, 110, 120, 130], index=dates), pd.Series([1, 1, -1, -1], index=dates)
    )
    assert bt.execution_position.tolist() == [0, 1, 1, -1]
    assert bt.position.tolist() == [0, 0, 1, 1]
    table = annual_performance(bt)
    assert table.loc[2020, "trades"] == 1
    assert table.loc[2021, "trades"] == 1
    assert table.loc["overall", "trades"] == 2
