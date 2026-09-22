"""纯绘图函数的结构及数据契约测试。"""

from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from src import plotting


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """保证测试不遗留图形。"""
    yield
    plt.close("all")


def test_signals_preserve_close_and_align_positions() -> None:
    """信号按行情日期对齐，曲线与两种背景均正确生成。"""
    dates = pd.date_range("2025-01-01", periods=4)
    close = pd.Series([100.0, 103.0, 102.0, 104.0], index=dates)
    signal = pd.Series([-1, 1, 0, 1], index=dates[::-1])
    close_before, signal_before = close.copy(), signal.copy()
    figure = plotting.plot_signals(close, signal)
    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 1
        axis = figure.axes[0]
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), close.to_numpy())
        assert axis.lines[0].get_color() == "black"
        assert len(axis.collections) == 2
        red, green = (item.get_facecolor()[0] for item in axis.collections)
        assert red[0] > red[1]
        assert green[1] > green[0]
        figure.canvas.draw()
        pd.testing.assert_series_equal(close, close_before)
        pd.testing.assert_series_equal(signal, signal_before)
    finally:
        plt.close(figure)


def test_nav_draws_three_strategies_and_one_benchmark() -> None:
    """净值曲线直接使用传入结果，且不修改回测表。"""
    dates = pd.date_range("2025-01-01", periods=3)
    backtests = {
        mode: pd.DataFrame(
            {"nav": [1.0, 1.1, value], "benchmark_nav": [1.0, 1.02, 1.03]},
            index=dates,
        )
        for mode, value in zip(("long_short", "long_only", "short_only"), (1.2, 1.15, 1.05))
    }
    before = {mode: frame.copy(deep=True) for mode, frame in backtests.items()}
    figure = plotting.plot_nav(backtests)
    try:
        assert len(figure.axes) == 1
        assert len(figure.axes[0].lines) == 4
        for line, mode in zip(figure.axes[0].lines[:3], backtests):
            np.testing.assert_allclose(line.get_ydata(), backtests[mode]["nav"])
        np.testing.assert_allclose(figure.axes[0].lines[3].get_ydata(), [1.0, 1.02, 1.03])
        figure.canvas.draw()
        for mode, frame in backtests.items():
            pd.testing.assert_frame_equal(frame, before[mode])
    finally:
        plt.close(figure)


def test_losses_show_latest_year_all_seeds_sorted_by_epoch() -> None:
    """四类损失保留种子区分，旧年份不能混入曲线。"""
    losses = pd.DataFrame(
        {
            "year": [2024, 2025, 2025, 2025, 2025],
            "seed": [0, 0, 0, 1, 1],
            "epoch": [1, 2, 1, 2, 1],
            "total": [99.0, 0.5, 1.0, 0.6, 1.1],
            "prediction": [99.0, 0.2, 0.4, 0.3, 0.5],
            "reconstruction": [99.0, 0.2, 0.4, 0.2, 0.4],
            "sparsity": [99.0, 0.1, 0.2, 0.1, 0.2],
        }
    )
    before = losses.copy(deep=True)
    figure = plotting.plot_losses(losses)
    try:
        assert len(figure.axes) == 4
        for axis in figure.axes:
            assert len(axis.lines) == 2
            np.testing.assert_array_equal(axis.lines[0].get_xdata(), [1, 2])
            assert max(axis.lines[0].get_ydata()) < 99
        np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [1.0, 0.5])
        assert "2025" in figure._suptitle.get_text()
        figure.canvas.draw()
        pd.testing.assert_frame_equal(losses, before)
    finally:
        plt.close(figure)


def test_folds_expand_training_and_leave_purge_gap() -> None:
    """训练条起点固定、终点逐年推进；训练条与测试条之间留出标签清洗空档。"""
    folds = pd.DataFrame(
        {
            "year": [2021, 2020],
            "train_start": pd.to_datetime(["2015-01-05", "2015-01-05"]),
            "train_end": pd.to_datetime(["2020-12-10", "2019-12-11"]),
            "test_start": pd.to_datetime(["2021-01-04", "2020-01-02"]),
            "test_end": pd.to_datetime(["2021-12-31", "2020-12-31"]),
        }
    )
    before = folds.copy(deep=True)
    figure = plotting.plot_folds(folds)
    try:
        axis = figure.axes[0]
        train, test = axis.containers
        assert [label.get_text() for label in axis.get_yticklabels()] == ["2020", "2021"]
        # 行序按年排序后，2020 在第 0 行；每行训练终点须早于测试起点（清洗空档）。
        for bar_train, bar_test in zip(train, test):
            assert bar_train.get_x() + bar_train.get_width() < bar_test.get_x()
        assert train[0].get_width() < train[1].get_width()
        assert train[0].get_x() == train[1].get_x()
        figure.canvas.draw()
        pd.testing.assert_frame_equal(folds, before)
    finally:
        plt.close(figure)


def test_correlation_uses_only_seed_columns() -> None:
    """热图排除集成预测与实际收益，仅计算种子之间的相关性。"""
    predictions = pd.DataFrame(
        {
            "seed_0": [1.0, 2.0, 3.0],
            "seed_1": [3.0, 2.0, 1.0],
            "prediction": [2.0, 2.0, 2.0],
            "target": [0.1, 0.2, 0.3],
        }
    )
    before = predictions.copy(deep=True)
    figure = plotting.plot_seed_correlation(predictions)
    try:
        assert len(figure.axes) == 2  # 主图和色标。
        axis = figure.axes[0]
        np.testing.assert_allclose(
            np.ravel(axis.collections[0].get_array()), [1, -1, -1, 1]
        )
        assert [text.get_text() for text in axis.texts] == ["1.00", "-1.00", "-1.00", "1.00"]
        assert [label.get_text() for label in axis.get_xticklabels()] == ["seed_0", "seed_1"]
        figure.canvas.draw()
        pd.testing.assert_frame_equal(predictions, before)
    finally:
        plt.close(figure)


def test_nav_drawdown_overlays_drawdown_on_right_axis() -> None:
    """左轴画净值、右轴填充回撤，且不修改传入的回测表。"""
    dates = pd.date_range("2025-01-01", periods=4)
    frame = pd.DataFrame(
        {"nav": [1.0, 1.2, 0.9, 1.1], "signal": [0, 1, 1, 1]}, index=dates
    )
    before = frame.copy(deep=True)
    figure = plotting.plot_nav_drawdown(frame, title="只做多策略净值与回撤")
    try:
        assert len(figure.axes) == 2
        left, right = figure.axes
        np.testing.assert_allclose(left.lines[0].get_ydata(), frame["nav"])
        assert left.get_title() == "只做多策略净值与回撤"
        expected = frame["nav"] / frame["nav"].cummax() - 1.0
        vertices = np.concatenate(
            [path.vertices[:, 1] for path in right.collections[0].get_paths()]
        )
        assert vertices.max() == pytest.approx(0.0)
        assert vertices.min() == pytest.approx(expected.min())
        assert isinstance(right.yaxis.get_major_formatter(), PercentFormatter)
        figure.canvas.draw()
        pd.testing.assert_frame_equal(frame, before)
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"signal": [1, 0, 1]}, index=pd.date_range("2025-01-01", periods=3)),
        pd.DataFrame({"nav": []}, index=pd.DatetimeIndex([])),
        pd.DataFrame(
            {"nav": [1.0, np.nan, 1.1]}, index=pd.date_range("2025-01-01", periods=3)
        ),
        pd.DataFrame(
            {"nav": [1.0, 0.0, 1.1]}, index=pd.date_range("2025-01-01", periods=3)
        ),
        pd.DataFrame(
            {"nav": [1.0, -0.2, 1.1]}, index=pd.date_range("2025-01-01", periods=3)
        ),
    ],
)
def test_nav_drawdown_rejects_invalid_nav(frame: pd.DataFrame) -> None:
    """缺列、空表、非有限或非正净值都抛 ValueError。"""
    with pytest.raises(ValueError):
        plotting.plot_nav_drawdown(frame, title="非法输入")


def test_nav_drawdown_zero_drawdown_keeps_flat_axis() -> None:
    """净值单调不亏时回撤恒为 0，右轴退化为固定 (-0.01, 0) 而非零高度。"""
    dates = pd.date_range("2025-01-01", periods=3)
    frame = pd.DataFrame({"nav": [1.0, 1.1, 1.2]}, index=dates)
    figure = plotting.plot_nav_drawdown(frame, title="无回撤")
    try:
        assert figure.axes[1].get_ylim() == pytest.approx((-0.01, 0.0))
    finally:
        plt.close(figure)


def test_nav_drawdown_benchmark_line_and_validation() -> None:
    """传 benchmark 时左轴多一条深灰虚线的基准净值；基准无效抛 ValueError。"""
    dates = pd.date_range("2025-01-01", periods=4)
    frame = pd.DataFrame(
        {"nav": [1.0, 1.2, 0.9, 1.1], "signal": [0, 1, 1, 1]}, index=dates
    )
    close = pd.Series([100.0, 110.0, 105.0, 120.0], index=dates)
    figure = plotting.plot_nav_drawdown(frame, title="x", benchmark=close)
    try:
        assert len(figure.axes) == 2
        left = figure.axes[0]
        assert len(left.lines) == 2
        benchmark_line = left.lines[1]
        np.testing.assert_allclose(benchmark_line.get_ydata(), [1.0, 1.1, 1.05, 1.2])
        assert benchmark_line.get_color() == "darkgray"
        assert benchmark_line.get_linestyle() == "--"
    finally:
        plt.close(figure)
    with pytest.raises(ValueError):
        plotting.plot_nav_drawdown(frame, title="x", benchmark=close.iloc[:2])
    with pytest.raises(ValueError):
        plotting.plot_nav_drawdown(frame, title="x", benchmark=close * 0)


def test_threshold_sensitivity_draws_four_facets() -> None:
    """四个指标各一个分面，x 为阈值百分比，且不修改输入表。"""
    frame = pd.DataFrame(
        {
            "threshold": [0.0, 0.002, 0.004],
            "annual_return": [0.4042, 0.4321, 0.3549],
            "trades": [378, 216, 126],
            "win_rate": [0.5471, 0.5326, 0.5320],
            "max_drawdown": [-0.1631, -0.14, -0.3049],
        }
    )
    before = frame.copy(deep=True)
    figure = plotting.plot_threshold_sensitivity(frame, title="阈值敏感性")
    try:
        assert len(figure.axes) == 4
        assert len(figure.axes[0].lines) == 1
        np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(), [0.0, 0.2, 0.4])
        np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [40.42, 43.21, 35.49])
        figure.canvas.draw()
        pd.testing.assert_frame_equal(frame, before)
    finally:
        plt.close(figure)
