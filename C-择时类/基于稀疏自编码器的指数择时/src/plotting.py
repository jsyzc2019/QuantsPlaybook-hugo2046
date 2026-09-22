"""SAE 实验的纯绘图函数；图形保存由调用方负责。"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


def _font_style() -> dict[str, object]:
    """从已注册字体中选择可用的中文字体，不修改全局配置。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = (
        "PingFang SC",
        "Arial Unicode MS",
        "Heiti SC",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
    )
    family = next((name for name in candidates if name in available), "DejaVu Sans")
    return {"font.family": family, "axes.unicode_minus": False, "font.size": 10}


def plot_signals(close: pd.Series, signal: pd.Series) -> Figure:
    """绘制指数收盘价及多空信号背景（研报图 6）。

    :param close: 以日期为索引的指数收盘价。
    :param signal: 日期索引的方向信号，正值做多、负值做空、零值空仓。
    :returns: 收盘价黑线、多仓红色背景、空头绿色背景的图形。
    """
    positions = signal.reindex(close.index).fillna(0)
    with plt.rc_context(_font_style()):
        figure, axis = plt.subplots(figsize=(12, 4), layout="constrained")
        # 背景使用坐标轴高度，避免价格极值影响仓位显示；零仓位留白。
        for mask, color, label in (
            (positions.gt(0), "#c94444", "多仓信号"),
            (positions.lt(0), "#3a9463", "空头信号"),
        ):
            axis.fill_between(
                close.index,
                0,
                mask.astype(float),
                step="post",
                transform=axis.get_xaxis_transform(),
                color=color,
                alpha=0.18,
                linewidth=0,
                label=label,
            )
        axis.plot(close.index, close.to_numpy(), color="black", linewidth=1, label="指数收盘价")
        axis.set(title="指数价格与择时信号", xlabel="日期", ylabel="收盘价")
        axis.margins(x=0)
        axis.legend(loc="upper left", ncols=3, frameon=False)
        return figure


def plot_folds(folds: pd.DataFrame) -> Figure:
    """绘制年度扩展训练的训练/清洗/测试区间（研报图 2，sklearn TimeSeriesSplit 风格）。

    每个测试年一行：训练集起点固定、终点逐年推进；训练终点到测试起点之间
    是标签清洗留出的空档（horizon 天标签必须在测试年前实现）。

    :param folds: ``ResearchResult.folds``，含 year、train_start、train_end、
        test_start、test_end 列。
    :returns: 蓝色训练条、红色测试条、行序自上而下按年递增的甘特图。
    """
    frame = folds.sort_values("year")
    rows = np.arange(len(frame))
    with plt.rc_context(_font_style()):
        figure, axis = plt.subplots(figsize=(12, 0.45 * len(frame) + 1.6), layout="constrained")
        for start, end, color, label in (
            ("train_start", "train_end", "#3b5fc4", "训练集"),
            ("test_start", "test_end", "#e0654f", "测试集"),
        ):
            left = pd.to_datetime(frame[start])
            width = pd.to_datetime(frame[end]) - left
            axis.barh(rows, width, left=left, height=0.6, color=color, label=label)
        axis.set_yticks(rows, labels=frame["year"].astype(str))
        axis.invert_yaxis()
        axis.set(title="年度扩展训练切分", xlabel="日期", ylabel="测试年份")
        axis.grid(axis="x", alpha=0.18)
        # 图例放在坐标区左上方之外，避免遮住最后一行的训练条。
        axis.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, frameon=False)
        return figure


def plot_nav(backtests: dict[str, pd.DataFrame]) -> Figure:
    """绘制多空、多头、空头策略及指数净值（研报图 8）。

    :param backtests: long_short、long_only、short_only 三个回测表，
        每表以日期为索引，包含 nav 和 benchmark_nav 列。
    :returns: 三种策略与指数基准的净值对比图形。
    """
    with plt.rc_context(_font_style()):
        figure, axis = plt.subplots(figsize=(12, 4.5), layout="constrained")
        for key, label, color in (
            ("long_short", "多空策略", "#c94444"),
            ("long_only", "仅做多策略", "#386a9c"),
            ("short_only", "仅做空策略", "#3a9463"),
        ):
            frame = backtests[key]
            axis.plot(frame.index, frame["nav"], label=label, color=color, linewidth=1.4)
        benchmark = backtests["long_short"]
        axis.plot(
            benchmark.index,
            benchmark["benchmark_nav"],
            label="指数基准",
            color="black",
            linewidth=1.1,
        )
        axis.set(title="策略与指数净值", xlabel="日期", ylabel="净值")
        axis.grid(axis="y", alpha=0.18)
        axis.legend(loc="upper left", ncols=4, frameon=False)
        return figure


def plot_nav_drawdown(
    frame: pd.DataFrame, *, title: str, benchmark: pd.Series | None = None
) -> Figure:
    """绘制单策略累积净值与最大回撤（研报图 11/12 风格）。

    :param frame: 单张回测表，来自 ``analyze.backtest_frame(portfolio, signal)``
        （``run_reproduction`` 里即 ``strategies[mode]``）。以交易日为索引，
        必须含 ``nav`` 列（归一化净值，完整回测表首日恰为 1.0，回撤以此为峰）；
        其余列不参与绘图。
    :param title: 图标题，如「只做多策略净值与回撤」。
    :param benchmark: 可选的指数收盘价序列（如回测所用 ``close``）。reindex 到
        ``frame.index`` 后按首值归一，画成深灰虚线的「指数基准(左轴)」，便于看
        策略相对基准的表现；为 ``None`` 时不画，行为与旧版一致。
    :returns: 左轴红色净值线、右轴灰色回撤填充的双轴图形；给 benchmark 时左轴另加基准线。
    :raises ValueError: ``frame`` 为空、缺 ``nav`` 列、``nav`` 非有限正数，或
        ``benchmark`` reindex 后含缺失、非正或非有限值。
    """
    if "nav" not in frame.columns:
        raise ValueError("frame 缺少 nav 列")
    nav = frame["nav"].astype(float)
    if nav.empty or not np.isfinite(nav.to_numpy()).all() or nav.le(0).any():
        raise ValueError("nav 须为非空、有限正数")
    benchmark_nav = None
    if benchmark is not None:
        aligned = benchmark.reindex(frame.index).astype(float)
        if aligned.isna().any() or not np.isfinite(aligned.to_numpy()).all() or aligned.le(0).any():
            raise ValueError("benchmark 须覆盖 frame 全部日期且为有限正数")
        benchmark_nav = aligned / aligned.iloc[0]
    drawdown = nav / nav.cummax() - 1.0
    with plt.rc_context(_font_style()):
        figure, axis = plt.subplots(figsize=(12, 4.5), layout="constrained")
        axis.plot(frame.index, nav, color="#c94444", linewidth=1.4, label="累积净值(左轴)")
        if benchmark_nav is not None:
            axis.plot(
                frame.index,
                benchmark_nav,
                color="darkgray",
                linestyle="--",
                linewidth=1.2,
                label="指数基准(左轴)",
            )
        axis.set(title=title, xlabel="日期", ylabel="净值")
        axis.grid(axis="y", alpha=0.10)
        drawdown_axis = axis.twinx()
        # 回撤填充在右轴，默认盖住左轴的净值/基准线；把左轴抬到上层并透明化其背景，
        # 让线始终可见，灰色回撤退为背景。
        axis.set_zorder(drawdown_axis.get_zorder() + 1)
        axis.patch.set_visible(False)
        drawdown_axis.fill_between(
            frame.index,
            0,
            drawdown.to_numpy(),
            color="#bdbdbd",
            alpha=0.4,
            linewidth=0,
            label="最大回撤(右轴)",
        )
        floor = float(drawdown.min()) * 1.05
        drawdown_axis.set_ylim(floor if floor < 0 else -0.01, 0)
        drawdown_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        drawdown_axis.set_ylabel("回撤")
        handles, labels = axis.get_legend_handles_labels()
        dd_handles, dd_labels = drawdown_axis.get_legend_handles_labels()
        axis.legend(
            handles + dd_handles,
            labels + dd_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncols=2,
            frameon=False,
        )
        return figure


def _validate_threshold_frame(frame: pd.DataFrame) -> None:
    """校验阈值绩效表含绘图必需列且非空。"""
    required = ("threshold", "annual_return", "trades", "win_rate", "max_drawdown")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"frame 缺少列：{missing}")
    if frame.empty:
        raise ValueError("frame 不能为空")


def plot_threshold_sensitivity(frame: pd.DataFrame, *, title: str) -> Figure:
    """绘制阈值 k 对四项绩效的影响（seaborn 2×2 分面，各指标独立 y 轴）。

    :param frame: 阈值绩效表，来自 ``analyze.threshold_performance``（``run_reproduction``
        里即 ``figure_5_thresholds``）。含 ``threshold`` 列（小数，如 0.002）及
        ``annual_return``、``trades``、``win_rate``、``max_drawdown`` 列。
    :param title: 图标题，如「中证500 阈值敏感性」。
    :returns: 年化收益(%)、交易次数(次)、胜率(%)、最大回撤(%) 四个分面折线图。
    :raises ValueError: ``frame`` 为空或缺必需列。
    """
    _validate_threshold_frame(frame)
    x = frame["threshold"].astype(float).to_numpy() * 100.0
    facets = (
        ("年化收益(%)", frame["annual_return"].astype(float).to_numpy() * 100.0),
        ("交易次数(次)", frame["trades"].astype(float).to_numpy()),
        ("胜率(%)", frame["win_rate"].astype(float).to_numpy() * 100.0),
        ("最大回撤(%)", frame["max_drawdown"].astype(float).to_numpy() * 100.0),
    )
    long = pd.DataFrame(
        {
            "阈值 k(%)": np.tile(x, len(facets)),
            "指标": np.repeat([name for name, _ in facets], len(x)),
            "数值": np.concatenate([values for _, values in facets]),
        }
    )
    with plt.rc_context(_font_style()):
        grid = sns.relplot(
            data=long,
            x="阈值 k(%)",
            y="数值",
            col="指标",
            col_wrap=2,
            kind="line",
            marker="o",
            height=3,
            aspect=1.6,
            facet_kws={"sharex": True, "sharey": False},
        )
        grid.set_titles("{col_name}")
        grid.figure.suptitle(title)
        return grid.figure


def plot_losses(losses: pd.DataFrame) -> Figure:
    """绘制最新训练年份的各随机种子分项损失（研报图 3）。

    :param losses: 包含 year、seed、epoch、total、prediction、
        reconstruction、sparsity 列的训练损失表；可含 activation 列，本图不绘制。
    :returns: 总损失、预测损失、重构损失和稀疏惩罚的四面板图形。
    """
    year = losses["year"].max()
    latest = losses.loc[losses["year"].eq(year)].sort_values(["seed", "epoch"])
    with plt.rc_context(_font_style()):
        figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for axis, (column, title) in zip(
            axes.flat,
            (
                ("total", "总损失"),
                ("prediction", "预测损失"),
                ("reconstruction", "重构损失"),
                ("sparsity", "稀疏惩罚"),
            ),
        ):
            # 每个种子单独成线，不能把不同年度或随机初始化连接起来。
            for seed, group in latest.groupby("seed", sort=True):
                axis.plot(group["epoch"], group[column], label=f"种子 {seed}", linewidth=1)
            axis.set(title=title, xlabel="训练轮次", ylabel="损失")
            axis.grid(alpha=0.18)
        axes.flat[0].legend(frameon=False, ncols=2)
        figure.suptitle(f"{year} 年各随机种子训练损失")
        return figure


def plot_seed_correlation(predictions: pd.DataFrame) -> Figure:
    """绘制样本外种子预测的 Pearson 相关系数热图。

    :param predictions: 预测结果表，只有列名以 seed_ 开头的列参与计算。
    :returns: 相关系数范围固定为 [-1, 1] 的热图，格内标注两位小数，附带色标。
    """
    seeds = predictions.loc[:, predictions.columns.str.startswith("seed_")]
    correlation = seeds.corr()
    with plt.rc_context(_font_style()):
        figure, axis = plt.subplots(figsize=(6, 5), layout="constrained")
        sns.heatmap(
            correlation,
            vmin=-1,
            vmax=1,
            cmap="RdBu_r",
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 8},
            square=True,
            cbar_kws={"label": "Pearson 相关系数"},
            ax=axis,
        )
        # seaborn 默认纵轴标签竖排，统一改为横排；横轴斜排避免种子名重叠。
        axis.tick_params(axis="x", labelrotation=45)
        plt.setp(axis.get_xticklabels(), ha="right")
        axis.tick_params(axis="y", labelrotation=0)
        axis.set_title("样本外种子预测相关性")
        return figure
