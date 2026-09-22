"""SAE纯算法：92维候选特征、因果小波、标签与迟滞信号。

完整特征配方为研究假设；MACD/RSI/OBV定义来自研报§2.2。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pywt
import talib
from loguru import logger

from ._validation import require_positive_int, validate_datetime_index

WINDOWS: tuple[int, ...] = (5, 10, 20, 60, 120, 250)
# 均线比值对(短, 长)，输出ma_短/ma_长；两端必须都在所用windows内。
MA_PAIRS: tuple[tuple[int, int], ...] = (
    (5, 10),
    (5, 20),
    (10, 20),
    (20, 60),
    (60, 120),
    (120, 250),
)


def build_features(
    quotes: pd.DataFrame,
    windows: tuple[int, ...] = WINDOWS,
    ma_pairs: tuple[tuple[int, int], ...] = MA_PAIRS,
    rsi_form: str = "bounded",
) -> pd.DataFrame:
    """构造单指数候选特征（默认92维），不填充warm-up与未定义的比值。

    列数为 ``8 + 13 * len(windows) + len(ma_pairs)``。

    :param quotes: datetime升序索引；含open/high/low/close/preclose/turn。
    :param windows: 滚动窗口（交易日），每个窗口派生13列；互不重复的正整数。
    :param ma_pairs: 均线比值对 ``(a, b)``，输出 ``ma_a_b = ma_a / ma_b``；两端须都在windows内。
    :param rsi_form: "bounded"为 ``up/(up+down)``∈[0,1]（默认；研报比值r的单调变换r/(1+r)，
        全涨=1、窗口内收盘全平取0.5）；"ratio"为研报§2.2原比值，分母为0时NaN，仅供对照。
        研报未定义分母为0的情形，有界形式为研究假设，见设计文档§3A。
    :returns: 同索引特征，缺失与无限比值为NaN。
    :raises ValueError: 日期重复或未排序；windows为空、重复或非正整数；
        ma_pairs重复或引用了windows以外的窗口；rsi_form不是bounded/ratio。
    """
    validate_datetime_index(quotes, allow_empty=True)
    if rsi_form not in {"bounded", "ratio"}:
        raise ValueError(f"未知rsi_form：{rsi_form}")
    if not windows or len(set(windows)) != len(windows):
        raise ValueError("windows不能为空或重复")
    for w in windows:
        require_positive_int(w, "windows元素")
    # 比值对引用未计算的均线时，提前报错而不是在循环里抛KeyError。
    missing = {w for pair in ma_pairs for w in pair} - set(windows)
    if missing or len(set(ma_pairs)) != len(ma_pairs):
        raise ValueError(f"ma_pairs不能重复或引用windows以外的窗口：{sorted(missing)}")
    q = quotes.astype(float)
    close, turn = q["close"], q["turn"]
    prev = q["preclose"].where(q["preclose"] > 0)
    base = q[["open", "high", "low", "close"]].div(prev, axis=0)
    base.columns = [f"{name}_prev" for name in base]
    base["open_close"] = q["open"] / close
    base["turn"] = turn
    base["amplitude"] = (q["high"] - q["low"]) / prev
    # TA-Lib第三项即DIF-DEA；不使用国内软件常见的二倍柱体。
    base["macd"] = talib.MACD(close.to_numpy(dtype=float))[2]
    ret = close.pct_change(fill_method=None)
    delta = close.diff()
    # 研报用涨跌幅绝对值，不是TA-Lib RSI，也不是绝对价格变化。
    up, down = ret.clip(lower=0), -ret.clip(upper=0)
    signed_turn = np.sign(delta) * turn
    features = dict(base.items())
    ma = {}
    flat_hits: dict[str, int] = {}
    # 只遍历配置的窗口；时间维度全部向量化。
    for w in windows:
        ma[w] = close.rolling(w).mean()
        roller = turn.rolling(w)
        up_sum, down_sum = up.rolling(w).sum(), down.rolling(w).sum()
        if rsi_form == "ratio":
            rsi = up_sum / down_sum.replace(0, np.nan)
        else:
            # 0/0（窗口内收盘全平）取中性0.5；warm-up的NaN保持NaN
            total = up_sum + down_sum
            flat = total.eq(0)
            rsi = (up_sum / total.replace(0, np.nan)).where(~flat, 0.5)
            if flat.any():
                flat_hits[f"rsi_{w}"] = int(flat.sum())
        derived = {
            "close_ma": close / ma[w],
            "return": close / close.shift(w) - 1,
            "ret_std": ret.rolling(w).std(ddof=1),
            "turn_mean": roller.mean(),
            "turn_std": roller.std(ddof=1),
            "turn_q20": roller.quantile(0.2),
            "turn_q80": roller.quantile(0.8),
            "amplitude_mean": base["amplitude"].rolling(w).mean(),
            "open_prev_mean": base["open_prev"].rolling(w).mean(),
            "high_prev_mean": base["high_prev"].rolling(w).mean(),
            "low_prev_mean": base["low_prev"].rolling(w).mean(),
            "rsi": rsi,
            "obv": signed_turn.rolling(w).sum(),
        }
        features.update({f"{key}_{w}": value for key, value in derived.items()})
    for a, b in ma_pairs:
        features[f"ma_{a}_{b}"] = ma[a] / ma[b]
    if flat_hits:
        # 指数上不应出现；出现多半是停牌填充或行情异常，汇总成一条便于排查
        logger.warning("窗口内收盘全平，有界RSI取中性0.5：各列命中行数{}", flat_hits)
    return pd.DataFrame(features, index=q.index).replace([np.inf, -np.inf], np.nan)


def causal_wavelet(
    features: pd.DataFrame,
    window: int = 256,
    wavelet: str = "db4",
    level: int = 4,
    mode: str = "reflect",
) -> pd.DataFrame:
    """历史窗口四级分解，去掉最高频D1，只保留重构末端。

    :param features: 已标准化、无缺失的时间×特征矩阵。
    :param window: 历史窗口长度，含当前日。
    :param wavelet: 小波基，默认db4为研究假设。
    :param level: 分解层数，研报披露为4。
    :param mode: PyWavelets边界延拓模式，默认reflect。只取末值时，symmetric/smooth
        会让末点的D1在边界处结构性趋零，末值几乎不去噪（白噪声只去掉约4%能量，
        窗口中段约53%）；periodization把末点卷绕到窗口首日，对持续性特征更差。
    :returns: 同形状矩阵，前window-1行NaN；末端不依赖未来。
    :raises ValueError: 非有限输入、未知边界模式或窗口不足以支持分解级数。
    """
    validate_datetime_index(features, allow_empty=True)
    if mode not in pywt.Modes.modes:
        raise ValueError(f"未知小波边界模式：{mode}")
    if window < 2 or level < 1 or pywt.dwt_max_level(window, pywt.Wavelet(wavelet).dec_len) < level:
        raise ValueError("窗口不足以支持指定小波级数")
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("小波输入必须有限；先用训练集统计量处理缺失")
    result = np.full_like(values, np.nan)
    if len(values) >= window:
        windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
        # 分块避免整段历史×92×256的临时数组同时驻留内存。
        for start in range(0, len(windows), 128):
            chunk = windows[start : start + 128]
            coeff = pywt.wavedec(chunk, wavelet, level=level, axis=-1, mode=mode)
            coeff[-1] = np.zeros_like(coeff[-1])
            reconstructed = pywt.waverec(coeff, wavelet, axis=-1, mode=mode)
            result[start + window - 1 : start + window - 1 + len(chunk)] = reconstructed[
                ..., window - 1
            ]
    return pd.DataFrame(result, index=features.index, columns=features.columns)


def forward_label(
    close: pd.Series, horizon: int = 5, skip: int = 1
) -> tuple[pd.Series, pd.Series]:
    """生成与成交时点对齐的未来累计收益及其实现日期，用于清除跨训练边界标签。

    T日收盘出信号、T+1收盘成交，可交易收益从C[t+1]起算，故默认skip=1：
    标签为 ``C[t+skip+horizon] / C[t+skip] - 1``。研报未披露预测周期与标签起点，
    两者均为研究假设。

    :param close: 升序收盘价。
    :param horizon: 持有交易日数，默认5。
    :param skip: 信号日到成交日的交易日数；0复现0.7.0及之前口径。
    :returns: (收益率标签, 标签实现日期)，最后skip+horizon行缺失。
    :raises ValueError: horizon非正整数或skip为负。
    """
    validate_datetime_index(close, allow_empty=True)
    require_positive_int(horizon, "horizon")
    if not isinstance(skip, int) or isinstance(skip, bool) or skip < 0:
        raise ValueError("skip须为非负整数")
    # 标签与实现日期必须同步移位，否则折末训练样本的标签会跨进测试年
    span = skip + horizon
    label = (close.shift(-span) / close.shift(-skip) - 1).rename("label")
    end = pd.Series(close.index, index=close.index, name="label_end").shift(-span)
    return label, end


def threshold_signal(score: pd.Series, threshold: float = 0.002) -> pd.Series:
    """按±阈值生成迟滞信号；等于阈值或死区保留前值。

    :param score: 日度收益率预测，不能包含缺失。
    :param threshold: 非负阈值，单位为收益小数。
    :returns: -1/0/1的signal序列；首次触发前保持空仓0。
    """
    validate_datetime_index(score, allow_empty=True)
    if not np.isfinite(threshold) or threshold < 0 or not np.isfinite(score).all():
        raise ValueError("阈值须非负且预测值必须有限")
    events = pd.Series(
        np.select([score > threshold, score < -threshold], [1.0, -1.0], default=np.nan),
        index=score.index,
    )
    return events.ffill().fillna(0).astype("int8").rename("signal")
