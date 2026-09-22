"""研报§2.2/2.3与§4的纯算法约束。"""

import numpy as np
import pandas as pd
import pytest
import talib
from pandas.testing import assert_frame_equal


def quotes(n=700):
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "open": close * 0.995,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "preclose": np.r_[100, close[:-1]],
            "turn": rng.uniform(0.3, 2, n),
            "volume": rng.uniform(100, 200, n),
            "amount": rng.uniform(10000, 20000, n),
        },
        index=pd.bdate_range("2017-01-02", periods=n, name="datetime"),
    )


def test_features_92_macd_and_no_future():
    from src.factor_algo import build_features

    q = quotes()
    original = q.copy(deep=True)
    x = build_features(q)
    assert x.shape == (len(q), 92)
    assert x.columns.is_unique
    np.testing.assert_allclose(x.macd, talib.MACD(q.close.to_numpy(dtype=float))[2], equal_nan=True)
    altered = q.copy()
    altered.iloc[550:] *= 3
    assert_frame_equal(x.iloc[:550], build_features(altered).iloc[:550])
    assert_frame_equal(q, original)


def test_report_rsi_obv_use_turnover():
    from src.factor_algo import build_features

    q = quotes()
    x = build_features(q, rsi_form="ratio")
    delta = q.close.pct_change(fill_method=None).iloc[6:11]
    assert x.rsi_5.iloc[10] == pytest.approx(
        delta.clip(lower=0).sum() / (-delta.clip(upper=0).sum())
    )
    assert x.obv_5.iloc[10] == pytest.approx((np.sign(delta) * q.turn.iloc[6:11]).sum())


def test_wavelet_prefix_invariance_and_constant():
    from src.factor_algo import causal_wavelet

    x = quotes()[["close", "turn"]]
    a = causal_wavelet(x)
    assert_frame_equal(a.iloc[:500], causal_wavelet(x.iloc[:500]))
    assert a.iloc[:255].isna().all().all()
    assert a.iloc[255:].notna().all().all()
    constant = x * 0 + 5
    np.testing.assert_allclose(causal_wavelet(constant).iloc[255:], 5, atol=1e-10)


def test_hysteresis_strict_boundary_and_initial_cash():
    from src.factor_algo import threshold_signal

    s = pd.Series(
        [0, 0.002, 0.003, 0, -0.002, -0.003, 0, 0.0021], index=pd.bdate_range("2020", periods=8)
    )
    np.testing.assert_array_equal(threshold_signal(s, 0.002), [0, 0, 1, 1, 1, -1, -1, 1])
    with pytest.raises(ValueError):
        threshold_signal(s, -0.001)
    with pytest.raises(ValueError):
        threshold_signal(s * float("nan"))


def test_forward_label_maturity():
    from src.factor_algo import forward_label

    c = pd.Series([100.0, 110, 121, 133.1], index=pd.bdate_range("2020", periods=4))
    y, end = forward_label(c, 2, skip=0)
    assert y.iloc[0] == pytest.approx(0.21)
    assert end.iloc[0] == c.index[2]
    assert y.iloc[-2:].isna().all()
    assert end.iloc[-2:].isna().all()


def test_forward_label_skip_aligns_with_t_plus_1_fill():
    from src.factor_algo import forward_label

    c = pd.Series(
        [100.0, 110, 121, 133.1, 146.41, 161.051, 177.1561, 194.87171],
        index=pd.bdate_range("2020", periods=8),
    )
    y, end = forward_label(c, horizon=5, skip=1)
    # 标签 = C[t+6]/C[t+1]-1，与T+1收盘成交后持有5日的收益一致
    assert y.iloc[0] == pytest.approx(c.iloc[6] / c.iloc[1] - 1)
    assert end.iloc[0] == c.index[6]
    # 末尾 skip+horizon 行没有完整标签
    assert y.iloc[-6:].isna().all() and end.iloc[-6:].isna().all()
    assert y.iloc[:2].notna().all()


def test_forward_label_skip_ignores_same_day_close():
    from src.factor_algo import forward_label

    c = pd.Series(np.linspace(100, 120, 12), index=pd.bdate_range("2020", periods=12))
    base, _ = forward_label(c, 3, skip=1)
    bumped = c.copy()
    bumped.iloc[2] *= 1.5  # 篡改t=2当日收盘
    other, _ = forward_label(bumped, 3, skip=1)
    # skip=1下第t行标签不含C[t]；只有以C[2]为起点(t=1)或终点的行受影响
    assert other.iloc[2] == pytest.approx(base.iloc[2])


def test_forward_label_rejects_negative_skip():
    from src.factor_algo import forward_label

    c = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2020", periods=3))
    with pytest.raises(ValueError):
        forward_label(c, 1, skip=-1)


def test_rsi_uses_percentage_changes_not_price_deltas():
    from src.factor_algo import build_features

    q = quotes(40)
    q.loc[q.index[:6], "close"] = [100, 200, 100, 200, 100, 200]
    assert build_features(q, rsi_form="ratio").rsi_5.iloc[5] == pytest.approx(3.0)


def test_bounded_rsi_has_no_nan_when_all_up_and_preserves_rank():
    from src.factor_algo import build_features

    q = quotes(400)
    # 构造连续8日全涨，使rsi_5分母为0
    q.iloc[300:308, q.columns.get_loc("close")] = q.close.iloc[299] * 1.01 ** np.arange(1, 9)
    q["preclose"] = q.close.shift(1).fillna(q.close.iloc[0])
    ratio = build_features(q, rsi_form="ratio")["rsi_5"].iloc[250:]
    bounded = build_features(q, rsi_form="bounded")["rsi_5"].iloc[250:]
    assert ratio.isna().any()                     # 研报比值在全涨时无定义
    assert bounded.notna().all()
    assert bounded.between(0, 1).all()
    assert bounded[ratio.isna()].eq(1.0).all()    # 全涨=1
    both = ratio.notna()
    # 有界形式是比值的单调变换 r/(1+r)
    assert np.allclose(bounded[both], ratio[both] / (1 + ratio[both]))


def test_bounded_rsi_flat_window_is_half():
    from src.factor_algo import build_features

    q = quotes(400)
    for col in ("open", "high", "low", "close", "preclose"):
        q.iloc[300:312, q.columns.get_loc(col)] = 100.0
    assert build_features(q, rsi_form="bounded")["rsi_5"].iloc[311] == pytest.approx(0.5)


def test_bounded_rsi_flat_window_logs_once_with_counts():
    """窗口内收盘全平取0.5属兜底，须留日志；正常行情不得产生该日志。"""
    from loguru import logger

    from src.factor_algo import build_features

    records: list[str] = []
    handler_id = logger.add(lambda msg: records.append(str(msg)), level="WARNING")
    try:
        build_features(quotes(400), rsi_form="bounded")
        assert not [r for r in records if "0.5" in r]
        q = quotes(400)
        for col in ("open", "high", "low", "close", "preclose"):
            q.iloc[300:312, q.columns.get_loc(col)] = 100.0
        build_features(q, rsi_form="bounded")
    finally:
        logger.remove(handler_id)
    hits = [r for r in records if "0.5" in r]
    # 多个窗口同时命中也只汇总成一条，并点名窗口
    assert len(hits) == 1 and "rsi_5" in hits[0]


def test_rsi_form_rejects_unknown():
    from src.factor_algo import build_features

    with pytest.raises(ValueError):
        build_features(quotes(300), rsi_form="talib")


def test_custom_windows_and_ma_pairs():
    """自定义窗口决定列数，比值对按所给窗口计算。"""
    from src.factor_algo import build_features

    q = quotes(100)
    x = build_features(q, windows=(3, 7), ma_pairs=((3, 7),))
    assert x.shape == (len(q), 8 + 13 * 2 + 1)
    assert "close_ma_250" not in x
    np.testing.assert_allclose(
        x.ma_3_7, q.close.rolling(3).mean() / q.close.rolling(7).mean(), equal_nan=True
    )


@pytest.mark.parametrize(
    ("windows", "ma_pairs"),
    [
        ((), ()),  # 空窗口
        ((3, 3), ()),  # 重复窗口
        ((3, 0), ()),  # 非正整数
        ((3, 7), ((3, 250),)),  # 比值对引用windows以外的窗口
        ((3, 7), ((3, 7), (3, 7))),  # 重复比值对会静默覆盖同名列
    ],
)
def test_invalid_windows_or_pairs_rejected(windows, ma_pairs):
    from src.factor_algo import build_features

    with pytest.raises(ValueError):
        build_features(quotes(100), windows=windows, ma_pairs=ma_pairs)


def test_reverse_time_is_rejected():
    from src.factor_algo import (
        causal_wavelet,
        forward_label,
        threshold_signal,
    )

    x = quotes().iloc[::-1]
    with pytest.raises(ValueError):
        forward_label(x.close)
    with pytest.raises(ValueError):
        causal_wavelet(x[["close"]])
    with pytest.raises(ValueError):
        threshold_signal(x.close)


def _endpoint_removed_share(**kwargs) -> float:
    """白噪声经因果小波后，被去掉的能量占原方差的比例。"""
    from src.factor_algo import causal_wavelet

    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        rng.standard_normal((1500, 3)), index=pd.bdate_range("2010-01-01", periods=1500)
    )
    y = causal_wavelet(x, **kwargs).dropna()
    xa = x.loc[y.index]
    return float(((xa - y) ** 2).to_numpy().mean() / xa.to_numpy().var())


def test_wavelet_default_actually_denoises_endpoint():
    """D1约占白噪声一半能量；只取末值时去噪比例须接近窗口中段水平。"""
    assert _endpoint_removed_share() > 0.4
    assert _endpoint_removed_share(mode="reflect") > 0.4
    # 钉住0.7.0前的缺陷：symmetric延拓下末点D1结构性趋零，消融基线依赖这一口径
    assert _endpoint_removed_share(mode="symmetric") < 0.1


def test_wavelet_endpoint_improves_persistent_signal():
    """持续性特征（滚动均值类）末值须比原始值更接近干净信号；periodization会在此失败。"""
    from src.factor_algo import causal_wavelet

    t = np.arange(2300)
    clean = 0.004 * t / 10 + 1.5 * np.sin(2 * np.pi * t / 400) + 0.6 * np.sin(2 * np.pi * t / 60)
    clean = (clean - clean.mean()) / clean.std()
    noisy = clean + 0.3 * np.random.default_rng(1).standard_normal(len(t))
    idx = pd.bdate_range("2010-01-01", periods=len(t))
    y = causal_wavelet(pd.DataFrame({"x": noisy}, index=idx))["x"].dropna()
    truth = pd.Series(clean, index=idx).loc[y.index]
    raw = pd.Series(noisy, index=idx).loc[y.index]

    def rmse(a: pd.Series) -> float:
        return float(np.sqrt(((a - truth) ** 2).mean()))

    assert rmse(y) < 0.9 * rmse(raw)


def test_wavelet_rejects_unknown_mode():
    from src.factor_algo import causal_wavelet

    with pytest.raises(ValueError, match="边界模式"):
        causal_wavelet(quotes()[["close"]], mode="bogus")
