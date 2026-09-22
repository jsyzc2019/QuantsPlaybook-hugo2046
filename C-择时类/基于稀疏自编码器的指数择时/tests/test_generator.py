"""自然年滚动、标签purge、未来扰动与长表出口。"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from .test_factor_algo import quotes


def _small_config(**kwargs):
    """折外小波相关用例的共用小配置：短小波窗口、单seed、1个epoch。"""
    from src.generator import ResearchConfig
    from src.model import ModelConfig

    base = {
        "train_start": "2019-01-01",
        "test_start": "2021-01-01",
        "test_end": "2021-12-31",
        "seeds": (0,),
        "wavelet_window": 128,
        "model": ModelConfig(hidden_dim=8, code_dim=3, epochs=1, batch_size=32),
    }
    base.update(kwargs)
    return ResearchConfig(**base)


def test_prepare_rejects_nan_features_after_warmup():
    """ratio形式的RSI在分母为0时产生NaN，折外小波无法处理，须明确报错。"""
    from dataclasses import replace as _replace

    from src.generator import SAETimingGenerator

    g = SAETimingGenerator(_replace(_small_config(), rsi_form="ratio"))
    q = quotes(1500)
    q.iloc[600:608, q.columns.get_loc("close")] = q.close.iloc[599] * 1.01 ** np.arange(1, 9)
    with pytest.raises(ValueError, match="rsi_form"):
        g._prepare(q)


def test_prepare_missing_non_rsi_feature_omits_rsi_hint(monkeypatch):
    """非rsi列缺失时报错消息不应附带rsi_form提示，成因判断须按实际缺失列而非写死。"""
    from src import generator as generator_module
    from src.generator import SAETimingGenerator

    g = SAETimingGenerator(_small_config())
    q = quotes(1500)
    original_build_features = generator_module.build_features

    def _poison_non_rsi_column(*args, **kwargs):
        features = original_build_features(*args, **kwargs)
        non_rsi = next(c for c in features.columns if not c.startswith("rsi_"))
        features = features.copy()
        features.iloc[-1, features.columns.get_loc(non_rsi)] = np.nan
        return features

    monkeypatch.setattr(generator_module, "build_features", _poison_non_rsi_column)
    with pytest.raises(ValueError) as excinfo:
        g._prepare(q)
    assert "rsi_form" not in str(excinfo.value)


def test_denoised_features_do_not_depend_on_future_rows():
    """折外小波仍严格因果：篡改末尾50行不改变更早任何一行的去噪特征。"""
    from src.generator import SAETimingGenerator

    g = SAETimingGenerator(_small_config(test_end="2022-12-31"))
    q = quotes(1500)
    full, *_ = g._prepare(q)
    bumped = q.copy()
    close = bumped.close.to_numpy().copy()
    close[-50:] *= 1.3
    # close的派生列须同步重算，否则"只有未来行变化"不成立（preclose错位一行）
    bumped["close"] = close
    bumped["open"] = close * 0.995
    bumped["high"] = close * 1.02
    bumped["low"] = close * 0.98
    bumped["preclose"] = np.r_[100, close[:-1]]
    other, *_ = g._prepare(bumped)
    common = full.index[:-60]
    np.testing.assert_allclose(full.loc[common], other.loc[common], rtol=0, atol=1e-12)


def test_folds_labels_are_realized_before_test_year():
    """新路径的折隔离：训练标签实现日严格早于测试首日，且训练标签无缺失。"""
    from src.generator import SAETimingGenerator

    g = SAETimingGenerator(_small_config(test_end="2022-12-31"))
    x, labels, maturity, folds = g._prepare(quotes(1500))
    assert len(folds) == 2
    for train, test in folds:
        assert maturity.iloc[train].max() < x.index[test[0]]
        assert np.isfinite(labels.iloc[train]).all()


def test_midyear_test_start_drops_leading_rows_of_first_test_year():
    """test_start落在年中时，测试年内早于它的行既不训练也不测试。"""
    from src.generator import SAETimingGenerator

    g = SAETimingGenerator(_small_config(test_start="2021-03-02"))
    x, _labels, _maturity, folds = g._prepare(quotes(1500))
    train, test = folds[0]
    assert x.index[test].min() >= pd.Timestamp("2021-03-02")
    assert (x.index[test] < pd.Timestamp("2022-01-01")).all()
    assert x.index[train].max() < pd.Timestamp("2021-01-01")
    # 年前最后6行（标签跨年）被清除
    assert not ((x.index[train] >= pd.Timestamp("2021-01-01")).any())
    # gap=label_skip+horizon=6：2021年前最后6个交易行不在训练集，第7个在
    before = x.index[x.index < pd.Timestamp("2021-01-01")]
    assert set(before[-6:]).isdisjoint(x.index[train])
    assert before[-7] in x.index[train]


def test_prepare_test_years_follow_rows_present_not_the_year_range():
    """样本外区间内某年整年无行时，折数与各折测试年须按实际存在的年份走。

    折数原先由 ``range(test_start.year, test_end.year+1)`` 里"有行的年数"反推，
    再交给划分器取"最后n年"——两者靠隐式一致成立。改为显式列出测试年后，
    本例（缺2021整年）锁定：2折，测试年恰为2020与2022。
    """
    from src.generator import SAETimingGenerator

    g = SAETimingGenerator(_small_config(test_start="2020-01-01", test_end="2022-06-30"))
    q = quotes(1500)
    q = q[q.index.year != 2021]
    x, _labels, _maturity, folds = g._prepare(q)
    assert [x.index[test[0]].year for _train, test in folds] == [2020, 2022]


def test_prepare_rejects_test_rows_lost_to_wavelet_warmup():
    """小波预热吃掉样本外前段时须明确报错，而不是静默少预测几天。"""
    from src.generator import SAETimingGenerator

    g = SAETimingGenerator(
        _small_config(train_start="2018-01-01", test_start="2019-01-01", wavelet_window=700)
    )
    with pytest.raises(ValueError, match="需要更早历史"):
        g._prepare(quotes(1200))


def test_end_to_end_no_future_and_single_day():
    from src.generator import ResearchConfig, SAETimingGenerator
    from src.model import ModelConfig

    q = quotes(1300)
    cfg = ResearchConfig(
        train_start="2017-01-01",
        test_start="2020-01-01",
        test_end="2020-12-31",
        seeds=(0, 1),
        model=ModelConfig(hidden_dim=8, code_dim=3, epochs=2, batch_size=256),
    )
    a = SAETimingGenerator(cfg).run(q)
    changed = q.copy()
    changed.loc["2021":] *= 7
    b = SAETimingGenerator(cfg).run(changed)
    assert_frame_equal(a.predictions, b.predictions)
    assert a.predictions.index.names == ["datetime", "code"]
    assert a.predictions.index.is_unique and a.predictions.index.is_monotonic_increasing
    assert a.predictions.score.notna().all()
    np.testing.assert_allclose(
        a.predictions.score, a.predictions[["seed_0", "seed_1"]].mean(axis=1)
    )
    assert len(a.folds) == 1
    assert a.folds.label_end.max() < pd.Timestamp("2020-01-01")
    assert len(a.losses) == 4
    one = SAETimingGenerator(cfg).generate(q, "2020-06-01", "2020-06-01")
    assert len(one) == 1 and one.notna().all()
    assert one.name == "HY_SAE"


@pytest.mark.parametrize(
    ("train_start", "test_start"),
    [
        ("2021-01-01", "2020-01-01"),  # 训练起点晚于样本外起点
        ("2020-03-01", "2020-06-01"),  # 早于test_start但不早于首个训练截止（当年年初）
    ],
)
def test_config_rejects_train_start_not_before_first_cutoff(train_start, test_start):
    """首个测试年没有可用训练样本的配置在构造时即报错，而非跑完特征后才在年度分段报错。"""
    from src.generator import ResearchConfig

    with pytest.raises(ValueError, match="训练起点"):
        ResearchConfig(train_start=train_start, test_start=test_start, test_end="2021-12-31")


def test_training_start_preserves_prior_wavelet_warmup():
    from src.generator import ResearchConfig, SAETimingGenerator
    from src.model import ModelConfig

    q = quotes(1500)
    cfg = ResearchConfig(
        train_start="2019-01-01",
        test_start="2021-01-01",
        test_end="2021-02-01",
        seeds=(0,),
        model=ModelConfig(epochs=1, batch_size=32),
    )
    result = SAETimingGenerator(cfg).run(q)
    assert result.folds.train_start.iloc[0] == pd.Timestamp("2019-01-01")


def test_predictions_before_midyear_unchanged_by_later_prices():
    from src.generator import ResearchConfig, SAETimingGenerator
    from src.model import ModelConfig

    q = quotes(1300)
    cfg = ResearchConfig(
        train_start="2017-01-01",
        test_start="2020-01-01",
        test_end="2020-12-31",
        seeds=(0,),
        model=ModelConfig(epochs=1, batch_size=32),
    )
    a = SAETimingGenerator(cfg).run(q).predictions
    changed = q.copy()
    changed.loc["2020-07-01":] *= 3
    b = SAETimingGenerator(cfg).run(changed).predictions
    dates = a.index.get_level_values("datetime")
    assert_frame_equal(a.loc[dates < "2020-07-01"], b.loc[dates < "2020-07-01"])


def test_config_feature_windows_pass_through():
    """ResearchConfig的窗口与比值对透传到特征层，模型输入维度随之变化。"""
    from src.generator import ResearchConfig, SAETimingGenerator
    from src.model import ModelConfig

    cfg = ResearchConfig(
        train_start="2017-01-01",
        test_start="2020-01-01",
        test_end="2020-03-31",
        seeds=(0,),
        feature_windows=(5, 20, 60),
        ma_pairs=((5, 20), (20, 60)),
        model=ModelConfig(epochs=1, batch_size=32),
    )
    result = SAETimingGenerator(cfg).run(quotes(1300))
    assert len(result.feature_names) == 8 + 13 * 3 + 2
    assert "ma_20_60" in result.feature_names
    assert "close_ma_250" not in result.feature_names
    assert result.predictions.score.notna().all()


def test_config_wavelet_mode_pass_through(monkeypatch):
    """ResearchConfig.wavelet_mode 透传到小波；默认reflect。"""
    from src import generator as gen
    from src.model import ModelConfig

    seen = []
    real = gen.causal_wavelet

    def spy(*args, **kwargs):
        seen.append(kwargs["mode"])
        return real(*args, **kwargs)

    monkeypatch.setattr(gen, "causal_wavelet", spy)
    cfg = gen.ResearchConfig(
        train_start="2017-01-01",
        test_start="2020-01-01",
        test_end="2020-03-31",
        seeds=(0,),
        wavelet_mode="symmetric",
        model=ModelConfig(epochs=1, batch_size=32),
    )
    gen.SAETimingGenerator(cfg).run(quotes(1300))
    assert seen == ["symmetric"]
    assert gen.ResearchConfig().wavelet_mode == "reflect"


def _selection_config(rho: float):
    from src.generator import ResearchConfig
    from src.model import ModelConfig

    return ResearchConfig(
        train_start="2017-01-01",
        test_start="2020-01-01",
        test_end="2020-12-31",
        seeds=(0, 1),
        model=ModelConfig(epochs=1, batch_size=32, rho=rho),
    )


def test_select_sparsity_weight_uses_first_fold_training_only():
    """预注册选择只看首折训练集：首个训练截止之后的行情改变不影响任何输出。"""
    from src.generator import SAETimingGenerator

    # rho=0.6 → 2ρ=1.2，任何sigmoid均值都达标，故选中最小候选，结果与训练随机性无关
    gen = SAETimingGenerator(_selection_config(rho=0.6))
    q = quotes(1300)
    chosen, detail = gen.select_sparsity_weight(q, candidates=(0.001, 0.1))
    changed = q.copy()
    changed.loc["2020-01-01":] *= 3
    chosen_b, detail_b = gen.select_sparsity_weight(changed, candidates=(0.001, 0.1))
    assert chosen == chosen_b == 0.001
    assert_frame_equal(detail, detail_b)
    assert list(detail.columns) == ["sparsity_weight", "seed", "activation", "prediction", "passed"]
    assert len(detail) == 4 and detail.passed.all()


def test_select_sparsity_weight_refuses_to_relax():
    """无候选达标时报错，不自动放宽判据或扩充候选。"""
    from src.generator import SAETimingGenerator

    gen = SAETimingGenerator(_selection_config(rho=0.01))
    with pytest.raises(ValueError, match="没有候选"):
        gen.select_sparsity_weight(quotes(1300), candidates=(0.001,))


def test_select_sparsity_weight_requires_ascending_candidates():
    from src.generator import SAETimingGenerator

    gen = SAETimingGenerator(_selection_config(rho=0.05))
    with pytest.raises(ValueError, match="升序"):
        gen.select_sparsity_weight(quotes(1300), candidates=(0.1, 0.01))


def test_select_sparsity_weight_chooses_non_minimal_candidate(monkeypatch):
    """替换真实训练为查表替身：只有最大候选达标时须选中它，而非机械选最小候选。"""
    from src import generator as gen_module

    activation_by_weight = {0.001: 0.5, 0.01: 0.3, 0.1: 0.08}

    class _StubRegressor:
        def __init__(self, **params):
            self._activation = activation_by_weight[params["sparsity_weight"]]

        def fit(self, features, label):
            self.history_ = pd.DataFrame(
                {"activation": [self._activation], "prediction": [0.0]}
            )
            return self

    monkeypatch.setattr(gen_module, "SAERegressor", _StubRegressor)
    gen = gen_module.SAETimingGenerator(_selection_config(rho=0.05))
    chosen, detail = gen.select_sparsity_weight(quotes(1300), candidates=(0.001, 0.01, 0.1))
    assert chosen == 0.1
    assert detail.loc[detail["sparsity_weight"] == 0.1, "passed"].all()
    assert not detail.loc[detail["sparsity_weight"] != 0.1, "passed"].any()
