"""研报各图独立锚点，不将内部差异抹平。"""

import pandas as pd
import pytest

from src.report_anchors import (
    REPORT_ANCHORS,
    anchor_acceptance,
    anchor_matches,
    compare_anchors,
)


def test_figure_five_and_seven_keep_different_overall_returns() -> None:
    fig5 = compare_anchors({"annual_return": 0.5}, figure=5, threshold=0.002)
    fig7 = compare_anchors({"annual_return": 0.5}, figure=7)
    row5 = fig5.loc[fig5.metric.eq("annual_return")].iloc[0]
    row7 = fig7.loc[fig7.metric.eq("annual_return") & fig7.period.eq("overall")].iloc[0]
    assert row5.target == 0.4321
    assert row7.target == 0.4386
    assert row5.difference == pytest.approx(0.0679)
    assert row7.difference == pytest.approx(0.0614)
    assert row5.source_figure == "图5"
    assert row7.source_figure == "图7"


def test_all_threshold_anchors_and_annual_rows_present() -> None:
    frame = REPORT_ANCHORS
    fig5 = frame.loc[frame.figure.eq(5)]
    assert set(fig5.threshold) == {0.0, 0.002, 0.004}
    assert fig5.loc[fig5.metric.eq("trades"), "target"].tolist() == [378, 216, 126]
    assert frame.loc[frame.figure.eq(7) & frame.metric.eq("annual_return"), "target"].tolist() == [
        0.2603,
        0.0688,
        0.3346,
        0.4409,
        0.8441,
        0.8476,
        0.4386,
    ]
    assert frame.loc[frame.figure.eq(9), "target"].item() == 0.233
    assert frame.loc[frame.figure.eq(10), "target"].item() == 0.1668


def test_missing_actual_stays_missing_and_annual_index_matches() -> None:
    actual = pd.DataFrame({"annual_return": [0.3, 0.4]}, index=[2020, "overall"])
    comparison = compare_anchors(actual, figure=7)
    row = comparison.loc[comparison.period.eq(2020) & comparison.metric.eq("annual_return")].iloc[0]
    assert row.actual == 0.3
    assert row.difference == pytest.approx(0.0397)
    assert comparison.loc[comparison.metric.eq("volatility"), "actual"].isna().all()


def test_threshold_must_be_explicit_for_figure_five() -> None:
    with pytest.raises(ValueError):
        compare_anchors({"annual_return": 0.4}, figure=5)
    with pytest.raises(ValueError):
        compare_anchors({"annual_return": 0.4}, figure=5, threshold=0.1)
    with pytest.raises(ValueError):
        compare_anchors({"annual_return": 0.4}, figure=8)


def _perfect(convention: str = "report_formula") -> pd.DataFrame:
    """中证500每个锚点actual=target的完整单口径对照表。"""
    anchors = REPORT_ANCHORS.loc[REPORT_ANCHORS.code.eq("000905.SH")].reset_index(drop=True)
    return anchors.assign(actual=anchors.target, return_convention=convention)


def _is(table: pd.DataFrame, figure: int, period: object = "overall") -> pd.Series:
    return (
        table.figure.eq(figure)
        & table.threshold.eq(0.002)
        & table.period.astype(str).eq(str(period))
        & table.metric.eq("annual_return")
    )


def test_conflict_group_marks_only_the_inconsistent_overall_returns() -> None:
    # 图5(k=0.2%)的43.21%与图7汇总的43.86%在代码里是同一个数，不可能同时吻合
    grouped = REPORT_ANCHORS.loc[
        REPORT_ANCHORS.conflict_group.notna() & REPORT_ANCHORS.code.eq("000905.SH")
    ]
    assert len(grouped) == 2
    assert grouped.conflict_group.nunique() == 1
    assert sorted(grouped.target) == [0.4321, 0.4386]
    assert set(grouped.figure) == {5, 7}
    assert "conflict_group" in compare_anchors({"annual_return": 0.4}, figure=9)


def test_acceptance_needs_only_one_match_inside_conflict_group() -> None:
    table = _perfect()
    assert anchor_acceptance(table)
    # 真实run里两行actual是同一个数：吻合图7就必然不吻合图5
    table.loc[_is(table, 5), "actual"] = 0.4386
    assert anchor_matches(table).sum() == len(table) - 1
    assert anchor_acceptance(table)
    table.loc[_is(table, 5) | _is(table, 7), "actual"] = 0.40  # 两项均不吻合
    assert not anchor_acceptance(table)


def test_acceptance_fails_on_non_conflict_mismatch_missing_rows_and_empty() -> None:
    table = _perfect()
    table.loc[_is(table, 5), "actual"] = 0.4386
    assert anchor_acceptance(table)
    broken = table.copy()
    broken.loc[_is(broken, 9), "actual"] = 0.2331
    assert not anchor_acceptance(broken)
    missing_value = table.copy()
    missing_value.loc[_is(missing_value, 10), "actual"] = float("nan")
    assert not anchor_acceptance(missing_value)
    assert not anchor_acceptance(table.loc[~_is(table, 7, 2021)])  # 缺非冲突行
    assert not anchor_acceptance(table.loc[~_is(table, 7)])  # 冲突组漏行：剩下一项吻合也不算
    assert not anchor_acceptance(table.iloc[0:0])  # 空表的.all()为True，必须拦住


def test_acceptance_rejects_duplicate_keys_and_mixed_conventions() -> None:
    table = _perfect()
    with pytest.raises(ValueError, match="重复"):
        anchor_acceptance(pd.concat([table, table.iloc[[0]]], ignore_index=True))
    mixed = pd.concat([table, _perfect("vectorbt_native_account")], ignore_index=True)
    with pytest.raises(ValueError, match="口径"):
        anchor_acceptance(mixed)


def test_matches_use_exact_trade_count_and_half_unit_tolerance() -> None:
    table = pd.DataFrame(
        {
            "metric": ["annual_return", "annual_return", "trades", "trades", "volatility"],
            "actual": [0.43864, 0.43866, 126, 125, None],
            "target": [0.4386, 0.4386, 126, 126, 0.2121],
        }
    )
    assert anchor_matches(table).tolist() == [True, False, True, False, False]


def test_non_numeric_extras_do_not_leak_into_actual() -> None:
    # 研报口径绩效字典带回撤起止时间戳；actual必须保持浮点，否则落库成字符串
    actual = {"annual_return": 0.4, "drawdown_start": pd.Timestamp("2022-03-02"), "note": "x"}
    comparison = compare_anchors(actual, figure=10)
    assert comparison.actual.dtype == float
    assert comparison.actual.item() == 0.4


# ---- 多指数锚点（0.10.0）：图号决定指数，单次验收只针对一个指数 ----


def test_csi1000_anchors_mirror_csi500_layout() -> None:
    from src.report_anchors import ANCHOR_FIGURES

    assert ANCHOR_FIGURES["000905.SH"] == {
        "thresholds": 5, "long_short": 7, "long_only": 9, "short_only": 10,
    }
    assert ANCHOR_FIGURES["000852.SH"] == {
        "thresholds": 13, "long_short": 15, "long_only": 17, "short_only": 18,
    }
    assert REPORT_ANCHORS.groupby("code").size().to_dict() == {"000852.SH": 35, "000905.SH": 35}
    csi1000 = REPORT_ANCHORS.loc[REPORT_ANCHORS.code.eq("000852.SH")]
    fig13 = csi1000.loc[csi1000.figure.eq(13)]
    assert fig13.loc[fig13.metric.eq("annual_return"), "target"].tolist() == [0.4618, 0.5047, 0.4378]
    assert fig13.loc[fig13.metric.eq("trades"), "target"].tolist() == [394, 246, 166]
    fig15 = csi1000.loc[csi1000.figure.eq(15)]
    assert fig15.loc[fig15.metric.eq("annual_return"), "target"].tolist() == [
        0.1881, 0.0396, 0.9363, 0.8346, 0.7131, 0.5825, 0.5121,
    ]
    assert fig15.loc[fig15.metric.eq("max_drawdown") & fig15.period.astype(str).eq("overall"), "target"].item() == -0.3003
    assert csi1000.loc[csi1000.figure.eq(17), "target"].item() == 0.26
    assert csi1000.loc[csi1000.figure.eq(18), "target"].item() == 0.2001
    # 图13(k=0.2%)的50.47%与图15汇总的51.21%同样互相矛盾
    grouped = csi1000.loc[csi1000.conflict_group.notna()]
    assert sorted(grouped.target) == [0.5047, 0.5121]
    assert grouped.conflict_group.unique().tolist() == ["csi1000_long_short_annual_return"]
    # 各指数的冲突组互不相干
    assert REPORT_ANCHORS.conflict_group.nunique() == 2


def test_threshold_figures_require_explicit_threshold_for_every_index() -> None:
    with pytest.raises(ValueError):
        compare_anchors({"annual_return": 0.4}, figure=13)
    row = compare_anchors({"annual_return": 0.5047}, figure=13, threshold=0.002)
    assert row.loc[row.metric.eq("annual_return"), "difference"].item() == pytest.approx(0.0)
    assert row.code.unique().tolist() == ["000852.SH"]


def test_acceptance_is_scoped_to_the_index_present_in_the_table() -> None:
    def perfect(code: str) -> pd.DataFrame:
        table = REPORT_ANCHORS.loc[REPORT_ANCHORS.code.eq(code)]
        return table.assign(actual=table.target, return_convention="report_formula")

    # 只跑了中证1000的run，不应因为缺中证500的35项而被判缺行
    assert anchor_acceptance(perfect("000852.SH"))
    assert anchor_acceptance(perfect("000905.SH"))
    csi1000 = perfect("000852.SH")
    assert not anchor_acceptance(csi1000.loc[~csi1000.figure.eq(18)])  # 缺本指数的行仍判失败
    with pytest.raises(ValueError, match="指数"):
        anchor_acceptance(pd.concat([perfect("000852.SH"), perfect("000905.SH")], ignore_index=True))
