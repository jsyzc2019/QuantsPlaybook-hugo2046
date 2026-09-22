"""华源 SAE 研报图表的固定核对值（中证500、中证1000），保留阈值表与分年表的内部差异。"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

# 单一收益口径内唯一确定一个锚点的组合键。
ANCHOR_KEY: list[str] = ["figure", "threshold", "period", "metric"]

# 各指数在研报里的图号：阈值表、多空分年表、只做多与只做空的汇总年化。图号决定指数。
ANCHOR_FIGURES: dict[str, dict[str, int]] = {
    "000905.SH": {"thresholds": 5, "long_short": 7, "long_only": 9, "short_only": 10},
    "000852.SH": {"thresholds": 13, "long_short": 15, "long_only": 17, "short_only": 18},
}
_PERIODS = [2020, 2021, 2022, 2023, 2024, 2025, "overall"]


def _index_anchors(
    code: str,
    conflict_group: str,
    thresholds: dict[str, list[float]],
    annual: dict[str, list[float]],
    long_only: float,
    short_only: float,
) -> pd.DataFrame:
    """按研报同构的四张表生成一个指数的锚点。

    研报对每个指数都同时给阈值表（k=0.2%行）和多空分年表（汇总行）的年化，两者互相矛盾，
    而复现里取自同一次绩效计算、不可能同时吻合；归为一个冲突组，验收时组内至少一项吻合。
    单边策略只录入已明确核对的汇总年化，不推断未录入的指标。

    :param code: 指数代码，须在ANCHOR_FIGURES内。
    :param conflict_group: 该指数多空汇总年化冲突组的名字。
    :param thresholds: 阈值表各列，按k=0、0.2%、0.4%排列。
    :param annual: 多空分年表各列，按2020–2025、汇总排列。
    :param long_only: 只做多汇总年化。
    :param short_only: 只做空汇总年化。
    :returns: 含code、figure、threshold、period、metric、target、conflict_group的长表。
    """
    figures = ANCHOR_FIGURES[code]
    parts = [
        pd.DataFrame({"threshold": [0.0, 0.002, 0.004], **thresholds})
        .melt(id_vars="threshold", var_name="metric", value_name="target")
        .assign(figure=figures["thresholds"], period="overall"),
        pd.DataFrame({"period": _PERIODS, **annual})
        .melt(id_vars="period", var_name="metric", value_name="target")
        .assign(figure=figures["long_short"], threshold=0.002),
        pd.DataFrame(
            {
                "figure": [figures["long_only"], figures["short_only"]],
                "threshold": 0.002,
                "period": "overall",
                "metric": "annual_return",
                "target": [long_only, short_only],
            }
        ),
    ]
    anchors = pd.concat(parts, ignore_index=True).assign(code=code, conflict_group=None)
    anchors.loc[
        anchors["figure"].isin([figures["thresholds"], figures["long_short"]])
        & anchors["threshold"].eq(0.002)
        & anchors["period"].astype(str).eq("overall")
        & anchors["metric"].eq("annual_return"),
        "conflict_group",
    ] = conflict_group
    return anchors


REPORT_ANCHORS = pd.concat(
    [
        _index_anchors(
            "000905.SH",
            "csi500_long_short_annual_return",  # 图5的43.21% vs 图7的43.86%
            thresholds={
                "annual_return": [0.4042, 0.4321, 0.3549],
                "trades": [378, 216, 126],
                "win_rate": [0.5471, 0.5326, 0.5320],
                "max_drawdown": [-0.1631, -0.14, -0.3049],
            },
            annual={
                "annual_return": [0.2603, 0.0688, 0.3346, 0.4409, 0.8441, 0.8476, 0.4386],
                "volatility": [0.2523, 0.1524, 0.2167, 0.1276, 0.2830, 0.1994, 0.2121],
                "max_drawdown": [-0.1361, -0.14, -0.1084, -0.0446, -0.1355, -0.08, -0.14],
            },
            long_only=0.233,
            short_only=0.1668,
        ),
        _index_anchors(
            "000852.SH",
            "csi1000_long_short_annual_return",  # 图13的50.47% vs 图15的51.21%
            thresholds={
                "annual_return": [0.4618, 0.5047, 0.4378],
                "trades": [394, 246, 166],
                "win_rate": [0.5430, 0.5464, 0.5402],
                "max_drawdown": [-0.2241, -0.3003, -0.2013],
            },
            annual={
                "annual_return": [0.1881, 0.0396, 0.9363, 0.8346, 0.7131, 0.5825, 0.5121],
                "volatility": [0.2726, 0.1888, 0.2481, 0.1440, 0.3299, 0.2235, 0.2419],
                "max_drawdown": [-0.1996, -0.1070, -0.0806, -0.0477, -0.1925, -0.1335, -0.3003],
            },
            long_only=0.26,
            short_only=0.2001,
        ),
    ],
    ignore_index=True,
)
REPORT_ANCHORS["source_figure"] = "图" + REPORT_ANCHORS["figure"].astype(str)


def anchor_matches(table: pd.DataFrame) -> pd.Series:
    """逐项判定实际值是否吻合研报值。

    交易次数须相等；其余指标容差0.00005，即研报百分数两位小数的半个单位。缺实际值不吻合。

    :param table: 含metric、actual、target列的对照表。
    :returns: 同索引布尔序列。
    """
    tolerance = np.where(table["metric"].eq("trades"), 0, 0.00005)
    return (table["actual"] - table["target"]).abs().le(tolerance)


def anchor_acceptance(table: pd.DataFrame) -> bool:
    """单一收益口径下的研报数值验收。

    通过条件：预期锚点齐全，非冲突项全部吻合，每个冲突组至少一项吻合。
    逐项是否全部吻合是另一个判断，用 :func:`anchor_matches`。

    预期锚点只取表内图号所属的那一个指数：只跑中证1000的run不因缺中证500的锚点而失败。

    :param table: `compare_anchors` 拼接出的单口径、单指数对照表；可带return_convention列。
    :returns: 是否通过；缺行、缺实际值、空表、图号不属任何已录入指数，一律不通过。
    :raises ValueError: 混入多种收益口径、多个指数，或组合键重复。
    """
    if "return_convention" in table and table["return_convention"].nunique() > 1:
        raise ValueError("验收须先按收益口径过滤，不能混合多种口径")
    # period经DuckDB往返后年份会变成字符串，统一按字符串比对
    keys = table[ANCHOR_KEY].assign(period=table["period"].astype(str))
    if keys.duplicated().any():
        raise ValueError("锚点组合键重复")
    codes = REPORT_ANCHORS.loc[REPORT_ANCHORS["figure"].isin(table["figure"]), "code"].unique()
    if len(codes) > 1:
        raise ValueError(f"验收一次只针对一个指数，表内混有{sorted(codes)}")
    if len(codes) == 0:
        return False
    expected = REPORT_ANCHORS.loc[REPORT_ANCHORS["code"].eq(codes[0])]
    expected = expected.assign(period=expected["period"].astype(str))
    merged = expected.merge(
        keys.assign(matched=anchor_matches(table).to_numpy()), on=ANCHOR_KEY, how="left"
    )
    # 漏行在左连接后为NaN；先判齐全再聚合，避免空表或漏行让.all()/.any()误通过
    if merged["matched"].isna().any():
        return False
    matched = merged["matched"].astype(bool)
    grouped = merged["conflict_group"].notna()
    return bool(
        matched[~grouped].all() and matched[grouped].groupby(merged["conflict_group"]).any().all()
    )


def compare_anchors(
    actual: Mapping[str, float] | pd.DataFrame,
    *,
    figure: int,
    threshold: float | None = None,
) -> pd.DataFrame:
    """返回指定图表逐项实际值、研报值及实际减目标的差异。

    不对缺失结果补零；阈值表（图5、图13）必须指定阈值，其余图对应默认阈值0.002。
    阈值表与分年表的汇总年化属于不同来源（如图5的0.4321与图7的0.4386），禁止互相覆盖。
    所有收益、胜率、波动、回撤均以小数表示；交易次数为次数。

    :param actual: overall 指标字典，或以年份整数/overall 为索引、指标为列的绩效表。
    :param figure: 研报图号，须在ANCHOR_FIGURES内（中证500为5/7/9/10，中证1000为13/15/17/18）。
    :param threshold: 阈值表对应阈值，其他图可省略或填0.002。
    :return: 包含code、figure、source_figure、threshold、period、metric、conflict_group、target、
        actual、difference的长表。
    :raises ValueError: 图号、阈值不存在，或实际表存在重复索引/列。
    """
    if threshold is None and figure in {f["thresholds"] for f in ANCHOR_FIGURES.values()}:
        raise ValueError(f"图{figure}是阈值表，必须明确指定 threshold，避免混淆各阈值结果")
    selected = REPORT_ANCHORS.loc[REPORT_ANCHORS["figure"].eq(figure)].copy()
    if threshold is not None:
        selected = selected.loc[selected["threshold"].eq(threshold)]
    if selected.empty:
        raise ValueError("没有对应图号和阈值的研报锚点")
    frame = (
        actual.copy()
        if isinstance(actual, pd.DataFrame)
        else pd.DataFrame([actual], index=["overall"])
    )
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("实际绩效表索引及列名必须唯一")
    frame.index.name = "period"
    # 只取被锚定的指标：绩效表可能夹带回撤起止日期等非数值列，混入会让actual落库成字符串
    frame = frame[frame.columns.intersection(selected["metric"].unique())].astype(float)
    observed = frame.reset_index().melt(id_vars="period", var_name="metric", value_name="actual")
    result = selected.merge(observed, on=["period", "metric"], how="left", validate="one_to_one")
    result["difference"] = result["actual"] - result["target"]
    return result[
        [
            "code",
            "figure",
            "source_figure",
            "threshold",
            "period",
            "metric",
            "conflict_group",
            "target",
            "actual",
            "difference",
        ]
    ]
