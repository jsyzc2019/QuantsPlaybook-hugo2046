"""自然年扩展训练、标签边界清除、五seed样本外集成。

流程：折外一次性算完特征与因果小波去噪 → `AnnualExpandingSplit` 按自然年划折 →
每折以训练行拟合 `Pipeline(SimpleImputer, StandardScaler)` → SAE 逐seed训练预测。
小波放在折外的依据见设计文档§3.4（无状态因果变换，与 `build_features` 同类）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .factor_algo import (
    MA_PAIRS,
    WINDOWS,
    build_features,
    causal_wavelet,
    forward_label,
    threshold_signal,
)
from .model import ModelConfig, SAERegressor, within_sparsity_target
from .splitter import AnnualExpandingSplit

# 稀疏权重预注册候选（升序）；判据见偏离分析文档「稀疏权重预注册判据」。
SPARSITY_CANDIDATES: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)


@dataclass(frozen=True)
class ResearchConfig:
    """可复现实验配置。

    :param code: 指数代码。
    :param train_start: 最早训练样本日期，须早于首个测试年年初。
    :param test_start: 样本外起点；训练截止仍是当年年初之前。
    :param test_end: 样本外结束日。
    :param horizon: 未来收益标签交易日周期。
    :param label_skip: 信号日到成交日的交易日数；0复现0.7.0前标签口径。
    :param seeds: 独立初始化种子；研报集成为五个。
    :param feature_windows: 特征滚动窗口（交易日）；最长窗口决定特征warm-up长度。
    :param ma_pairs: 均线比值对 ``(短, 长)``，两端须都在feature_windows内。
    :param rsi_form: "bounded"（默认，研究假设）或"ratio"（研报§2.2原比值，仅供对照）。
    :param wavelet_window: 历史小波窗口。
    :param wavelet: 小波基。
    :param wavelet_level: 分解级数。
    :param wavelet_mode: 小波边界延拓模式；复现0.7.0前结果用"symmetric"。
    :param threshold: 信号阈值。
    :param annual_days: 绩效年化天数。
    :param model: 模型与训练参数。
    """

    code: str = "000905.SH"
    train_start: str = "2005-01-01"
    test_start: str = "2020-01-01"
    test_end: str = "2025-12-31"
    horizon: int = 5
    label_skip: int = 1
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    feature_windows: tuple[int, ...] = WINDOWS
    ma_pairs: tuple[tuple[int, int], ...] = MA_PAIRS
    rsi_form: str = "bounded"
    wavelet_window: int = 256
    wavelet: str = "db4"
    wavelet_level: int = 4
    wavelet_mode: str = "reflect"
    threshold: float = 0.002
    annual_days: int = 252
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seed不能为空或重复")
        if (
            self.horizon < 1
            or self.label_skip < 0
            or pd.Timestamp(self.test_start) > pd.Timestamp(self.test_end)
        ):
            raise ValueError("标签周期或样本外范围无效")
        # 训练截止是首个测试年的年初（见AnnualExpandingSplit的gap清除），不是test_start本身。
        first_cutoff = pd.Timestamp(year=pd.Timestamp(self.test_start).year, month=1, day=1)
        if pd.Timestamp(self.train_start) >= first_cutoff:
            raise ValueError(f"训练起点{self.train_start}须早于首个训练截止{first_cutoff.date()}")
        if self.threshold < 0 or self.annual_days < 1:
            raise ValueError("阈值/年化天数无效")


@dataclass
class ResearchResult:
    """完整审计结果。

    :param predictions: datetime/code长表，含各seed及均值score。
    :param folds: 每轮实际训练/测试和标签可见边界。
    :param losses: 每轮每seed每epoch分项损失。
    :param preprocessors: 每年预处理统计量；median为**去噪后**特征的训练集中位数
        （小波在折外已算完，填补发生在去噪特征上），mean/std同为去噪后特征的训练集统计量。
    :param checkpoints: 每年每seed模型权重与标签缩放。
    :param feature_names: 有序特征清单（默认92维）。
    """

    predictions: pd.DataFrame
    folds: pd.DataFrame
    losses: pd.DataFrame
    preprocessors: dict[int, pd.DataFrame]
    checkpoints: dict[str, dict]
    feature_names: list[str]


@dataclass(frozen=True)
class FoldInputs:
    """单个年度折的模型输入；预处理只以本折训练集拟合。

    :param train_dates: 训练日期。
    :param x_train: 训练特征矩阵。
    :param y_train: 训练标签，原始收益单位。
    :param x_test: 样本外特征矩阵。
    :param test_dates: 本折样本外日期，与x_test逐行对应。
    :param statistics: 本折median/mean/std，均在**去噪后**特征上按训练行统计
        （median即去噪后特征的中位数，非原始特征）。
    """

    train_dates: pd.DatetimeIndex
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    test_dates: pd.DatetimeIndex
    statistics: pd.DataFrame


class SAETimingGenerator:
    """编排特征、逐年训练与样本外预测；自身不做数据IO。

    :param config: 明确记录的研究配置。
    """

    def __init__(self, config: ResearchConfig | None = None) -> None:
        self.config = config or ResearchConfig()

    def _prepare(
        self, quotes: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[tuple[np.ndarray, np.ndarray]]]:
        """裁剪到实验结束日，构造特征、折外因果去噪、标签，并切分年度折。

        :param quotes: 经数据层验证、含warm-up的单指数行情。
        :returns: (去噪特征, 未来收益标签, 标签实现日期, 位置下标折)；前三者索引相同。
        :raises ValueError: 样本外有行情缺口、warm-up之后仍有缺失特征，或历史不足。
        """
        cfg = self.config
        # 所有后续计算先裁到实验结束日，绝不读取未请求的未来行。
        q = quotes.loc[: cfg.test_end]
        expected = quotes.attrs.get("expected_test_dates")
        if expected is not None:
            actual = q.loc[cfg.test_start : cfg.test_end].index
            if not actual.equals(pd.DatetimeIndex(expected)):
                raise ValueError("样本外交易日不完整")
        # 丢弃最长滚动窗口的warm-up行（默认250）。
        features = build_features(q, cfg.feature_windows, cfg.ma_pairs, cfg.rsi_form).iloc[
            max(cfg.feature_windows) :
        ]
        if features.isna().any().any():
            bad = features.columns[features.isna().any()].tolist()
            message = f"warm-up之后仍有缺失特征{bad}；小波在折外计算无法按折填补。"
            if any(col.startswith("rsi_") for col in bad):
                message += "rsi_form='ratio'在分母为0时缺失，请用'bounded'"
            raise ValueError(message)
        # 小波是无状态因果变换（每行只用自己的历史窗口、无拟合参数），与build_features同属折外；
        # 去D1为线性滤波，在同一组μ/σ下与标准化可交换（实测差2e-12，设计文档§3.4）。但这不是与
        # 研报「标准化→去噪」等价：缩放器改在去噪后数据上拟合，6列日度噪声大的输入被放大至多约1.25倍。
        denoised = causal_wavelet(
            features, cfg.wavelet_window, cfg.wavelet, cfg.wavelet_level, mode=cfg.wavelet_mode
        ).iloc[cfg.wavelet_window - 1 :]
        # 保留train_start以前的特征作为小波预热，只限制候选训练样本。
        x = denoised.loc[cfg.train_start :]
        labels, maturity = forward_label(q.close, cfg.horizon, cfg.label_skip)
        # 划分器只认年份：测试年内早于test_start的行既不训练也不测试，先剔除
        test_start = pd.Timestamp(cfg.test_start)
        first = pd.Timestamp(year=test_start.year, month=1, day=1)
        x = x[(x.index < first) | (x.index >= test_start)]
        wanted = features.loc[test_start : cfg.test_end].index
        if not wanted.difference(x.index).empty:
            raise ValueError("样本外存在warm-up空缺；需要更早历史")
        # 显式列出实际有行的测试年，而不是由年份区间反推个数：划分器取的是groups里
        # 「最后n个年份」，两者必须指同一批年份。x已裁到test_end且测试年都在训练年之后，
        # 正常情况下二者恒等；不等说明上游裁剪改变了假设，宁可报错也不静默错位。
        present = np.unique(x.index.year)
        test_years = present[
            (present >= test_start.year) & (present <= pd.Timestamp(cfg.test_end).year)
        ]
        if len(test_years) == 0:
            raise ValueError("样本外区间没有数据")
        if not np.array_equal(test_years, present[-len(test_years) :]):
            raise ValueError(f"测试年{test_years.tolist()}不是样本里最后{len(test_years)}个年份")
        cv = AnnualExpandingSplit(len(test_years), gap=cfg.horizon + cfg.label_skip)
        folds = list(cv.split(x, groups=x.index.year.to_numpy()))
        return x, labels.reindex(x.index), maturity.reindex(x.index), folds

    def _fold_inputs(
        self, x: pd.DataFrame, labels: pd.Series, fold: tuple[np.ndarray, np.ndarray]
    ) -> FoldInputs:
        """仅以本折训练行拟合填补+标准化Pipeline，返回模型输入。

        :param x: `_prepare` 产出的去噪特征。
        :param labels: `_prepare` 产出的标签，索引与x一致。
        :param fold: `(训练行下标, 测试行下标)`。
        :returns: 本折训练/测试矩阵与预处理统计量。
        :raises ValueError: 训练样本不足一个batch，或训练标签含缺失。
        """
        train, test = fold
        if len(train) < self.config.model.batch_size:
            raise ValueError(f"{x.index[test[0]].year}训练样本不足一个batch")
        y_train = labels.iloc[train].to_numpy()
        # gap已清掉年末skip+horizon行；test_end贴近数据末尾时仍兜底，避免静默喂入NaN
        if not np.isfinite(y_train).all():
            raise ValueError(f"{x.index[test[0]].year}训练标签存在缺失")
        # keep_empty_features=True：全空列不被静默丢弃，维度与feature_names保持一致
        pre = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
        ).fit(x.iloc[train])
        imputer, scaler = pre.named_steps["simpleimputer"], pre.named_steps["standardscaler"]
        statistics = pd.DataFrame(
            {"median": imputer.statistics_, "mean": scaler.mean_, "std": scaler.scale_},
            index=x.columns,
        )
        return FoldInputs(
            train_dates=x.index[train],
            x_train=pre.transform(x.iloc[train]),
            y_train=y_train,
            x_test=pre.transform(x.iloc[test]),
            test_dates=x.index[test],
            statistics=statistics,
        )

    def run(self, quotes: pd.DataFrame) -> ResearchResult:
        """对完整历史行情执行年度扩展训练。

        :param quotes: 经数据层验证、含warm-up的单指数行情。
        :returns: 逐日预测及可审计的损失、分段、统计量、模型权重。
        :raises ValueError: 历史不足或样本外有行情缺口。
        """
        cfg = self.config
        x, labels, maturity, folds = self._prepare(quotes)
        predictions, histories, boundaries = [], [], []
        preprocessors, checkpoints = {}, {}
        for fold in folds:
            inputs = self._fold_inputs(x, labels, fold)
            year = inputs.test_dates[0].year
            logger.info(
                "训练{}：{}个训练日，{}个预测日，{} seeds",
                year,
                len(inputs.train_dates),
                len(inputs.test_dates),
                len(cfg.seeds),
            )
            frame = pd.DataFrame(index=inputs.test_dates)
            for seed in cfg.seeds:
                model = SAERegressor(**asdict(cfg.model), seed=seed)
                model.fit(inputs.x_train, inputs.y_train)
                frame[f"seed_{seed}"] = model.predict(inputs.x_test)
                histories.append(model.history_.assign(year=year, seed=seed))
                checkpoints[f"{year}_{seed}"] = {
                    "state_dict": model.network_.state_dict(),
                    "label_scale": model.label_scale_,
                }
            frame["score"] = frame.mean(axis=1)
            predictions.append(frame)
            preprocessors[year] = inputs.statistics
            train_dates = inputs.train_dates
            boundaries.append(
                {
                    "year": year,
                    "train_start": train_dates.min(),
                    "train_end": train_dates.max(),
                    "label_end": maturity.loc[train_dates].max(),
                    "test_start": inputs.test_dates.min(),
                    "test_end": inputs.test_dates.max(),
                    "train_count": len(train_dates),
                    "test_count": len(inputs.test_dates),
                }
            )
        prediction = pd.concat(predictions).rename_axis("datetime")
        prediction["code"] = cfg.code
        prediction = prediction.set_index("code", append=True)
        return ResearchResult(
            prediction,
            pd.DataFrame(boundaries),
            pd.concat(histories, ignore_index=True),
            preprocessors,
            checkpoints,
            x.columns.tolist(),
        )

    def select_sparsity_weight(
        self, quotes: pd.DataFrame, candidates: tuple[float, ...] = SPARSITY_CANDIDATES
    ) -> tuple[float, pd.DataFrame]:
        """按预注册判据选稀疏权重：只用首个年度折训练集，不接触任何样本外数据。

        每个候选按配置的全部seed各训练一次，取末epoch编码平均激活的seed均值，
        选均值不超过 ``2 * rho`` 的最小候选。

        :param quotes: 与run相同的完整行情。
        :param candidates: 非空、唯一、升序的候选权重。
        :returns: (选中权重, 每候选每seed的末epoch activation/prediction明细，含passed列)。
        :raises ValueError: 候选不合法，或没有任何候选达标（不放宽判据）。
        """
        if not candidates or list(candidates) != sorted(set(candidates)):
            raise ValueError("候选权重须非空、唯一且升序")
        cfg = self.config
        x, labels, _, folds = self._prepare(quotes)
        # 只取首折：训练标签全部在首个测试年年初前实现
        inputs = self._fold_inputs(x, labels, folds[0])
        rows = []
        for weight in candidates:
            model_config = replace(cfg.model, sparsity_weight=weight)
            for seed in cfg.seeds:
                model = SAERegressor(**asdict(model_config), seed=seed)
                last = model.fit(inputs.x_train, inputs.y_train).history_.iloc[-1]
                rows.append(
                    {
                        "sparsity_weight": weight,
                        "seed": seed,
                        "activation": float(last.activation),
                        "prediction": float(last.prediction),
                    }
                )
        detail = pd.DataFrame(rows)
        mean_activation = detail.groupby("sparsity_weight")["activation"].mean()
        ok = within_sparsity_target(mean_activation, cfg.model.rho)
        detail["passed"] = detail["sparsity_weight"].map(ok)
        if not ok.any():
            raise ValueError(f"没有候选权重使训练末平均激活≤2ρ：{mean_activation.round(4).to_dict()}")
        # groupby 键已升序，第一个达标者即最小候选
        chosen = float(ok[ok].index[0])
        logger.info("稀疏权重预注册选择={}；各候选训练末平均激活{}", chosen, mean_activation.round(4).to_dict())
        return chosen, detail

    def generate(
        self, quotes: pd.DataFrame, start_date: str | None = None, end_date: str | None = None
    ) -> pd.Series:
        """输出标准长表信号，先算完整样本外状态再裁请求区间。

        :param quotes: 含全部训练历史的行情。
        :param start_date: 返回信号起点，默认配置样本外起点。
        :param end_date: 返回信号终点，默认配置样本外终点。
        :returns: 名为HY_SAE的datetime/code索引Series。
        """
        result = self.run(quotes)
        score = result.predictions["score"]
        signal = threshold_signal(score, self.config.threshold).rename("HY_SAE")
        dates = signal.index.get_level_values("datetime")
        return signal[
            (dates >= pd.Timestamp(start_date or self.config.test_start))
            & (dates <= pd.Timestamp(end_date or self.config.test_end))
        ]
