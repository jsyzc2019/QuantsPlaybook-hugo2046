"""监督稀疏自编码器；网络宽度等均为显式研究假设。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np  # 本机先初始化NumPy，避免PyTorch先加载时的OpenMP冲突。
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn

from ._validation import require_positive_int


@dataclass(frozen=True)
class ModelConfig:
    """SAE研究配置，非作者公布参数。

    :param hidden_dim: 编码/解码隐藏层宽度。
    :param code_dim: sigmoid稀疏编码宽度。
    :param epochs: 固定训练轮数，不用样本外表现早停。
    :param batch_size: 批大小。
    :param learning_rate: Adam学习率。
    :param reconstruction_weight: 重构MSE权重。
    :param sparsity_weight: KL稀疏损失权重；0.1按首折训练集预注册判据选定（训练末平均激活≤2ρ），见偏离分析文档。
    :param rho: 平均编码激活目标。
    """

    hidden_dim: int = 64
    code_dim: int = 16
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.001
    reconstruction_weight: float = 0.1
    sparsity_weight: float = 0.1
    rho: float = 0.05

    def __post_init__(self) -> None:
        for name in ("hidden_dim", "code_dim", "epochs", "batch_size"):
            require_positive_int(getattr(self, name), name)
        if not np.isfinite(
            [self.rho, self.learning_rate, self.reconstruction_weight, self.sparsity_weight]
        ).all():
            raise ValueError("模型参数必须为有限值")
        if not 0 < self.rho < 1 or self.learning_rate <= 0:
            raise ValueError("rho须在(0,1)，学习率须为正")
        if self.reconstruction_weight < 0 or self.sparsity_weight < 0:
            raise ValueError("损失权重不能为负")


def sparse_kl(code: torch.Tensor, rho: float) -> torch.Tensor:
    """计算研报KL稀疏惩罚，避免log(0)。

    :param code: batch×神经元的sigmoid激活。
    :param rho: 目标激活概率。
    :returns: 按神经元求和的KL损失标量。
    """
    average = code.mean(dim=0).clamp(1e-6, 1 - 1e-6)
    return (rho * torch.log(rho / average) + (1 - rho) * torch.log((1 - rho) / (1 - average))).sum()


def within_sparsity_target(activation: pd.Series, rho: float) -> pd.Series:
    """判断编码平均激活是否达到稀疏目标（≤2ρ）；预注册选择与run闸门共用此判据。

    :param activation: 训练末编码平均激活，按候选或年份索引。
    :param rho: 目标激活概率。
    :returns: 同索引的布尔序列。
    """
    return activation <= 2 * rho


class _SAE(nn.Module):
    def __init__(self, n_features: int, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.code_dim),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.code_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, n_features),
        )
        self.predictor = nn.Sequential(nn.Linear(config.code_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        code = self.encoder(x)
        return self.predictor(code).squeeze(-1), self.decoder(code), code


def _build_network(n_features: int, config: ModelConfig, seed: int) -> _SAE:
    """按seed确定性初始化网络，不污染全局随机状态。

    :param n_features: 输入特征数。
    :param config: 训练配置。
    :param seed: 初始化随机种子。
    :returns: 初始化后的网络。
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return _SAE(n_features, config)


class SAERegressor(RegressorMixin, BaseEstimator):
    """按单seed拟合预测MSE+重构MSE+稀疏KL；遵循sklearn estimator约定。

    构造函数只保存参数（get_params/clone依赖此约定）；参数校验与网络构建在fit。
    参数含义同 :class:`ModelConfig`。

    :param seed: 初始化及批次排序随机种子。
    :ivar network_: 拟合后的网络。
    :ivar history_: 每epoch按样本加权的分项训练损失及编码平均激活activation（与rho对照判断稀疏是否生效）。
    :ivar label_scale_: 训练标签标准差，预测时乘回。
    :ivar n_features_in_: 训练特征数。
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        code_dim: int = 16,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        reconstruction_weight: float = 0.1,
        sparsity_weight: float = 0.1,
        rho: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.reconstruction_weight = reconstruction_weight
        self.sparsity_weight = sparsity_weight
        self.rho = rho
        self.seed = seed

    def fit(self, features: np.ndarray, label: np.ndarray) -> SAERegressor:
        """只用传入训练样本优化模型，标签按训练标准差缩放。

        :param features: n×p有限特征矩阵。
        :param label: n个真实未来收益率。
        :returns: 当前实例。
        :raises ValueError: 参数非法，或输入为空、长度不一致、含非有限值。
        """
        params = self.get_params()
        seed = params.pop("seed")
        config = ModelConfig(**params)  # 复用既有校验，单一来源
        features, label = np.asarray(features), np.asarray(label)
        if features.ndim != 2 or label.ndim != 1:
            raise ValueError("训练输入维度错误")
        if (
            len(features) < 2
            or len(features) != len(label)
            or not np.isfinite(features).all()
            or not np.isfinite(label).all()
        ):
            raise ValueError("训练输入为空、不对齐或包含非有限值")
        self.n_features_in_ = features.shape[1]
        self.network_ = _build_network(self.n_features_in_, config, seed)
        self.label_scale_ = max(float(np.std(label, ddof=1)), 1e-6)
        x = torch.as_tensor(np.asarray(features, dtype=np.float32))
        y = torch.as_tensor(np.asarray(label / self.label_scale_, dtype=np.float32))
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=config.learning_rate)
        rng = np.random.default_rng(seed)
        history = []
        self.network_.train()
        for epoch in range(config.epochs):
            totals = np.zeros(5)
            permutation = rng.permutation(len(x))
            for start in range(0, len(x), config.batch_size):
                ids = permutation[start : start + config.batch_size]
                pred, recon, code = self.network_(x[ids])
                prediction = nn.functional.mse_loss(pred, y[ids])
                reconstruction = nn.functional.mse_loss(recon, x[ids])
                sparsity = sparse_kl(code, config.rho)
                loss = (
                    prediction
                    + config.reconstruction_weight * reconstruction
                    + config.sparsity_weight * sparsity
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("SAE损失非有限值")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                totals += len(ids) * np.array(
                    [
                        z.detach().item()
                        for z in (loss, prediction, reconstruction, sparsity, code.mean())
                    ]
                )
            history.append([epoch + 1, *(totals / len(x))])
        self.history_ = pd.DataFrame(
            history,
            columns=["epoch", "total", "prediction", "reconstruction", "sparsity", "activation"],
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """输出恢复到原始收益率单位的预测。

        :param features: 有限的n×p特征矩阵。
        :returns: n个未来累计收益率预测。
        :raises sklearn.exceptions.NotFittedError: 尚未训练。
        """
        check_is_fitted(self, "network_")
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError("预测输入维度错误")
        if not np.isfinite(features).all():
            raise ValueError("预测输入含非有限值")
        self.network_.eval()
        with torch.no_grad():
            pred, _, _ = self.network_(torch.as_tensor(np.asarray(features, dtype=np.float32)))
        return pred.numpy() * self.label_scale_
