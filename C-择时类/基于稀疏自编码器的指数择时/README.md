# 基于稀疏自编码器的指数择时

复现华源证券 2026-02-02《量化择时系列研究之一：基于稀疏自编码器的指数择时》。从 AlphaFarmer 的 hy_sae_timing 项目移植，保留原算法与 notebook 输出。

## 新人从这里开始

| 你想做什么 | 文档 |
|---|---|
| 安装环境、配置数据并跑完一次 notebook | [新人使用指南](docs/使用指南_20260922.md) |
| 离线读取、调整参数、导出结果、排查错误 | [常用操作](docs/常用操作_20260922.md) |
| 查研究参数、模型默认值和结果结构 | [参数与常用接口参考](docs/参数参考_20260922.md) |
| 理解数据到回测的流程与口径 | [核心流程图](docs/核心流程图_20260921/核心流程.html) · [流程说明](docs/核心流程图_20260921/核心流程图说明_20260921.md) |

## 快速开始

使用 Python 3.10+，在本策略目录（包含本 README）执行：

```bash
python -m pip install -r requirements.txt
python -m pip install notebook ipykernel
cp -n .env.example .env
```

编辑 `.env`，填写自己的 TuShare Pro `TS_TOKEN`，然后启动：

```bash
python -m notebook notebook/hy_sae_timing.ipynb
```

选择同一个 Python 环境的内核，从上到下运行。DataFeed 是 **Hugo 私有的数据访问模块**，不随项目提供；不可用时自动使用开源 `tushare` 客户端。已有 `TS_TOKEN` 环境变量优先于 `.env`。模板不含真实 token，本地 `.env` 被 Git 忽略。

当前 notebook 使用中证500、2020–2025样本外区间、10个 seed；模块默认5个。数据库自动生成于本策略目录的 `data/hy_sae_timing.duckdb`，缺数据时会联网。训练结果留在内存，notebook 不自动保存完整实验或执行研报验收。

严格离线读取应使用 `load_index_dataset`，当前 `FETCH` 变量不是联网开关。详细步骤及常见错误见[常用操作](docs/常用操作_20260922.md)。

## 保留内容

- `src/`：数据、指标、模型、回测与绘图模块。
- `data/`：不搬运源项目已有数据；后续运行时在本项目目录下自动生成，沿用本仓库规则，不纳入 Git。
- `notebook/hy_sae_timing.ipynb`：唯一保留的 notebook。
- `docs/`：新人教程、常用操作、参数参考、原研报、核心流程图及历史偏离评估。
- `tests/`：可独立运行的原有模块测试，导入路径已调整。依赖被排除脚本或验证产物的五个测试文件未搬运。

未搬运 `scripts/`、其他 notebook、其他文档及源项目的代理配置和历史记录。研报复现偏离评估属于原始研究快照，其中对源仓库脚本、其他文档或验证产物的引用不保证在本移植目录可用；新人请优先阅读上述现行使用文档。

## 阅读入口

- [研究 notebook](notebook/hy_sae_timing.ipynb)
- [核心流程图](docs/核心流程图_20260921/核心流程图说明_20260921.md)
- [研报复现偏离评估](docs/研报复现偏离评估_20260921.md)

回测结果与研报存在偏离，请结合保留的评估文档理解结果。
