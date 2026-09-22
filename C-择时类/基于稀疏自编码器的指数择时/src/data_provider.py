"""指数行情数据提供者：TuShare拉取落库（ADR-0001）与DuckDB读取校验出口。"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from loguru import logger

from ._validation import validate_datetime_index
from .store import DEFAULT_DB_PATH, SAEStore

FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "preclose", "volume", "amount", "turn")


def prepare_quotes(raw: pd.DataFrame, *, turnover_policy: str = "require") -> pd.DataFrame:
    """验证行情并按显式选择处理换手率，绝不静默填零。

    :param raw: 单指数时间索引行情，列名无美元符号。
    :param turnover_policy: require要求真实换手率；volume_proxy明确用量比代理。
    :returns: 验证后的副本，attrs保存换手率来源。
    :raises ValueError: 核心字段缺失、非法价格、日期错误或真实换手率缺失。
    """
    if turnover_policy not in {"require", "volume_proxy"}:
        raise ValueError("未知换手率政策")
    q = raw.loc[:, list(FIELDS)].astype(float).copy()
    validate_datetime_index(q, name="行情")
    prices = q[["open", "high", "low", "close", "preclose"]]
    if not np.isfinite(prices).all().all() or (prices <= 0).any().any():
        raise ValueError("价格缺失或非正，不能前填后用于收益计算")
    if (
        not np.isfinite(q[["volume", "amount"]]).all().all()
        or (q[["volume", "amount"]] <= 0).any().any()
    ):
        raise ValueError("指数成交量/额缺失或非正")
    if turnover_policy == "require":
        if not np.isfinite(q.turn).all() or (q.turn <= 0).any():
            raise ValueError("真实换手率缺失或无效；补齐数据或显式选择volume_proxy研究偏离")
    else:
        q["turn"] = q.volume / q.volume.rolling(20).mean()
        logger.warning("研究偏离：以成交量/20日均量代理换手率，非研报原始字段")
    q.attrs["turnover_policy"] = turnover_policy
    return q


def _migrate_legacy_columns(db_path: str | Path) -> None:
    """旧库（0.6.0前的instrument列）经可写打开完成就地列名迁移。

    先以只读连接探测，确认是旧schema才拿写锁——已迁移/新建库在并发
    只读连接（如另一notebook）持有下不会因构造取写锁而冲突；探测异常
    （文件不是DuckDB等）不视为旧schema，交由真实写入路径统一处理。
    """
    path = Path(db_path)
    if not path.exists():
        return
    try:
        with SAEStore(path, read_only=True) as store:
            legacy = store.query(
                "SELECT 1 FROM information_schema.columns"
                " WHERE table_name IN ('market_quotes', 'index_ranges')"
                " AND column_name='instrument' LIMIT 1"
            )
    except duckdb.Error:
        return
    if legacy.empty:
        return
    try:
        # __init__内完成ALTER TABLE RENAME COLUMN，行情与水位原封不动
        with SAEStore(path):
            pass
    except duckdb.IOException as exc:
        raise RuntimeError(
            f"旧库列名迁移需要独占可写打开（{path}），"
            f"请先关闭其他占用该库的进程后重试：{exc}"
        ) from exc


def ensure_index_coverage(
    db_path: str | Path,
    code: str,
    start_date: str,
    end_date: str,
    *,
    client=None,
) -> pd.DataFrame:
    """确保库内覆盖请求区间：起点钳制到已固化有效起点后经行情源补拉。

    供研究流程（CLI、探针、notebook）在读取校验出口前调用。请求起点早于
    库内固化起点时按固化起点钳制而非报错——warm-up回溯不足截断到数据
    起点是既有研究语义，由读取出口如实报告；严格拒绝越界请直接用
    ``TuShareIndexSource.get_quotes``。

    :param db_path: ``SAEStore``数据库路径。
    :param code: 指数TS代码，如``000905.SH``。
    :param start_date: 请求开始日，``YYYY-MM-DD``或``YYYYMMDD``。
    :param end_date: 请求结束日，格式同上。
    :param client: 注入的TuShare pro客户端，测试用；缺省优先DataFeed封装，回退根``.env``的``TS_TOKEN``。
    :returns: 行情源返回的请求区间行情。
    """
    begin = pd.Timestamp(start_date)
    if Path(db_path).exists():
        # 旧库（0.6.0前instrument列）先完成就地列名迁移，再只读取固化起点
        _migrate_legacy_columns(db_path)
        with SAEStore(db_path, read_only=True) as store:
            first, _ = store.read_index_range(code)
        if first is not None and begin < first:
            logger.info(
                "请求起点{}早于指数{}固化起点{}，按固化起点确保覆盖",
                begin.date(),
                code,
                first.date(),
            )
            begin = first
    return TuShareIndexSource(db_path, client=client).get_quotes(
        code, str(begin.date()), end_date
    )


def load_index_dataset(
    db_path: str | Path,
    start_date: str,
    end_date: str,
    *,
    code: str = "000905.SH",
    turnover_policy: str = "require",
) -> pd.DataFrame:
    """从模块DuckDB读取原始指数行情，并执行与入库一致的校验。

    :param db_path: ``SAEStore``数据库路径。
    :param start_date: 请求起始日；读取时会回溯600个交易日以供warm-up。
    :param end_date: 请求结束日。
    :param code: 指数代码。
    :param turnover_policy: ``require``或显式``volume_proxy``。
    :returns: 含warm-up的验证行情，换手率代理只存在于内存。
    :raises ValueError: 区间倒置、数据库行情不足、交易日缺失或换手率无效。
    """
    # 须在warm-up回溯之前检查：回溯会把起点挪到终点之前，倒置区间会被掩盖成合法读取。
    if pd.Timestamp(end_date) < pd.Timestamp(start_date):
        raise ValueError(f"请求结束日{end_date}早于开始日{start_date}，区间无效")
    with SAEStore(db_path, read_only=True) as store:
        sessions = store.read_calendar()
        if sessions.empty:
            raise ValueError("DuckDB交易日历为空")
        start_pos = int(sessions.searchsorted(pd.Timestamp(start_date), side="left"))
        begin_pos = max(0, start_pos - 600)
        begin = sessions[begin_pos]
        raw = store.read_quotes(code, str(begin.date()), end_date)
    if raw.empty:
        raise ValueError(f"DuckDB没有{code}的行情")
    validate_sessions(raw, sessions)
    q = prepare_quotes(raw, turnover_policy=turnover_policy)
    validate_sessions(q, sessions)
    q.attrs.update(
        calendar_verified=True,
        code=code,
        source=f"SAEStore:{Path(db_path).expanduser().resolve()}",
        requested_start=start_date,
        requested_end=end_date,
    )
    logger.info(
        "从DuckDB读取指数{}：{}至{}，{}行",
        code,
        q.index[0].date(),
        q.index[-1].date(),
        len(q),
    )
    return q


def get_index_dataset(
    start_date: str,
    end_date: str,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    code: str = "000905.SH",
    turnover_policy: str = "require",
    client: Any = None,
) -> pd.DataFrame:
    """确保库内覆盖后读取校验行情：缺失才联网补拉，缓存命中零网络。

    编排层日常取数入口：先经 :func:`ensure_index_coverage` 确保库内覆盖
    （请求起点自动钳制到固化有效起点，缺头/尾才触网），再走
    :func:`load_index_dataset` 的只读校验出口（含600交易日warm-up）。
    需要保证绝不触网、缺数据即报错的离线/复现场景，请直接调用
    :func:`load_index_dataset`。

    :param start_date: 请求起始日，``YYYY-MM-DD``或``YYYYMMDD``。
    :param end_date: 请求结束日，格式同上。
    :param db_path: ``SAEStore``数据库路径；缺省为模块默认库。
    :param code: 指数TS代码，如``000905.SH``。
    :param turnover_policy: ``require``或显式``volume_proxy``。
    :param client: 注入的TuShare pro客户端，测试用。
    :returns: 含warm-up的验证行情，换手率代理只存在于内存。
    :raises ValueError: 覆盖补拉或读取校验失败。
    :raises RuntimeError: 覆盖补拉时接口返回无列空表（疑似HTTP失败/限流），
        或旧库列名迁移需独占可写打开。
    """
    ensure_index_coverage(db_path, code, start_date, end_date, client=client)
    return load_index_dataset(
        db_path,
        start_date,
        end_date,
        code=code,
        turnover_policy=turnover_policy,
    )


def validate_sessions(
    quotes: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """验证行情起止之间没有漏掉真实交易日。

    :param quotes: 日期索引行情。
    :param sessions: 权威交易日历（库内``trade_cal``日历），含节假日规则。
    :param start_date: 显式请求起点，检查区间首部缺口。
    :param end_date: 显式请求终点，检查区间尾部缺口。
    :returns: 无返回值，遇缺口抛错。
    :raises ValueError: 行情区间存在缺失或非交易日。
    """
    if quotes.empty:
        raise ValueError("行情为空")
    start = pd.Timestamp(start_date) if start_date else quotes.index.min()
    end = pd.Timestamp(end_date) if end_date else quotes.index.max()
    expected = sessions[(sessions >= start) & (sessions <= end)]
    observed = quotes.loc[start:end].index
    if not observed.equals(expected):
        missing = expected.difference(observed)
        extra = observed.difference(expected)
        raise ValueError(f"交易日缺口/异常日期：missing={missing.tolist()}, extra={extra.tolist()}")


DATAFEED_TUSHARE = "DataFeed.dataserver.apis.tushare.tushare_api"


def _datafeed_client() -> Any | None:
    """尝试构造DataFeed的TuShare封装；不可用时返回None交由调用方回退。

    DataFeed经网关代理转发请求，token读其``api_config.ini``且只对网关有效，
    故整体复用其client，而不是取token给官方``pro_api``。该模块在导入期即读
    配置：缺包抛ImportError、缺ini抛FileNotFoundError、缺``[tushare]``段抛KeyError。

    :returns: DataFeed ``TuShare``实例；导入失败或token为空时为None。
    """
    try:
        module = importlib.import_module(DATAFEED_TUSHARE)
    except (ImportError, FileNotFoundError, KeyError) as exc:
        logger.warning("DataFeed TuShare不可用（{}: {}），回退tushare+TS_TOKEN", type(exc).__name__, exc)
        return None
    if not module.TS_TOKEN:
        logger.warning("DataFeed未配置tushare token，回退tushare+TS_TOKEN")
        return None
    logger.info("TuShare客户端：DataFeed封装（网关代理）")
    # 封装默认max_retry=0表示无限重试，断网会静默卡死，显式设上限
    return module.TuShare(token=module.TS_TOKEN, max_retry=3)


def _tushare_env_client() -> Any:
    """回退路径：加载仓库根``.env``读``TS_TOKEN``，创建官方tushare客户端。

    :returns: ``tushare.pro_api()``客户端。
    :raises ValueError: 未配置``TS_TOKEN``（此时DataFeed也已不可用）。
    """
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())
    token = os.environ.get("TS_TOKEN", "")
    if not token:
        raise ValueError(
            "无可用TuShare客户端：DataFeed不可用（见上方告警）且未配置TS_TOKEN；"
            "请修复DataFeed的api_config.ini，或在仓库根.env中设置TS_TOKEN"
        )
    import tushare as ts

    logger.info("TuShare客户端：tushare官方接口（根.env的TS_TOKEN）")
    ts.set_token(token)
    return ts.pro_api()


class TuShareIndexSource:
    """TuShare指数行情源：拉取两接口行情inner拼接后落DuckDB（ADR-0001）。

    ``index_daily``与``index_dailybasic``按交易日inner拼接，原始表换手率
    自拼接有效起点起恒完整；缓存未命中时仅补缺失头/尾段并upsert合并，
    超上限长区间按交易日自动分段；各指数固化起点与拉取水位持久化于
    ``index_ranges``表：水位仅在(库内最新行, 请求终点]区间内没有交易日
    （权威交易日历``calendar_sessions``判定）时才推进到请求终点，否则
    停在库内最新行——尾部有交易日却没拿到数据，不能当成已覆盖；且推进
    的上限钳制在``trade_cal``本轮确认的日历发布边界（不超过实际发布到的
    最远日期，而非仅看开市日的``sessions.max()``），避免请求终点越过日历
    尚未发布的区间时把水位误判到未来——请求终点落在周末/假期等已发布的
    休市日不受影响，水位仍如实推进到请求终点。
    """

    CALENDAR_START = "20040101"
    DAILY_MAX_ROWS = 7800  # index_daily单次上限8000行，留余量
    BASIC_MAX_ROWS = 2800  # index_dailybasic单次上限3000行，留余量

    def __init__(self, db_path: str | Path, *, client=None) -> None:
        """初始化行情源。

        :param db_path: ``SAEStore``数据库路径，作缓存与原始行情库。
        :param client: 注入的TuShare pro客户端；缺省优先DataFeed封装，回退根``.env``的``TS_TOKEN``。
        """
        self.db_path = Path(db_path)
        self._client = client
        # 旧库（instrument列）在此触发就地列名迁移；已迁移库只读探测、不取写锁
        _migrate_legacy_columns(self.db_path)

    @property
    def client(self) -> Any:
        """惰性创建TuShare客户端：优先DataFeed封装，不可用再回退tushare+根``.env``。"""
        if self._client is None:
            client = _datafeed_client()
            self._client = client if client is not None else _tushare_env_client()
        return self._client

    def get_quotes(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """按区间取行情：缓存命中零网络调用；未命中仅补缺失段并upsert落库。

        起点治理三段式：发布日为预期起点（首拉从发布日起、请求整体早于
        发布日直接报错）；从发布日起的拉取发现实际数据起点晚于发布日则
        告警并固化；此后请求起点早于固化起点直接报错，不发起注定缺头的
        拉取。请求区间被库内水位完全覆盖时直接读库返回；否则仅对缺失的
        头/尾段发起调用，已有区间零重拉；请求终点超过库内最新时自动补到
        请求终点：尾部确认只剩非交易日时水位如实写到请求终点，尾部还有
        交易日却没拿到数据（可能当日未发布，也可能接口静默失败）时水位
        停在库内最新行，同区间下次请求会再次发起调用而非静默复用；水位
        推进上限钳制在``trade_cal``本轮确认的日历发布边界（``min(请求终点,
        published_through)``，``published_through``取自不带``is_open``过滤
        的日历原始返回，能区分"区间内确认无交易日"与"日历没发布到那么
        远"两种情况），请求终点落在已发布的周末/假期不受影响，只有真正
        超出日历发布范围时才会钳制、避免把水位误判到未发布的未来日期。

        :param code: 指数TS代码，如``000905.SH``。
        :param start_date: 请求开始日，``YYYY-MM-DD``或``YYYYMMDD``。
        :param end_date: 请求结束日，格式同上。
        :returns: DatetimeIndex升序、列为``FIELDS``的行情；数据晚于请求
            起点或早于请求终点时为实际可得短区间，极端情况下为空表。
        :raises ValueError: 请求起点早于固化起点、请求整体早于发布日、区间倒置、首拉区间无交易日、
            TS代码无效（index_basic空返回）或DataFeed与TS_TOKEN均不可用。
        :raises RuntimeError: 接口某块返回``None``或无列空表（疑似HTTP失败/
            限流），或首次拉取最终0行（疑似TS代码无数据或调用失败）。
        """
        begin = start_date.replace("-", "")
        end = end_date.replace("-", "")
        if pd.Timestamp(end) < pd.Timestamp(begin):
            raise ValueError(f"请求结束日{end}早于开始日{begin}，区间无效")
        stored = self._read_stored(code)
        first, until = self._read_range(code)
        if not stored.empty:
            if first is None:
                logger.warning(
                    "指数{}库内存量行情无拉取水位记录：疑似0.3.0的Qlib口径库"
                    "（换手率与量纲和TuShare不同，继续增量补拉会混合口径），"
                    "建议删除或更换db文件后重新拉取",
                    code,
                )
            elif pd.Timestamp(begin) < first:
                raise ValueError(
                    f"指数{code}有效起点为{first.date()}（以库内最早行固化），"
                    f"请求起点{begin}早于有效起点，拒绝发起注定缺头的拉取"
                )
            covered_end = until if until is not None else stored.index.max()
            if pd.Timestamp(begin) >= stored.index.min() and pd.Timestamp(end) <= covered_end:
                logger.info("指数{}缓存命中：{}至{}直接读库返回", code, begin, end)
                return stored.loc[begin:end]
        info = self.client.index_basic(ts_code=code)
        if info.empty:
            raise ValueError(f"index_basic未返回指数{code}元数据：请检查TS代码与接口权限")
        list_date = str(info.iloc[0]["list_date"])
        publish = pd.to_datetime(list_date, format="%Y%m%d")
        if pd.Timestamp(end) < publish:
            raise ValueError(
                f"指数{code}发布于{publish.date()}，"
                f"请求区间{begin}至{end}整体早于发布日，无可拉数据"
            )
        sessions, published_through = self._ensure_calendar(end)
        solidify: pd.Timestamp | None = None
        frames: list[pd.DataFrame] = []
        with SAEStore(self.db_path) as store:
            for kind, seg_start, seg_end in self._missing_segments(stored, begin, end, list_date):
                from_publish = seg_start == list_date
                merged = self._fetch_segment(code, seg_start, seg_end, sessions)
                actual = self._actual_start(merged, stored, kind, from_publish)
                if actual is not None:
                    solidify = actual if solidify is None else min(solidify, actual)
                if from_publish and actual is not None and actual > publish:
                    logger.warning(
                        "换手率告警：指数{}实际数据起点{}晚于发布日{}，"
                        "换手率（index_dailybasic）自实际起点才有数据；"
                        "已按实际起点固化，更早请求将直接报错",
                        code,
                        actual.date(),
                        publish.date(),
                    )
                if not merged.empty:
                    frames.append(merged)
                    store.save_market_data(code, merged, sessions)
            if stored.empty and not frames:
                # 区间内没有交易日时根本不会发起行情调用，报「疑似TS代码无数据、请重试」
                # 会误导且重试不可能成功；按日历如实报错。
                window = (sessions >= pd.Timestamp(max(begin, list_date))) & (
                    sessions <= pd.Timestamp(end)
                )
                if not window.any():
                    raise ValueError(
                        f"指数{code}请求{begin}至{end}内无交易日（全为休市日，或超出"
                        f"交易日历已发布范围{published_through.date()}），无可拉数据"
                    )
                # 首拉最终0行：两接口调用本身没报错，但拿不到任何数据。若仍按
                # 旧逻辑记INFO返回空表，水位会照样写到end，之后同区间请求永远
                # 缓存命中返回空表，问题被彻底掩盖，故改为报错、不写水位。
                raise RuntimeError(
                    f"TuShare指数{code}请求{begin}至{end}未获取到任何行情，"
                    "疑似TS代码无数据或接口调用失败，水位未推进，请重试"
                )
            latest = store.read_quotes(code)
            last_row = latest.index.max() if not latest.empty else None
            if last_row is None:
                # 理论上不会走到这里（上面已对首拉0行报错拦截）；保留兜底避免
                # 向save_index_range传入None。
                watermark = pd.Timestamp(end)
            else:
                pending = sessions[(sessions > last_row) & (sessions <= pd.Timestamp(end))]
                # 尾部只剩非交易日（pending为空）才能确认"已覆盖"，水位写到
                # 请求终点；否则水位停在库内最新行——尾部还有交易日却没拿到
                # 数据，可能是当日未发布，也可能是网关静默失败，都不能算已
                # 覆盖，代价是下次同区间请求会重新触网重试。
                # 但"已覆盖"的确认只能到日历实际发布的边界（published_through）
                # 为止：pending为空也可能是因为请求终点超出了TuShare已发布的
                # trade_cal范围，而不是真的确认了区间内无交易日（周末/假期
                # 属于此情形之外——它们本身就在已发布范围内，published_through
                # 会如实覆盖到请求终点），此时不能把水位写到请求终点本身，
                # 否则之后日历/数据真的补齐时会被max语义卡在这个虚假水位上，
                # 同区间请求永远缓存命中、静默返回旧数据。
                watermark = (
                    min(pd.Timestamp(end), published_through) if pending.empty else last_row
                )
            store.save_index_range(code, solidify, watermark)
            if not frames:
                store.save_calendar(sessions)
        result = self._read_stored(code)
        logger.info(
            "TuShare指数{}请求{}至{}补{}段落库，库内{}至{}共{}行",
            code,
            begin,
            end,
            len(frames),
            result.index[0].date(),
            result.index[-1].date(),
            len(result),
        )
        return result.loc[begin:end]

    def _actual_start(
        self,
        merged: pd.DataFrame,
        stored: pd.DataFrame,
        kind: str,
        from_publish: bool,
    ) -> pd.Timestamp | None:
        """从拉取结果推断实际数据起点证据，供固化判定。

        只有触达发布日的段内最早行才是真实起点证据；头部段拉空说明库内
        最早行之前无inner数据，实际起点即库内最早行；尾部段结果不构成
        头部证据，首拉段拉空亦无可固化证据。

        :param merged: 单段inner拼接结果，可能为空。
        :param stored: 拉取前库内行情，可能为空。
        :param kind: 段类型，``first``/``head``/``tail``。
        :param from_publish: 段起点是否触达发布日。
        :returns: 实际数据起点；无证据时为None。
        """
        if from_publish and not merged.empty:
            return merged.index.min()
        if kind == "head" and merged.empty and not stored.empty:
            return stored.index.min()
        return None

    def _missing_segments(
        self, stored: pd.DataFrame, begin: str, end: str, list_date: str
    ) -> list[tuple[str, str, str]]:
        """计算需拉取的缺失段，已有区间零重拉。

        首拉返回整段（起点截断到发布日）；否则仅返回头部/尾部缺口段，
        段边界与库内已有区间相差一个自然日，避免重拉已有数据。

        :param stored: 拉取前库内行情，可能为空。
        :param begin: 归一化请求起点，``YYYYMMDD``。
        :param end: 归一化请求终点。
        :param list_date: 发布日，``YYYYMMDD``。
        :returns: (段类型first/head/tail, 段起点, 段终点)列表。
        """
        if stored.empty:
            return [("first", max(begin, list_date), end)]
        segments: list[tuple[str, str, str]] = []
        lo, hi = stored.index.min(), stored.index.max()
        if pd.Timestamp(begin) < lo:
            head_from = max(begin, list_date)
            head_to = (lo - pd.Timedelta(days=1)).strftime("%Y%m%d")
            if head_from <= head_to:
                segments.append(("head", head_from, head_to))
        if pd.Timestamp(end) > hi:
            segments.append(("tail", (hi + pd.Timedelta(days=1)).strftime("%Y%m%d"), end))
        return segments

    def _fetch_segment(
        self, code: str, seg_start: str, seg_end: str, sessions: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """拉取单个缺失段：两接口按交易日分块调用后inner拼接。

        :param code: 指数TS代码。
        :param seg_start: 段起点，``YYYYMMDD``。
        :param seg_end: 段终点，``YYYYMMDD``。
        :param sessions: 完整交易日历，兼作分块基准。
        :returns: 该段行情；段内无数据时为空表。
        """
        dates = sessions[(sessions >= pd.Timestamp(seg_start)) & (sessions <= pd.Timestamp(seg_end))]
        if dates.empty:
            return pd.DataFrame()
        daily = self._fetch_api("index_daily", code, dates, self.DAILY_MAX_ROWS)
        basic = self._fetch_api("index_dailybasic", code, dates, self.BASIC_MAX_ROWS)
        if daily.empty or basic.empty:
            return pd.DataFrame()
        return self._merge(daily, basic)

    def _fetch_api(
        self, api: str, code: str, dates: pd.DatetimeIndex, limit: int
    ) -> pd.DataFrame:
        """按limit行上限将交易日切块调用指定接口并合并，规避单次截断。

        :param api: TuShare接口名，``index_daily``或``index_dailybasic``。
        :param code: 指数TS代码。
        :param dates: 该段全部交易日，升序。
        :param limit: 单次调用行数上限。
        :returns: 各块结果按行合并的原始返回。
        :raises RuntimeError: 某块返回``None``或无列空表。官方tushare客户端
            （``tushare/pro/client.py``）在HTTP状态码≥400时把
            ``requests.Response``判假，被``if res: ... else: return
            pd.DataFrame()``吞成无列空表而非抛异常，与合法的「当日未发布」
            响应（带FIELDS对应列、只是0行）不同；DataFeed封装在属性不存在
            时也返回``None``。一旦命中说明本次拉取不可信，禁止用它推进水位。
        """
        fetch = getattr(self.client, api)
        frames: list[pd.DataFrame] = []
        for i in range(0, len(dates), limit):
            frame = fetch(
                ts_code=code,
                start_date=dates[i].strftime("%Y%m%d"),
                end_date=dates[min(i + limit, len(dates)) - 1].strftime("%Y%m%d"),
            )
            if frame is None or frame.columns.empty:
                raise RuntimeError(
                    f"TuShare {api} 返回无列空表：疑似HTTP失败（网关/限流），水位未推进，请重试"
                )
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def _read_stored(self, code: str) -> pd.DataFrame:
        """读库内该指数全量行情，供固化起点判定与缓存命中。

        :param code: 指数TS代码。
        :returns: ``DatetimeIndex``升序行情；库文件不存在时返回空表。
        """
        if not self.db_path.exists():
            return pd.DataFrame()
        with SAEStore(self.db_path, read_only=True) as store:
            return store.read_quotes(code)

    def _read_range(self, code: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """读该指数固化起点与拉取水位；库文件不存在时为(None, None)。

        :param code: 指数TS代码。
        :returns: (固化起点, 拉取水位)。
        """
        if not self.db_path.exists():
            return None, None
        with SAEStore(self.db_path, read_only=True) as store:
            return store.read_index_range(code)

    def _read_calendar(self) -> pd.DatetimeIndex:
        """读库内交易日历；库文件不存在时返回空日历。

        :returns: 升序唯一``DatetimeIndex``。
        """
        if not self.db_path.exists():
            return pd.DatetimeIndex([])
        with SAEStore(self.db_path, read_only=True) as store:
            return store.read_calendar()

    def _ensure_calendar(self, end: str) -> tuple[pd.DatetimeIndex, pd.Timestamp]:
        """增量维护SSE交易日历至请求终点，返回开市日历与本轮确认的日历发布边界。

        不再用``is_open="1"``过滤请求——只看开市日无法区分"区间内确认无
        交易日"与"日历根本没发布到那么远"，两者都表现为空结果。改为拉取
        区间内全部日历行（开市+休市），据此计算两样东西：①落库的开市
        交易日历``sessions``（内容不变，仍只含开市日，供缺失段计算使用）；
        ②``published_through``——TuShare日历本轮确认已发布到的最远日期，
        供调用方钳制拉取水位，不把"未发布"误判为"已确认覆盖"。

        :param end: 归一化请求终点，``YYYYMMDD``。
        :returns: ``(sessions, published_through)``；``sessions``为覆盖
            [日历起点, 请求终点]的开市交易日序列；``published_through``
            为本轮确认的日历发布边界，下限保证不早于已有存量日历末尾。
        """
        existing = self._read_calendar()
        if not existing.empty and existing.max() >= pd.Timestamp(end):
            # 已有开市交易日>=请求终点，本身即证明日历早已发布覆盖到此。
            return existing, pd.Timestamp(end)
        start = (
            self.CALENDAR_START
            if existing.empty
            else (existing.max() + pd.Timedelta(days=1)).strftime("%Y%m%d")
        )
        cal = self.client.trade_cal(exchange="SSE", start_date=start, end_date=end)
        if cal is None or cal.columns.empty:
            # 官方tushare客户端把HTTP≥400的失败响应吞成无列空表而非抛异常
            # （与_fetch_api对index_daily/index_dailybasic的处理同源）；
            # 与「区间内合法返回0行」（有cal_date/is_open列，只是没有行）
            # 不同，一旦命中说明本次拉取不可信，不能当成"日历未发布"静默
            # 放行，否则会带着过期水位返回陈旧数据。
            raise RuntimeError(
                "TuShare trade_cal 返回无列空表：疑似HTTP失败（网关/限流），水位未推进，请重试"
            )
        if cal.empty:
            # TuShare对这个区间合法地没有返回任何日历行（开市或休市都没有），
            # 说明日历尚未发布到这么远，只能确认到已有存量日历的末尾。
            published_through = (
                existing.max() if not existing.empty else pd.Timestamp(start) - pd.Timedelta(days=1)
            )
            return existing, published_through
        dates = pd.to_datetime(cal["cal_date"].astype(str), format="%Y%m%d")
        published_through = dates.max()
        # 生产网关按cal_date降序返回；_fetch_segment按位置切块要求升序，
        # 否则每块的start_date/end_date会倒置，两接口按区间过滤恒空。
        fetched = pd.DatetimeIndex(dates[cal["is_open"].astype(int) == 1]).sort_values()
        sessions = fetched if existing.empty else existing.union(fetched).sort_values()
        return sessions, published_through

    def _merge(self, daily: pd.DataFrame, basic: pd.DataFrame) -> pd.DataFrame:
        """两接口按交易日inner拼接并映射为FIELDS列。"""
        d = daily.copy()
        b = basic.copy()
        d["trade_date"] = d["trade_date"].astype(str)
        b["trade_date"] = b["trade_date"].astype(str)
        merged = pd.merge(
            d, b[["ts_code", "trade_date", "turnover_rate"]], on=["ts_code", "trade_date"], how="inner"
        )
        merged = merged.rename(
            columns={"pre_close": "preclose", "vol": "volume", "turnover_rate": "turn"}
        )
        merged["trade_date"] = pd.to_datetime(merged["trade_date"], format="%Y%m%d")
        joined = merged.set_index("trade_date").rename_axis("datetime")
        return joined.sort_index()[list(FIELDS)].astype(float)
