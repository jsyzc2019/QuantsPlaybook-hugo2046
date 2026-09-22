"""TuShareIndexSource 取数与缓存契约测试（seam：get_quotes 公共接口，注入 fake client）。"""

import numpy as np
import pandas as pd
import pytest

from .test_factor_algo import quotes

INDEX = "000905.SH"


class FakeClient:
    """内存 TuShare 客户端：按需切片返回 daily/basic/日历，记录调用。

    ``trade_cal`` 模拟真实 TuShare 语义：不传 ``is_open`` 时返回区间内
    每一个自然日（含休市日），``is_open`` 按该日是否属于开市交易日
    （``cal``，缺省由行情反推）标 1/0；传 ``is_open`` 则按其过滤，与真实
    接口一致。``published_until`` 模拟"日历尚未发布到那么远"：缺省
    ``None``表示对任意请求终点都视为已发布（无上限）。``descending``模拟
    生产网关按``cal_date``降序返回的真实行为（缺省``False``保持升序）。
    ``trade_cal_fails``模拟HTTP失败：返回无列空表（官方tushare客户端把
    HTTP≥400响应吞成``pd.DataFrame()``的已知行为）。
    """

    def __init__(
        self,
        daily: pd.DataFrame,
        basic: pd.DataFrame,
        list_date: str,
        cal: pd.Series | None = None,
        published_until: str | None = None,
        descending: bool = False,
        trade_cal_fails: bool = False,
    ):
        self.daily, self.basic, self.list_date = daily, basic, list_date
        # 独立开市日历数据；缺省由行情日期反推（仅够对齐检查，勿用于日历语义用例）
        self.cal = cal
        # TuShare日历实际发布到的最远日期；None表示不设上限（永远已发布）
        self.published_until = published_until
        # 生产网关trade_cal按cal_date降序返回；测试默认升序，显式开启复现降序
        self.descending = descending
        # 模拟trade_cal本身HTTP失败（无列空表），可在用例中期启用
        self.trade_cal_fails = trade_cal_fails
        self.calls: list[tuple[str, dict]] = []

    def _slice(self, df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        d = df["trade_date"].astype(str)
        return df[(d >= start) & (d <= end)].copy()

    def index_basic(self, **kwargs):
        self.calls.append(("index_basic", kwargs))
        return pd.DataFrame([{"ts_code": INDEX, "list_date": self.list_date}])

    def index_daily(self, **kwargs):
        self.calls.append(("index_daily", kwargs))
        return self._slice(self.daily, kwargs["start_date"], kwargs["end_date"])

    def index_dailybasic(self, **kwargs):
        self.calls.append(("index_dailybasic", kwargs))
        return self._slice(self.basic, kwargs["start_date"], kwargs["end_date"])

    def trade_cal(self, **kwargs):
        self.calls.append(("trade_cal", kwargs))
        if self.trade_cal_fails:
            return pd.DataFrame()  # 模拟HTTP失败：无列空表
        start, end = kwargs["start_date"], kwargs["end_date"]
        if self.published_until is not None and self.published_until < end:
            end = self.published_until  # 日历只发布到这里，超出部分模拟为"没有这一行"
        if end < start:
            return pd.DataFrame({"cal_date": pd.Series(dtype=str), "is_open": pd.Series(dtype=int)})
        if self.cal is None:
            open_days = set(pd.concat([self.daily["trade_date"], self.basic["trade_date"]]).unique())
        else:
            open_days = set(self.cal)
        all_days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D").strftime("%Y%m%d")
        frame = pd.DataFrame(
            {"cal_date": all_days, "is_open": [1 if d in open_days else 0 for d in all_days]}
        )
        if "is_open" in kwargs:
            frame = frame[frame["is_open"] == int(kwargs["is_open"])]
        frame = frame.reset_index(drop=True)
        if self.descending:
            frame = frame.sort_values("cal_date", ascending=False).reset_index(drop=True)
        return frame


def _make_raw(days: int = 24, start: str = "2007-01-15"):
    """构造 daily 与 basic 两个内存接口数据。"""
    idx = pd.bdate_range(start, periods=days)
    dates = idx.strftime("%Y%m%d")
    daily = pd.DataFrame(
        {
            "ts_code": INDEX,
            "trade_date": dates,
            "open": np.arange(1, days + 1, dtype=float),
            "high": np.arange(2, days + 2, dtype=float),
            "low": np.arange(0.5, days + 0.5, dtype=float),
            "close": np.arange(1.5, days + 1.5, dtype=float),
            "pre_close": np.arange(1.4, days + 1.4, dtype=float),
            "vol": np.arange(100, 100 + days, dtype=float),
            "amount": np.arange(1000, 1000 + days, dtype=float),
        }
    )
    basic = pd.DataFrame(
        {
            "ts_code": INDEX,
            "trade_date": dates,
            "turnover_rate": np.linspace(0.8, 1.2, days),
        }
    )
    return daily, basic, idx


def test_first_fetch_starts_at_list_date_and_persists(tmp_path):
    """首次拉取：起点截断到发布日、字段映射正确、落库可读回。"""
    from src.data_provider import TuShareIndexSource
    from src.store import SAEStore

    daily, basic, idx = _make_raw()
    client = FakeClient(daily, basic, list_date="20070115")
    db = tmp_path / "ts.duckdb"
    source = TuShareIndexSource(db, client=client)

    # 请求起点早于发布日：应从 list_date 起拉，而非 20070101
    q = source.get_quotes(INDEX, "2007-01-01", "2007-02-28")

    assert isinstance(q.index, pd.DatetimeIndex)
    assert q.index.equals(pd.DatetimeIndex(idx))
    from src.data_provider import FIELDS

    assert list(q.columns) == list(FIELDS)
    np.testing.assert_allclose(q.turn.to_numpy(), basic.turnover_rate.to_numpy())
    np.testing.assert_allclose(q.volume.to_numpy(), daily.vol.to_numpy())
    np.testing.assert_allclose(q.preclose.to_numpy(), daily.pre_close.to_numpy())
    daily_calls = [kw for name, kw in client.calls if name == "index_daily"]
    assert daily_calls[0]["start_date"] == "20070115"
    basic_calls = [kw for name, kw in client.calls if name == "index_dailybasic"]
    assert basic_calls[0]["start_date"] == "20070115"
    with SAEStore(db, read_only=True) as store:
        stored = store.read_quotes(INDEX, "2007-01-01", "2007-03-01")
    assert len(stored) == len(daily)


def test_inner_join_drops_daily_only_head(tmp_path):
    """index_daily 独有的头部日期（basic 无换手率）不出现在返回与库中。"""
    from src.data_provider import TuShareIndexSource
    from src.store import SAEStore

    daily, _, idx = _make_raw()
    basic = daily.iloc[8:][["ts_code", "trade_date"]].assign(
        turnover_rate=np.linspace(0.8, 1.2, len(daily) - 8)
    )
    client = FakeClient(daily, basic, list_date="20070115")
    db = tmp_path / "ts.duckdb"
    source = TuShareIndexSource(db, client=client)

    q = source.get_quotes(INDEX, "2007-01-01", "2007-02-28")

    assert q.index.min() == idx[8]
    assert len(q) == len(daily) - 8
    with SAEStore(db, read_only=True) as store:
        stored = store.read_quotes(INDEX, "2007-01-01", "2007-02-28")
    assert stored.index.min() == idx[8]
    assert stored.index.equals(q.index)


def test_calendar_written_from_trade_cal(tmp_path):
    """日历表含 trade_cal 返回的全部会话，请求起点自 2004-01-01。"""
    from src.data_provider import TuShareIndexSource
    from src.store import SAEStore

    daily, basic, _ = _make_raw()
    cal = pd.Series(pd.bdate_range("2004-01-02", periods=1200).strftime("%Y%m%d"))
    client = FakeClient(daily, basic, list_date="20070115", cal=cal)
    db = tmp_path / "ts.duckdb"
    source = TuShareIndexSource(db, client=client)

    source.get_quotes(INDEX, "2007-01-01", "2007-02-28")

    cal_calls = [kw for name, kw in client.calls if name == "trade_cal"]
    assert cal_calls[0]["start_date"] == "20040101"
    with SAEStore(db, read_only=True) as store:
        sessions = store.read_calendar()
    expected = pd.DatetimeIndex(pd.to_datetime(cal, format="%Y%m%d"))
    expected = expected[expected <= pd.Timestamp("2007-02-28")]
    assert sessions.equals(expected)


DATAFEED_TOKEN_OK = "gw-token"


def _patch_datafeed(monkeypatch, outcome):
    """把 data_provider 里的 importlib 换成替身：outcome 为异常则抛出，否则返回假 DataFeed 模块。"""
    import types

    from src import data_provider as dp

    def import_module(name):
        assert name == dp.DATAFEED_TUSHARE
        if isinstance(outcome, BaseException):
            raise outcome
        module = types.ModuleType(name)
        module.TS_TOKEN = outcome

        class TuShare:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        module.TuShare = TuShare
        return module

    monkeypatch.setattr(dp, "importlib", types.SimpleNamespace(import_module=import_module))
    return dp


def _patch_tushare(monkeypatch, record: dict):
    """替身 tushare 模块：记录 set_token，pro_api 返回哨兵字符串。"""
    import sys
    import types

    module = types.ModuleType("tushare")
    module.set_token = lambda token: record.setdefault("token", token)
    module.pro_api = lambda: "official-pro"
    monkeypatch.setitem(sys.modules, "tushare", module)


def test_client_prefers_datafeed(tmp_path, monkeypatch):
    """DataFeed 可用时整体复用其封装（网关 token 只对网关有效），不触碰官方 tushare。"""
    import sys

    dp = _patch_datafeed(monkeypatch, DATAFEED_TOKEN_OK)
    monkeypatch.setitem(sys.modules, "tushare", None)  # 走到官方路径即 ImportError
    client = dp.TuShareIndexSource(tmp_path / "ts.duckdb").client
    assert client.kwargs == {"token": DATAFEED_TOKEN_OK, "max_retry": 3}


@pytest.mark.parametrize(
    "outcome",
    [ImportError("no DataFeed"), FileNotFoundError("api_config.ini"), KeyError("tushare"), ""],
    ids=["缺包", "缺ini", "缺tushare段", "token为空"],
)
def test_client_falls_back_to_env_token(tmp_path, monkeypatch, outcome):
    """DataFeed 导入期失败或 token 为空时，回退 tushare + 根 .env 的 TS_TOKEN。"""
    dp = _patch_datafeed(monkeypatch, outcome)
    record: dict = {}
    _patch_tushare(monkeypatch, record)
    monkeypatch.setenv("TS_TOKEN", "env-token")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    assert dp.TuShareIndexSource(tmp_path / "ts.duckdb").client == "official-pro"
    assert record == {"token": "env-token"}


def test_missing_token_raises_with_guidance(tmp_path, monkeypatch):
    """DataFeed 不可用且无 TS_TOKEN 时，经 get_quotes 报错并同时指引两条配置路径。"""
    dp = _patch_datafeed(monkeypatch, ImportError("no DataFeed"))
    monkeypatch.delenv("TS_TOKEN", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    source = dp.TuShareIndexSource(tmp_path / "ts.duckdb")
    with pytest.raises(ValueError, match="DataFeed.*TS_TOKEN"):
        source.get_quotes(INDEX, "2007-01-01", "2007-02-28")


@pytest.fixture
def warning_records():
    """收集 WARNING 及以上 loguru 日志，供告警断言。"""
    from loguru import logger

    records: list[str] = []
    handler_id = logger.add(lambda msg: records.append(str(msg)), level="WARNING")
    yield records
    logger.remove(handler_id)


def test_actual_start_later_than_list_date_warns_and_solidifies(tmp_path, warning_records):
    """实际起点（basic 自 2006 年末才有）晚于发布日：首拉告警并按实际起点固化。"""
    from src.data_provider import TuShareIndexSource
    from src.store import SAEStore

    daily, _, idx = _make_raw(days=700, start="2004-01-02")
    basic = daily.iloc[-30:][["ts_code", "trade_date"]].assign(
        turnover_rate=np.linspace(0.8, 1.2, 30)
    )
    client = FakeClient(daily, basic, list_date="20040102")
    db = tmp_path / "ts.duckdb"
    source = TuShareIndexSource(db, client=client)

    q = source.get_quotes(INDEX, "2004-01-01", "2006-12-31")

    actual_start = idx[-30]
    assert q.index.min() == actual_start
    assert any("换手率" in r and str(actual_start.date()) in r for r in warning_records)
    with SAEStore(db, read_only=True) as store:  # 固化：库内最早行即拼接有效起点
        stored = store.read_quotes(INDEX)
    assert stored.index.min() == actual_start


def test_request_before_solidified_start_raises_zero_calls(tmp_path):
    """库非空且请求起点早于固化起点：ValueError 含有效起点，fake client 零调用。"""
    from src.data_provider import TuShareIndexSource

    daily, basic, _ = _make_raw()
    client = FakeClient(daily, basic, list_date="20070115")
    source = TuShareIndexSource(tmp_path / "ts.duckdb", client=client)
    source.get_quotes(INDEX, "2007-01-15", "2007-02-28")  # 首拉固化起点 2007-01-15

    calls_before = len(client.calls)
    with pytest.raises(ValueError, match=f"有效起点.*{pd.Timestamp('2007-01-15').date()}"):
        source.get_quotes(INDEX, "2007-01-01", "2007-02-28")
    assert len(client.calls) == calls_before


def test_covered_range_served_from_cache_zero_calls(tmp_path):
    """请求区间被库内完全覆盖：直接读库返回，fake client 零调用，内容与库内一致。"""
    from src.data_provider import TuShareIndexSource
    from src.store import SAEStore

    daily, basic, _idx = _make_raw()
    client = FakeClient(daily, basic, list_date="20070115")
    db = tmp_path / "ts.duckdb"
    source = TuShareIndexSource(db, client=client)
    first = source.get_quotes(INDEX, "2007-01-15", "2007-02-28")

    calls_before = len(client.calls)
    hit = source.get_quotes(INDEX, "2007-01-18", "2007-02-15")  # 子区间
    same = source.get_quotes(INDEX, "2007-01-15", "2007-02-15")  # 库内完整区间
    assert len(client.calls) == calls_before
    pd.testing.assert_frame_equal(hit, first.loc["2007-01-18":"2007-02-15"])
    pd.testing.assert_frame_equal(same, first)
    with SAEStore(db, read_only=True) as store:
        stored = store.read_quotes(INDEX, "2007-01-18", "2007-02-15")
    pd.testing.assert_frame_equal(hit, stored)


def test_request_entirely_before_list_date_no_quote_calls(tmp_path):
    """请求区间整体早于发布日：显式报错，不触发两接口与日历调用。"""
    from src.data_provider import TuShareIndexSource

    daily, basic, _ = _make_raw()
    client = FakeClient(daily, basic, list_date="20070115")
    source = TuShareIndexSource(tmp_path / "ts.duckdb", client=client)

    with pytest.raises(ValueError, match="发布日"):
        source.get_quotes(INDEX, "2006-12-01", "2007-01-10")
    assert [name for name, _ in client.calls] == ["index_basic"]


def test_no_origin_warning_when_request_starts_after_publish(tmp_path, warning_records):
    """请求起点晚于发布日时实际起点晚于发布日属正常截断，不告警。"""
    from src.data_provider import TuShareIndexSource

    daily, basic, _ = _make_raw()
    client = FakeClient(daily, basic, list_date="20070101")
    source = TuShareIndexSource(tmp_path / "ts.duckdb", client=client)

    q = source.get_quotes(INDEX, "2007-01-15", "2007-02-28")

    assert q.index.min() == pd.Timestamp("2007-01-15")
    assert warning_records == []


def _quote_calls(client: FakeClient) -> list[tuple[str, dict]]:
    """过滤行情两接口（index_daily/index_dailybasic）的调用记录。"""
    return [(name, kw) for name, kw in client.calls if name in ("index_daily", "index_dailybasic")]


def test_tail_gap_fetches_only_missing_segment(tmp_path):
    """尾部缺段：仅对[库内末日+1, 请求终点]发起两接口调用，已有区间零重拉。"""
    from src.data_provider import TuShareIndexSource

    daily, basic, idx = _make_raw(days=40)  # 2007-01-15起40个交易日，末交易日2007-03-09
    client = FakeClient(daily, basic, list_date="20070115")
    source = TuShareIndexSource(tmp_path / "ts.duckdb", client=client)
    source.get_quotes(INDEX, "2007-01-15", "2007-02-15")  # 首拉前24个交易日

    client.calls.clear()
    q = source.get_quotes(INDEX, "2007-01-15", "2007-03-15")  # 缺尾段2007-02-16起

    calls = _quote_calls(client)
    assert len(calls) == 2  # 两接口各恰好一次，无已有区间重拉
    for _, kw in calls:
        assert kw["start_date"] == "20070216"
        assert kw["end_date"] == idx[-1].strftime("%Y%m%d")
    assert q.index.equals(pd.DatetimeIndex(idx))  # 合并后40日无缺无重


def test_head_gap_fetches_only_missing_segment(tmp_path):
    """头部缺段（首拉未从发布日起）：仅补[请求起点, 库内首日-1]，已有区间零重拉。"""
    from src.data_provider import TuShareIndexSource

    daily, basic, idx = _make_raw(days=170)
    client = FakeClient(daily, basic, list_date="20070115")
    source = TuShareIndexSource(tmp_path / "ts.duckdb", client=client)
    source.get_quotes(INDEX, "2007-06-01", "2007-07-31")  # 首拉未触发布日，未固化起点

    client.calls.clear()
    q = source.get_quotes(INDEX, "2007-03-01", "2007-07-31")  # 头部扩展到3月

    calls = _quote_calls(client)
    assert len(calls) == 2
    for _, kw in calls:
        assert kw["start_date"] == "20070301"
        assert kw["end_date"] == "20070531"  # 库内首日2007-06-01的前一自然日
    expected = idx[(idx >= "2007-03-01") & (idx <= "2007-07-31")]
    assert q.index.equals(expected)  # 补头后返回区间连续无缺日


def test_head_gap_to_publish_solidifies_and_blocks_earlier(tmp_path, warning_records):
    """头部扩展到发布日：按实际数据起点告警固化，此后更早请求零调用报错。"""
    from src.data_provider import TuShareIndexSource

    daily, _, _ = _make_raw(days=170)
    has_turn = daily["trade_date"] >= "20070402"  # 换手率自2007-04-02才有
    basic = daily.loc[has_turn, ["ts_code", "trade_date"]].assign(
        turnover_rate=np.linspace(0.8, 1.2, int(has_turn.sum()))
    )
    client = FakeClient(daily, basic, list_date="20070115")
    source = TuShareIndexSource(tmp_path / "ts.duckdb", client=client)
    source.get_quotes(INDEX, "2007-06-01", "2007-07-31")

    q = source.get_quotes(INDEX, "2007-01-01", "2007-07-31")  # 头部扩展到发布日

    assert q.index.min() == pd.Timestamp("2007-04-02")
    assert any("换手率" in r and "2007-04-02" in r for r in warning_records)
    calls_before = len(client.calls)
    with pytest.raises(ValueError, match="有效起点.*2007-04-02"):
        source.get_quotes(INDEX, "2007-03-01", "2007-07-31")
    assert len(client.calls) == calls_before


def test_head_gap_empty_solidifies_to_stored_min(tmp_path):
    """头部段拉回为空：实际起点即库内最早行，固化后更早请求零调用报错。"""
    from src.data_provider import TuShareIndexSource

    daily, basic, _ = _make_raw(days=60, start="2007-06-01")  # 数据自2007-06-01起
    cal = pd.Series(pd.bdate_range("2007-01-15", "2007-07-31").strftime("%Y%m%d"))
    client = FakeClient(daily, basic, list_date="20070115", cal=cal)
    source = TuShareIndexSource(tmp_path / "ts.duckdb", client=client)
    source.get_quotes(INDEX, "2007-06-01", "2007-07-31")

    q = source.get_quotes(INDEX, "2007-03-01", "2007-07-31")  # 头部[3-1, 5-31]无数据

    assert q.index.min() == pd.Timestamp("2007-06-01")  # 如实返回库内实际起点
    starts = [kw["start_date"] for _, kw in _quote_calls(client)]
    assert starts == ["20070601", "20070601", "20070301", "20070301"]  # 首拉+头段各一对
    calls_before = len(client.calls)
    with pytest.raises(ValueError, match="有效起点.*2007-06-01"):
        source.get_quotes(INDEX, "2007-04-01", "2007-07-31")
    assert len(client.calls) == calls_before


def test_long_range_segmented_within_limits(tmp_path):
    """8200交易日长区间：两接口按各自上限分段多次调用，合并无缺日无重复。"""
    import math

    from src.data_provider import TuShareIndexSource

    days = 8200
    daily, basic, idx = _make_raw(days=days, start="2004-01-02")
    client = FakeClient(daily, basic, list_date="20040102")
    source = TuShareIndexSource(tmp_path / "ts.duckdb", client=client)

    q = source.get_quotes(INDEX, "2004-01-02", idx[-1].strftime("%Y-%m-%d"))

    daily_calls = [kw for name, kw in client.calls if name == "index_daily"]
    basic_calls = [kw for name, kw in client.calls if name == "index_dailybasic"]
    assert len(daily_calls) == math.ceil(days / source.DAILY_MAX_ROWS)
    assert len(basic_calls) == math.ceil(days / source.BASIC_MAX_ROWS)
    for kw in daily_calls:
        in_range = daily["trade_date"].between(kw["start_date"], kw["end_date"])
        assert int(in_range.sum()) <= source.DAILY_MAX_ROWS
    for kw in basic_calls:
        in_range = basic["trade_date"].between(kw["start_date"], kw["end_date"])
        assert int(in_range.sum()) <= source.BASIC_MAX_ROWS
    assert len(q) == days
    assert q.index.is_unique
    assert q.index.equals(pd.DatetimeIndex(idx))


def test_end_beyond_latest_returns_actual_tail(tmp_path):
    """终点超库内最新且尾段无数据：自动补拉，正常返回库内实际末尾。"""
    from src.data_provider import TuShareIndexSource

    daily, basic, _ = _make_raw(days=24)  # 数据实际到2007-02-15
    cal = pd.Series(pd.bdate_range("2007-01-15", "2007-02-28").strftime("%Y%m%d"))
    client = FakeClient(daily, basic, list_date="20070115", cal=cal)
    source = TuShareIndexSource(tmp_path / "ts.duckdb", client=client)
    source.get_quotes(INDEX, "2007-01-15", "2007-02-15")

    q = source.get_quotes(INDEX, "2007-01-15", "2007-02-28")  # 尾段2-16起无数据

    assert q.index.max() == pd.Timestamp("2007-02-15")  # 返回库内实际末尾
    assert len(q) == 24
    tail = [kw for name, kw in client.calls if name == "index_daily" and kw["start_date"] == "20070216"]
    assert len(tail) == 1  # 确实对缺失尾段发起了补拉调用


def test_tail_gap_with_trading_days_retries_and_watermark_stalls(tmp_path):
    """尾部缺段内仍有交易日却拿不到数据：水位停在库内最新行，同区间会再次调用（不再静默截断）。"""
    from src.data_provider import TuShareIndexSource
    from src.store import SAEStore

    daily, basic, _ = _make_raw(days=24)  # 数据实际到2007-02-15
    # 独立日历显式收录2007-02-16至2007-02-28的交易日：尾部并非非交易日
    cal = pd.Series(pd.bdate_range("2007-01-15", "2007-02-28").strftime("%Y%m%d"))
    client = FakeClient(daily, basic, list_date="20070115", cal=cal)
    db = tmp_path / "ts.duckdb"
    source = TuShareIndexSource(db, client=client)
    source.get_quotes(INDEX, "2007-01-15", "2007-02-15")

    q = source.get_quotes(INDEX, "2007-01-15", "2007-02-28")  # 尾段2-16起有交易日但两接口拿不到数据

    assert q.index.max() == pd.Timestamp("2007-02-15")
    with SAEStore(db, read_only=True) as store:
        _, until = store.read_index_range(INDEX)
    assert until == pd.Timestamp("2007-02-15")  # 水位停在库内最新行，不写到请求终点

    calls_before = len(_quote_calls(client))
    again = source.get_quotes(INDEX, "2007-01-15", "2007-02-28")  # 同区间第二次请求
    assert len(_quote_calls(client)) > calls_before  # 尾部仍有缺失交易日，两接口再次被调用
    assert again.index.max() == pd.Timestamp("2007-02-15")


def test_watermark_capped_at_calendar_published_boundary(tmp_path):
    """请求终点远超trade_cal实际发布边界：水位钳制在该边界，而非跳到未发布的请求终点。

    复现：数据到2007-02-15，日历（``published_until``）只发布到2007-02-15，
    请求到2007-06-30。若不钳制，(last_row, end]区间内日历没有交易日会被
    误判为"尾部已覆盖"，水位直接跳到2007-06-30；之后日历真的发布到
    2007-03-30、数据也扩展到2007-03-30后，同样请求2007-03-30会因为
    水位(2007-06-30)>=请求终点而缓存命中，零调用静默返回只到
    2007-02-15的旧数据。
    """
    from src.data_provider import TuShareIndexSource
    from src.store import SAEStore

    daily, basic, _idx = _make_raw(days=24)  # 数据实际到2007-02-15
    client = FakeClient(daily, basic, list_date="20070115", published_until="20070215")
    db = tmp_path / "ts.duckdb"
    source = TuShareIndexSource(db, client=client)

    q = source.get_quotes(INDEX, "2007-01-15", "2007-06-30")  # 请求终点远超日历发布边界

    assert q.index.max() == pd.Timestamp("2007-02-15")
    with SAEStore(db, read_only=True) as store:
        _, until = store.read_index_range(INDEX)
    assert until == pd.Timestamp("2007-02-15")  # 水位钳制在日历发布边界，不越过未发布区间

    # 日历与数据都真实扩展到2007-03-30之后，同请求应重新触网并取到新数据
    daily2, basic2, idx2 = _make_raw(days=34, start="2007-01-15")
    client.daily, client.basic = daily2, basic2
    client.published_until = "20070330"
    calls_before = len(_quote_calls(client))
    q2 = source.get_quotes(INDEX, "2007-01-15", "2007-03-30")
    assert len(_quote_calls(client)) > calls_before  # 确实重新发起了两接口调用
    assert q2.index.max() == idx2.max()


def test_watermark_reaches_request_end_across_weekend(tmp_path):
    """请求终点是日历已发布的周末（真实休市日）：水位应如实推进到请求终点，不该卡在周五。

    回归用例（17eb613曾把这个场景搞错）：钳制若误用``sessions.max()``
    （只含开市日），会让水位停在周五——即使trade_cal早已发布到年底、
    真正确认了周末不开市——导致同一个"区间已完全覆盖"的请求每次都
    重新触发index_basic+trade_cal调用，而不是零调用缓存命中。
    """
    from src.data_provider import TuShareIndexSource
    from src.store import SAEStore

    daily, basic, idx = _make_raw(days=25)  # 数据实际到周五2007-02-16
    friday = idx[-1]
    assert friday.day_name() == "Friday"  # 前置条件：确保末尾确实是周五
    sunday = friday + pd.Timedelta(days=2)
    # 日历真实发布到年底，远超请求终点，证明周日没有交易日是"确认休市"
    # 而非"日历还没发布到那么远"。
    client = FakeClient(daily, basic, list_date="20070115", published_until="20071231")
    db = tmp_path / "ts.duckdb"
    source = TuShareIndexSource(db, client=client)

    q = source.get_quotes(INDEX, "2007-01-15", sunday.strftime("%Y-%m-%d"))

    assert q.index.max() == friday
    with SAEStore(db, read_only=True) as store:
        _, until = store.read_index_range(INDEX)
    assert until == sunday  # 水位如实推进到请求终点（周日），不卡在周五

    calls_before = len(client.calls)
    again = source.get_quotes(INDEX, "2007-01-15", sunday.strftime("%Y-%m-%d"))
    assert len(client.calls) == calls_before  # 区间已完全覆盖，零调用
    assert again.index.max() == friday


def test_ensure_calendar_incremental_http_failure_raises(tmp_path):
    """trade_cal增量请求遇HTTP失败（无列空表）：显式报错，不静默当成"日历未发布"。"""
    from src.data_provider import TuShareIndexSource
    from src.store import SAEStore

    daily, basic, _idx = _make_raw(days=24)  # 数据实际到2007-02-15
    client = FakeClient(daily, basic, list_date="20070115")
    db = tmp_path / "ts.duckdb"
    source = TuShareIndexSource(db, client=client)
    source.get_quotes(INDEX, "2007-01-15", "2007-02-15")  # 首拉成功，日历落库到2007-02-15

    with SAEStore(db, read_only=True) as store:
        before_until = store.read_index_range(INDEX)[1]
        before_rows = len(store.read_quotes(INDEX))

    # 数据源真的扩展了，但本轮trade_cal增量请求模拟网关HTTP失败
    daily2, basic2, _ = _make_raw(days=34, start="2007-01-15")
    client.daily, client.basic = daily2, basic2
    client.trade_cal_fails = True

    with pytest.raises(RuntimeError, match="trade_cal"):
        source.get_quotes(INDEX, "2007-01-15", "2007-03-30")

    with SAEStore(db, read_only=True) as store:  # 水位与已落库行情都未被污染
        assert store.read_index_range(INDEX)[1] == before_until
        assert len(store.read_quotes(INDEX)) == before_rows


def test_first_fetch_with_descending_trade_cal_produces_ascending_chunks(tmp_path, monkeypatch):
    """网关trade_cal按cal_date降序返回：首拉日历切块仍须升序，起止不倒置、不重叠、覆盖全区间。"""
    from src.data_provider import TuShareIndexSource

    days = 50
    daily, basic, idx = _make_raw(days=days, start="2007-01-15")
    client = FakeClient(daily, basic, list_date="20070115", descending=True)
    source = TuShareIndexSource(tmp_path / "ts.duckdb", client=client)
    monkeypatch.setattr(source, "DAILY_MAX_ROWS", 10)
    monkeypatch.setattr(source, "BASIC_MAX_ROWS", 7)

    q = source.get_quotes(INDEX, "2007-01-15", idx[-1].strftime("%Y-%m-%d"))

    daily_calls = [kw for name, kw in client.calls if name == "index_daily"]
    basic_calls = [kw for name, kw in client.calls if name == "index_dailybasic"]
    assert daily_calls and basic_calls

    def assert_ascending_non_overlapping_coverage(calls, expected_start, expected_end):
        starts = [kw["start_date"] for kw in calls]
        ends = [kw["end_date"] for kw in calls]
        for s, e in zip(starts, ends):
            assert s <= e  # 单块起止未倒置
        assert starts == sorted(starts)  # 块与块之间升序排列
        assert starts[0] == expected_start
        assert ends[-1] == expected_end
        for prev_end, next_start in zip(ends, starts[1:]):
            assert next_start > prev_end  # 不重叠、首尾相接覆盖全区间

    expected_start, expected_end = idx[0].strftime("%Y%m%d"), idx[-1].strftime("%Y%m%d")
    assert_ascending_non_overlapping_coverage(daily_calls, expected_start, expected_end)
    assert_ascending_non_overlapping_coverage(basic_calls, expected_start, expected_end)
    assert len(q) == days
    assert q.index.equals(pd.DatetimeIndex(idx))


def test_tail_gap_only_non_trading_days_zero_calls_watermark_at_end(tmp_path):
    """尾部缺段内只剩非交易日（日历里没有）：水位写到请求终点，同区间第二次请求零调用。"""
    from src.data_provider import TuShareIndexSource
    from src.store import SAEStore

    daily, basic, idx = _make_raw(days=24)  # 数据实际到2007-02-15
    # 独立日历只收录实际有数据的24个交易日：2007-02-16之后没有任何交易日
    cal = pd.Series(idx.strftime("%Y%m%d"))
    client = FakeClient(daily, basic, list_date="20070115", cal=cal)
    db = tmp_path / "ts.duckdb"
    source = TuShareIndexSource(db, client=client)
    source.get_quotes(INDEX, "2007-01-15", "2007-02-15")

    q = source.get_quotes(INDEX, "2007-01-15", "2007-02-20")  # 尾段2-16起全是非交易日

    assert q.index.max() == pd.Timestamp("2007-02-15")
    with SAEStore(db, read_only=True) as store:
        _, until = store.read_index_range(INDEX)
    assert until == pd.Timestamp("2007-02-20")  # 尾部确实已覆盖，水位如实写到请求终点

    calls_before = len(client.calls)
    again = source.get_quotes(INDEX, "2007-01-15", "2007-02-20")  # 同区间第二次请求
    assert len(client.calls) == calls_before  # 缓存命中，零调用（含index_basic等全部接口）
    assert again.index.max() == pd.Timestamp("2007-02-15")


def test_refetch_after_backfill_hits_cache_zero_calls(tmp_path):
    """尾部补拉落库后再次请求同区间：命中缓存零调用，内容一致。"""
    from src.data_provider import TuShareIndexSource

    daily, basic, _ = _make_raw(days=40)
    client = FakeClient(daily, basic, list_date="20070115")
    source = TuShareIndexSource(tmp_path / "ts.duckdb", client=client)
    source.get_quotes(INDEX, "2007-01-15", "2007-02-15")
    q = source.get_quotes(INDEX, "2007-01-15", "2007-03-15")  # 补拉尾段

    calls_before = len(client.calls)
    again = source.get_quotes(INDEX, "2007-01-15", "2007-03-15")

    assert len(client.calls) == calls_before
    pd.testing.assert_frame_equal(again, q)


def test_ensure_coverage_clamps_request_to_solidified_start(tmp_path):
    """ensure_index_coverage：请求起点早于固化起点时钳制后补拉，不触发越界报错。"""
    from src.data_provider import (
        TuShareIndexSource,
        ensure_index_coverage,
    )

    daily, basic, _ = _make_raw(days=40)
    client = FakeClient(daily, basic, list_date="20070115")
    db = tmp_path / "ts.duckdb"
    source = TuShareIndexSource(db, client=client)
    source.get_quotes(INDEX, "2007-01-15", "2007-02-15")  # 首拉固化起点 2007-01-15

    client.calls.clear()
    q = ensure_index_coverage(db, INDEX, "2005-01-01", "2007-03-15", client=client)

    # 直接 get_quotes 会因 2005-01-01 早于固化起点报错；经 ensure 钳制到 2007-01-15 后仅补尾段
    starts = {kw["start_date"] for _, kw in _quote_calls(client)}
    assert starts == {"20070216"}
    assert q.index.min() == pd.Timestamp("2007-01-15")
    assert q.index.max() == pd.Timestamp(daily["trade_date"].iloc[-1])


def test_ensure_coverage_without_db_passes_request_through(tmp_path):
    """ensure_index_coverage：库不存在时不钳制，透传请求起点给行情源。"""
    from src.data_provider import ensure_index_coverage

    daily, basic, _ = _make_raw()
    client = FakeClient(daily, basic, list_date="20070115")

    q = ensure_index_coverage(
        tmp_path / "ts.duckdb", INDEX, "2007-01-01", "2007-02-28", client=client
    )

    starts = [kw["start_date"] for name, kw in client.calls if name == "index_daily"]
    assert starts == ["20070115"]  # 透传后由行情源截断到发布日
    assert q.index.min() == pd.Timestamp("2007-01-15")


def test_end_before_start_raises_without_side_effects(tmp_path):
    """区间倒置直接报错，不创建库、不发起任何调用。"""
    from src.data_provider import TuShareIndexSource

    daily, basic, _ = _make_raw()
    client = FakeClient(daily, basic, list_date="20070115")
    source = TuShareIndexSource(tmp_path / "ts.duckdb", client=client)

    with pytest.raises(ValueError, match="早于开始日.*区间无效"):
        source.get_quotes(INDEX, "2007-02-01", "2007-01-01")
    assert client.calls == []
    assert not (tmp_path / "ts.duckdb").exists()


def test_first_fetch_without_sessions_raises_value_error_without_quote_calls(tmp_path):
    """首拉区间全为休市日：按日历如实报「无交易日」，不误导为TS代码无数据或让用户重试。"""
    from src.data_provider import TuShareIndexSource

    daily, basic, _ = _make_raw()
    client = FakeClient(daily, basic, list_date="20070115")
    source = TuShareIndexSource(tmp_path / "ts.duckdb", client=client)

    with pytest.raises(ValueError, match="无交易日"):
        source.get_quotes(INDEX, "2007-01-20", "2007-01-21")  # 周六至周日
    assert not [name for name, _ in client.calls if name in ("index_daily", "index_dailybasic")]


def test_index_basic_empty_raises_with_guidance(tmp_path):
    """无效TS代码或无接口权限时index_basic空返回：带指引报错而非裸IndexError。"""
    from src.data_provider import TuShareIndexSource

    daily, basic, _ = _make_raw()
    client = FakeClient(daily, basic, list_date="20070115")
    client.index_basic = lambda **kwargs: pd.DataFrame()
    source = TuShareIndexSource(tmp_path / "ts.duckdb", client=client)

    with pytest.raises(ValueError, match="index_basic未返回指数000905.SH元数据"):
        source.get_quotes(INDEX, "2007-01-15", "2007-02-15")


def test_legacy_quotes_without_range_record_warns_migration(tmp_path, warning_records):
    """0.3.0旧库：有存量行情但无拉取水位记录——告警提示口径混合风险。"""
    from src.data_provider import TuShareIndexSource
    from src.store import SAEStore

    daily, basic, _idx = _make_raw()
    legacy = quotes(30)  # Qlib口径存量行情，仅落market_quotes不落水位
    db = tmp_path / "legacy.duckdb"
    with SAEStore(db) as store:
        store.save_market_data(INDEX, legacy, legacy.index)

    source = TuShareIndexSource(db, client=FakeClient(daily, basic, list_date="20070115"))
    source.get_quotes(INDEX, str(legacy.index[0].date()), str(legacy.index[-1].date()))

    assert any("0.3.0" in r and "口径" in r for r in warning_records)


def test_retry_after_midway_failure_is_idempotent(tmp_path):
    """中途网络失败后重试：首段数据完好、水位未推进、只补尾部缺段。"""
    from src.data_provider import TuShareIndexSource
    from src.store import SAEStore

    daily, basic, idx = _make_raw()
    client = FakeClient(daily, basic, list_date="20070115")
    db = tmp_path / "ts.duckdb"
    source = TuShareIndexSource(db, client=client)
    source.get_quotes(INDEX, "2007-01-15", str(idx[-1].date()))  # 首拉24日

    daily2, basic2, idx2 = _make_raw(days=30)  # 数据延长至30日
    client.daily, client.basic = daily2, basic2
    original_daily = client.index_daily
    client.index_daily = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("网络中断"))
    with pytest.raises(RuntimeError, match="网络中断"):
        source.get_quotes(INDEX, "2007-01-15", str(idx2[-1].date()))
    with SAEStore(db, read_only=True) as store:  # 失败后：库内仍只有首拉段
        stored = store.read_quotes(INDEX, "2004-01-01", "2030-01-01")
    assert stored.index.max() == idx[-1]

    client.index_daily = original_daily
    client.calls.clear()
    q = source.get_quotes(INDEX, "2007-01-15", str(idx2[-1].date()))
    assert q.index.max() == idx2[-1]
    retry_starts = [kw["start_date"] for name, kw in client.calls if name == "index_daily"]
    # 重试只拉尾部缺段（起点为首拉末日之后），首段零重拉
    assert retry_starts and all(s > idx[-1].strftime("%Y%m%d") for s in retry_starts)


def test_tail_backfill_empty_columns_response_raises_and_state_unchanged(tmp_path):
    """尾部补拉时index_dailybasic返回无列空表（HTTP失败特征）：报错，水位与库内行数不变。"""
    from src.data_provider import TuShareIndexSource
    from src.store import SAEStore

    daily, basic, _ = _make_raw(days=24)  # 数据实际到2007-02-15
    # 独立日历显式收录2007-02-16至2007-03-15的交易日，确保尾段会真正发起两接口调用
    cal = pd.Series(pd.bdate_range("2007-01-15", "2007-03-15").strftime("%Y%m%d"))
    client = FakeClient(daily, basic, list_date="20070115", cal=cal)
    db = tmp_path / "ts.duckdb"
    source = TuShareIndexSource(db, client=client)
    source.get_quotes(INDEX, "2007-01-15", "2007-02-15")  # 首拉24日

    with SAEStore(db, read_only=True) as store:
        range_before = store.read_index_range(INDEX)
        rows_before = len(store.read_quotes(INDEX))

    client.index_dailybasic = lambda **kwargs: pd.DataFrame()  # 无列空表，模拟网关429/5xx被吞成空表
    with pytest.raises(RuntimeError, match="index_dailybasic.*无列空表"):
        source.get_quotes(INDEX, "2007-01-15", "2007-03-15")  # 尾段2-16起触发补拉

    with SAEStore(db, read_only=True) as store:
        range_after = store.read_index_range(INDEX)
        rows_after = len(store.read_quotes(INDEX))
    assert range_after == range_before  # 水位未推进
    assert rows_after == rows_before  # 未误写入任何行


def test_first_fetch_first_basic_chunk_empty_columns_raises_before_any_write(tmp_path):
    """首拉basic分两块，第一块无列空表：报错，库内该指数无任何行情、水位为空。"""
    from src.data_provider import TuShareIndexSource
    from src.store import SAEStore

    daily, basic, _ = _make_raw(days=24)
    client = FakeClient(daily, basic, list_date="20070115")
    db = tmp_path / "ts.duckdb"
    source = TuShareIndexSource(db, client=client)
    source.BASIC_MAX_ROWS = 12  # 实例属性覆盖：强制24个交易日的basic恰好分两块拉取

    original_basic = client.index_dailybasic
    calls = {"n": 0}

    def flaky_basic(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:  # 第一块模拟HTTP失败特征（无列空表）
            client.calls.append(("index_dailybasic", kwargs))
            return pd.DataFrame()
        return original_basic(**kwargs)

    client.index_dailybasic = flaky_basic

    with pytest.raises(RuntimeError, match="index_dailybasic.*无列空表"):
        source.get_quotes(INDEX, "2007-01-15", "2007-02-15")

    with SAEStore(db, read_only=True) as store:
        assert store.read_index_range(INDEX) == (None, None)
        assert store.read_quotes(INDEX).empty


def test_first_fetch_zero_rows_raises_instead_of_silent_empty(tmp_path):
    """首拉两接口均合法返回（带列）但0行：不再返回空表，而是报错、不写水位。"""
    from src.data_provider import TuShareIndexSource
    from src.store import SAEStore

    daily, basic, _ = _make_raw(days=24)
    client = FakeClient(daily, basic, list_date="20070115")
    empty_daily, empty_basic = daily.iloc[0:0], basic.iloc[0:0]  # 带列、0行的合法空返回

    def legacy_daily(**kwargs):
        client.calls.append(("index_daily", kwargs))
        return empty_daily

    def legacy_basic(**kwargs):
        client.calls.append(("index_dailybasic", kwargs))
        return empty_basic

    client.index_daily = legacy_daily
    client.index_dailybasic = legacy_basic
    db = tmp_path / "ts.duckdb"
    source = TuShareIndexSource(db, client=client)

    with pytest.raises(RuntimeError, match="未获取到任何行情"):
        source.get_quotes(INDEX, "2007-01-15", "2007-02-15")

    with SAEStore(db, read_only=True) as store:
        assert store.read_index_range(INDEX) == (None, None)
        assert store.read_quotes(INDEX).empty


def test_cache_hit_coexists_with_concurrent_read_only_holder(tmp_path):
    """已迁移库被并发只读连接持有时，行情源构造与缓存命中不取写锁、不失败。"""
    import duckdb

    from src.data_provider import TuShareIndexSource

    daily, basic, _ = _make_raw(days=30)
    client = FakeClient(daily, basic, list_date="20070115")
    db = tmp_path / "research.duckdb"
    TuShareIndexSource(db, client=client).get_quotes(INDEX, "2007-01-15", "2007-02-28")
    calls_before = len(client.calls)
    # 模拟另一notebook持有的只读连接：DuckDB多读可共存、读写互斥
    holder = duckdb.connect(str(db), read_only=True)
    try:
        source = TuShareIndexSource(db, client=client)  # 构造只探测、不拿写锁
        fetched = source.get_quotes(INDEX, "2007-01-15", "2007-02-28")  # 缓存命中
        assert len(fetched) == 30
        assert len(client.calls) == calls_before  # 零网络调用
    finally:
        holder.close()
