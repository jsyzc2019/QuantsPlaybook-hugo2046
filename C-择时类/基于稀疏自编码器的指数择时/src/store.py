"""SAE研究数据的DuckDB边界；短连接、原始行情与实验快照分离。"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from types import TracebackType

import duckdb
import numpy as np
import pandas as pd
from typing_extensions import Self

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "hy_sae_timing.duckdb"
QUOTE_FIELDS = ("open", "high", "low", "close", "preclose", "volume", "amount", "turn")


def _json_text(value: object) -> str:
    def clean(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(v) for v in item]
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        if item is pd.NaT or item is pd.NA:
            return None
        if isinstance(item, (Path, pd.Timestamp, pd.Timedelta)):
            return str(item)
        return item

    return json.dumps(clean(value), ensure_ascii=False, allow_nan=False)


def _identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _table_name(name: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name):
        raise ValueError("实验表名须为小写字母/数字/下划线，长度不超过64")
    return "run_" + name


class SAEStore:
    """本模块DuckDB存储；用with语句及时释放文件锁。

    :param db_path: 数据库文件路径。
    :param read_only: 只读打开已存在的数据库。
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, *, read_only: bool = False) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        # 产物登记以 DB 所在目录为锚，data/ 连库带产物可整体搬迁（0.14.0）
        self._artifact_root = self.db_path.parent
        if not read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self.db_path), read_only=read_only)
        if not read_only:
            self._con.execute("""
                CREATE TABLE IF NOT EXISTS market_quotes (
                    code VARCHAR, datetime TIMESTAMP,
                    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, preclose DOUBLE,
                    volume DOUBLE, amount DOUBLE, turn DOUBLE,
                    PRIMARY KEY (code, datetime)
                );
                CREATE TABLE IF NOT EXISTS calendar_sessions (datetime TIMESTAMP PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS index_ranges (
                    code VARCHAR PRIMARY KEY,
                    first_date TIMESTAMP,
                    fetched_until TIMESTAMP NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id VARCHAR PRIMARY KEY, created_at TIMESTAMP DEFAULT current_timestamp,
                    config JSON NOT NULL, metadata JSON NOT NULL, artifact_dir VARCHAR
                );
                CREATE TABLE IF NOT EXISTS run_tables (
                    run_id VARCHAR, dataset VARCHAR, columns_json JSON, dtypes_json JSON,
                    row_count BIGINT, PRIMARY KEY (run_id, dataset)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    run_id VARCHAR, name VARCHAR, path VARCHAR, sha256 VARCHAR,
                    PRIMARY KEY (run_id, name)
                );
            """)
            # 0.6.0列名instrument更名为code；旧库就地改名迁移，保住已拉取行情与水位
            for table in ("market_quotes", "index_ranges"):
                legacy = self._con.execute(
                    "SELECT 1 FROM information_schema.columns"
                    " WHERE table_name=? AND column_name='instrument'",
                    [table],
                ).fetchone()
                if legacy:
                    self._con.execute(f"ALTER TABLE {table} RENAME COLUMN instrument TO code")
            # 0.14.0：产物登记由绝对路径改为相对DB目录，旧库开库时就地迁移
            self._migrate_artifact_paths()

    def _store_path(self, path: str | Path) -> str:
        """产物路径入库前的规范化：库目录树之内存相对，之外存绝对。

        :param path: 产物文件或目录路径。
        :returns: 相对于DB所在目录的路径；不在该目录树下时返回绝对路径。
        """
        resolved = Path(path).expanduser().resolve()
        try:
            return resolved.relative_to(self._artifact_root).as_posix()
        except ValueError:
            # 库目录树之外的产物保持绝对，不写成 ../.. ——那种路径搬迁后同样会断
            return str(resolved)

    def _resolve_path(self, stored: str) -> Path:
        """把登记值还原成本机绝对路径。

        :param stored: ``artifacts.path`` 或 ``runs.artifact_dir`` 的登记值。
        :returns: 绝对路径；登记值本就是绝对路径时原样返回。
        """
        path = Path(stored)
        return path if path.is_absolute() else self._artifact_root / path

    def _migrate_artifact_paths(self) -> None:
        """把旧库的绝对路径登记就地改写为相对DB目录的形式。

        0.14.0 之前登记的是绝对路径，一旦 ``data/`` 被搬走或 worktree 被删除即断链。
        改写规则：取路径的最长后缀，使其在当前DB目录下确实存在；**找不到对应文件的
        行原样保留**，不凭字符串猜测制造假登记。

        :returns: 无返回值。
        """
        rows = self._con.execute(
            "SELECT run_id, name, path FROM artifacts WHERE starts_with(path, '/')"
        ).fetchall()
        for run_id, name, stored in rows:
            relative = self._relative_suffix(stored)
            if relative is not None:
                self._con.execute(
                    "UPDATE artifacts SET path=? WHERE run_id=? AND name=?",
                    [relative, run_id, name],
                )
        rows = self._con.execute(
            "SELECT run_id, artifact_dir FROM runs"
            " WHERE artifact_dir IS NOT NULL AND starts_with(artifact_dir, '/')"
        ).fetchall()
        for run_id, stored in rows:
            relative = self._relative_suffix(stored)
            if relative is not None:
                self._con.execute(
                    "UPDATE runs SET artifact_dir=? WHERE run_id=?", [relative, run_id]
                )

    def _relative_suffix(self, stored: str) -> str | None:
        """在DB目录下寻找与登记路径匹配的最长存在后缀。

        :param stored: 绝对路径登记值。
        :returns: 相对路径字符串；DB目录下找不到对应文件时返回None。
        """
        parts = Path(stored).parts[1:]
        for start in range(len(parts)):
            candidate = Path(*parts[start:])
            if (self._artifact_root / candidate).exists():
                return candidate.as_posix()
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """关闭连接，释放数据库文件锁。

        :returns: 无返回值。
        """
        self._con.close()

    def save_market_data(
        self, code: str, raw: pd.DataFrame, sessions: pd.DatetimeIndex
    ) -> None:
        """在同一事务中更新原始行情和同源交易日历。

        :param code: 指数代码。
        :param raw: 唯一升序DatetimeIndex、8个原始字段；turn允许缺失。
        :param sessions: 与行情同源的权威交易日历（如``trade_cal``日历）。
        :returns: 无返回值；重复日期更新，不重复添加。
        :raises ValueError: 代理行情、非法日期或日历覆盖不足。
        """
        if raw.attrs.get("turnover_policy") == "volume_proxy":
            raise ValueError("代理行情不能写入原始market_quotes")
        if raw.empty or not isinstance(raw.index, pd.DatetimeIndex) or raw.index.hasnans:
            raise ValueError("原始行情必须使用非空DatetimeIndex")
        if not raw.index.is_unique or not raw.index.is_monotonic_increasing:
            raise ValueError("原始行情日期必须唯一升序")
        sessions = pd.DatetimeIndex(sessions).sort_values().unique()
        expected = sessions[(sessions >= raw.index.min()) & (sessions <= raw.index.max())]
        if sessions.hasnans or not expected.equals(raw.index):
            raise ValueError("行情与同源交易日历未对齐")
        frame = raw.loc[:, list(QUOTE_FIELDS)].astype(float).rename_axis("datetime").reset_index()
        frame.insert(0, "code", code)
        self._con.register("_quotes", frame)
        try:
            self._con.execute("BEGIN")
            self._con.execute("INSERT OR REPLACE INTO market_quotes SELECT * FROM _quotes")
            self._insert_calendar(sessions)
            self._con.execute("COMMIT")
        except Exception:
            self._con.execute("ROLLBACK")
            raise
        finally:
            self._con.unregister("_quotes")

    def _insert_calendar(self, sessions: pd.DatetimeIndex) -> None:
        """在当前事务内增量写入交易日历；重复日期被忽略。

        :param sessions: 交易日序列。
        :returns: 无返回值。
        """
        calendar = pd.DataFrame({"datetime": pd.DatetimeIndex(sessions).sort_values().unique()})
        self._con.register("_calendar", calendar)
        try:
            self._con.execute("INSERT OR IGNORE INTO calendar_sessions SELECT * FROM _calendar")
        finally:
            self._con.unregister("_calendar")

    def save_calendar(self, sessions: pd.DatetimeIndex) -> None:
        """独立增量写入交易日历，供行情无增量时日历仍前进。

        :param sessions: 交易日序列；重复日期被忽略。
        :returns: 无返回值。
        """
        self._con.execute("BEGIN")
        try:
            self._insert_calendar(sessions)
            self._con.execute("COMMIT")
        except Exception:
            self._con.execute("ROLLBACK")
            raise

    def read_index_range(self, code: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """读单指数的固化起点与拉取水位。

        :param code: 指数代码。
        :returns: (固化起点, 拉取水位)；从未拉取或旧库无表时为(None, None)。
        """
        try:
            row = self._con.execute(
                "SELECT first_date, fetched_until FROM index_ranges WHERE code = ?",
                [code],
            ).fetchone()
        except duckdb.Error:  # 版本升级前的旧库没有index_ranges表
            return None, None
        if row is None:
            return None, None
        return (pd.Timestamp(row[0]) if row[0] is not None else None, pd.Timestamp(row[1]))

    def save_index_range(
        self, code: str, first_date: pd.Timestamp | None, fetched_until: pd.Timestamp
    ) -> None:
        """合并保存单指数拉取覆盖区间：起点取更早、水位取更晚后upsert。

        :param code: 指数代码。
        :param first_date: 本轮验证到的实际数据起点；None表示本轮无新证据。
        :param fetched_until: 本轮拉取抵达的请求终点，尾段无数据也推进。
        :returns: 无返回值。
        """
        old_first, old_until = self.read_index_range(code)
        candidates = [d for d in (first_date, old_first) if d is not None]
        first = min(candidates) if candidates else None
        until = max(fetched_until, old_until) if old_until is not None else fetched_until
        self._con.execute(
            "INSERT OR REPLACE INTO index_ranges VALUES (?, ?, ?)", [code, first, until]
        )

    def read_quotes(
        self, code: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """从数据库读取单指数原始行情，不应用换手率代理或填充。

        :param code: 指数代码。
        :param start: 可选开始日期，含当天。
        :param end: 可选结束日期，含当天。
        :returns: DatetimeIndex×8字段DataFrame。
        """
        frame = self._con.execute(
            """
            SELECT datetime, open, high, low, close, preclose, volume, amount, turn
            FROM market_quotes WHERE code = ?
            AND (? IS NULL OR datetime >= CAST(? AS TIMESTAMP))
            AND (? IS NULL OR datetime <= CAST(? AS TIMESTAMP)) ORDER BY datetime
        """,
            [code, start, start, end, end],
        ).df()
        frame["datetime"] = frame["datetime"].astype("datetime64[ns]")
        return frame.set_index("datetime")

    def read_calendar(self) -> pd.DatetimeIndex:
        """读取已落库的同源交易日历。

        :returns: 升序、唯一的DatetimeIndex。
        """
        frame = self._con.execute("SELECT datetime FROM calendar_sessions ORDER BY datetime").df()
        return pd.DatetimeIndex(frame["datetime"].astype("datetime64[ns]"), name="datetime")

    def save_run(
        self,
        run_id: str,
        config: dict,
        metadata: dict,
        tables: Mapping[str, pd.DataFrame],
        *,
        artifacts: Mapping[str, Path] | None = None,
        artifact_dir: str | Path | None = None,
    ) -> None:
        """原子保存完整实验；重复run_id拒绝覆盖。

        :param run_id: 用户可读且非空的唯一实验标识。
        :param config: 实验配置，JSON可序列化。
        :param metadata: 数据口径、指纹与汇总等元信息。
        :param tables: 表名到普通DataFrame；索引需先reset_index成为列。
        :param artifacts: 模型、Portfolio或图片的文件名到路径。
        :param artifact_dir: 本轮文件目录。
        :returns: 无返回值；全部表及目录在同一事务提交。
        :raises ValueError: 重复run_id、表名/列/索引不合法。
        """
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id不能为空")
        if self._con.execute("SELECT 1 FROM runs WHERE run_id=?", [run_id]).fetchone():
            raise ValueError(f"实验已存在：{run_id}")
        self._con.execute("BEGIN")
        try:
            self._con.execute(
                "INSERT INTO runs(run_id,config,metadata,artifact_dir) VALUES (?,?,?,?)",
                [
                    run_id,
                    _json_text(config),
                    _json_text(metadata),
                    self._store_path(artifact_dir) if artifact_dir else None,
                ],
            )
            for name, frame in tables.items():
                self._append_frame(run_id, name, frame)
            for name, path in (artifacts or {}).items():
                path = Path(path).resolve()
                digest = sha256(path.read_bytes()).hexdigest()
                self._con.execute(
                    "INSERT INTO artifacts VALUES (?,?,?,?)",
                    [run_id, name, self._store_path(path), digest],
                )
            self._con.execute("COMMIT")
        except Exception:
            self._con.execute("ROLLBACK")
            raise

    def _append_frame(self, run_id: str, name: str, frame: pd.DataFrame) -> None:
        table = _table_name(name)
        columns = frame.columns.tolist()
        if (
            not columns
            or any(not isinstance(c, str) for c in columns)
            or not frame.columns.is_unique
        ):
            raise ValueError("实验DataFrame须有唯一字符串列名")
        if {"run_id", "_row"} & set(columns):
            raise ValueError("run_id和_row为存储保留列")
        if not frame.index.equals(pd.RangeIndex(len(frame))):
            raise ValueError("实验表请先reset_index，将索引显式保存为列")
        incoming = frame.copy()
        incoming.insert(0, "_row", np.arange(len(frame), dtype=np.int64))
        incoming.insert(0, "run_id", run_id)
        for column in columns:
            if pd.api.types.is_object_dtype(incoming[column].dtype):
                incoming[column] = incoming[column].astype("string")
        self._con.register("_incoming", incoming)
        try:
            self._con.execute(
                f"CREATE TABLE IF NOT EXISTS {_identifier(table)} AS SELECT * FROM _incoming WHERE FALSE"
            )
            existing = {
                row[0] for row in self._con.execute(f"DESCRIBE {_identifier(table)}").fetchall()
            }
            # 不同实验的seed数可变；按名字扩展列，旧实验读取仍只返回自己的列。
            for column, dtype, *_ in self._con.execute(
                "DESCRIBE SELECT * FROM _incoming"
            ).fetchall():
                if column not in existing:
                    self._con.execute(
                        f"ALTER TABLE {_identifier(table)} ADD COLUMN {_identifier(column)} {dtype}"
                    )
            self._con.execute(f"INSERT INTO {_identifier(table)} BY NAME SELECT * FROM _incoming")
            self._con.execute(
                "INSERT INTO run_tables VALUES (?,?,?,?,?)",
                [
                    run_id,
                    name,
                    _json_text(columns),
                    _json_text({c: str(frame[c].dtype) for c in columns}),
                    len(frame),
                ],
            )
        finally:
            self._con.unregister("_incoming")

    def list_runs(self) -> pd.DataFrame:
        """列出实验ID、时间、配置、元数据与文件目录。

        :returns: 每行一个已完整提交实验的DataFrame。
        """
        return self._con.execute("SELECT * FROM runs ORDER BY created_at, run_id").df()

    def run_metadata(self, run_id: str) -> dict:
        """读取实验配置与元数据。

        :param run_id: 实验ID。
        :returns: 含config/metadata/artifact_dir的字典。
        :raises KeyError: 实验不存在。
        """
        row = self._con.execute(
            "SELECT config,metadata,artifact_dir FROM runs WHERE run_id=?", [run_id]
        ).fetchone()
        if row is None:
            raise KeyError(f"实验不存在：{run_id}")
        return {
            "run_id": run_id,
            "config": json.loads(row[0]),
            "metadata": json.loads(row[1]),
            "artifact_dir": str(self._resolve_path(row[2])) if row[2] else None,
        }

    def read_table(self, run_id: str, name: str) -> pd.DataFrame:
        """按run_id读取结构化实验表，恢复列顺序和pandas类型。

        :param run_id: 实验ID。
        :param name: 写入时的表名，例如predictions或losses。
        :returns: 普通DataFrame；日期/标的为显式列。
        :raises KeyError: 实验中不存在该表。
        """
        table = _table_name(name)
        info = self._con.execute(
            "SELECT columns_json,dtypes_json FROM run_tables WHERE run_id=? AND dataset=?",
            [run_id, name],
        ).fetchone()
        if info is None:
            raise KeyError(f"{run_id}没有表{name}")
        columns, dtypes = json.loads(info[0]), json.loads(info[1])
        projection = ",".join(_identifier(c) for c in columns)
        frame = self._con.execute(
            f"SELECT {projection} FROM {_identifier(table)} WHERE run_id=? ORDER BY _row", [run_id]
        ).df()
        return frame.astype(dtypes)

    def read_predictions(self, run_id: str) -> pd.DataFrame:
        """读取标准长表预测，可直接继续阈值与回测。

        :param run_id: 实验ID。
        :returns: datetime/code MultiIndex、各seed及score列。
        """
        frame = self.read_table(run_id, "predictions")
        # 0.6.0前旧实验的predictions列名为instrument，读取时兼容回退；
        # 归一层名为code，保证新旧实验索引层一致、可直接xs(level="code")
        level = "code" if "code" in frame.columns else "instrument"
        return frame.set_index(["datetime", level]).rename_axis(index=["datetime", "code"])

    def artifact_path(self, run_id: str, name: str) -> Path:
        """查找模型、Portfolio或图片文件的原生路径。

        :param run_id: 实验ID。
        :param name: 保存时登记的名称，如portfolio_long_short.pkl。
        :returns: 本机文件绝对路径。
        :raises KeyError: 未登记该产物。
        """
        row = self._con.execute(
            "SELECT path FROM artifacts WHERE run_id=? AND name=?", [run_id, name]
        ).fetchone()
        if row is None:
            raise KeyError(f"{run_id}没有产物{name}")
        return self._resolve_path(row[0])

    def query(self, sql: str, parameters: list | None = None) -> pd.DataFrame:
        """执行只读SELECT查询，支持参数绑定。

        :param sql: 单条SELECT语句。
        :param parameters: 可选参数值列表。
        :returns: SQL结果DataFrame。
        :raises ValueError: 多语句或非SELECT请求。
        """
        statements = self._con.extract_statements(sql)
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            raise ValueError("query只接受单条SELECT")
        return self._con.execute(sql, parameters or []).df()
