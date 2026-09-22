"""真实DuckDB公开接口的持久化和隔离约束。"""

import numpy as np
import pandas as pd
import pytest

from .test_factor_algo import quotes


def test_market_roundtrip_update_and_proxy_rejection(tmp_path):
    from src.store import SAEStore

    path = tmp_path / "research.duckdb"
    raw = quotes(40)
    raw["turn"] = np.nan
    with SAEStore(path) as store:
        store.save_market_data("000905.SH", raw, raw.index)
        pd.testing.assert_frame_equal(
            store.read_quotes("000905.SH"), raw, check_freq=False, check_like=True
        )
        pd.testing.assert_index_equal(store.read_calendar(), raw.index, check_names=False)
        corrected = raw.iloc[-2:].copy()
        corrected["amount"] *= 2
        store.save_market_data("000905.SH", corrected, raw.index)
        assert len(store.read_quotes("000905.SH")) == 40
        assert store.read_quotes("000905.SH").amount.iloc[-1] == corrected.amount.iloc[-1]
        corrected.attrs["turnover_policy"] = "volume_proxy"
        with pytest.raises(ValueError, match="代理"):
            store.save_market_data("000905.SH", corrected, raw.index)
    with SAEStore(path, read_only=True) as store:
        assert len(store.read_quotes("000905.SH", str(raw.index[-1].date()))) == 1
        assert store.read_quotes("000905.SH").turn.isna().all()


def test_runs_are_immutable_and_seed_columns_do_not_leak(tmp_path):
    from src.store import SAEStore

    first = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-02"]),
            "code": ["000905.SH"],
            "seed_0": [0.01],
            "score": [0.01],
        }
    )
    second = first.assign(seed_1=0.03, score=0.02)
    with SAEStore(tmp_path / "research.duckdb") as store:
        store.save_run("first", {"seeds": [0]}, {"policy": "volume_proxy"}, {"predictions": first})
        store.save_run("second", {"seeds": [0, 1]}, {}, {"predictions": second})
        pd.testing.assert_frame_equal(store.read_table("first", "predictions"), first)
        assert (
            store.read_predictions("second").loc[(pd.Timestamp("2020-01-02"), "000905.SH"), "score"]
            == 0.02
        )
        assert store.run_metadata("first")["config"]["seeds"] == [0]
        with pytest.raises(ValueError, match="已存在"):
            store.save_run("first", {}, {}, {"predictions": second})
        assert len(store.list_runs()) == 2
        pd.testing.assert_frame_equal(store.read_table("first", "predictions"), first)


def test_failed_run_rolls_back_all_tables_and_registered_id(tmp_path):
    from src.store import SAEStore

    frame = pd.DataFrame({"value": [1.0]})
    with SAEStore(tmp_path / "research.duckdb") as store:
        with pytest.raises(ValueError):
            store.save_run("bad", {}, {}, {"losses": frame, "invalid-table": frame})
        assert store.list_runs().empty
        store.save_run("bad", {}, {}, {"losses": frame})
        assert len(store.read_table("bad", "losses")) == 1


def test_artifact_lookup_and_read_only_connection(tmp_path):
    from src.store import SAEStore

    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"research checkpoint")
    path = tmp_path / "research.duckdb"
    with SAEStore(path) as store:
        store.save_run("model-run", {}, {}, {}, artifacts={"model": artifact})
    with SAEStore(path, read_only=True) as store:
        assert store.artifact_path("model-run", "model") == artifact
        with pytest.raises(KeyError):
            store.read_table("model-run", "losses")


def test_query_accepts_one_select_and_rejects_writes(tmp_path):
    from src.store import SAEStore

    with SAEStore(tmp_path / "research.duckdb") as store:
        output = store.query("SELECT 1 AS value")
        assert output.iloc[0, 0] == 1
        with pytest.raises(ValueError, match="单条SELECT"):
            store.query("CREATE TABLE bad AS SELECT 1")
        with pytest.raises(ValueError, match="单条SELECT"):
            store.query("SELECT 1; SELECT 2")


def test_legacy_instrument_columns_migrated_in_place(tmp_path):
    """0.6.0前旧库的instrument列首次可写打开时就地改名，已拉取行情与水位不丢。"""
    import duckdb

    from src.store import SAEStore

    path = tmp_path / "legacy.duckdb"
    raw = quotes(3)
    legacy_frame = raw.rename_axis("datetime").reset_index()
    # INSERT按位置对齐，须显式按schema列序（code, datetime, open...turn）排列
    legacy_frame = legacy_frame[
        ["datetime", "open", "high", "low", "close", "preclose", "volume", "amount", "turn"]
    ]
    legacy_frame.insert(0, "instrument", "000905.SH")
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE market_quotes (instrument VARCHAR, datetime TIMESTAMP,"
        " open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, preclose DOUBLE,"
        " volume DOUBLE, amount DOUBLE, turn DOUBLE, PRIMARY KEY (instrument, datetime));"
        " CREATE TABLE index_ranges (instrument VARCHAR PRIMARY KEY,"
        " first_date TIMESTAMP, fetched_until TIMESTAMP NOT NULL);"
    )
    con.register("_legacy", legacy_frame)
    con.execute("INSERT INTO market_quotes SELECT * FROM _legacy")
    con.execute("INSERT INTO index_ranges VALUES ('000905.SH', NULL, '2020-01-05')")
    con.unregister("_legacy")
    con.close()

    with SAEStore(path) as store:
        # 可写打开即完成迁移：列名已改为code，行情与水位原封不动
        columns = {
            row[0]
            for row in store._con.execute(
                "SELECT column_name FROM information_schema.columns"
                " WHERE table_name IN ('market_quotes', 'index_ranges')"
            ).fetchall()
        }
        assert "instrument" not in columns and "code" in columns
        pd.testing.assert_frame_equal(
            store.read_quotes("000905.SH"), raw, check_freq=False, check_like=True
        )
        assert store.read_index_range("000905.SH") == (None, pd.Timestamp("2020-01-05"))
    # 已迁移库再次可写打开不重复报错
    with SAEStore(path) as store:
        assert len(store.read_quotes("000905.SH")) == len(raw)


def test_ensure_index_coverage_migrates_legacy_db_and_hits_cache(tmp_path):
    """0.6.0前旧库经ensure_index_coverage使用时自动迁移，缓存命中路径零网络。"""
    import duckdb

    from src.data_provider import ensure_index_coverage
    from src.store import SAEStore

    path = tmp_path / "legacy.duckdb"
    raw = quotes(3)
    legacy_frame = raw.rename_axis("datetime").reset_index()
    legacy_frame = legacy_frame[
        ["datetime", "open", "high", "low", "close", "preclose", "volume", "amount", "turn"]
    ]
    legacy_frame.insert(0, "instrument", "000905.SH")
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE market_quotes (instrument VARCHAR, datetime TIMESTAMP,"
        " open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, preclose DOUBLE,"
        " volume DOUBLE, amount DOUBLE, turn DOUBLE, PRIMARY KEY (instrument, datetime));"
        " CREATE TABLE index_ranges (instrument VARCHAR PRIMARY KEY,"
        " first_date TIMESTAMP, fetched_until TIMESTAMP NOT NULL);"
        " CREATE TABLE calendar_sessions (datetime TIMESTAMP PRIMARY KEY);"
    )
    con.register("_legacy", legacy_frame)
    con.execute("INSERT INTO market_quotes SELECT * FROM _legacy")
    con.execute(
        "INSERT INTO index_ranges VALUES ('000905.SH', '2017-01-02', '2030-12-31')"
    )
    con.unregister("_legacy")
    con.close()

    # 缺省client：若未命中缓存而惰性建客户端会失败，故通过即证明零网络
    fetched = ensure_index_coverage(
        path, "000905.SH", str(raw.index[0].date()), str(raw.index[-1].date())
    )
    pd.testing.assert_frame_equal(fetched, raw, check_freq=False, check_like=True)
    with SAEStore(path, read_only=True) as store:
        assert store.read_quotes("000905.SH").index.equals(raw.index)


# --- 产物路径锚定（0.14.0：相对 DB 所在目录）---


def _raw_artifact_row(db_path, run_id="model-run", name="model"):
    """绕开 artifact_path 的解析，读库里原始存的那个字符串。"""
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(
            "SELECT path FROM artifacts WHERE run_id=? AND name=?", [run_id, name]
        ).fetchone()[0]
    finally:
        con.close()


def test_artifact_under_db_dir_stored_relative(tmp_path):
    """DB 同目录树下的产物存相对路径，使 data/ 可整体搬迁。"""
    from src.store import SAEStore

    run_dir = tmp_path / "artifacts_demo"
    run_dir.mkdir()
    artifact = run_dir / "models.pt"
    artifact.write_bytes(b"checkpoint")
    path = tmp_path / "research.duckdb"
    with SAEStore(path) as store:
        store.save_run("model-run", {}, {}, {}, artifacts={"model": artifact}, artifact_dir=run_dir)

    assert _raw_artifact_row(path) == "artifacts_demo/models.pt"
    with SAEStore(path, read_only=True) as store:
        assert store.artifact_path("model-run", "model") == artifact
        assert store.run_metadata("model-run")["artifact_dir"] == str(run_dir)


def test_artifacts_resolve_after_moving_whole_data_dir(tmp_path):
    """把库与产物作为一个整体搬走后，登记仍然解析得到正确文件。"""
    import shutil

    from src.store import SAEStore

    src = tmp_path / "data"
    (src / "artifacts_demo").mkdir(parents=True)
    artifact = src / "artifacts_demo" / "models.pt"
    artifact.write_bytes(b"checkpoint")
    with SAEStore(src / "research.duckdb") as store:
        store.save_run("model-run", {}, {}, {}, artifacts={"model": artifact})

    moved = tmp_path / "elsewhere"
    shutil.move(str(src), str(moved))

    with SAEStore(moved / "research.duckdb", read_only=True) as store:
        found = store.artifact_path("model-run", "model")
    assert found == moved / "artifacts_demo" / "models.pt"
    assert found.read_bytes() == b"checkpoint"


def test_artifact_outside_db_dir_stays_absolute(tmp_path):
    """DB 目录树之外的产物保持绝对路径，不写成 ../.. 形式。"""
    from src.store import SAEStore

    outside = tmp_path / "outside"
    outside.mkdir()
    artifact = outside / "models.pt"
    artifact.write_bytes(b"checkpoint")
    path = tmp_path / "db" / "research.duckdb"
    with SAEStore(path) as store:
        store.save_run("model-run", {}, {}, {}, artifacts={"model": artifact})

    assert _raw_artifact_row(path) == str(artifact)
    with SAEStore(path, read_only=True) as store:
        assert store.artifact_path("model-run", "model") == artifact


def test_legacy_absolute_paths_migrated_on_open(tmp_path):
    """旧库里指向已消失前缀的绝对路径，开库时按文件实际存在与否就地改写。"""
    import duckdb

    from src.store import SAEStore

    db = tmp_path / "research.duckdb"
    run_dir = tmp_path / "artifacts_demo"
    run_dir.mkdir()
    (run_dir / "models.pt").write_bytes(b"checkpoint")
    with SAEStore(db) as store:
        store.save_run("model-run", {}, {}, {})
    con = duckdb.connect(str(db))
    stale = "/gone/worktree/timing_signals/hy_sae_timing/data/artifacts_demo/models.pt"
    con.execute("INSERT INTO artifacts VALUES (?,?,?,?)", ["model-run", "model", stale, "x"])
    con.execute("UPDATE runs SET artifact_dir=? WHERE run_id=?",
                ["/gone/worktree/timing_signals/hy_sae_timing/data/artifacts_demo", "model-run"])
    con.close()

    with SAEStore(db) as store:  # 开库即迁移
        assert store.artifact_path("model-run", "model") == run_dir / "models.pt"
        assert store.run_metadata("model-run")["artifact_dir"] == str(run_dir)
    assert _raw_artifact_row(db) == "artifacts_demo/models.pt"


def test_legacy_path_left_alone_when_file_absent(tmp_path):
    """迁移只在目标文件确实存在时改写，否则原样保留，不制造假登记。"""
    import duckdb

    from src.store import SAEStore

    db = tmp_path / "research.duckdb"
    with SAEStore(db) as store:
        store.save_run("model-run", {}, {}, {})
    stale = "/gone/worktree/timing_signals/hy_sae_timing/data/artifacts_demo/models.pt"
    con = duckdb.connect(str(db))
    con.execute("INSERT INTO artifacts VALUES (?,?,?,?)", ["model-run", "model", stale, "x"])
    con.close()

    with SAEStore(db):
        pass
    assert _raw_artifact_row(db) == stale
