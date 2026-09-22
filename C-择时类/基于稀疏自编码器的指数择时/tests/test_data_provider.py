"""真实字段缺失、数据出口、warm-up路径验证。"""

import numpy as np
import pytest

from .test_factor_algo import quotes


def test_missing_turn_is_rejected_unless_explicit_proxy():
    from src.data_provider import prepare_quotes

    q = quotes()
    q["turn"] = np.nan
    with pytest.raises(ValueError, match="换手率"):
        prepare_quotes(q)
    p = prepare_quotes(q, turnover_policy="volume_proxy")
    assert p.attrs["turnover_policy"] == "volume_proxy"
    np.testing.assert_allclose(p.turn.iloc[19:], (q.volume / q.volume.rolling(20).mean()).iloc[19:])
    q["turn"] = 0.0
    with pytest.raises(ValueError, match="换手率"):
        prepare_quotes(q)


def test_invalid_ohlc_fails():
    from src.data_provider import prepare_quotes

    q = quotes()
    q.loc[q.index[3], "close"] = np.nan
    with pytest.raises(ValueError):
        prepare_quotes(q)


def test_missing_session_fails_even_without_nan():
    from src.data_provider import validate_sessions

    q = quotes(50)
    with pytest.raises(ValueError, match="交易日缺口"):
        validate_sessions(q.drop(q.index[20]), q.index)
    validate_sessions(q, q.index)


def test_requested_end_missing_is_rejected():
    from src.data_provider import validate_sessions

    q = quotes(50)
    with pytest.raises(ValueError, match="交易日缺口"):
        validate_sessions(
            q.iloc[:-3],
            q.index,
            start_date=str(q.index[10].date()),
            end_date=str(q.index[-1].date()),
        )
