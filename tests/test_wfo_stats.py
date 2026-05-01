import numpy as np
import pandas as pd
import pytest

from src.stats import compute_stats, extract_trades
from src.walk_forward import WFOConfig, iter_quarters


def test_extract_trades_assigns_exit_bar_pnl_and_conserves_total():
    pnl = np.array([10.0, 5.0, -3.0, 2.0, -4.0])
    pos = np.array([1, 1, 0, -1, 0], dtype=np.int8)

    trades = extract_trades(pnl, pos)

    assert [t.side for t in trades] == [1, -1]
    assert [t.pnl for t in trades] == pytest.approx([12.0, -2.0])
    assert sum(t.pnl for t in trades) == pytest.approx(float(pnl.sum()))


def test_extract_trades_handles_reversal_and_flat_roundtrip():
    pnl = np.array([-2.0, 10.0, -6.0, 4.0])
    pos = np.array([0, 1, -1, -1], dtype=np.int8)

    trades = extract_trades(pnl, pos)

    assert [t.side for t in trades] == [0, 1, -1]
    assert [t.pnl for t in trades] == pytest.approx([-2.0, 4.0, 4.0])
    assert sum(t.pnl for t in trades) == pytest.approx(float(pnl.sum()))


def test_compute_stats_uses_return_series_and_conserved_trade_pnl():
    pnl = np.array([10.0, 5.0, -3.0, 2.0, -4.0])
    equity = 1000.0 + np.cumsum(pnl)
    pos = np.array([1, 1, 0, -1, 0], dtype=np.int8)
    dt = pd.date_range("2020-01-01", periods=len(pnl), freq="D").values

    stats = compute_stats(equity, pnl, pos, dt, bars_per_year=252.0)

    returns = pnl / np.array([1000.0, 1010.0, 1015.0, 1012.0, 1014.0])
    assert stats.total_profit == pytest.approx(10.0)
    
    assert stats.avg_return_per_bar == pytest.approx(float(returns.mean()))
    assert stats.std_return_per_bar == pytest.approx(float(returns.std(ddof=1)))
    assert stats.num_trades == 2
    assert stats.profit_factor == pytest.approx(12.0 / 2.0)


def test_iter_quarters_respects_end_as_latest_os_end():
    dt = pd.date_range("2000-01-01", "2011-12-31", freq="D").values
    cfg = WFOConfig(is_years=4, os_months=3)

    tasks = list(iter_quarters(dt, cfg, start="2010-01-01", end="2010-04-01"))

    assert [task[0] for task in tasks] == ["2010Q1"]


def test_iter_quarters_supports_monthly_tau_without_skipping_months():
    dt = pd.date_range("2000-01-01", "2011-12-31", freq="D").values
    cfg = WFOConfig(is_years=4, os_months=1)

    tasks = list(iter_quarters(dt, cfg, start="2010-01-01", end="2010-04-01"))

    assert [task[0] for task in tasks] == ["2010-01", "2010-02", "2010-03"]
