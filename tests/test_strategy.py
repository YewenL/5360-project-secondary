# unit tests for src/strategy.py
# tiny hand-crafted series so every trade is verifiable against main.m

import math

import numpy as np
import pytest

from src.strategy import (
    rolling_max_prev,
    rolling_min_prev,
    
    run_strategy,
)


def test_rolling_max_prev_matches_brute_force():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(500)
    w = 17

    fast = rolling_max_prev(x, w)
    for k in range(len(x)):
        if k < w:
            assert math.isinf(fast[k]) and fast[k] < 0
        else:
            assert fast[k] == x[k - w : k].max()



def test_rolling_min_prev_matches_brute_force():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(500)
    
    w = 23
    fast = rolling_min_prev(x, w)

    for k in range(len(x)):
        if k < w:
            assert math.isinf(fast[k]) and fast[k] > 0
        else:
            assert fast[k] == x[k - w : k].min()


def test_rolling_handles_duplicates():
    x = np.array([1, 2, 2, 2, 3, 1, 1, 4, 4, 0], dtype=np.float64)
    w = 3
    fmax = rolling_max_prev(x, w)
    fmin = rolling_min_prev(x, w)

    for k in range(len(x)):
        if k < w:
            continue
        assert fmax[k] == x[k - w : k].max()
        assert fmin[k] == x[k - w : k].min()


CHN = 3
PV = 10.0
SLPG = 2.0

def _flat_bars(n, price=100.0):
    H = np.full(n, price)
    L = np.full(n, price)
    C = np.full(n, price)
    return H, L, C


def test_flat_series_triggers_same_bar_breakout_each_bar():
    # flat series: H==HH and L==LL every bar, main.m books HH->LL round trip
    # each bar, stays flat, bleeds slpg per bar
    H, L, C = _flat_bars(20, price=100.0)
    res = run_strategy(H, L, C, chn_len=CHN, stp_pct=0.01, pv=PV, slpg=SLPG, e0=1000.0)

    trading_bars = 20 - CHN
    assert res.trades.sum() == pytest.approx(float(trading_bars))
    np.testing.assert_array_equal(res.pos_end, 0)
    assert res.E[-1] == pytest.approx(1000.0 - trading_bars * SLPG)



def test_single_long_entry_then_trailing_exit():
    # k=3 long entry at HH=100, bench_long=110.
    # k=5 L=105 <= 108.9 -> trailing exit. net PnL = 10*8.9 - 2 = 87.
    
    H = np.array([100, 100, 100, 110, 110, 105, 107], dtype=np.float64)
    L = H.copy()
    C = H.copy()

    res = run_strategy(
        H, L, C,
        chn_len=CHN, stp_pct=0.01, pv=PV, slpg=SLPG,
        bars_back=CHN, e0=1000.0,
    )

    assert res.trades.sum() == pytest.approx(1)
    assert res.trades[3] == pytest.approx(0.5)
    assert res.trades[5] == pytest.approx(0.5)

    np.testing.assert_allclose(res.E[:3], 1000.0)
    assert res.E[3] == pytest.approx(1099.0)
    assert res.E[4] == pytest.approx(1099.0)
    assert res.E[5] == pytest.approx(1087.0)
    
    assert res.E[-1] - res.e0 == pytest.approx(87.0)


def test_single_short_entry_then_trailing_exit():
    # mirror: breakout down, rebound to trailing stop
    H = np.array([100, 100, 100, 90, 90, 95, 93], dtype=np.float64)
    L = H.copy()
    C = H.copy()
    res = run_strategy(
        H, L, C,
        chn_len=CHN, stp_pct=0.02, pv=PV, slpg=SLPG,
        bars_back=CHN, e0=1000.0,
    )
    assert res.trades.sum() == pytest.approx(1)
    assert res.E[-1] - res.e0 == pytest.approx(80.0)


def test_same_bar_double_breakout_while_flat():
    # wide bar while flat -> book HH->LL roundtrip, stay flat, full slpg
    H = np.array([100, 100, 100, 115], dtype=np.float64)
    L = np.array([100, 100, 100,  85], dtype=np.float64)
    C = np.array([100, 100, 100, 100], dtype=np.float64)
    res = run_strategy(
        H, L, C,
        chn_len=CHN, stp_pct=0.01, pv=PV, slpg=SLPG,
        bars_back=CHN, e0=1000.0,
    )
    assert res.trades[3] == pytest.approx(1)
    assert res.E[3] == pytest.approx(1000.0 - 2.0)
    assert res.pos_end[3] == 0


def test_long_reversal_to_short_in_one_bar():
    # enter long, later bar crashes through both trailing stop AND channel low
    # main.m: reverse to short at LL with full slpg
    H = np.array([100, 100, 100, 110, 110, 110, 85], dtype=np.float64)
    L = np.array([100, 100, 100, 110, 110, 110, 85], dtype=np.float64)
    C = np.array([100, 100, 100, 110, 110, 110, 85], dtype=np.float64)
    res = run_strategy(
        H, L, C,
        chn_len=CHN, stp_pct=0.01, pv=PV, slpg=SLPG,
        bars_back=CHN, e0=1000.0,
    )

    assert res.trades[6] == pytest.approx(1)
    assert res.pos_end[6] == -1
    assert res.E[3] == pytest.approx(1099.0)
    assert res.E[6] == pytest.approx(1347.0)


def test_short_reversal_to_long_in_one_bar():
    H = np.array([100, 100, 100,  90,  90,  90, 115], dtype=np.float64)
    L = np.array([100, 100, 100,  90,  90,  90, 115], dtype=np.float64)
    C = np.array([100, 100, 100,  90,  90,  90, 115], dtype=np.float64)
    

    res = run_strategy(
        H, L, C,
        chn_len=CHN, stp_pct=0.01, pv=PV, slpg=SLPG,
        bars_back=CHN, e0=1000.0,
    )
    assert res.trades[6] == pytest.approx(1)
    assert res.pos_end[6] == 1
    assert res.E[6] == pytest.approx(1099.0 + 248.0)


def test_bars_back_smaller_than_chn_len_rejected():
    H, L, C = _flat_bars(10)
    with pytest.raises(ValueError):
        run_strategy(H, L, C, chn_len=5, stp_pct=0.01, pv=PV, slpg=SLPG, bars_back=3)


def test_window_stats_matches_matlab_formulas():
    H = np.array([100, 100, 100, 110, 110, 105, 107], dtype=np.float64)
    L = H.copy(); C = H.copy()
    res = run_strategy(H, L, C, chn_len=CHN, stp_pct=0.01, pv=PV, slpg=SLPG,
                       bars_back=CHN, e0=1000.0)
    w = res.window_stats(0, 6)
    assert w.profit == pytest.approx(res.E[6] - res.E[0])
    assert w.worst_dd == pytest.approx(res.DD[0:7].min())
    assert w.trades == pytest.approx(res.trades[0:7].sum())
    pnl = np.empty(7)
    pnl[0] = 0.0
    pnl[1:] = np.diff(res.E[0:7])
    assert w.stdev == pytest.approx(float(np.std(pnl, ddof=1)))




@pytest.mark.parametrize(
    "seed,stp_pct,chn",
    [
        (0, 0.02, 20),
        (1, 0.01, 50),
        (2, 0.05, 10),
        (3, 0.0, 15),   # stp=0 edge case
        (4, 0.10, 30),
    ],
)
def test_numba_matches_literal_port_on_random_slice(seed, stp_pct, chn):
    # sanity check: numba core must match the literal port
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.mainm_literal import run_literal

    rng = np.random.default_rng(seed)
    n = 400
    bars_back = chn + 5

    C = 100.0 + np.cumsum(rng.standard_normal(n)) * 0.5
    H = C + np.abs(rng.standard_normal(n)) * 0.1
    L = C - np.abs(rng.standard_normal(n)) * 0.1

    E_lit, DD_lit, tr_lit, _, _ = run_literal(
        H, L, C, bars_back=bars_back, chn_len=chn, stp_pct=stp_pct,
        pv=100.0, slpg=0.1, e0=1000.0,
    )
    res = run_strategy(H, L, C, chn_len=chn, stp_pct=stp_pct, pv=100.0, slpg=0.1,
                       bars_back=bars_back, e0=1000.0)

    np.testing.assert_allclose(res.E, E_lit, rtol=0, atol=1e-12)
    np.testing.assert_allclose(res.DD, DD_lit, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(res.trades, tr_lit)

def test_trailing_stop_not_triggered_on_entry_bar():
    # main.m sets traded=True on entry so trailing block is skipped same bar
    H = np.array([100, 100, 100, 110, 109, 109], dtype=np.float64)
    L = np.array([100, 100, 100, 107, 109, 109], dtype=np.float64)
    C = np.array([100, 100, 100, 110, 109, 109], dtype=np.float64)
    res = run_strategy(
        H, L, C,
        chn_len=CHN, stp_pct=0.01, pv=PV, slpg=SLPG,
        bars_back=CHN, e0=1000.0,
    )
    assert res.trades[3] == pytest.approx(0.5)
    assert res.pos_end[3] == 1
    assert res.pos_end[4] == 1
    assert res.trades[4] == pytest.approx(0)


def test_channel_breakout_without_trailing_hit_triggers_reversal():
    # sell_short True but sell_exit False -> reverse to short at LL anyway
    H = np.array([100, 100, 100, 110, 110, 110, 109.5], dtype=np.float64)
    L = np.array([100, 100, 100, 110, 110, 110, 109.5], dtype=np.float64)
    C = np.array([100, 100, 100, 110, 110, 110, 109.5], dtype=np.float64)
    res = run_strategy(
        H, L, C,
        chn_len=CHN, stp_pct=0.01, pv=PV, slpg=SLPG,
        bars_back=CHN, e0=1000.0,
    )
    assert res.trades[6] == pytest.approx(1)
    assert res.pos_end[6] == -1
    assert res.E[6] - res.E[5] == pytest.approx(3.0)



def test_drawdown_invariants():
    rng = np.random.default_rng(123)
    n = 300

    C = 100.0 + np.cumsum(rng.standard_normal(n)) * 0.4
    H = C + np.abs(rng.standard_normal(n)) * 0.1
    L = C - np.abs(rng.standard_normal(n)) * 0.1
    res = run_strategy(H, L, C, chn_len=10, stp_pct=0.02, pv=100.0, slpg=0.1,
                       bars_back=15, e0=1000.0)

    assert (res.DD <= 1e-12).all()

    running_max = np.maximum.accumulate(res.E)
    at_peak = res.DD >= -1e-12
    np.testing.assert_array_equal(at_peak, res.E >= running_max - 1e-12)




def test_run_strategy_keep_arrays_flag():
    H, L, C = _flat_bars(10)
    res_default = run_strategy(H, L, C, chn_len=3, stp_pct=0.01, pv=PV, slpg=SLPG,
                               bars_back=3, e0=1000.0)
    
    assert res_default.HH is None
    assert res_default.LL is None

    res_keep = run_strategy(H, L, C, chn_len=3, stp_pct=0.01, pv=PV, slpg=SLPG,
                            bars_back=3, e0=1000.0, keep_arrays=True)
    
    assert isinstance(res_keep.HH, np.ndarray)
    assert isinstance(res_keep.LL, np.ndarray)
    assert res_keep.HH.shape == (10,)
