# Channel WithDDControl, port of main.m. Numba version; a slow python line-for-line port in scripts/mainm_literal.py gives identical output so this one should match main.m too.
#
# NB: running on HO-5minHLV.csv, main.m's output (and therefore this engine) does not exactly match HO Optimization.xlsx - IS trade count is ~2.3x,
# OS ~1.3x. Ratio is stable across the (L,S) grid which rules out a bar-level bug. Reference xlsx most likely came from a different main.m that includes some DD-pause / CanTrade-reset logic not present in the main.m we were given. See scripts/semantic_variants.py.

from dataclasses import dataclass

import numpy as np
from numba import njit


@njit(cache=True)
def rolling_max_prev(x, window):
    # HH[k] = max(x[k-window .. k-1]); left at -inf for k<window
    n = x.shape[0]
    out = np.full(n, -np.inf, dtype=np.float64)
    dq = np.empty(n + 1, dtype=np.int64)
    head = 0
    tail = 0
    for k in range(n):
        while head < tail and dq[head] < k - window:
            head += 1
        if k >= window and head < tail:
            out[k] = x[dq[head]]
        while head < tail and x[dq[tail - 1]] <= x[k]:
            tail -= 1
        dq[tail] = k
        tail += 1
    return out


@njit(cache=True)
def rolling_min_prev(x, window):
    n = x.shape[0]
    out = np.full(n, np.inf, dtype=np.float64)
    dq = np.empty(n + 1, dtype=np.int64)
    head = 0
    tail = 0
    for k in range(n):
        while head < tail and dq[head] < k - window:
            head += 1
        if k >= window and head < tail:
            out[k] = x[dq[head]]
        while head < tail and x[dq[tail - 1]] >= x[k]:
            tail -= 1
        dq[tail] = k
        tail += 1
    return out


@njit(cache=True)
def _run_core(H, L, C, HH, LL, stp_pct, pv, slpg, bars_back, e0):
    n = H.shape[0]
    E = np.full(n, e0, dtype=np.float64)
    DD = np.zeros(n, dtype=np.float64)
    trades = np.zeros(n, dtype=np.float64)
    pos_end = np.zeros(n, dtype=np.int8)

    position = 0
    bL = 0.0
    bS = 0.0

    emax = e0

    for k in range(bars_back, n):
        traded = False
        delta = pv * (C[k] - C[k - 1]) * position

        if position == 0:
            buy = H[k] >= HH[k]
            sell = L[k] <= LL[k]

            if buy and sell:
                # both fire on same bar -> book HH->LL round trip, stay flat
                delta += -slpg + pv * (LL[k] - HH[k])
                trades[k] = 1
            else:
                if buy:
                    delta += -slpg / 2 + pv * (C[k] - HH[k])
                    position = 1
                    traded = True
                    bL = H[k]
                    trades[k] = 0.5

                if sell:
                    delta += -slpg / 2 - pv * (C[k] - LL[k])
                    position = -1
                    traded = True
                    bS = L[k]
                    trades[k] = 0.5


        if position == 1 and not traded:
            sell_short = L[k] <= LL[k]
            sell_exit = L[k] <= bL * (1 - stp_pct)

            if sell_short and sell_exit:
                delta += -slpg - 2 * pv * (C[k] - LL[k])
                position = -1
                bS = L[k]
                trades[k] = 1

            else:
                if sell_exit:
                    delta += -slpg / 2 - pv * (C[k] - bL * (1 - stp_pct))
                    position = 0
                    trades[k] = 0.5
                if sell_short:
                    
                    delta += -slpg - 2 * pv * (C[k] - LL[k])
                    position = -1
                    bS = L[k]
                    trades[k] = 1

            # bump running high after stop check, matches main.m order
            if H[k] > bL:
                bL = H[k]


        if position == -1 and not traded:
            buy_long = H[k] >= HH[k]
            buy_exit = H[k] >= bS * (1 + stp_pct)

            if buy_long and buy_exit:
                delta += -slpg + 2 * pv * (C[k] - HH[k])
                position = 1
                bL = H[k]
                trades[k] = 1
            else:
                if buy_exit:
                    delta += -slpg / 2 + pv * (C[k] - bS * (1 + stp_pct))
                    position = 0
                    trades[k] = 0.5
                if buy_long:
                    delta += -slpg + 2 * pv * (C[k] - HH[k])
                    position = 1
                    bL = H[k]
                    trades[k] = 1

            if L[k] < bS:
                bS = L[k]


        E[k] = E[k - 1] + delta
        if E[k] > emax:
            emax = E[k]
        DD[k] = E[k] - emax
        pos_end[k] = position

    return E, DD, trades, pos_end


@dataclass
class WindowStats:
    profit: float
    worst_dd: float
    stdev: float
    trades: float


@dataclass
class RunResult:
    E: np.ndarray
    DD: np.ndarray
    trades: np.ndarray
    pos_end: np.ndarray
    chn_len: int
    stp_pct: float
    pv: float
    slpg: float
    bars_back: int
    e0: float
    HH: np.ndarray = None
    LL: np.ndarray = None

    def window_stats(self, i0, i1):
        # matlab post-loop stats: Profit=E(i1)-E(i0), WorstDD=min(DD),
        # StDev=std(PnL, ddof=1), Trades=sum(trades).
        # PnL is padded with zeros for k<barsBack per main.m.
        pnl = np.empty(i1 - i0 + 1, dtype=np.float64)
        pnl[0] = 0 if i0 <= self.bars_back else self.E[i0] - self.E[i0 - 1]
        pnl[1:] = np.diff(self.E[i0 : i1 + 1])
        return WindowStats(
            profit=float(self.E[i1] - self.E[i0]),
            worst_dd=float(self.DD[i0 : i1 + 1].min()),
            stdev=float(np.std(pnl, ddof=1)),
            trades=float(self.trades[i0 : i1 + 1].sum()),
        )


def run_strategy(H, L, C, chn_len, stp_pct, pv, slpg,
                 bars_back=None, e0=100_000.0, keep_arrays=False):
    H = np.ascontiguousarray(H, dtype=np.float64)
    L = np.ascontiguousarray(L, dtype=np.float64)
    C = np.ascontiguousarray(C, dtype=np.float64)

    if not (H.shape == L.shape == C.shape):
        raise ValueError("H/L/C shape mismatch")
    if bars_back is None:
        bars_back = chn_len
    if bars_back < chn_len:
        raise ValueError("bars_back < chn_len")

    HH = rolling_max_prev(H, chn_len)
    LL = rolling_min_prev(L, chn_len)

    E, DD, trades, pos_end = _run_core(
        H, L, C, HH, LL,
        float(stp_pct), float(pv), float(slpg), int(bars_back), float(e0),
    )

    return RunResult(
        E=E, DD=DD, trades=trades, pos_end=pos_end,
        chn_len=int(chn_len), stp_pct=float(stp_pct),
        pv=float(pv), slpg=float(slpg),
        bars_back=int(bars_back), e0=float(e0),
        HH=HH if keep_arrays else None,
        LL=LL if keep_arrays else None,
    )



def load_hlv_csv(path):
    import pandas as pd

    df = pd.read_csv(
        path,
        usecols=["Date", "Time", "High", "Low", "Close"],
        dtype={"Date": "string", "Time": "string"},
    )
    dt = pd.to_datetime(df["Date"] + " " + df["Time"], format="%m/%d/%Y %H:%M")

    return (
        dt.values.astype("datetime64[m]"),
        df["High"].to_numpy(dtype=np.float64),
        df["Low"].to_numpy(dtype=np.float64),
        df["Close"].to_numpy(dtype=np.float64),
    )


def index_at_or_after(dt64, when_str):
    # matlab: sum(numTime < when) + 1   -> 0-based
    return int(np.searchsorted(dt64, np.datetime64(when_str), side="left"))

def index_at_or_before(dt64, when_str):
    return int(np.searchsorted(dt64, np.datetime64(when_str), side="right")) - 1
