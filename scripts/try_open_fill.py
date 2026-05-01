# try two things flagged by careful re-reading:
#   1. CSV has an Open column we ignored. Main.m comments on line 129/135 note
#      fill = min(Open, stopPrice) / min(Open, LL) but author short-circuits to just stop/LL.
#   2. pseudo-code in BasicTradingSystems.doc has no channel-based reversal while positioned;
#      only the trailing stop exit. Re-entry only after return to flat.
#
# so test three variants on top of the main.m engine:
#   V1 = main.m logic + Open-based fills (prices only, trade count unchanged)
#   V2 = pseudo-code literal: close-based trailing, init at HH, 1-bar-lag orders, no mid-position reversal
#   V3 = V2 + Open-based fills

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_ohlc(path):
    df = pd.read_csv(path,
                     usecols=["Date", "Time", "Open", "High", "Low", "Close"],
                     dtype={"Date": "string", "Time": "string"})
    dt = pd.to_datetime(df["Date"] + " " + df["Time"], format="%m/%d/%Y %H:%M")
    return (dt.values.astype("datetime64[m]"),
            df["Open"].to_numpy(dtype=np.float64),
            df["High"].to_numpy(dtype=np.float64),
            df["Low"].to_numpy(dtype=np.float64),
            df["Close"].to_numpy(dtype=np.float64))


def rolling_max_prev(x, w):
    n = len(x)
    out = np.full(n, -np.inf)
    for k in range(w, n):
        out[k] = x[k - w:k].max()
    return out


def rolling_min_prev(x, w):
    n = len(x)
    out = np.full(n, np.inf)
    for k in range(w, n):
        out[k] = x[k - w:k].min()
    return out


def run_v1_open_fills(O, H, L, C, bars_back, chn_len, stp_pct, pv, slpg, e0):
    # main.m logic, but every fill price uses the %min(Open, stop) / %max(Open, trigger) convention
    n = len(H)
    HH = rolling_max_prev(H, chn_len)
    LL = rolling_min_prev(L, chn_len)

    E = np.full(n, e0); DD = np.zeros(n); trades = np.zeros(n)
    Emax = e0
    position = 0
    bL = bS = 0.0

    for k in range(bars_back, n):
        traded = False
        delta = pv * (C[k] - C[k - 1]) * position

        if position == 0:
            buy = H[k] >= HH[k]
            sell = L[k] <= LL[k]
            if buy and sell:
                bf = max(HH[k], O[k])
                sf = min(LL[k], O[k])
                delta += -slpg + pv * (sf - bf)
                trades[k] = 1
            else:
                if buy:
                    fill = max(HH[k], O[k])
                    delta += -slpg / 2 + pv * (C[k] - fill)
                    position = 1; traded = True; bL = H[k]; trades[k] = 0.5
                if sell:
                    fill = min(LL[k], O[k])
                    delta += -slpg / 2 - pv * (C[k] - fill)
                    position = -1; traded = True; bS = L[k]; trades[k] = 0.5

        if position == 1 and not traded:
            stop_px = bL * (1 - stp_pct)
            sell_short = L[k] <= LL[k]
            sell_exit = L[k] <= stop_px
            if sell_short and sell_exit:
                fill = min(LL[k], O[k])
                delta += -slpg - 2 * pv * (C[k] - fill)
                position = -1; bS = L[k]; trades[k] = 1
            else:
                if sell_exit:
                    fill = min(stop_px, O[k])
                    delta += -slpg / 2 - pv * (C[k] - fill)
                    position = 0; trades[k] = 0.5
                if sell_short:
                    fill = min(LL[k], O[k])
                    delta += -slpg - 2 * pv * (C[k] - fill)
                    position = -1; bS = L[k]; trades[k] = 1
            if H[k] > bL: bL = H[k]

        if position == -1 and not traded:
            stop_px = bS * (1 + stp_pct)
            buy_long = H[k] >= HH[k]
            buy_exit = H[k] >= stop_px
            if buy_long and buy_exit:
                fill = max(HH[k], O[k])
                delta += -slpg + 2 * pv * (C[k] - fill)
                position = 1; bL = H[k]; trades[k] = 1
            else:
                if buy_exit:
                    fill = max(stop_px, O[k])
                    delta += -slpg / 2 + pv * (C[k] - fill)
                    position = 0; trades[k] = 0.5
                if buy_long:
                    fill = max(HH[k], O[k])
                    delta += -slpg + 2 * pv * (C[k] - fill)
                    position = 1; bL = H[k]; trades[k] = 1
            if L[k] < bS: bS = L[k]

        E[k] = E[k - 1] + delta
        Emax = max(Emax, E[k])
        DD[k] = E[k] - Emax

    return E, DD, trades


def run_v2_pseudo_el(O, H, L, C, bars_back, chn_len, stp_pct, pv, slpg, e0,
                     *, use_open_fill=False):
    # pseudo-code Channel, literal interpretation:
    #   - orders placed at end of bar k, fire on bar k+1
    #   - HighestHigh at bar k uses H[k-L+1..k] (includes current bar in EL convention)
    #   - trailing stop PrevPeak init = EntryPrice; update via Close
    #   - no channel-based reversal while positioned; only trailing stop exit
    n = len(H)

    # EL-convention rolling max over last chn_len bars INCLUDING current
    def hh_incl(k):
        return H[max(0, k - chn_len + 1):k + 1].max()

    def ll_incl(k):
        return L[max(0, k - chn_len + 1):k + 1].min()

    E = np.full(n, e0); DD = np.zeros(n); trades = np.zeros(n)
    Emax = e0
    position = 0
    prev_peak = 0.0
    prev_trough = 0.0

    # orders pending for next bar
    pend_buy = None
    pend_sell = None
    pend_long_exit = None
    pend_short_exit = None

    for k in range(bars_back, n):
        delta = pv * (C[k] - C[k - 1]) * position

        if position == 0:
            buy_fires = pend_buy is not None and H[k] >= pend_buy
            sell_fires = pend_sell is not None and L[k] <= pend_sell
            if buy_fires and sell_fires:
                bf = max(pend_buy, O[k]) if use_open_fill else pend_buy
                sf = min(pend_sell, O[k]) if use_open_fill else pend_sell
                delta += -slpg + pv * (sf - bf)
                trades[k] = 1
            elif buy_fires:
                fill = max(pend_buy, O[k]) if use_open_fill else pend_buy
                delta += -slpg / 2 + pv * (C[k] - fill)
                position = 1
                prev_peak = fill
                if C[k] > prev_peak: prev_peak = C[k]
                trades[k] = 0.5
            elif sell_fires:
                fill = min(pend_sell, O[k]) if use_open_fill else pend_sell
                delta += -slpg / 2 - pv * (C[k] - fill)
                position = -1
                prev_trough = fill
                if C[k] < prev_trough: prev_trough = C[k]
                trades[k] = 0.5

        elif position == 1:
            if pend_long_exit is not None and L[k] <= pend_long_exit:
                fill = min(pend_long_exit, O[k]) if use_open_fill else pend_long_exit
                delta += -slpg / 2 - pv * (C[k] - fill)
                position = 0
                trades[k] = 0.5
            else:
                if C[k] > prev_peak: prev_peak = C[k]

        elif position == -1:
            if pend_short_exit is not None and H[k] >= pend_short_exit:
                fill = max(pend_short_exit, O[k]) if use_open_fill else pend_short_exit
                delta += -slpg / 2 + pv * (C[k] - fill)
                position = 0
                trades[k] = 0.5
            else:
                if C[k] < prev_trough: prev_trough = C[k]

        E[k] = E[k - 1] + delta
        Emax = max(Emax, E[k])
        DD[k] = E[k] - Emax

        # place orders for next bar
        pend_buy = pend_sell = pend_long_exit = pend_short_exit = None
        if k + 1 < n:
            if position == 0 and k + 1 >= chn_len:
                pend_buy = hh_incl(k)
                pend_sell = ll_incl(k)
            elif position == 1:
                pend_long_exit = prev_peak * (1 - stp_pct)
            elif position == -1:
                pend_short_exit = prev_trough * (1 + stp_pct)

    return E, DD, trades


def stats(E, DD, trades, i0, i1):
    pnl = np.concatenate(([0.0], np.diff(E[i0:i1 + 1])))
    return (E[i1] - E[i0], DD[i0:i1 + 1].min(),
            float(np.std(pnl, ddof=1)), float(trades[i0:i1 + 1].sum()))


def main():
    path = ROOT / 'instruction_and_data' / 'HO-5minHLV.csv'
    print('loading OHLC...')
    dt, O, H, L, C = load_ohlc(str(path))
    print(f'  {len(H)} bars')

    bars_back = 17001
    is_lo, is_hi = 17000, 226119
    os_lo, os_hi = 226120, 611836
    pv = 42_000.0; slpg = 47.0; e0 = 100_000.0

    # reference values from HO Optimization.xlsx
    refs = {
        (12700, 0.010): {'is': (19988.688, -3280.682, 19.633, 123),
                         'os': (172337.836, -14299.070, 56.860, 466)},
        (11500, 0.019): {'is': (26528.309, -6537.505, 25.494, 63),
                         'os': (181458.926, -15300.564, 70.699, 262)},
    }

    cases = [
        ('V1 main.m + Open fills',
         lambda L_, S: run_v1_open_fills(O, H, L, C, bars_back, L_, S, pv, slpg, e0)),
        ('V2 pseudo-code EL literal',
         lambda L_, S: run_v2_pseudo_el(O, H, L, C, bars_back, L_, S, pv, slpg, e0,
                                        use_open_fill=False)),
        ('V3 pseudo-code EL + Open fills',
         lambda L_, S: run_v2_pseudo_el(O, H, L, C, bars_back, L_, S, pv, slpg, e0,
                                        use_open_fill=True)),
    ]

    for (L_, S), ref in refs.items():
        print(f'\n=== (L={L_}, S={S}) ===')
        rip, rid, ris, rit = ref['is']
        rop, rod, ros, rot = ref['os']
        print(f'  ref IS: profit={rip:>10.1f} dd={rid:>8.1f} std={ris:>6.2f} trades={rit:>5.0f}')
        print(f'  ref OS: profit={rop:>10.1f} dd={rod:>8.1f} std={ros:>6.2f} trades={rot:>5.0f}')
        for name, fn in cases:
            E, DD, trades = fn(L_, S)
            ip, id_, ist, it = stats(E, DD, trades, is_lo, is_hi)
            op, od, ost, ot = stats(E, DD, trades, os_lo, os_hi)
            print(f'  {name}')
            print(f'    IS: profit={ip:>10.1f} dd={id_:>8.1f} std={ist:>6.2f} trades={it:>5.0f}')
            print(f'    OS: profit={op:>10.1f} dd={od:>8.1f} std={ost:>6.2f} trades={ot:>5.0f}')


if __name__ == '__main__':
    main()
