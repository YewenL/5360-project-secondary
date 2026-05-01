# test: trailing stop only engages after trade becomes profitable.
# PrevPeak = EntryPrice at entry; only update when Close> PrevPeak.
# When close is below entry, stop stays at EntryPrice*(1-S).

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.strategy import load_hlv_csv


def run(H, L, C, bars_back, chn_len, stp_pct, pv, slpg, e0):
    n = len(H)
    HH = np.zeros(n); LL = np.zeros(n)

    for k in range(bars_back, n):
        HH[k] = H[k - chn_len : k].max()
        LL[k] = L[k - chn_len : k].min()

    E = np.full(n, e0); DD = np.zeros(n); trades = np.zeros(n)

    Emax = e0; position = 0
    prev_peak = 0
    prev_trough = 0
    entry_price_long = 0
    
    entry_price_short = 0

    for k in range(bars_back, n):
        traded = False
        delta = pv * (C[k] - C[k - 1]) * position

        if position == 0:
            buy = H[k] >= HH[k]
            sell = L[k] <= LL[k]
            if buy and sell:
                delta = -slpg + pv * (LL[k] - HH[k])
                trades[k] = 1
            else:
                if buy:
                    delta = -slpg / 2 + pv * (C[k] - HH[k])
                    position = 1; traded = True
                    entry_price_long = HH[k]
                    prev_peak = entry_price_long
                    trades[k] = 0.5
                if sell:
                    delta = -slpg / 2 - pv * (C[k] - LL[k])
                    position = -1; traded = True
                    entry_price_short = LL[k]
                    prev_trough = entry_price_short
                    trades[k] = 0.5

        if position == 1 and not traded:
            sell_short = L[k] <= LL[k]
            sell_exit = L[k] <= prev_peak * (1 - stp_pct)
            if sell_short and sell_exit:
                delta = delta - slpg - 2 * pv * (C[k] - LL[k])
                position = -1
                entry_price_short = LL[k]
                prev_trough = entry_price_short
                trades[k] = 1
            else:
                if sell_exit:
                    delta = delta - slpg / 2 - pv * (C[k] - prev_peak * (1 - stp_pct))
                    position = 0; trades[k] = 0.5
                if sell_short:
                    delta = delta - slpg - 2 * pv * (C[k] - LL[k])
                    position = -1
                    entry_price_short = LL[k]
                    prev_trough = entry_price_short
                    trades[k] = 1
            # pseudo-code: if Close > PrevPeak then PrevPeak = Close
            if C[k] > prev_peak:
                prev_peak = C[k]

        if position == -1 and not traded:
            buy_long = H[k] >= HH[k]
            buy_exit = H[k] >= prev_trough * (1 + stp_pct)
            if buy_long and buy_exit:
                delta = delta - slpg + 2 * pv * (C[k] - HH[k])
                position = 1
                entry_price_long = HH[k]
                prev_peak = entry_price_long
                trades[k] = 1
            else:
                if buy_exit:
                    delta = delta - slpg / 2 + pv * (C[k] - prev_trough * (1 + stp_pct))
                    position = 0; trades[k] = 0.5
                if buy_long:
                    delta = delta - slpg + 2 * pv * (C[k] - HH[k])
                    position = 1
                    entry_price_long = HH[k]
                    prev_peak = entry_price_long
                    trades[k] = 1
            if C[k] < prev_trough:
                prev_trough = C[k]

        E[k] = E[k - 1] + delta
        Emax = max(Emax, E[k]); DD[k] = E[k] - Emax

    return E, DD, trades


def main():
    dt, H, L, C = load_hlv_csv('instruction_and_data/HO-5minHLV.csv')
    L_, S = 12700, 0.010
    bars_back = 17001

    is_lo, is_hi = 17000, 226119
    os_lo, os_hi = 226120, 611836

    E, DD, trades = run(H, L, C, bars_back, L_, S, 42000, 47, 100000)

    for name, lo, hi in [('IS', is_lo, is_hi), ('OS', os_lo, os_hi)]:
        pnl = np.concatenate(([0], np.diff(E[lo:hi+1])))
        print(name,
              'profit', round(E[hi]-E[lo], 2),
              'dd', round(DD[lo:hi+1].min(), 2),
              'std', round(np.std(pnl, ddof=1), 3),
              'trades', trades[lo:hi+1].sum())
    print('\nref: IS trades=123 profit=19988.69, OS trades=466')


if __name__ == '__main__':
    main()
