# literal line-for-line python port of main.m. slow but used to
# cross-check the numba version in src/strategy.py.

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.strategy import load_hlv_csv


def run_literal(H, L, C, bars_back, chn_len, stp_pct, pv, slpg, e0):
    n = len(H)
    HH = np.zeros(n, dtype=np.float64)
    
    LL = np.zeros(n, dtype=np.float64)

    # matlab k=barsBack+1 -> python k=bars_back
    # matlab H(k-L:k-1) -> python H[k-chn_len : k]
    for k in range(bars_back, n):
        HH[k] = H[k - chn_len : k].max()
        LL[k] = L[k - chn_len : k].min()

    E = np.full(n, e0, dtype=np.float64)
    DD = np.zeros(n, dtype=np.float64)
    trades = np.zeros(n, dtype=np.float64)
    Emax = e0

    position = 0
    bench_long = 0.0
    
    bench_short = 0.0

    for k in range(bars_back, n):
        traded = False
        delta = pv * (C[k] - C[k - 1]) * position

        if position == 0:
            buy = H[k] >= HH[k]
            sell = L[k] <= LL[k]
            if buy and sell:
                delta = -slpg + pv * (LL[k] - HH[k])
                trades[k] = 1.0
            else:
                if buy:
                    delta = -slpg / 2.0 + pv * (C[k] - HH[k])
                    position = 1
                    traded = True
                    bench_long = H[k]
                    trades[k] = 0.5
                    
                if sell:
                    delta = -slpg / 2.0 - pv * (C[k] - LL[k])
                    position = -1
                    traded = True
                    bench_short = L[k]
                    trades[k] = 0.5

        if position == 1 and not traded:
            sell_short = L[k] <= LL[k]
            sell_exit = L[k] <= bench_long * (1.0 - stp_pct)
            if sell_short and sell_exit:
                delta = delta - slpg - 2.0 * pv * (C[k] - LL[k])
                position = -1
                bench_short = L[k]
                
                trades[k] = 1.0
            else:
                if sell_exit:
                    delta = delta - slpg / 2.0 - pv * (C[k] - bench_long * (1.0 - stp_pct))
                    position = 0
                    trades[k] = 0.5
                if sell_short:
                    delta = delta - slpg - 2.0 * pv * (C[k] - LL[k])
                    position = -1
                    bench_short = L[k]
                    trades[k] = 1.0
            bench_long = max(H[k], bench_long)

        if position == -1 and not traded:
            buy_long = H[k] >= HH[k]
            buy_exit = H[k] >= bench_short * (1.0 + stp_pct)
            if buy_long and buy_exit:
                delta = delta - slpg + 2.0 * pv * (C[k] - HH[k])
                position = 1
                bench_long = H[k]
                trades[k] = 1.0
            else:
                if buy_exit:
                    delta = delta - slpg / 2.0 + pv * (C[k] - bench_short * (1.0 + stp_pct))
                    position = 0
                    
                    trades[k] = 0.5
                if buy_long:
                    delta = delta - slpg + 2.0 * pv * (C[k] - HH[k])
                    position = 1
                    bench_long = H[k]
                    trades[k] = 1.0
            bench_short = min(L[k], bench_short)

        E[k] = E[k - 1] + delta
        Emax = max(Emax, E[k])
        DD[k] = E[k] - Emax

    return E, DD, trades, HH, LL


def main():
    dt, H, L, C = load_hlv_csv('instruction_and_data/HO-5minHLV.csv')

    L_, S = 12700, 0.010
    bars_back = 17001
    
    pv = 42_000.0
    slpg = 47.0
    e0 = 100_000.0

    is_lo = 17000   # matlab 17001 - 1
    is_hi = 226119  # matlab 226120 - 1

    import time
    t0 = time.time()
    E, DD, trades, HH, LL = run_literal(H, L, C, bars_back, L_, S, pv, slpg, e0)
    print(f'Literal run: {time.time()-t0:.1f}s')

    pnl = np.concatenate(([0.0], np.diff(E[is_lo : is_hi + 1])))
    profit = E[is_hi] - E[is_lo]
    worst_dd = DD[is_lo : is_hi + 1].min()
    stdev = np.std(pnl, ddof=1)
    tr = trades[is_lo : is_hi + 1].sum()
    
    print(f'L={L_} S={S} IS: profit {profit:.4f}, dd {worst_dd:.4f}, std {stdev:.4f}, trades {tr}')
    print('expected  (xlsx): profit 19988.688, dd -3280.682, std 19.633, trades 123')


if __name__ == "__main__":
    main()
