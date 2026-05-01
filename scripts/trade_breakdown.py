# breakdown of trade events in IS window -- shows where the 284 vs 123 gap
# comes from. prints per-type counts + first few entry bars.

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.strategy import load_hlv_csv


def run(H, L, C, bars_back, chn_len, stp_pct, pv, slpg, e0):
    n = len(H)
    HH = np.zeros(n)
    LL = np.zeros(n)
    for k in range(bars_back, n):
        HH[k] = H[k - chn_len : k].max()
        LL[k] = L[k - chn_len : k].min()

    E = np.full(n, e0)
    DD = np.zeros(n)
    trades = np.zeros(n)
    Emax = e0

    position = 0
    bench_long = 0
    bench_short = 0

    # event kind: 0 none, 1 flat-roundtrip, 2 enter long, 3 enter short,
    # 4 exit long (trailing), 5 exit short (trailing),
    # 6 reverse L->S, 7 reverse S->L
    evt = np.zeros(n, dtype=np.int8)


    for k in range(bars_back, n):
        traded = False
        delta = pv * (C[k] - C[k - 1]) * position

        if position == 0:
            buy = H[k] >= HH[k]
            sell = L[k] <= LL[k]

            if buy and sell:
                delta = -slpg + pv * (LL[k] - HH[k])
                trades[k] = 1
                evt[k] = 1
            else:
                if buy:
                    delta = -slpg / 2 + pv * (C[k] - HH[k])
                    position = 1
                    traded = True
                    bench_long = H[k]
                    trades[k] = 0.5
                    evt[k] = 2
                if sell:
                    delta = -slpg / 2 - pv * (C[k] - LL[k])
                    position = -1
                    traded = True
                    bench_short = L[k]
                    trades[k] = 0.5
                    evt[k] = 3

        if position == 1 and not traded:
            sell_short = L[k] <= LL[k]
            sell_exit = L[k] <= bench_long * (1 - stp_pct)
            if sell_short and sell_exit:
                delta = delta - slpg - 2 * pv * (C[k] - LL[k])
                position = -1
                bench_short = L[k]
                trades[k] = 1
                evt[k] = 6
            else:
                if sell_exit:
                    delta = delta - slpg / 2 - pv * (C[k] - bench_long * (1 - stp_pct))
                    position = 0
                    trades[k] = 0.5
                    evt[k] = 4
                if sell_short:
                    delta = delta - slpg - 2 * pv * (C[k] - LL[k])
                    position = -1
                    bench_short = L[k]
                    trades[k] = 1
                    evt[k] = 6
            bench_long = max(H[k], bench_long)

        if position == -1 and not traded:
            buy_long = H[k] >= HH[k]
            buy_exit = H[k] >= bench_short * (1 + stp_pct)
            if buy_long and buy_exit:
                delta = delta - slpg + 2 * pv * (C[k] - HH[k])
                position = 1
                bench_long = H[k]
                trades[k] = 1
                evt[k] = 7
            else:
                if buy_exit:
                    delta = delta - slpg / 2 + pv * (C[k] - bench_short * (1 + stp_pct))
                    position = 0
                    trades[k] = 0.5
                    evt[k] = 5
                if buy_long:
                    delta = delta - slpg + 2 * pv * (C[k] - HH[k])
                    position = 1
                    bench_long = H[k]
                    
                    trades[k] = 1
                    evt[k] = 7
            bench_short = min(L[k], bench_short)

        E[k] = E[k - 1] + delta
        Emax = max(Emax, E[k])
        DD[k] = E[k] - Emax

    return E, DD, trades, evt, HH, LL



def main():
    dt, H, L, C = load_hlv_csv('instruction_and_data/HO-5minHLV.csv')

    L_, S = 12700, 0.010
    bars_back = 17001
    pv = 42_000.0
    slpg = 47.0
    e0 = 100_000.0
    is_lo, is_hi = 17000, 226119
    os_lo, os_hi = 226120, 611836

    print("Running (L=12700, S=0.010)...")
    E, DD, trades, evt, HH, LL = run(H, L, C, bars_back, L_, S, pv, slpg, e0)

    labels = {
        0: 'none',
        1: 'flat round-trip (both triggered)',
        2: 'enter long',
        3: 'enter short',
        4: 'exit long (trailing stop)',
        5: 'exit short (trailing stop)',
        6: 'reverse long -> short',
        7: 'reverse short -> long',
    }
    for name, lo, hi in [('IS', is_lo, is_hi), ('OS', os_lo, os_hi)]:
        print(f'\n{name} window [{lo}, {hi}]')
        slc = evt[lo:hi+1]
        counts = {k: int((slc == k).sum()) for k in labels}
        total_tr = float(trades[lo:hi+1].sum())

        for k in sorted(labels):
            if counts[k]:
                print(f'  {labels[k]}: {counts[k]}')
        print(f'  total trades sum = {total_tr}')

    print('\nfirst 20 IS entry bars')
    entries = [i for i in range(is_lo, is_hi+1) if evt[i] in (2, 3, 6, 7)]
    for i in entries[:20]:
        print(f'[{i}] {dt[i]} evt={labels[evt[i]]} '
              f'HH={HH[i]:.4f} LL={LL[i]:.4f} H={H[i]:.4f} L={L[i]:.4f} C={C[i]:.4f}')


if __name__ == '__main__':
    main()
