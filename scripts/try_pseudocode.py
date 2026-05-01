# okay so i have talked to prof and was told yeah the pseudocode and main.m are a bit diff we can choose whatever we want, so we go for the main.m version.
#

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.strategy import load_hlv_csv


def run(H, L, C, bars_back, chn_len, stp_pct, pv, slpg, e0, *, variant):
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
    entry_price_long = 0
    entry_price_short = 0


    def init_bench_long(k, HH_k):
        if variant in ('A',):
            return H[k], H[k]
        if variant in ('B', 'E'):
            return HH_k, HH_k
        if variant in ('C',):
            return HH_k, HH_k
        if variant == 'D':
            return H[k], H[k]
        raise ValueError(variant)

    def init_bench_short(k, LL_k):
        if variant in ('A',):
            return L[k], L[k]
        if variant in ('B', 'E'):
            return LL_k, LL_k
        if variant in ('C',):
            return LL_k, LL_k
        if variant == 'D':
            return L[k], L[k]
        raise ValueError(variant)

    def update_bench_long(k, bench, entry):
        if variant == 'A':
            return max(H[k], bench)
        if variant in ('B', 'D'):
            return max(C[k], bench)
        if variant == 'E':
            return max(H[k], bench)
        if variant == 'C':
            return max(entry, C[k])
        raise ValueError(variant)

    def update_bench_short(k, bench, entry):
        if variant == 'A':
            return min(L[k], bench)
        if variant in ('B', 'D'):
            return min(C[k], bench)
        if variant == 'E':
            return min(L[k], bench)
        if variant == 'C':
            return min(entry, C[k])
        raise ValueError(variant)

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
                    position = 1
                    traded = True
                    bench_long, entry_price_long = init_bench_long(k, HH[k])
                    trades[k] = 0.5
                if sell:
                    delta = -slpg / 2 - pv * (C[k] - LL[k])
                    position = -1
                    traded = True
                    bench_short, entry_price_short = init_bench_short(k, LL[k])
                    trades[k] = 0.5

        if position == 1 and not traded:
            sell_short = L[k] <= LL[k]
            sell_exit = L[k] <= bench_long * (1 - stp_pct)
            if sell_short and sell_exit:
                delta = delta - slpg - 2 * pv * (C[k] - LL[k])
                position = -1
                bench_short, entry_price_short = init_bench_short(k, LL[k])
                trades[k] = 1
            else:
                if sell_exit:
                    delta = delta - slpg / 2 - pv * (C[k] - bench_long * (1 - stp_pct))
                    position = 0
                    trades[k] = 0.5
                if sell_short:
                    delta = delta - slpg - 2 * pv * (C[k] - LL[k])
                    position = -1
                    bench_short, entry_price_short = init_bench_short(k, LL[k])
                    trades[k] = 1
            bench_long = update_bench_long(k, bench_long, entry_price_long)

        if position == -1 and not traded:
            buy_long = H[k] >= HH[k]
            buy_exit = H[k] >= bench_short * (1 + stp_pct)
            if buy_long and buy_exit:
                delta = delta - slpg + 2 * pv * (C[k] - HH[k])
                position = 1
                bench_long, entry_price_long = init_bench_long(k, HH[k])
                trades[k] = 1
            else:
                if buy_exit:
                    delta = delta - slpg / 2 + pv * (C[k] - bench_short * (1 + stp_pct))
                    position = 0
                    trades[k] = 0.5
                if buy_long:
                    delta = delta - slpg + 2 * pv * (C[k] - HH[k])
                    position = 1
                    bench_long, entry_price_long = init_bench_long(k, HH[k])
                    trades[k] = 1
            bench_short = update_bench_short(k, bench_short, entry_price_short)

        E[k] = E[k - 1] + delta
        Emax = max(Emax, E[k])
        DD[k] = E[k] - Emax

    return E, DD, trades


def stats(E, DD, trades, i0, i1):
    pnl = np.concatenate(([0.0], np.diff(E[i0:i1+1])))
    return (E[i1] - E[i0], DD[i0:i1+1].min(), np.std(pnl, ddof=1), trades[i0:i1+1].sum())


def main():
    dt, H, L, C = load_hlv_csv('instruction_and_data/HO-5minHLV.csv')
    L_, S = 12700, 0.010
    bars_back = 17001
    pv = 42_000.0
    slpg = 47.0
    e0 = 100_000.0
    is_lo, is_hi = 17000, 226119
    os_lo, os_hi = 226120, 611836

    print(f"Expected: IS trades=123 profit=19988.688  |  OS trades=466 profit=172337.836\n")
    descs = {
        'A': "main.m: running max(High), init=H[entry]",
        'B': "running max(Close), init=HH[entry]",
        'C': "pseudo-code literal: max(EntryPrice=HH, Close(k)), no memory",
        'D': "running max(Close), init=H[entry]",
        'E': "running max(High), init=HH[entry]",
    }

    for v in ['A', 'B', 'C', 'D', 'E']:
        E, DD, trades = run(H, L, C, bars_back, L_, S, pv, slpg, e0, variant=v)
        ip, id_, ist, it = stats(E, DD, trades, is_lo, is_hi)
        op, od, ost, ot = stats(E, DD, trades, os_lo, os_hi)
        print(f'{v}: {descs[v]}')
        print(f'  IS: profit {ip:.2f}, dd {id_:.2f}, std {ist:.3f}, trades {it}')
        print(f'  OS: profit {op:.2f}, dd {od:.2f}, std {ost:.3f}, trades {ot}')


if __name__ == '__main__':
    main()
