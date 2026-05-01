# try alternative breakout conventions to see which one matches reference.

import sys
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.strategy import load_hlv_csv


def run_variant(H, L, C, bars_back, chn_len, stp_pct, pv, slpg, e0,
                *, strict_breakout=False,
                window_include_k=False,
                
                use_close_for_bench=False):
    # strict_breakout: > / < instead of >= / <=
    
    # window_include_k: HH(k) = max over H[k-L+1..k] (inclusive)
    # use_close_for_bench: bench_long tracks max(close), not max(high)
    n = len(H)
    HH = np.zeros(n)
    LL = np.zeros(n)
    for k in range(bars_back, n):
        if window_include_k:
            HH[k] = H[k - chn_len + 1 : k + 1].max()
            LL[k] = L[k - chn_len + 1 : k + 1].min()
        else:
            
            HH[k] = H[k - chn_len : k].max()
            LL[k] = L[k - chn_len : k].min()

    E = np.full(n, e0)
    DD = np.zeros(n)
    trades = np.zeros(n)
    Emax = e0

    position = 0
    bench_long = 0
    bench_short = 0

    if strict_breakout:
        ge = lambda a, b: a > b
        le = lambda a, b: a < b
    else:
        ge = lambda a, b: a >= b
        le = lambda a, b: a <= b


    for k in range(bars_back, n):
        traded = False
        delta = pv * (C[k] - C[k - 1]) * position

        if position == 0:
            buy = ge(H[k], HH[k])
            sell = le(L[k], LL[k])
            
            if buy and sell:
                delta = -slpg + pv * (LL[k] - HH[k])
                trades[k] = 1
            else:
                if buy:
                    delta = -slpg / 2 + pv * (C[k] - HH[k])
                    position = 1
                    traded = True
                    bench_long = H[k] if not use_close_for_bench else C[k]
                    trades[k] = 0.5
                if sell:
                    delta = -slpg / 2 - pv * (C[k] - LL[k])
                    position = -1
                    traded = True
                    bench_short = L[k] if not use_close_for_bench else C[k]
                    trades[k] = 0.5

        if position == 1 and not traded:
            sell_short = le(L[k], LL[k])
            sell_exit = L[k] <= bench_long * (1 - stp_pct)
            if sell_short and sell_exit:
                
                delta = delta - slpg - 2 * pv * (C[k] - LL[k])
                position = -1
                
                bench_short = L[k] if not use_close_for_bench else C[k]
                trades[k] = 1
            else:
                if sell_exit:
                    delta = delta - slpg / 2 - pv * (C[k] - bench_long * (1 - stp_pct))
                    position = 0
                    trades[k] = 0.5
                if sell_short:
                    delta = delta - slpg - 2 * pv * (C[k] - LL[k])
                    position = -1
                    bench_short = L[k] if not use_close_for_bench else C[k]
                    trades[k] = 1
            if use_close_for_bench:
                bench_long = max(C[k], bench_long)
            else:
                bench_long = max(H[k], bench_long)

        if position == -1 and not traded:
            buy_long = ge(H[k], HH[k])
            buy_exit = H[k] >= bench_short * (1 + stp_pct)
            if buy_long and buy_exit:
                delta = delta - slpg + 2 * pv * (C[k] - HH[k])
                position = 1
                bench_long = H[k] if not use_close_for_bench else C[k]
                trades[k] = 1
            else:
                if buy_exit:
                    delta = delta - slpg / 2 + pv * (C[k] - bench_short * (1 + stp_pct))
                    position = 0
                    trades[k] = 0.5
                    
                if buy_long:
                    delta = delta - slpg + 2 * pv * (C[k] - HH[k])
                    position = 1
                    bench_long = H[k] if not use_close_for_bench else C[k]
                    trades[k] = 1
            if use_close_for_bench:
                bench_short = min(C[k], bench_short)
            else:
                bench_short = min(L[k], bench_short)

        E[k] = E[k - 1] + delta
        Emax = max(Emax, E[k])
        DD[k] = E[k] - Emax

    return E, DD, trades


def stats(E, DD, trades, i0, i1):
    pnl = np.concatenate(([0.0], np.diff(E[i0:i1+1])))
    return {
        'profit': E[i1] - E[i0],
        'worst_dd': DD[i0:i1+1].min(),
        'stdev': np.std(pnl, ddof=1),
        'trades': trades[i0:i1+1].sum(),
    }


def main():
    dt, H, L, C = load_hlv_csv('instruction_and_data/HO-5minHLV.csv')
    L_, S = 12700, 0.010
    bars_back = 17001
    pv = 42_000.0
    slpg = 47.0
    e0 = 100_000.0
    is_lo, is_hi = 17000, 226119
    os_lo, os_hi = 226120, 611836

    print("Expected (HO Optimization.xlsx, L=12700, S=0.01):")
    print("IS profit=19988.688, dd=-3280.682, std=19.633, trades=123")
    print("OS profit=172337.836, dd=-14299.070, std=56.860, trades=466")

    print()

    variants = [
        ("baseline (main.m literal)", {}),
        ("strict_breakout (>)", {"strict_breakout": True}),
        ("window_include_k", {"window_include_k": True}),
        ("use_close_for_bench", {"use_close_for_bench": True}),
        ("strict + close bench", {"strict_breakout": True, "use_close_for_bench": True}),
    ]

    for name, kwargs in variants:
        t0 = time.time()
        E, DD, trades = run_variant(H, L, C, bars_back, L_, S, pv, slpg, e0, **kwargs)
        isS = stats(E, DD, trades, is_lo, is_hi)
        osS = stats(E, DD, trades, os_lo, os_hi)
        dt_s = time.time() - t0

        print(f"{name} ({dt_s:.1f}s)")
        print(f"IS: profit={isS['profit']:.2f} dd={isS['worst_dd']:.2f} std={isS['stdev']:.3f} trades={isS['trades']}")
        print(f"OS: profit={osS['profit']:.2f} dd={osS['worst_dd']:.2f} std={osS['stdev']:.3f} trades={osS['trades']}")


if __name__ == "__main__":
    main()
