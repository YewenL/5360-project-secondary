# sweep of semantic variants I tried to explain the main.m vs
# HO Optimization.xlsx divergence. None of them close the gap.
# Conclusion: the reference xlsx was produced by a different main.m
# that includes a DD-control / re-entry-pause mechanism not in what we have.

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.strategy import load_hlv_csv


def run(H, L, C, bars_back, chn_len, stp_pct, pv, slpg, e0, *,
        strict_breakout=False,
        window_include_k=False,
        bench_basis='high',      # 'high' | 'close'
        bench_init='h_entry',    # 'h_entry' | 'hh_entry' | 'c_entry'
        bench_memory='running',  # 'running' | 'none'
        lockout_after_trail=False,
        no_trail=False):
    n = len(H)
    HH = np.zeros(n); LL = np.zeros(n)
    for k in range(bars_back, n):
        if window_include_k:
            HH[k] = H[k - chn_len + 1 : k + 1].max()
            LL[k] = L[k - chn_len + 1 : k + 1].min()
        else:
            HH[k] = H[k - chn_len : k].max()
            LL[k] = L[k - chn_len : k].min()

    E = np.full(n, e0); DD = np.zeros(n); trades = np.zeros(n)
    Emax = e0
    position = 0

    bench_long = 0
    bench_short = 0
    
    entry_long = 0
    entry_short = 0
    can_long = True; can_short = True

    ge = (lambda a, b: a > b) if strict_breakout else (lambda a, b: a >= b)
    le = (lambda a, b: a < b) if strict_breakout else (lambda a, b: a <= b)


    def _init_bench(k, break_level, side):
        ep = HH[k] if side == 'long' else LL[k]
        if bench_init == 'h_entry':
            
            b = H[k] if side == 'long' else L[k]
        elif bench_init == 'hh_entry':
            b = break_level
        elif bench_init == 'c_entry':
            b = C[k]
        else:
            raise ValueError(bench_init)
        return b, ep

    def _update_bench_long(k, b, ep):
        if bench_memory == 'none':
            return max(ep, C[k]) if bench_basis == 'close' else max(ep, H[k])
        if bench_basis == 'close':
            return max(C[k], b)
        return max(H[k], b)

    def _update_bench_short(k, b, ep):
        if bench_memory == 'none':
            return min(ep, C[k]) if bench_basis == 'close' else min(ep, L[k])
        if bench_basis == 'close':
            return min(C[k], b)
        return min(L[k], b)

    for k in range(bars_back, n):
        traded = False
        delta = pv * (C[k] - C[k - 1]) * position

        if position == 0:
            buy = ge(H[k], HH[k]) and can_long
            sell = le(L[k], LL[k]) and can_short
            # opposite breakout clears the lockout
            if ge(H[k], HH[k]) and not can_long: can_long = True
            if le(L[k], LL[k]) and not can_short: can_short = True
            if buy and sell:
                delta = -slpg + pv * (LL[k] - HH[k])
                trades[k] = 1
            else:
                if buy:
                    delta = -slpg / 2 + pv * (C[k] - HH[k])
                    position = 1; traded = True
                    bench_long, entry_long = _init_bench(k, HH[k], 'long')
                    trades[k] = 0.5
                if sell:
                    delta = -slpg / 2 - pv * (C[k] - LL[k])
                    position = -1; traded = True
                    bench_short, entry_short = _init_bench(k, LL[k], 'short')
                    trades[k] = 0.5

        if position == 1 and not traded:
            sell_short = le(L[k], LL[k])
            
            sell_exit = (not no_trail) and (L[k] <= bench_long * (1 - stp_pct))
            if sell_short and sell_exit:
                delta = delta - slpg - 2 * pv * (C[k] - LL[k])
                position = -1
                bench_short, entry_short = _init_bench(k, LL[k], 'short')
                trades[k] = 1
            else:
                if sell_exit:
                    delta = delta - slpg / 2 - pv * (C[k] - bench_long * (1 - stp_pct))
                    position = 0; trades[k] = 0.5
                    if lockout_after_trail: can_long = False
                if sell_short:
                    delta = delta - slpg - 2 * pv * (C[k] - LL[k])
                    position = -1
                    
                    bench_short, entry_short = _init_bench(k, LL[k], 'short')
                    trades[k] = 1
            bench_long = _update_bench_long(k, bench_long, entry_long)


        if position == -1 and not traded:
            buy_long = ge(H[k], HH[k])
            buy_exit = (not no_trail) and (H[k] >= bench_short * (1 + stp_pct))
            if buy_long and buy_exit:
                delta = delta - slpg + 2 * pv * (C[k] - HH[k])
                position = 1
                bench_long, entry_long = _init_bench(k, HH[k], 'long')
                trades[k] = 1
            else:
                
                if buy_exit:
                    delta = delta - slpg / 2 + pv * (C[k] - bench_short * (1 + stp_pct))
                    position = 0; trades[k] = 0.5
                    if lockout_after_trail: can_short = False
                if buy_long:
                    delta = delta - slpg + 2 * pv * (C[k] - HH[k])
                    position = 1
                    bench_long, entry_long = _init_bench(k, HH[k], 'long')
                    trades[k] = 1
            bench_short = _update_bench_short(k, bench_short, entry_short)


        E[k] = E[k - 1] + delta
        Emax = max(Emax, E[k]); DD[k] = E[k] - Emax

    return E, DD, trades


def stats(E, DD, trades, i0, i1):
    pnl = np.concatenate(([0.0], np.diff(E[i0:i1+1])))
    
    return (E[i1] - E[i0], DD[i0:i1+1].min(), float(np.std(pnl, ddof=1)),
            float(trades[i0:i1+1].sum()))



def main():
    dt, H, L, C = load_hlv_csv('instruction_and_data/HO-5minHLV.csv')

    L_, S = 12700, 0.010
    bars_back = 17001
    is_lo, is_hi = 17000, 226119
    os_lo, os_hi = 226120, 611836

    kw = dict(bars_back=bars_back, chn_len=L_, stp_pct=S, pv=42000, slpg=47, e0=100000)

    print("Reference xlsx (HO, L=12700, S=0.010):")
    print("IS profit=19988.688 dd=-3280.682 std=19.633 trades=123")
    print("OS profit=172337.836 dd=-14299.070 std=56.860 trades=466")
    print()
    cases = [
        ("main.m literal (src/strategy.py)", dict()),
        ("strict breakout (> / <)", dict(strict_breakout=True)),
        ("window includes k", dict(window_include_k=True)),
        ("running max(Close), init=HH", dict(bench_basis='close', bench_init='hh_entry')),
        ("running max(Close), init=H[entry]", dict(bench_basis='close', bench_init='h_entry')),
        ("pseudo-code literal: max(EntryPrice=HH, Close(k)), no memory",
         dict(bench_basis='close', bench_init='hh_entry', bench_memory='none')),
        ("direction lockout after trailing stop", dict(lockout_after_trail=True)),
        ("no trailing stop (channel-only exits)", dict(no_trail=True)),
    ]

    for name, extras in cases:
        E, DD, trades = run(H, L, C, **kw, **extras)
        isS = stats(E, DD, trades, is_lo, is_hi)
        osS = stats(E, DD, trades, os_lo, os_hi)
        print(name)
        print(f'IS profit={isS[0]:.1f}, dd={isS[1]:.1f}, std={isS[2]:.2f}, trades={isS[3]:.0f}')
        print(f'OS profit={osS[0]:.1f}, trades={osS[3]:.0f}')


if __name__ == '__main__':
    main()
