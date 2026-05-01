# test: after trailing-stop exit, block same-direction re-entry.
# hypothesis: the "WithDDControl" mechanism is a hysteresis that prevents
# whipsaw re-entries after being stopped out.

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
    Emax = e0
    position = 0
    bench_long = 0
    bench_short = 0
    # direction lockout flags: set after trailing stop, cleared on opposite breakout
    can_long = True
    can_short = True
    #hiiiiiiiiiiiiiiiiiii
    
    for k in range(bars_back, n):
        traded = False
        delta = pv * (C[k] - C[k - 1]) * position

        if position == 0:
            buy = H[k] >= HH[k]
            sell = L[k] <= LL[k]
            buy_ok = buy and can_long
            
            sell_ok = sell and can_short
            if buy and not can_long:
                can_long = True  # reset lockout even though we don't trade
            if sell and not can_short:
                can_short = True
            if buy_ok and sell_ok:
                delta = -slpg + pv * (LL[k] - HH[k])
                trades[k] = 1
            else:
                if buy_ok:
                    delta = -slpg / 2 + pv * (C[k] - HH[k])
                    position = 1; traded = True
                    bench_long = H[k]; trades[k] = 0.5
                    can_short = True
                if sell_ok:
                    delta = -slpg / 2 - pv * (C[k] - LL[k])
                    position = -1; traded = True
                    bench_short = L[k]; trades[k] = 0.5
                    can_long = True

        if position == 1 and not traded:
            sell_short = L[k] <= LL[k]
            
            sell_exit = L[k] <= bench_long * (1 - stp_pct)
            if sell_short and sell_exit:
                delta = delta - slpg - 2 * pv * (C[k] - LL[k])
                position = -1; bench_short = L[k]; trades[k] = 1
                can_long = True
            else:
                if sell_exit:
                    delta = delta - slpg / 2 - pv * (C[k] - bench_long * (1 - stp_pct))
                    position = 0; trades[k] = 0.5
                    can_long = False  # stopped out -> can't re-enter long until opposite
                if sell_short:
                    delta = delta - slpg - 2 * pv * (C[k] - LL[k])
                    position = -1; bench_short = L[k]; trades[k] = 1
                    can_long = True
            bench_long = max(H[k], bench_long)

        if position == -1 and not traded:
            buy_long = H[k] >= HH[k]
            buy_exit = H[k] >= bench_short * (1 + stp_pct)
            if buy_long and buy_exit:
                delta = delta - slpg + 2 * pv * (C[k] - HH[k])
                position = 1; bench_long = H[k]; trades[k] = 1
                can_short = True
            else:
                if buy_exit:
                    delta = delta - slpg / 2 + pv * (C[k] - bench_short * (1 + stp_pct))
                    position = 0; trades[k] = 0.5
                    can_short = False
                if buy_long:
                    delta = delta - slpg + 2 * pv * (C[k] - HH[k])
                    position = 1; bench_long = H[k]; trades[k] = 1
                    can_short = True
            bench_short = min(L[k], bench_short)

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
        p = E[hi] - E[lo]
        dd = DD[lo:hi+1].min()
        sd = np.std(pnl, ddof=1)
        tr = trades[lo:hi+1].sum()
        print(f'{name}: profit {p:.2f}, dd {dd:.2f}, std {sd:.3f}, trades {tr}')

    print('ref IS: trades=123 profit=19988.69 dd=-3280.68 std=19.63')
    print('ref OS: trades=466 profit=172337.84')


if __name__ == '__main__':
    main()
