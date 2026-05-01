# test: drop trailing stop entirely - only exit on opposite channel breakout

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.strategy import load_hlv_csv


def run_no_trail(H, L, C, bars_back, chn_len, stp_pct, pv, slpg, e0):
    n = len(H)
    HH = np.zeros(n); LL = np.zeros(n)
    for k in range(bars_back, n):
        HH[k] = H[k - chn_len : k].max()
        LL[k] = L[k - chn_len : k].min()

    E = np.full(n, e0); DD = np.zeros(n); trades = np.zeros(n)
    Emax = e0; position = 0
    for k in range(bars_back, n):
        traded = False
        delta = pv * (C[k] - C[k - 1]) * position

        if position == 0:
            buy = H[k] >= HH[k]; sell = L[k] <= LL[k]
            if buy and sell:
                delta = -slpg + pv * (LL[k] - HH[k])
                trades[k] = 1
            else:
                if buy:
                    delta = -slpg / 2 + pv * (C[k] - HH[k])
                    position = 1; traded = True; trades[k] = 0.5
                if sell:
                    delta = -slpg / 2 - pv * (C[k] - LL[k])
                    position = -1; traded = True; trades[k] = 0.5

        if position == 1 and not traded:
            if L[k] <= LL[k]:  # short breakout -> reverse
                delta = delta - slpg - 2 * pv * (C[k] - LL[k])
                position = -1; trades[k] = 1

        if position == -1 and not traded:
            if H[k] >= HH[k]:
                delta = delta - slpg + 2 * pv * (C[k] - HH[k])
                position = 1; trades[k] = 1

        E[k] = E[k - 1] + delta
        Emax = max(Emax, E[k]); DD[k] = E[k] - Emax

    return E, DD, trades


def main():
    dt, H, L, C = load_hlv_csv('instruction_and_data/HO-5minHLV.csv')
    L_, S = 12700, 0.010
    bars_back = 17001
    is_lo, is_hi = 17000, 226119
    os_lo, os_hi = 226120, 611836

    E, DD, trades = run_no_trail(H, L, C, bars_back, L_, S, 42000, 47, 100000)

    for name, lo, hi in [('IS', is_lo, is_hi), ('OS', os_lo, os_hi)]:
        pnl = np.concatenate(([0], np.diff(E[lo:hi+1])))
        print(f'{name} profit={E[hi]-E[lo]:.2f} dd={DD[lo:hi+1].min():.2f} '
              f'std={np.std(pnl,ddof=1):.3f} trades={trades[lo:hi+1].sum()}')

    print()
    print('reference: IS 123 trades / profit 19988.69 / dd -3280.68 / std 19.63')
    print('OS 466 trades / profit 172337.84')


if __name__ == '__main__':
    main()
