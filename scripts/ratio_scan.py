# scan trade-count ratio (mine / reference) across (L,S) grid.
# if the ratio is tight everywhere -> systematic bug (double counting etc).
# if it varies a lot -> semantic difference that interacts with data.

import sys
from pathlib import Path
import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.strategy import load_hlv_csv, run_strategy
from scripts.validate_ho import load_reference_grid


def main():
    dt, H, L, C = load_hlv_csv('instruction_and_data/HO-5minHLV.csv')
    is_lo, is_hi = 17000, 226119
    os_lo, os_hi = 226120, 611836

    opt = load_reference_grid(ROOT / 'instruction_and_data' / 'HO Optimization.xlsx')
    sample = sorted(opt.keys())[::37]  # ~25 points spread across the grid

    print('L,S -> IS: ref / mine / ratio | OS: ref / mine / ratio')

    for (L_, S) in sample:
        res = run_strategy(H, L, C, chn_len=L_, stp_pct=S, pv=42000, slpg=47,
                          bars_back=17001, e0=100000)
        wis = res.window_stats(is_lo, is_hi)
        wos = res.window_stats(os_lo, os_hi)

        ref = opt[(L_, round(S,3))]
        rti, rto = ref['is'][3], ref['os'][3]
        ri = wis.trades / rti if rti else float('nan')
        
        
        ro = wos.trades / rto if rto else float('nan')
        print(f'L={L_} S={S:.3f}: IS {rti:.0f}/{wis.trades:.0f}/{ri:.2f}x  '
              f'OS {rto:.0f}/{wos.trades:.0f}/{ro:.2f}x')


if __name__ == '__main__':
    main()
