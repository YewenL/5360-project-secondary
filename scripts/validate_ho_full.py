# run main.m engine (= src/strategy.py) on every (L, S) point that appears
# in HO MatLab Test.xlsx (66 points), and compare to both reference xlsx files
# point-by-point. prints a compact per-point table plus summary stats.

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.strategy import (
    load_hlv_csv,
    index_at_or_after,
    index_at_or_before,
    run_strategy,
)
from scripts.validate_ho import load_reference_grid, OPT_XLSX, TEST_XLSX, DATA

PV = 42_000.0
SLPG = 47.0
BARS_BACK = 17001
E0 = 100_000.0

IS_START = "1980-01-01"
IS_END = "2000-01-01"
OS_START = "2000-01-01"
OS_END = "2023-03-23"


def main():
    t0 = time.time()
    print(f"loading {DATA.name} ...")
    dt, H, L, C = load_hlv_csv(str(DATA))
    print(f"  {len(H):,} bars  ({time.time()-t0:.1f}s)")

    is_lo = max(index_at_or_after(dt, IS_START), BARS_BACK - 1)
    is_hi = max(index_at_or_before(dt, IS_END), BARS_BACK - 1)
    os_lo = max(index_at_or_after(dt, OS_START), BARS_BACK - 1)
    os_hi = max(index_at_or_before(dt, OS_END), BARS_BACK - 1)

    ref_opt = load_reference_grid(OPT_XLSX)
    ref_test = load_reference_grid(TEST_XLSX)

    # iterate over HO Test's 66 points (all are also in Opt)
    keys = sorted(ref_test.keys())
    print(f"  scanning {len(keys)} (L, S) points\n")

    # collect ratios
    is_trade_ratio_opt = []
    is_trade_ratio_test = []
    os_trade_ratio_opt = []
    is_profit_diff_opt = []
    is_profit_diff_test = []
    os_profit_diff_opt = []

    print(f"{'L':>6} {'S':>6} | {'ours IS P/T':>14} | {'opt IS P/T':>13} "
          f"{'test IS P/T':>13} | {'ours OS P/T':>14} {'opt OS P/T':>14} | "
          f"{'IS TrRatio':>10}")
    print("-" * 120)

    for L_, S in keys:
        res = run_strategy(H, L, C, chn_len=L_, stp_pct=S, pv=PV, slpg=SLPG,
                           bars_back=BARS_BACK, e0=E0)
        wis = res.window_stats(is_lo, is_hi)
        wos = res.window_stats(os_lo, os_hi)
        optIS, optOS = ref_opt[(L_, S)]["is"], ref_opt[(L_, S)]["os"]
        testIS = ref_test[(L_, S)]["is"]

        is_trade_ratio_opt.append(wis.trades / optIS[3])
        is_trade_ratio_test.append(wis.trades / testIS[3])
        os_trade_ratio_opt.append(wos.trades / optOS[3])
        is_profit_diff_opt.append(wis.profit - optIS[0])
        is_profit_diff_test.append(wis.profit - testIS[0])
        os_profit_diff_opt.append(wos.profit - optOS[0])

        print(f"{L_:>6} {S:>6.3f} | "
              f"{wis.profit:>8.0f}/{wis.trades:>4.0f} | "
              f"{optIS[0]:>7.0f}/{optIS[3]:>4.0f} "
              f"{testIS[0]:>7.0f}/{testIS[3]:>4.0f} | "
              f"{wos.profit:>8.0f}/{wos.trades:>4.0f} "
              f"{optOS[0]:>8.0f}/{optOS[3]:>4.0f} | "
              f"{wis.trades/optIS[3]:>10.2f}")

    def stats(arr, name):
        a = np.array(arr)
        print(f"  {name:<30} min={a.min():>8.3f}  med={np.median(a):>8.3f}  "
              f"mean={a.mean():>8.3f}  max={a.max():>8.3f}  std={a.std():>6.3f}")

    print("\n=== summary across 66 (L, S) points ===")
    stats(is_trade_ratio_opt, "IS trades ratio (ours/opt)")
    stats(is_trade_ratio_test, "IS trades ratio (ours/test)")
    stats(os_trade_ratio_opt, "OS trades ratio (ours/opt)")
    stats(is_profit_diff_opt, "IS profit diff (ours-opt)")
    stats(is_profit_diff_test, "IS profit diff (ours-test)")
    stats(os_profit_diff_opt, "OS profit diff (ours-opt)")

    # are opt and test consistent with each other?
    print("\n=== opt vs test consistency (reference-only) ===")
    t_diffs = []
    p_diffs = []
    for k in keys:
        tA, tB = ref_opt[k]["is"], ref_test[k]["is"]
        t_diffs.append(tA[3] - tB[3])
        p_diffs.append(tA[0] - tB[0])
    print(f"IS trades (opt - test): max abs diff = {max(abs(x) for x in t_diffs):.1f}, mean diff = {np.mean(t_diffs):+.2f}")
    print(f"IS profit (opt - test): max abs diff = {max(abs(x) for x in p_diffs):.0f}, mean diff = {np.mean(p_diffs):+.1f}, pct = {100*np.mean(p_diffs)/np.mean([ref_opt[k]['is'][0] for k in keys]):+.1f}%")

    print(f"\ntotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
