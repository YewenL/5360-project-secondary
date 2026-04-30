"""Command-line entry point for walk-forward optimization."""
# how you run ex:
#     .venv/bin/python scripts/run_wfo.py \
#     --data instruction_and_data/AUG-5minHLV.csv \
#     --pv 1000 \
#     --slpg 0.07\ 
#     -- originally in CNY, convert to USD
#     Conversion rate = 6.8370 (CNY → USD) from yahoo finance on 2026-04-29
#     Convert PV & slippage
#     --full \
#     --parallel \
#     --workers 8 \
#     --out-dir results/aug_full_is

#config workers is the core of your CPU, 8 workers would take less than 10 mins to run so feel free to test.



import argparse
from dataclasses import asdict
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.stats import compute_stats, extract_trades, format_stats
from src.strategy import load_hlv_csv
from src.walk_forward import WFOConfig, WFOGrid, _run_window_for_grid, _best_from_grid, _run_single_window

DATA = ROOT / "instruction_and_data" / "AUG-5minHLV.csv"


def main():
    FX_RATE = 6.8370
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="use PDF 91,296-point grid")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--pv", type=float, default= 1000/FX_RATE,
                    help="AU Gold point value converted from CNY to USD using 6.8370 CNY/USD)")
    ap.add_argument("--slpg", type=float, default=0.07/FX_RATE,
                    help="full round-turn slippage cost converted from CNY to USD using 6.8370 CNY/USD (default: AUG reference)")
    ap.add_argument("--e0", type=float, default=100_000.0,
                    help="starting equity per optimization/run window")
    ap.add_argument("--out-dir", default=None,
                    help="optional directory for params, equity, and trades CSVs")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print(f"loading {args.data} ...")
    dt, H, L, C = load_hlv_csv(args.data)
    print(f"  {len(H):,} bars  {dt[0]} -> {dt[-1]}   ({time.time()-t0:.1f}s)")

    grid = WFOGrid.pdf_default() if args.full else WFOGrid.smoke()
    cfg = WFOConfig(pv=args.pv, slpg=args.slpg, e0=args.e0)
    print(f"grid size = {grid.size():,}  (L: {len(grid.l_values)}, S: {len(grid.s_values)})")    
    print(f"PV = {cfg.pv:g}  Slpg = {cfg.slpg:g}  E0 = {cfg.e0:g}")
    print("mode: full in-sample\n")
    
    win_lo = int(max(grid.l_values))
    win_hi = len(H) - 1
    t1 = time.time()
    grid_res = _run_window_for_grid(H, L, C, win_lo, win_hi, grid, cfg)
    best = _best_from_grid(grid_res)

    if best is None:
        print("no valid full in-sample result -- check data length/grid")
        return
    best_L, best_S, is_profit, is_dd, is_trades, is_npdd = best

    print(f"optimization done in {time.time() - t1:.1f}s")
    print()
    print("Full in-sample best params")
    print(f"  L         = {best_L}")
    print(f"  S         = {best_S:.3f}")
    print(f"  IS profit = {is_profit:.2f}")
    print(f"  IS DD     = {is_dd:.2f}")
    print(f"  IS NP/DD  = {is_npdd:.4f}")
    print(f"  IS trades = {is_trades:.1f}")
    profit, dd, trades_n, pnl, trades_bar, pos = _run_single_window(
        H, L, C, win_lo, win_hi, best_L, best_S, cfg
    )
    equity = cfg.e0 + np.cumsum(pnl)

    stats = compute_stats(equity, pnl, pos,
                          os_dt=dt[win_lo:win_hi + 1])
    print()
    print(format_stats(stats, title="Full in-sample performance"))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "best_L": best_L,
            "best_S": best_S,
            "is_profit": is_profit,
            "is_dd": is_dd,
            "is_npdd": is_npdd,
            "is_trades": is_trades,
            "start_time": str(dt[win_lo]),
            "end_time": str(dt[win_hi]),
        }]).to_csv(
            out_dir / "full_is_params_aug.csv", index=False)
        
        pd.DataFrame({
            "DateTime": dt[win_lo:win_hi + 1].astype("datetime64[s]").astype(str),
            "Equity": equity,
            "PnL": pnl,
            "Trades": trades_bar,
            "PosEnd": pos,
        }).to_csv(out_dir / "full_is_equity_aug.csv", index=False)
        
        full_trades = extract_trades(pnl, pos)
        pd.DataFrame([
            {
                "entry_bar": t.entry_bar,
                "exit_bar": t.exit_bar,
                "entry_time": str(dt[win_lo + t.entry_bar]),
                "exit_time": str(dt[win_lo + t.exit_bar]),
                "side": t.side,
                "pnl": t.pnl,
            }
            for t in full_trades 
        ]).to_csv(out_dir / "full_is_trades_aug.csv", index=False)
        print(f"\nsaved{out_dir}")

    print(f"\ntotal wall time:{time.time()-t0:.1f}s")


if __name__ == "__main__":
    
    main()
