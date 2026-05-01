"""Command-line entry point for walk-forward optimization."""
# how you run ex:
#     .venv/bin/python scripts/run_wfo.py \
#     --data instruction_and_data/PL-5minHLV/PL-5minHLV.csv \
#     --pv 50 \
#     --slpg 148 \
#     --full \
#     --parallel \
#     --workers 8 \
#     --out-dir results/pl_wfo_full

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
from src.walk_forward import WFOConfig, WFOGrid, rolling_wfo, rolling_wfo_parallel

DATA = ROOT / "instruction_and_data" / "PL-5minHLV.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="use PDF 91,296-point grid (slow; use --parallel)")
    ap.add_argument("--parallel", action="store_true",
                    help="run quarters in parallel across CPU cores")
    ap.add_argument("--workers", type=int, default=None,
                    help="# worker processes (default: all cores)")
    ap.add_argument("--from", dest="start", default=None,
                    help="earliest OS quarter start date (YYYY-MM-DD)")
    ap.add_argument("--to", dest="end", default=None,
                    help="latest OS quarter end date (YYYY-MM-DD)")
    ap.add_argument("--is-years", type=int, default=4)
    ap.add_argument("--os-months", type=int, default=3)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--pv", type=float, default=42_000.0,
                    help="contract point value / multiplier (default: HO reference)")
    ap.add_argument("--slpg", type=float, default=47.0,
                    help="full round-turn slippage cost (default: HO reference)")
    ap.add_argument("--e0", type=float, default=100_000.0,
                    help="starting equity per optimization/run window")
    ap.add_argument("--out-dir", default=None,
                    help="optional directory for quarter params, OS equity, and trades CSVs")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print(f"loading {args.data} ...")
    dt, H, L, C = load_hlv_csv(args.data)
    print(f"  {len(H):,} bars  {dt[0]} -> {dt[-1]}   ({time.time()-t0:.1f}s)")

    grid = WFOGrid.pdf_default() if args.full else WFOGrid.smoke()
    cfg = WFOConfig(is_years=args.is_years, os_months=args.os_months,
                    pv=args.pv, slpg=args.slpg, e0=args.e0)
    print(f"grid size = {grid.size():,}  (L: {len(grid.l_values)}, S: {len(grid.s_values)})")
    print(f"IS = {cfg.is_years} years  OS = {cfg.os_months} months")
    
    print(f"PV = {cfg.pv:g}  Slpg = {cfg.slpg:g}  E0 = {cfg.e0:g}")
    print(f"mode: {'parallel' if args.parallel else 'serial'}\n")

    t1 = time.time()
    runner = rolling_wfo_parallel if args.parallel else rolling_wfo
    kwargs = {"start": args.start, "end": args.end,
              "progress": not args.no_progress}
    
    if args.parallel:
        kwargs["n_workers"] = args.workers
    res = runner(dt, H, L, C, grid, cfg, **kwargs)
    elapsed = time.time() - t1
    print(f"\nWFO done in {elapsed:.1f}s  ({len(res.quarters)} quarters)")

    if not res.quarters:
        print("no quarters produced -- adjust date range or check data")
        return

    stats = compute_stats(res.os_equity, res.os_pnl, res.os_pos_end,
                          os_dt=res.os_dt)
    print()
    print(format_stats(stats, title="OS aggregated performance"))

    print("per-quarter optimal params(head)")
    print(f"  {'Q':<8} {'L':>6} {'S':>6}  {'IS npdd':>9}  {'IS profit':>10}  {'OS profit':>10}  {'OS trades':>9}")
    for q in res.quarters[:20]:
        print(f"  {q.q_label:<8} {q.best_L:>6} {q.best_S:>6.3f}  "
              f"{q.is_npdd:>9.3f}  {q.is_profit:>10.1f}  {q.os_profit:>10.1f}  {q.os_trades:>9.1f}")
    if len(res.quarters) > 20:
        print(f"... and {len(res.quarters)- 20} more quarters")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(q) for q in res.quarters]).to_csv(
            out_dir / "wfo_quarter_params.csv", index=False)
        
        pd.DataFrame({
            "DateTime": res.os_dt.astype("datetime64[s]").astype(str),
            "Equity": res.os_equity,
            "PnL": res.os_pnl,
            "Trades": res.os_trades_bar,
            "PosEnd": res.os_pos_end,
        }).to_csv(out_dir / "wfo_os_equity.csv", index=False)
        
        trades = extract_trades(res.os_pnl, res.os_pos_end)
        pd.DataFrame([
            {
                "entry_bar": t.entry_bar,
                "exit_bar": t.exit_bar,
                "entry_time": str(res.os_dt[t.entry_bar]),
                "exit_time": str(res.os_dt[t.exit_bar]),
                "side": t.side,
                "pnl": t.pnl,
            }
            for t in trades
            
        ]).to_csv(out_dir / "wfo_os_trades.csv", index=False)
        print(f"\nsaved{out_dir}")

    print(f"\ntotal wall time:{time.time()-t0:.1f}s")


if __name__ == "__main__":
    
    main()
