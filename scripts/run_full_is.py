"""Full-period in-sample optimization.

Treats the entire data sample as a single in-sample window, scans the full
91,296-point (L, S) grid, and reports the global IS-NPDD-optimal parameters.
This is the upper-bound benchmark used to compute the decay coefficient
(OOS performance vs full-IS performance).

Per the project PDF: "After you are done with [the rolling WFO] you can also
solve the in-sample optimization with the whole period being the in-sample
length."

Example:
    .venv/bin/python scripts/run_full_is.py \
        --data instruction_and_data/PL-5minHLV/PL-5minHLV.csv \
        --pv 50 --slpg 148 \
        --full --parallel --workers 8 \
        --out-dir results/pl_full_is
"""

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.stats import compute_stats, extract_trades, format_stats
from src.strategy import load_hlv_csv, run_strategy
from src.walk_forward import (
    WFOConfig,
    WFOGrid,
    _best_from_grid,
    _run_window_for_grid,
)

DATA = ROOT / "instruction_and_data" / "HO-5minHLV.csv"


def _grid_chunk_indices(n_l: int, n_workers: int):
    bounds = np.linspace(0, n_l, n_workers + 1, dtype=int)
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_workers)
            if bounds[i + 1] > bounds[i]]


_WORKER_STATE = {}


def _pool_init(H, L_arr, C, win_lo, win_hi, cfg, l_values, s_values):
    _WORKER_STATE["H"] = H
    _WORKER_STATE["L"] = L_arr
    _WORKER_STATE["C"] = C
    _WORKER_STATE["win_lo"] = win_lo
    _WORKER_STATE["win_hi"] = win_hi
    _WORKER_STATE["cfg"] = cfg
    _WORKER_STATE["l_values"] = l_values
    _WORKER_STATE["s_values"] = s_values


def _pool_run(chunk):
    lo, hi = chunk
    sub_grid = WFOGrid(
        l_values=_WORKER_STATE["l_values"][lo:hi],
        s_values=_WORKER_STATE["s_values"],
    )
    return _run_window_for_grid(
        _WORKER_STATE["H"], _WORKER_STATE["L"], _WORKER_STATE["C"],
        _WORKER_STATE["win_lo"], _WORKER_STATE["win_hi"], sub_grid, _WORKER_STATE["cfg"],
    )


def _scan_grid_parallel(H, L_arr, C, win_lo, win_hi, grid, cfg, n_workers):
    import multiprocessing as mp

    chunks = _grid_chunk_indices(len(grid.l_values), n_workers)
    if len(chunks) <= 1:
        return _run_window_for_grid(H, L_arr, C, win_lo, win_hi, grid, cfg)

    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=len(chunks),
        initializer=_pool_init,
        initargs=(H, L_arr, C, win_lo, win_hi, cfg, grid.l_values, grid.s_values),
    ) as pool:
        results = pool.map(_pool_run, chunks)
    flat = []
    for chunk_results in results:
        flat.extend(chunk_results)
    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="use PDF 91,296-point grid (slow without --parallel)")
    ap.add_argument("--parallel", action="store_true",
                    help="split grid across CPU cores")
    ap.add_argument("--workers", type=int, default=None,
                    help="# worker processes (default: all cores)")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--pv", type=float, default=42_000.0,
                    help="contract point value / multiplier (default: HO reference)")
    ap.add_argument("--slpg", type=float, default=47.0,
                    help="full round-turn slippage cost (default: HO reference)")
    ap.add_argument("--e0", type=float, default=100_000.0,
                    help="starting equity")
    ap.add_argument("--out-dir", default=None,
                    help="optional directory for best_params, equity, and trades CSVs")
    args = ap.parse_args()

    t0 = time.time()
    print(f"loading {args.data} ...")
    dt, H, L_arr, C = load_hlv_csv(args.data)
    n = len(H)
    print(f" {n:,} bars  {dt[0]} -> {dt[-1]}   ({time.time()-t0:.1f}s)")

    grid = WFOGrid.pdf_default() if args.full else WFOGrid.smoke()
    cfg = WFOConfig(pv=args.pv, slpg=args.slpg, e0=args.e0)
    eval_lo = int(grid.l_values.max())
    eval_hi = n - 1
    eval_bars = eval_hi - eval_lo + 1
    print(f"grid size = {grid.size():,}  (L: {len(grid.l_values)}, S: {len(grid.s_values)})")
    print(f"IS window  = bars [{eval_lo:,}, {eval_hi:,}]  ({eval_bars:,} bars; "
          f"{pd.Timestamp(dt[eval_lo])} -> {pd.Timestamp(dt[eval_hi])})")
    print(f"  warmup   = bars [0, {eval_lo - 1:,}]  ({eval_lo:,} bars, used for L<=L_max channels)")
    print(f"PV = {cfg.pv:g}  Slpg = {cfg.slpg:g}  E0 = {cfg.e0:g}")
    print(f"mode: {'parallel' if args.parallel else 'serial'}\n")

    t1 = time.time()
    if args.parallel:
        import os as _os
        n_workers = args.workers or _os.cpu_count() or 1
        n_workers = max(1, min(n_workers, len(grid.l_values)))
        print(f"scanning grid with {n_workers} workers ...")
        grid_results = _scan_grid_parallel(H, L_arr, C, eval_lo, eval_hi, grid, cfg, n_workers)
    else:
        print("scanning grid (serial) ...")
        grid_results = _run_window_for_grid(H, L_arr, C, eval_lo, eval_hi, grid, cfg)
    elapsed = time.time() - t1
    print(f"grid scan done in {elapsed:.1f}s")

    best = _best_from_grid(grid_results)
    if best is None:
        print("no valid (L, S) combination found -- check grid / data")
        return
    bL, bS, is_p, is_dd, is_tr, is_npdd = best

    print()
    print("Full-IS optimum")
    print(f"L* = {bL}")
    print(f"S*= {bS:.4f}")
    print(f"Profit = {is_p:>14,.2f}")
    print(f"WorstDD = {is_dd:>14,.2f}")
    print(f"NPDD = {is_npdd:>14,.4f}")
    print(f"#trades= {is_tr:>14,.1f}")

    print("\nrerunning at (L*, S*) on full sample for equity / trade output ...")
    res = run_strategy(H, L_arr, C, chn_len=bL, stp_pct=bS,
                      pv=args.pv, slpg=args.slpg, e0=args.e0)
    pnl = np.empty(eval_hi - eval_lo + 1, dtype=np.float64)
    pnl[0] = res.E[eval_lo] - (res.e0 if eval_lo == 0 else res.E[eval_lo - 1])
    pnl[1:] = np.diff(res.E[eval_lo : eval_hi + 1])
    seg_equity = res.E[eval_lo : eval_hi + 1]
    seg_trades = res.trades[eval_lo : eval_hi + 1]
    seg_pos = res.pos_end[eval_lo : eval_hi + 1]
    seg_dt = dt[eval_lo : eval_hi + 1]

    stats = compute_stats(seg_equity, pnl, seg_pos, os_dt=seg_dt)
    print()
    print(format_stats(stats, title="Full-IS performance at (L*, S*)"))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame([{
            "best_L": bL,
            "best_S": bS,
            "is_lo": eval_lo,
            "is_hi": eval_hi,
            "is_start": str(pd.Timestamp(dt[eval_lo])),
            "is_end": str(pd.Timestamp(dt[eval_hi])),
            "is_bars": eval_bars,
            "profit": is_p,
            "worst_dd": is_dd,
            "npdd": is_npdd,
            "trades": is_tr,
            "pv": args.pv,
            "slpg": args.slpg,
            "e0": args.e0,
        }]).to_csv(out_dir / "best_params.csv", index=False)

        pd.DataFrame({
            "DateTime": seg_dt.astype("datetime64[s]").astype(str),
            "Equity": seg_equity,
            "PnL": pnl,
            "Trades": seg_trades,
            "PosEnd": seg_pos,
        }).to_csv(out_dir / "equity.csv", index=False)

        trades = extract_trades(pnl, seg_pos)
        pd.DataFrame([
            {
                "entry_bar": t.entry_bar + eval_lo,
                "exit_bar": t.exit_bar + eval_lo,
                "entry_time": str(seg_dt[t.entry_bar]),
                "exit_time": str(seg_dt[t.exit_bar]),
                "side": t.side,
                "pnl": t.pnl,
            }
            for t in trades
        ]).to_csv(out_dir / "trades.csv", index=False)
        print(f"saved -> {out_dir}")

    print(f"\ntotal wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
