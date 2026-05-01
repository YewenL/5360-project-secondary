"""Walk-forward optimization for the Channel strategy."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from src.strategy import (
    _run_core,
    rolling_max_prev,
    rolling_min_prev,
)


@dataclass
class WFOGrid:
    
    l_values: np.ndarray
    s_values: np.ndarray

    @classmethod
    def pdf_default(cls):
        return cls(
            l_values=np.arange(500, 10001, 10, dtype=np.int64),
            s_values=np.round(np.arange(0.005, 0.10 + 1e-9, 0.001), 6),
        )

    @classmethod
    def smoke(cls):
        return cls(
            l_values=np.array([2000, 5000, 8000], dtype=np.int64),
            s_values=np.array([0.005, 0.010, 0.020], dtype=np.float64),
        )

    def size(self):
        return len(self.l_values) * len(self.s_values)


@dataclass
class WFOConfig:
    is_years: int = 4
    os_months: int = 3
    pv: float = 42_000.0
    slpg: float = 47.0
    e0: float = 100_000.0


@dataclass
class WFOQuarter:
    q_label: str
    is_lo: int
    is_hi: int
    os_lo: int
    os_hi: int
    best_L: int
    best_S: float
    is_profit: float
    is_dd: float
    is_npdd: float
    is_trades: float
    os_profit: float
    os_dd: float
    os_trades: float


@dataclass
class WFOResult:
    quarters: List[WFOQuarter]
    os_equity: np.ndarray
    os_pnl: np.ndarray
    os_trades_bar: np.ndarray
    os_pos_end: np.ndarray
    os_dt: np.ndarray


def _np_dd_ratio(profit: float, worst_dd: float) -> float:
    if worst_dd >= -1e-12:
        return 1e18 if profit > 0 else (0.0 if profit == 0 else -1e18)
    return profit / abs(worst_dd)


def _slice_with_warmup(H, L, C, win_lo: int, win_hi: int, chn_len: int):
    start = max(0, win_lo - chn_len)
    
    H_s = H[start : win_hi + 1]
    L_s = L[start : win_hi + 1]
    C_s = C[start : win_hi + 1]
    
    bars_back_local = win_lo - start
    return H_s, L_s, C_s, bars_back_local, start


def _run_window_for_grid(H_arr, L_arr, C_arr,
                         win_lo: int, win_hi: int,
                         grid: WFOGrid, cfg: WFOConfig):
    out = []
    for L_ in grid.l_values:
        L_int = int(L_)
        if win_lo < L_int:
            continue
        H_s, L_s, C_s, bb, start = _slice_with_warmup(H_arr, L_arr, C_arr,
                                                     win_lo, win_hi, L_int)
        if bb < L_int:
            continue
        HH = rolling_max_prev(H_s, L_int)
        LL = rolling_min_prev(L_s, L_int)
        
        local_lo = bb
        local_hi = win_hi - start
        per_s = {}
        
        for S in grid.s_values:
            E, DD, tr, _ = _run_core(
                H_s, L_s, C_s, HH, LL,
                float(S), float(cfg.pv), float(cfg.slpg),
                int(bb), float(cfg.e0),
            )
            base_equity = float(cfg.e0) if local_lo == 0 else float(E[local_lo - 1])
            profit = float(E[local_hi] - base_equity)
            worst_dd = float(DD[local_lo : local_hi + 1].min())
            
            trades = float(tr[local_lo : local_hi + 1].sum())
            per_s[float(S)] = (profit, worst_dd, trades)
        out.append((L_int, per_s))
    return out


def _best_from_grid(grid_results):
    best_key = None
    best_payload = None
    for L_int, per_s in grid_results:
        for S, (profit, worst_dd, trades) in per_s.items():
            npdd = _np_dd_ratio(profit, worst_dd)
            
            key = (npdd, -L_int, -S)
            if best_key is None or key > best_key:
                best_key = key
                best_payload = (L_int, float(S), profit, worst_dd, trades, npdd)
    return best_payload


def _run_single_window(H_arr, L_arr, C_arr,
                       win_lo: int, win_hi: int,
                       chn_len: int, stp_pct: float, cfg: WFOConfig):
    H_s, L_s, C_s, bb, start = _slice_with_warmup(H_arr, L_arr, C_arr,
                                                 win_lo, win_hi, chn_len)
    HH = rolling_max_prev(H_s, chn_len)
    LL = rolling_min_prev(L_s, chn_len)
    
    E, DD, tr, pos = _run_core(
        H_s, L_s, C_s, HH, LL,
        float(stp_pct), float(cfg.pv), float(cfg.slpg),
        int(bb), float(cfg.e0),
    )
    local_lo = bb
    local_hi = win_hi - start
    base_equity = float(cfg.e0) if local_lo == 0 else float(E[local_lo - 1])
    profit = float(E[local_hi] - base_equity)
    
    worst_dd = float(DD[local_lo : local_hi + 1].min())
    trades = float(tr[local_lo : local_hi + 1].sum())

    seg_E = E[local_lo : local_hi + 1]
    seg_pnl = np.empty_like(seg_E)
    seg_pnl[0] = seg_E[0] - base_equity
    
    seg_pnl[1:] = np.diff(seg_E)
    seg_tr = tr[local_lo : local_hi + 1].copy()
    seg_pos = pos[local_lo : local_hi + 1].copy()
    return profit, worst_dd, trades, seg_pnl, seg_tr, seg_pos


def _bars_per_year(dt: np.ndarray) -> float:
    span_seconds = (pd.Timestamp(dt[-1]) - pd.Timestamp(dt[0])).total_seconds()
    span_years = span_seconds / (365.25 * 24 * 3600)
    if span_years <= 0:
        raise ValueError("dt must span a positive interval")
    return len(dt) / span_years


def iter_quarters(dt: np.ndarray, cfg: WFOConfig,
                  start: Optional[str] = None, end: Optional[str] = None):
    """Yield (label, is_lo, is_hi, os_lo, os_hi) for non-overlapping OS windows.
    Windows are sized by **bar count**, not calendar dates
    
    """
    if cfg.os_months <= 0:
        raise ValueError("os_months must be positive")
    n = len(dt)
    if n < 2:
        return

    bars_per_year = _bars_per_year(dt)
    is_bars = int(round(cfg.is_years * bars_per_year))
    step_bars = int(round((cfg.os_months / 12.0) * bars_per_year))
    if is_bars <= 0 or step_bars <= 0:
        return

    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None

    os_lo = is_bars
    while os_lo + step_bars <= n:
        os_hi = os_lo + step_bars - 1
        is_lo = os_lo - is_bars
        is_hi = os_lo - 1

        os_start_ts = pd.Timestamp(dt[os_lo])
        os_end_ts = pd.Timestamp(dt[os_hi])
        if start_ts is not None and os_start_ts < start_ts:
            os_lo += step_bars
            continue
        if end_ts is not None and os_end_ts > end_ts:
            break

        yield (_period_label(os_start_ts, cfg.os_months), is_lo, is_hi, os_lo, os_hi)
        os_lo += step_bars


def _period_label(start_ts: pd.Timestamp, os_months: int) -> str:
    if os_months == 3:
        return f"{start_ts.year}Q{((start_ts.month - 1) // 3) + 1}"
    return f"{start_ts:%Y-%m}"


def _process_quarter(task, H, L, C, grid: WFOGrid, cfg: WFOConfig):
    q_label, is_lo, is_hi, os_lo, os_hi = task
    grid_res = _run_window_for_grid(H, L, C, is_lo, is_hi, grid, cfg)
    
    if not grid_res:
        return None
    best = _best_from_grid(grid_res)
    if best is None:
        return None
    bL, bS, is_p, is_dd, is_tr, is_npdd = best

    os_p, os_dd, os_tr, seg_pnl, seg_tr, seg_pos = _run_single_window(
        H, L, C, os_lo, os_hi, bL, bS, cfg)

    quarter = WFOQuarter(
        q_label=q_label,
        is_lo=is_lo, is_hi=is_hi, os_lo=os_lo, os_hi=os_hi,
        best_L=bL, best_S=bS,
        is_profit=is_p, is_dd=is_dd, is_npdd=is_npdd, is_trades=is_tr,
        os_profit=os_p, os_dd=os_dd, os_trades=os_tr,
    )
    return quarter, seg_pnl, seg_tr, seg_pos


def _assemble_result(dt, quarter_outputs, cfg: WFOConfig) -> "WFOResult":
    quarters = [q for q, _, _, _ in quarter_outputs]
    if not quarters:
        return WFOResult(quarters=[], os_equity=np.array([]),
                         os_pnl=np.array([]), os_trades_bar=np.array([]),
                         os_pos_end=np.array([], dtype=np.int8),
                         os_dt=np.array([], dtype="datetime64[m]"))
        
    pnl = np.concatenate([p for _, p, _, _ in quarter_outputs])
    tr = np.concatenate([t for _, _, t, _ in quarter_outputs])
    
    pos = np.concatenate([s for _, _, _, s in quarter_outputs])
    dt_segs = [dt[q.os_lo : q.os_hi + 1] for q in quarters]
    
    os_dt = np.concatenate(dt_segs)
    os_equity = cfg.e0 + np.cumsum(pnl)
    return WFOResult(
        quarters=quarters,
        os_equity=os_equity,
        os_pnl=pnl,
        os_trades_bar=tr,
        os_pos_end=pos,
        os_dt=os_dt,
    )


def rolling_wfo(dt, H, L, C, grid: WFOGrid, cfg: WFOConfig,
                start: Optional[str] = None, end: Optional[str] = None,
                progress: bool = False):
    q_iter = list(iter_quarters(dt, cfg, start=start, end=end))
    outputs = []
    for qi, task in enumerate(q_iter):
        out = _process_quarter(task, H, L, C, grid, cfg)
        if out is None:
            continue
        outputs.append(out)
        if progress:
            q = out[0]
            
            print(f"[{qi+1:>3}/{len(q_iter)}] {q.q_label}: L={q.best_L:>5} "
                  f"S={q.best_S:.3f} ISnpdd={q.is_npdd:>7.2f}  "
                  f"OSp={q.os_profit:>10.1f}  OStr={q.os_trades:>5.1f}")
    return _assemble_result(dt, outputs, cfg)


_WORKER_STATE = {}


def _pool_init(H, L, C):
    _WORKER_STATE["H"] = H
    _WORKER_STATE["L"] = L
    _WORKER_STATE["C"] = C


def _pool_run(args):
    task, grid, cfg = args
    return _process_quarter(
        task, _WORKER_STATE["H"], _WORKER_STATE["L"], _WORKER_STATE["C"],
        grid, cfg,
    )


def rolling_wfo_parallel(dt, H, L, C, grid: WFOGrid, cfg: WFOConfig,
                         start: Optional[str] = None, end: Optional[str] = None,
                         n_workers: Optional[int] = None,
                         progress: bool = False):
    import multiprocessing as mp
    import os as _os

    q_iter = list(iter_quarters(dt, cfg, start=start, end=end))
    n_workers = n_workers or _os.cpu_count() or 1
    n_workers = min(n_workers, len(q_iter))
    if n_workers <= 1 or len(q_iter) <= 1:
        return rolling_wfo(dt, H, L, C, grid, cfg, start=start, end=end,
                          progress=progress)

    args = [(task, grid, cfg) for task in q_iter]
    ctx = mp.get_context("fork")
    outputs = []
    done = 0
    with ctx.Pool(processes=n_workers,
                  initializer=_pool_init,
                  initargs=(H, L, C)) as pool:
        for out in pool.imap_unordered(_pool_run, args, chunksize=1):
            done += 1
            if out is None:
                continue
            outputs.append(out)
            if progress:
                q = out[0]
                print(f"[{done:>3}/{len(q_iter)}] {q.q_label}: L={q.best_L:>5} "
                      f"S={q.best_S:.3f} ISnpdd={q.is_npdd:>7.2f}  "
                      f"OSp={q.os_profit:>10.1f}  OStr={q.os_trades:>5.1f}")


    outputs.sort(key=lambda o: o[0].os_lo)
    return _assemble_result(dt, outputs, cfg)
