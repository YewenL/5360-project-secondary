"""Performance statistics for walk-forward out-of-sample results."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    side: int
    pnl: float


@dataclass
class PerfStats:
    total_profit: float
    worst_drawdown: float
    worst_dd_duration_bars: int
    return_on_account: float

    avg_return_per_bar: float
    std_return_per_bar: float
    sharpe_per_bar: float
    sharpe_annual: float
    avg_return_annual: float

    num_trades: int
    num_winners: int
    num_losers: int
    pct_winners: float
    avg_winner: float
    avg_loser: float
    largest_winner: float
    largest_loser: float
    profit_factor: float


def extract_trades(pnl: np.ndarray, pos_end: np.ndarray) -> List[Trade]:
    """Reconstruct additive trade P&L from bar P&L and end-of-bar position."""
    n = len(pos_end)
    if n == 0:
        return []
    out: List[Trade] = []
    cur_start: Optional[int] = None
    cur_side: int = 0
    prev_pos = 0
    eps = 1e-12
    for k in range(n):
        cur = int(pos_end[k])
        if prev_pos == 0 and cur == 0:
            if abs(float(pnl[k])) > eps:
                out.append(Trade(entry_bar=k, exit_bar=k, side=0, pnl=float(pnl[k])))
        elif prev_pos == 0 and cur != 0:
            cur_start = k
            cur_side = cur
        elif prev_pos != 0 and cur == prev_pos:
            pass
        elif prev_pos != 0 and cur == 0:
            if cur_start is not None:
                pnl_trade = float(pnl[cur_start : k + 1].sum())
                out.append(Trade(entry_bar=cur_start, exit_bar=k,
                                 side=cur_side, pnl=pnl_trade))
            cur_start = None
            cur_side = 0
        else:
            if cur_start is not None:
                pnl_trade = float(pnl[cur_start : k + 1].sum())
                out.append(Trade(entry_bar=cur_start, exit_bar=k,
                                 side=cur_side, pnl=pnl_trade))
            if k + 1 < n:
                cur_start = k + 1
                cur_side = cur
            else:
                cur_start = None
                cur_side = 0
        prev_pos = cur
    if cur_start is not None and cur_side != 0:
        pnl_trade = float(pnl[cur_start:n].sum())
        out.append(Trade(entry_bar=cur_start, exit_bar=n - 1,
                         side=cur_side, pnl=pnl_trade))
    return out


def _worst_dd_duration(equity: np.ndarray) -> int:
    if equity.size == 0:
        return 0
    running_max = np.maximum.accumulate(equity)
    under = equity < running_max - 1e-9

    longest = 0
    cur = 0
    for flag in under:
        if flag:
            cur += 1
            if cur > longest:
                longest = cur
        else:
            cur = 0
    return int(longest)


def _estimate_bars_per_year(dt: np.ndarray) -> float:
    """Empirical bars-per-year from the timestamp array of OS bars."""
    if dt.size < 2:
        return 252.0 * 78.0  # fallback (~RTH ~6.5h * 12 bars/h)
    span_days = (pd.Timestamp(dt[-1]) - pd.Timestamp(dt[0])).total_seconds() / 86400.0
    
    if span_days <= 0:
        return 252.0 * 78.0
    return dt.size * 365.25 / span_days


def compute_stats(os_equity: np.ndarray, os_pnl: np.ndarray,
                  os_pos_end: np.ndarray, os_dt: Optional[np.ndarray] = None,
                  bars_per_year: Optional[float] = None) -> PerfStats:
    if os_equity.size == 0:
        return PerfStats(
            total_profit=0.0, worst_drawdown=0.0, worst_dd_duration_bars=0,
            return_on_account=0.0,
            avg_return_per_bar=0.0, std_return_per_bar=0.0,
            sharpe_per_bar=0.0, sharpe_annual=0.0, avg_return_annual=0.0,
            num_trades=0, num_winners=0, num_losers=0, pct_winners=0.0,
            avg_winner=0.0, avg_loser=0.0,
            largest_winner=0.0, largest_loser=0.0,
            profit_factor=0.0,
        )

    initial_equity = float(os_equity[0] - os_pnl[0])
    total_profit = float(os_pnl.sum())
    equity_path = np.concatenate((np.array([initial_equity], dtype=np.float64), os_equity))
    
    running_max = np.maximum.accumulate(equity_path)
    dd = equity_path - running_max
    
    mdd = float(dd.min())
    mdd_dur = _worst_dd_duration(equity_path)
    roa = total_profit / abs(mdd) if mdd < -1e-9 else float("inf") if total_profit > 0 else 0.0

    prev_equity = equity_path[:-1]
    returns = np.divide(
        os_pnl,
        prev_equity,
        out=np.zeros_like(os_pnl, dtype=np.float64),
        where=np.abs(prev_equity) > 1e-12,
    )
    mean_ret = float(returns.mean())
    std_ret = float(returns.std(ddof=1)) if returns.size > 1 else 0.0
    sharpe_pb = mean_ret / std_ret if std_ret > 0 else 0.0

    if bars_per_year is None:
        bars_per_year = _estimate_bars_per_year(os_dt) if os_dt is not None else 252.0 * 78.0
    sharpe_ann = sharpe_pb * np.sqrt(bars_per_year)
    avg_ret_ann = mean_ret * bars_per_year

    trades = extract_trades(os_pnl, os_pos_end)
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl < 0]
    
    n_tr = len(trades)
    pct_w = 100.0 * len(wins) / n_tr if n_tr else 0.0
    avg_w = float(np.mean(wins)) if wins else 0.0
    
    avg_l = float(np.mean(losses)) if losses else 0.0
    lw = float(max(wins)) if wins else 0.0
    ll = float(min(losses)) if losses else 0.0
    
    gross_w = float(sum(wins))
    gross_l = -float(sum(losses))
    if gross_l > 1e-12:
        pf = gross_w / gross_l
    else:
        pf = float("inf") if gross_w > 0 else 0.0

    return PerfStats(
        total_profit=total_profit,
        worst_drawdown=mdd,
        worst_dd_duration_bars=mdd_dur,
        return_on_account=roa,
        avg_return_per_bar=mean_ret,
        std_return_per_bar=std_ret,
        sharpe_per_bar=sharpe_pb,
        sharpe_annual=sharpe_ann,
        avg_return_annual=avg_ret_ann,
        num_trades=n_tr,
        num_winners=len(wins),
        num_losers=len(losses),
        pct_winners=pct_w,
        avg_winner=avg_w,
        avg_loser=avg_l,
        largest_winner=lw,
        largest_loser=ll,
        profit_factor=pf,
    )


def format_stats(s: PerfStats, title: str = "Performance") -> str:
    return (
        f"{title}\n"
        f"Total profit: {s.total_profit:>14,.2f}\n"
        f"Worst drawdown: {s.worst_drawdown:>14,.2f}\n"
        f"Worst DD duration (bars): {s.worst_dd_duration_bars:>14,}\n"
        f"Return on Account: {s.return_on_account:>14.4f}\n"
        f"Avg return / bar: {100*s.avg_return_per_bar:>13.6f}%\n"
        f"Std return / bar: {100*s.std_return_per_bar:>13.6f}%\n"
        f"Sharpe / bar: {s.sharpe_per_bar:>14.4f}\n"
        f"Sharpe (annualized): {s.sharpe_annual:>14.4f}\n"
        f"Avg return (annualized): {100*s.avg_return_annual:>13.2f}%\n"
        f"# trades: {s.num_trades:>14,}\n"
        f"# winners: {s.num_winners:>14,}\n"
        f"# losers: {s.num_losers:>14,}\n"
        f"% winners: {s.pct_winners:>14.2f}\n"
        f"Avg winner: {s.avg_winner:>14,.2f}\n"
        f"Avg loser: {s.avg_loser:>14,.2f}\n"
        f"Largest winner: {s.largest_winner:>14,.2f}\n"
        f"Largest loser: {s.largest_loser:>14,.2f}\n"
        f"Profit factor: {s.profit_factor:>14.4f}\n"
    )
