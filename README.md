# MATH GR5360 Project Primary GROUP 3


Hey guys, I'm not sure if the task assignment here covers everything we need to do. Could everyone read through the instruction PDF and confirm? Thanks!
## Task Assignment

| Whom | Responsibilities | Deliverables |
| --- | --- | --- |
| **P1** | PL market intro, Variance Ratio test, Push-Response test, interpretation | Intro slides, two test implementations, time-scale conclusion |
| **P2** | Implement Channel WithDDControl in Python + Numba; **validate against instructor HO reference** (`HO MatLab Test.xlsx`, `Equity 11500 0.019.pdf`, `Equity 12700 0.010.pdf`) | `strategy.py` with a clean `run_strategy(...)` API + unit tests |
| **P3** | Rolling 4-yr IS → next-quarter OOS, 91 296-point grid search with multi-core parallelism, per-quarter parameter log | `walk_forward.py`, OOS equity CSV, optimal-params table |
| **P4** | All required and bonus statistics, decay coefficients (OOS vs full-IS), equity/DD plots | Stats tables, plots, analysis slides |
| **P5** | T×τ sensitivity sweep, parameter heatmaps, final PPT assembly, dry-run coordination | Sensitivity results, final deck, rehearsal schedule |

---

## Task Assignment (Secondary Market — AUG Gold)

| Whom | Responsibilities | Deliverables |
| --- | --- | --- |
| **S1** | Introduce Gold futures market (contract specs, trading hours, point value, tick size, liquidity); prepare and verify 5-min HLV data | Market introduction slides, data description |
| **S2** | Reuse Primary group’s statistical tests: Variance Ratio Test and Push-Response Test; analyze short-term mean reversion and long-term trend behavior in Gold | VR(q) plots, Push-Response plots, interpretation |
| **S3** | Execute core backtesting: run **Walk-Forward Optimization (4-year IS → 3-month OOS)** and write **Full In-Sample optimization** on AUG data; apply USD conversion (PV, slippage) | `run_wfo_secondary.py`, Full IS runner, WFO output CSVs (equity/trades/params), Full IS results |
| **S4** | Compute performance metrics and decay analysis: compare WFO vs Full IS (Net Profit, Max Drawdown, NP/DD, Sharpe, etc.); calculate decay coefficients | Performance tables, decay analysis tables, comparison charts |
| **S5** | Integrate PPT and cross-market analysis: organize AUG results; compare with Primary (PL) results and provide insights on differences in performance and robustness | Secondary slides, cross-market comparison visuals, final presentation script |
---

## Main Goals for Secondary Market

1. Replace primary-market data and parameters with secondary-market data.
2. Run both statistical tests required by the project.
3. Run WFO using the full grid.
4. Run full in-sample optimization.
5. Compare WFO vs Full IS to measure performance decay.
6. Prepare slides explaining whether the strategy works well on the secondary market and why.
