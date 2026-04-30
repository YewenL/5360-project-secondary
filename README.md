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

# MATH GR5360 Project Secondary Market GROUP 3

Since the primary team already built most of the code framework, our focus should be on replacing the market data/parameters, running the required experiments, analyzing the results, and preparing the secondary-market slides.

## Task Assignment

| Whom | Responsibilities | Deliverables |
| --- | --- | --- |
| **S1** | Secondary market intro: contract description, trading hours, point value, tick size, slippage, currency conversion if needed | Market intro slides, contract specification table |
| **S2** | Run Variance Ratio test and Push-Response test on secondary market data; interpret inefficiency type and time-scale | VR plots, Push-Response plots, trend-following / mean-reversion interpretation |
| **S3** | Prepare secondary market data and parameters; run Channel WithDDControl on secondary market using existing Python/Numba framework | Clean secondary-market config, validated strategy output |
| **S4** | Run rolling WFO: 4-year IS → next-quarter OOS, full 91,296-point grid search, save OOS results | OOS equity CSV, trade table, per-quarter optimal parameter table |
| **S5** | Run full in-sample optimization and compare with WFO results | Full-IS equity CSV, best full-sample parameters, IS vs OOS comparison table |
| **S6** | Compute performance statistics and decay coefficients; prepare plots and final PPT slides | Stats table, equity/DD plots, decay analysis, final secondary-market slides |

---

## Main Goals for Secondary Market

1. Replace primary-market data and parameters with secondary-market data.
2. Run both statistical tests required by the project.
3. Run WFO using the full grid.
4. Run full in-sample optimization.
5. Compare WFO vs Full IS to measure performance decay.
6. Prepare slides explaining whether the strategy works well on the secondary market and why.
