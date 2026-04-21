# MATH GR5360 Final Project — Primary Market


## Task Assignment


| Role | Responsibilities | Deliverables |
| --- | --- | --- | --- |
| **P1 – Market & Stats** | PL market intro, Variance Ratio test, Push-Response test, interpretation | Intro slides, two test implementations, time-scale conclusion |
| **P2 – Strategy Core Engine** | Implement Channel WithDDControl in Python + Numba; **validate against instructor HO reference** (`HO MatLab Test.xlsx`, `Equity 11500 0.019.pdf`, `Equity 12700 0.010.pdf`) | `strategy.py` with a clean `run_strategy(...)` API + unit tests |
| **P3 – Walk-Forward Framework** | Rolling 4-yr IS → next-quarter OOS, 91 296-point grid search with multi-core parallelism, per-quarter parameter log | `walk_forward.py`, OOS equity CSV, optimal-params table |
| **P4 – Performance Analytics** | All required and bonus statistics, decay coefficients (OOS vs full-IS), equity/DD plots | Stats tables, plots, analysis slides |
| **P5 – Sensitivity + Integration + Presentation** | T×τ sensitivity sweep, parameter heatmaps, final PPT assembly, dry-run coordination | Sensitivity results, final deck, rehearsal schedule |

---

