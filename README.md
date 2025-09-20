# TQQQ Research Toolkit

## Overview
This repository assembles a research pipeline for studying the ProShares UltraPro QQQ (TQQQ) and related leveraged-Nasdaq strategies. It provides tools to stitch together a long-run Nasdaq proxy, fit a constant-growth baseline, derive a "temperature" metric, simulate TQQQ dynamics under financing costs, and backtest allocation rules that blend leveraged exposure with a cash reserve.

Key capabilities include:
- Building a unified daily price series spanning 1971 to today by scaling QQQ, ^NDX, and ^IXIC and recording the seam metadata for reproducibility.
- Fitting and visualising a constant-growth curve to the unified series with iterative outlier rejection, and exposing the derived parameters for downstream use.
- Defining a Nasdaq "temperature" as actual price relative to the fitted curve, with a CLI and importable API for evaluating valuation regimes.
- Simulating a levered TQQQ-like path that incorporates management fees and daily borrowing costs inferred from FRED rates, alongside downloads of the actual ETF for comparison.
- Backtesting a suite of TQQQ + reserve allocation experiments that combine valuation, momentum, macro, and rate filters and report portfolio growth metrics with diagnostic plots.

## Project Layout
```
.
├── build_unified_nasdaq.py        # Create the long-run Nasdaq proxy & metadata
├── fit_constant_growth.py         # Fit a robust constant-growth curve
├── nasdaq_temperature.py          # Compute temperature series & API
├── simulate_tqqq.py               # Compare simulated vs actual TQQQ
├── strategy_tqqq_reserve.py       # Allocation strategy & experiment engine
├── EXPERIMENTS.md                 # Narrative documentation of experiment family
├── analyze_strategy_debug.py      # Helper for inspecting rebalance debug dumps
├── test_iterative_fit.py          # Unit test for the iterative curve fitting
├── test_strategy_experiments.py   # Smoke tests pinning experiment CAGR targets
└── unified_nasdaq.csv/.json       # Sample unified data & seam metadata
```

## Getting Started
1. **Create an environment** (Python 3.9+ recommended) and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install pandas numpy matplotlib yfinance pandas_datareader fredapi pytest
   ```
   The scripts import these packages on demand and fall back gracefully when optional sources (such as `fredapi`) are unavailable. Provide a `FRED_API_KEY` environment variable to unlock authenticated FRED requests; otherwise the tooling automatically tries public CSV downloads.

2. **Build or refresh the unified Nasdaq history** (writes `unified_nasdaq.csv` and `unified_nasdaq_meta.json`):
   ```bash
   python build_unified_nasdaq.py --out unified_nasdaq.csv
   ```
   The script downloads adjusted closes for QQQ, ^NDX, and ^IXIC, scales them at seam dates, and logs the seam levels and scaling factors in the JSON sidecar for auditability.

3. **Inspect the baseline growth fit** (optional but useful sanity check):
   ```bash
   python fit_constant_growth.py --csv unified_nasdaq.csv --save-plot fit_constant_growth.png
   ```
   This performs a three-step iterative log-linear fit, prints the starting level and annualised growth rate, and renders a log-scale comparison chart.

4. **Evaluate the Nasdaq temperature** for any date or save the full plot:
   ```bash
   python nasdaq_temperature.py --csv unified_nasdaq.csv --date 2024-01-02 --save-plot nasdaq_temperature.png
   ```
   The CLI reports `T` for the requested session and visualises the time-series with reference bands at 0.5, 1.0, and 1.5; the module can also be imported to call `get_nasdaq_temperature()` programmatically.

5. **Simulate TQQQ performance** relative to the ETF:
   ```bash
   python simulate_tqqq.py --csv unified_nasdaq.csv --save-plot simulate_tqqq.png --save-csv tqqq_sim_vs_actual.csv
   ```
   The simulator multiplies unified daily returns by the specified leverage, subtracts daily management fees, applies borrowing costs derived from FRED rates, and overlays the result on actual auto-adjusted TQQQ data fetched from Yahoo Finance.

6. **Run the TQQQ + reserve strategy**:
   ```bash
   python strategy_tqqq_reserve.py --base-symbol QQQ --csv unified_nasdaq.csv --experiment A36 --save-plot strategy_tqqq_reserve.png
   ```
   By default the CLI executes experiment **A36 – High-Leverage Ramp**, produces two diagnostic plots, and prints summary metrics such as CAGR and rebalance activity; pass `--print-rebalances` or the debug flags to emit detailed trade logs and a per-day CSV for deeper analysis. The helper `analyze_strategy_debug.py` can then surface deployment statistics and rule violations from a debug dump.
   Provide a different `--base-symbol` (for example `SPY`, `BTC-USD`, `NVDA`) to drive the strategy with another asset. When a non-QQQ base is requested the script downloads the adjusted history via Yahoo Finance, caches it under `symbol_data/`, and stores the fitted temperature parameters plus diagnostic PNGs for reuse.

## Strategy Experiments
Strategy configurations are encapsulated in the `EXPERIMENTS` dictionary so features such as cold-temperature leverage boosts, hot-momentum overlays, macro filters, leverage ladders, and rate tapers can be combined without cluttering the core simulation loop. Each entry corresponds to an experiment documented in depth in `EXPERIMENTS.md`, which traces the evolution from the baseline A1 configuration (~29.7% CAGR) to the current A36 high-leverage variant (~45.31% CAGR through 2025‑01‑10). Use `--experiment A#` on the CLI to reproduce any particular configuration.

### Base experiment walkthrough (A1 foundation)

The strategy engine always starts from the base profile described at the top of `strategy_tqqq_reserve.py`. Later experiments add overlays, but A1 provides the core behaviour:

1. **Pair a simulated TQQQ sleeve with a cash reserve.** Daily returns for the TQQQ sleeve are synthesised from the unified Nasdaq series using 3× leverage, a 0.95% annual fee, and borrowing costs that charge the two borrowed turns of leverage at the prevailing FRED rate divided by a 0.7 borrow divisor. Cash compounds at the same interest rate.
2. **Measure "temperature" (`T`).** For every trading day, compute the ratio of the actual unified close to the constant-growth fit produced by `nasdaq_temperature.py`. `T = 1.0` means the index sits on the trend line, 1.5 is 50% above the curve, and 0.5 is 50% below.
3. **Derive the target allocation from temperature anchors.** A1 linearly interpolates between three waypoints: 100% TQQQ when `T ≤ 0.9`, 80% at `T = 1.0`, and 20% when `T ≥ 1.5`. Between the anchors the allocation slides smoothly; the engine never increases exposure on a day where `T > 1.3`, although selling is still allowed.
4. **Enforce a monthly rebalance cadence with momentum gates.** The portfolio aims to rebalance every 22 trading days. If the due day is blocked, the simulator checks again each subsequent session until trades are permitted and then resets the 22‑day clock. Buy attempts are delayed when QQQ has dropped by at least 3%/3 days, 3%/6 days, 2%/12 days, or 0%/22 days, or when `T > 1.3`. Sell attempts are similarly delayed after gains of 3%/3 days, 3%/6 days, 2.25%/12 days, or 0.75%/22 days.
5. **Apply crash and rate safety valves.** A drawdown worse than 15.25% over the trailing 22 trading days immediately moves the portfolio to 100% cash, regardless of the rebalance schedule. Separately, the target allocation is tapered when policy rates rise: exposure is untouched below 10%, fades linearly between 10% and 12%, and is forced to zero at 12%.
6. **Execute trades and compound returns.** When a rebalance is allowed, the engine shifts capital between the cash and TQQQ sleeves toward the temperature- and filter-adjusted target, then compounds both sleeves using that day’s simulated returns and interest accrual. Diagnostic flags (for example `--print-rebalances`) expose the resulting trades for inspection.

Every subsequent experiment in `EXPERIMENTS` tweaks pieces of this base loop—changing the anchors, introducing leverage boosts when the market is cold, adding macro filters, or altering the borrowing assumptions—but the decision flow above always remains the starting point.

Notable overlays include:
- **Temperature-driven base allocation** with buy/sell blocks and a crash de-risk rule triggered by a 22-day drawdown greater than 15.25%.
- **Interest-rate tapering** that linearly reduces exposure between 10% and 12% policy rates, alongside options to override leverage, fees, and borrow-cost modelling per experiment.
- **Cold leverage and momentum overlays** that dynamically raise exposure when valuation, rate, and momentum conditions align, governed by experiment-specific parameters.

Consult the experiment narrative for performance highlights, trade-offs, and the rationale behind each incremental change.

## Data Artifacts
Running the pipeline produces a small set of persisted assets:
- `unified_nasdaq.csv` – unified adjusted closes with their source segment (`QQQ`, `NDX_scaled`, `IXIC_scaled`, etc.).
- `unified_nasdaq_meta.json` – seam dates, scaling multipliers, and endpoint levels for the unified series, useful when sharing or rerunning analyses.
- Optional PNG/CSV outputs from the simulators and strategy runs (examples are checked in for reference).
- `symbol_data/` – cached auto-adjusted closes for any non-QQQ symbols requested by the strategy CLI.
- `temperature_cache/` – JSON metadata plus PNG diagnostics for per-symbol temperature fits generated on demand.

Because upstream providers revise historical adjustments, re-running `build_unified_nasdaq.py` periodically will refresh the data with updated values; the metadata file records the seam factors applied in each run for transparency.

## Testing
Automated tests cover critical invariants:
- `pytest test_iterative_fit.py` ensures the iterative fitter rejects configurations that would discard all data after thresholding.
- `pytest test_strategy_experiments.py` runs selected experiments end-to-end and verifies that the reported CAGR still matches the documented targets, alerting you if changes in logic alter headline performance.

Run the full suite with:
```bash
pytest
```
These checks take several minutes because they invoke the full strategy simulator for multiple experiment profiles.

## Troubleshooting & Tips
- **Missing FRED credentials:** The scripts automatically fall back to unauthenticated CSV downloads if `fredapi` or an API key is unavailable, but network failures will halt the run; consider caching rates locally when working offline.
- **Yahoo Finance throttling:** Building the unified series and fetching TQQQ data relies on `yfinance`. If you encounter throttling, rerun with `--no-plot` and consider enabling the `--upgrade` flag to grab the latest dependencies before retrying.
- **Deep dives:** Use `--debug-start/--debug-end` with the strategy simulator to capture per-day decision traces, then explore them with `analyze_strategy_debug.py` to validate that guards and blocks behaved as expected.

---
Leverage these tools to explore valuation-aware leverage, stress-test new overlays, or simply monitor how current market conditions stack up against decades of Nasdaq history. Pull requests and experiment ideas are welcome!
