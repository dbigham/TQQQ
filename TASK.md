# Task Plan

## Objectives
1. Allow choosing any base symbol/index instead of hard-coded Nasdaq/QQQ when running the strategy.
2. Fetch and fit data on-demand for new symbols, caching the curve fit and associated visualisations for reuse.
3. Ensure plots and saved assets reflect the selected base symbol and remain backward compatible for QQQ runs.
4. Document the new workflow and verify automated tests still pass.

## Plan
- [x] Review current data loading and temperature fitting flow in `strategy_tqqq_reserve.py` and related modules to identify integration points for symbol overrides.
- [x] Design caching/layout for per-symbol fits (parameters + PNG outputs) and determine where to generate these assets.
- [x] Implement CLI updates and data/fit orchestration to support arbitrary symbols, including on-demand yfinance downloads when needed.
- [x] Update documentation/tests as required and validate the pipeline with pytest.

## Progress
- [x] Initial analysis complete.
- [x] Implementation underway.
- [x] Final validation and documentation updates complete.
