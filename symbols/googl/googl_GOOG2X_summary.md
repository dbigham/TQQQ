# GOOGL – Strategy GOOG2X Summary

- **Span**: 2004-08-19 → 2025-09-19 (21.08 years)
- **Underlying (GOOGL price) CAGR**: 24.54%
- **Fitted curve CAGR**: 19.05%
- **Strategy CAGR**: 65.89%
- **Max drawdown**: -65.66%
- **Rebalances executed**: 121
- **P/E ratio (as of 2025-09-21)**: 27.18
- **PEG ratio (as of 2025-09-21)**: 1.73
- **Current growth rate (as of 2025-09-21)**: 15.69%

## Parameter comparison vs A1g

The tables below contrast the GOOG2X overrides with the original A1g defaults. Return
thresholds are simple price changes over the stated lookback; buy filters block new
buys when the return is **at or below** the threshold, while sell filters defer trims
when the return is **at or above** the threshold.

### Temperature anchors (leveraged sleeve allocation)

| Anchor | A1g (T → allocation) | GOOG2X (T → allocation) |
| --- | --- | --- |
| Cold anchor | T = 0.927 → 90% | T = 0.911 → 150% |
| On-curve anchor | T = 1.000 → 69.1% | T = 1.000 → 100.0% |
| Hot anchor | T = 1.390 → 12.4% | T = 1.414 → 8.8% |

### Momentum buy filters (block buys when…)

| Filter | A1g threshold | GOOG2X threshold |
| --- | --- | --- |
| 3-day return | ≤ -0.85% | ≤ -2.63% |
| 6-day return | ≤ -1.11% | ≤ -7.75% |
| 12-day return | ≤ -0.50% | ≤ -0.47% |
| 22-day return | ≤ 0.24% | ≤ 0.37% |
| Temperature | T > 1.445 | T > 1.260 |

### Momentum sell filters (block trims when…)

| Filter | A1g threshold | GOOG2X threshold |
| --- | --- | --- |
| 3-day return | ≥ 7.16% | ≥ 0.77% |
| 6-day return | ≥ 0.78% | ≥ 2.02% |
| 12-day return | ≥ 0.40% | ≥ 1.56% |
| 22-day return | ≥ 0.33% | ≥ -0.15% |

### Rate taper and leverage

| Setting | A1g | GOOG2X |
| --- | --- | --- |
| Taper start | 10.58% | 10.00% |
| Taper end | 11.65% | 12.44% |
| Minimum allocation | 1.68% | 17.48% |
| Leverage override | Uses CLI/default (3×) | Forces 2× |

Both configurations keep crash de-risking disabled and retain the 22-trading-day
rebalance cadence from A1g.

Notes:

- Temperatures and anchors per experiment govern deployment; see EXPERIMENTS.md for details.
- Figures use simulated leveraged sleeve with fees and borrow costs.
