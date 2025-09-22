# Leveraged Exposure Replication

Many experiments in this project express the portfolio target as the fraction of
capital that should sit inside a daily-reset leveraged ETF such as TQQQ. The
remainder is implicitly held as cash. When the target calls for a partial
allocation (for example 40% in a 2× fund) it is possible to deliver the same
net exposure to the underlying index with far less capital inside the leveraged
vehicle. Holding more of the unlevered asset cuts volatility drag and reduces
the share of the portfolio that pays the ETF’s embedded financing costs.

Let

- `m` = leverage factor of the ETF (3 for TQQQ, 2 for many single-stock funds),
- `a` = capital fraction the strategy wants in the leveraged product, and
- `E = m × a` = desired exposure to the underlying index (in "unlevered" units).

We want portfolio weights `(w_1x, w_mx, w_cash)` that sum to one and satisfy
`w_1x + m × w_mx = E` while using as little of the leveraged sleeve as
possible.

The optimal solution without borrowing follows two regimes:

1. **Sub-1× exposure (`E ≤ 1`):**
   Allocate `w_1x = E` to the unlevered asset, keep `w_mx = 0`, and hold the
   balance in cash (`w_cash = 1 - E`).
2. **Between 1× and `m×` exposure (`1 < E ≤ m`):**
   Invest `w_mx = (E - 1) / (m - 1)` in the leveraged fund and place the rest of
   the capital in the 1× asset (`w_1x = 1 - w_mx`). No cash is required in this
   regime.

When the strategy demands more than `m×` exposure (`a > 1` for an `m×`
product) matching the target while keeping the same borrowing footprint requires
holding more of the leveraged ETF. The simulator therefore falls back to the
original "leveraged ETF + cash" mix in that case. This ensures the replication
layer never increases the amount of borrowed capital compared with the legacy
behaviour.

`strategy_tqqq_reserve.py` exposes this feature through the
`use_leverage_replication` flag. The global default lives in
`DEFAULT_USE_LEVERAGE_REPLICATION`, while individual experiments can override it
by setting the flag inside their configuration dictionary. Experiment **A1**
enables replication to make the baseline behaviour align with the allocation
logic described above.
