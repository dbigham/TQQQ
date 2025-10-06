# InvestmentsView integration API

The strategy simulator now exposes a JSON bridge that InvestmentsView can call
directly. Instead of embedding the full rebalance loop, InvestmentsView sends a
single payload describing the current portfolio and receives back a
machine-readable decision: rebalance now or hold, along with recommended dollar
moves, share counts, and human-friendly explanations.

The bridge lives in `strategy_tqqq_reserve.py` and is available in multiple forms:

* `evaluate_integration_request(payload)` – importable function that returns a
  dictionary ready to serialise to JSON.【F:strategy_tqqq_reserve.py†L4378-L4697】
* `evaluate_temperature_chart_request(payload)` – importable helper that returns
  the fitted growth curve, temperature series, and pricing/anchor metadata for a
  requested date range so the frontend can render the temperature chart.【F:strategy_tqqq_reserve.py†L4339-L4475】
* `python strategy_tqqq_reserve.py --integration-request payload.json` – CLI
  wrapper that reads JSON from a file or stdin and prints the response.【F:strategy_tqqq_reserve.py†L4703-L4725】

## Request schema

Every request must include the model to evaluate and the current holdings
expressed in dollars (share counts are optional but improve the share-level
output). Providing the date of the last completed rebalance lets the simulator
reconstruct drift accurately, but when InvestmentsView is onboarding a fresh
account the field may be omitted and the bridge will assume the last rebalance
occurred on the request date. All symbol and market data settings default to the
chosen experiment, but the caller may override
them when necessary (e.g., to test a custom CSV or a different reserve asset).

```jsonc
{
  "experiment": "A1",               // required – strategy experiment key
  "request_date": "2025-09-19",     // required – evaluation date (trading day)
  "last_rebalance": "2025-09-01",   // optional – last executed rebalance (string or object)
  "positions": [                     // required – current holdings as of request_date
    {"symbol": "TQQQ", "dollars": 2150.25, "shares": 31.5},
    {"symbol": "SGOV", "dollars": 3150.00}
  ]
}
```

Optional keys mirror the CLI arguments and override the experiment defaults:

```jsonc
{
  "base_symbol": "QQQ",             // overrides experiment's base symbol
  "leveraged_symbol": "TQQQ",       // overrides experiment's leveraged sleeve
  "reserve_symbol": "SGOV",         // overrides experiment's reserve asset
  "fred_series": "FEDFUNDS",        // FRED policy-rate series identifier
  "csv": "./custom_prices.csv",     // path to price CSV with date/close columns
  "leverage": 3.0,
  "annual_fee": 0.0095,
  "borrow_divisor": 0.7,
  "trading_days": 252
}
```

When richer audit detail is available, `last_rebalance` may be provided as an
object (e.g., `{ "date": "2025-09-01", "note": "Quarterly rebalance" }`). Only
the `date` field is required; extra keys are ignored by the bridge. If the field
is omitted entirely, the simulator seeds its state using the request date so
newly created portfolios can obtain their first trade recommendation without
supplying historical activity.

The simulator reconstructs the original rebalance state by inferring share
counts from the supplied dollar amounts and each symbol’s adjusted close on the
last rebalance date.【F:strategy_tqqq_reserve.py†L4428-L4458】 It then seeds the
internal state so rebalances are deferred until the integration request date –
missed trades are treated as if a human skipped them, mirroring the desired
behaviour.【F:strategy_tqqq_reserve.py†L2836-L2853】【F:strategy_tqqq_reserve.py†L3124-L3156】

## Response structure

The response always contains the original metadata, the current portfolio, a
decision block, model diagnostics, and (when applicable) the trades required to
rebalance. A typical rebalance recommendation looks like:

```jsonc
{
  "request_date": "2025-09-19",
  "experiment": "A1",
  "base_symbol": "QQQ",
  "leveraged_symbol": "TQQQ",
  "reserve_symbol": "SGOV",
  "current_value": 12725.35,
  "current_allocation": 0.64,
  "current_positions": [
    {"symbol": "QQQ",  "dollars": 7425.10, "shares": 19.0, "price": 391.85},
    {"symbol": "TQQQ", "dollars": 2150.25, "shares": 31.5, "price": 68.26},
    {"symbol": "SGOV", "dollars": 3150.00, "shares": 315.0, "price": 10.0}
  ],
  "decision": {
    "action": "rebalance",
    "reason": "Rebalance required by strategy rules.",
    "details": {
      "action": "rebalance_buy",
      "date": "2025-09-19",
      "direction": "buy",
      "target_p": 0.82,
      "due": true,
      "rebalance_cadence": 22,
      "days_since_last_rebalance": 22
    }
  },
  "target_positions": [
    {"symbol": "QQQ",  "dollars": 5845.40, "shares": 14.92, "price": 391.85},
    {"symbol": "TQQQ", "dollars": 5223.62, "shares": 76.54, "price": 68.26},
    {"symbol": "SGOV", "dollars": 1656.33, "shares": 165.63, "price": 10.0}
  ],
  "trades": [
    {"symbol": "QQQ",  "action": "sell", "dollars": -1579.70, "shares": -4.08, "price": 391.85},
    {"symbol": "TQQQ", "action": "buy",  "dollars": 3073.37, "shares": 45.04, "price": 68.26},
    {"symbol": "SGOV", "action": "sell", "dollars": -1493.67, "shares": -149.37, "price": 10.0}
  ],
  "model": {
    "rebalance_cadence": 22,
    "days_since_last_rebalance": 22,
    "due": true,
    "target_allocation": 0.82,
    "base_allocation": 0.75
  },
  "recent_rebalance": {
    "date": "2025-09-19",
    "reason": "rebalance"
  },
  "price_source": "unified_nasdaq.csv"
}
```

When the model elects to hold, the response still includes the model’s desired
weights so InvestmentsView can surface drift information, and the `reason`
string explains which cadence or momentum rule blocked trading (e.g. “Next
rebalance window opens in 5 trading day(s).”, “Buy-side momentum guard active”
when a drawdown filter is engaged, or “Buy prevented by violation of maximum
temperature of 1.30.” when only the temperature limit blocks the trade).

## Consuming the bridge

From Python:

```python
import json
from strategy_tqqq_reserve import evaluate_integration_request

with open("payload.json", "r", encoding="utf-8") as fh:
    payload = json.load(fh)

response = evaluate_integration_request(payload)
print(json.dumps(response, indent=2))
```

From the command line:

```bash
python strategy_tqqq_reserve.py --integration-request payload.json > response.json
```

Both entry points share the same code path and therefore the same defaults for
experiments, cadence, and rate series.【F:strategy_tqqq_reserve.py†L4703-L4725】

### Temperature chart bridge

InvestmentsView can request temperature chart data by supplying the model and a
date window. The bridge resolves the experiment defaults, computes the fitted
growth curve (`A * e^(r * t)`), derives the temperature series (`price / fit`),
and returns the data required to plot the baseline curve alongside reference
temperatures and optional allocation anchors.【F:strategy_tqqq_reserve.py†L4339-L4475】

#### Request payload

```jsonc
{
  "experiment": "A1",              // required – strategy experiment key
  "start_date": "2024-01-02",     // required – beginning of chart window (trading day)
  "end_date": "2024-06-28"        // required – end of chart window (inclusive trading day)
}
```

Optional overrides mirror the integration request (e.g., `csv`, `base_symbol`,
`leveraged_symbol`, etc.) and allow InvestmentsView to chart alternative data
sources or manual fits.【F:strategy_tqqq_reserve.py†L4403-L4475】

#### Response payload

```jsonc
{
  "experiment": "A1",
  "start_date": "2024-01-02",
  "end_date": "2024-06-28",
  "growth_curve": [
    {"date": "2024-01-02", "fit": 1.92, "price": 375.04, "temperature": 0.98},
    {"date": "2024-01-03", "fit": 1.93, "price": 378.12, "temperature": 0.99}
    // … one entry per trading day
  ],
  "reference_temperatures": [0.5, 1.0, 1.5],
  "manual_override": false,
  "temperature_allocation": [
    {"temperature": 0.7, "allocation": 1.0},
    {"temperature": 1.2, "allocation": 0.5}
  ]
}
```

* `growth_curve` – daily entries that include the fitted curve value, observed
  price, and derived temperature for each trading session in the requested
  range.【F:strategy_tqqq_reserve.py†L4415-L4454】
* `reference_temperatures` – standard guideposts (0.5, 1.0, 1.5) to plot the
  horizontal reference lines in the chart.【F:strategy_tqqq_reserve.py†L4452-L4455】
* `manual_override` – flag indicating whether the experiment supplied a manual
  temperature model override (useful for labelling the chart).【F:strategy_tqqq_reserve.py†L4442-L4444】
* `temperature_allocation` – when the experiment defines allocation anchors, the
  bridge echoes them back so the frontend can draw the allocation ladder; absent
  when no anchors are defined.【F:strategy_tqqq_reserve.py†L4458-L4474】

## Implementation notes

* The bridge honours all momentum, rate, and crash guards defined in the
  experiment; it simply prevents the simulator from trading before the
  integration date by treating earlier days as user-skipped rebalances.【F:strategy_tqqq_reserve.py†L3124-L3206】
* Forced derisk events still occur immediately and are reported via the
  `recent_rebalance` field so InvestmentsView can surface that emergency action
  even when no new trade is due today.【F:strategy_tqqq_reserve.py†L3297-L3329】【F:strategy_tqqq_reserve.py†L4576-L4610】
* All dollar and share figures are calculated using the closing prices on the
  request date; providing share counts in the request improves numerical
  stability but is not required.【F:strategy_tqqq_reserve.py†L4430-L4462】【F:strategy_tqqq_reserve.py†L4564-L4569】

This interface keeps InvestmentsView in sync with the research code while
avoiding the complexity of replicating the strategy loop in a second code base.
