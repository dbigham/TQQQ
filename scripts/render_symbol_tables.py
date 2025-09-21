#!/usr/bin/env python3
"""Render Markdown tables for the cached A1 and A1g symbol statistics."""

from __future__ import annotations

import os
import sys
from typing import Mapping

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from a1_symbol_stats import A1_SYMBOL_STATS
from a1g_symbol_stats import A1G_SYMBOL_STATS


def _format_percent(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.2f}%"
    return "—"


def _format_chart_link(path: object) -> str:
    if isinstance(path, str) and path:
        return f"[Strategy Chart]({path})"
    return "—"


def _render_table(stats: Mapping[str, Mapping[str, object]]) -> str:
    headers = ["Symbol", "Name", "Strategy CAGR", "Max Drawdown", "Strategy Chart"]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for symbol, payload in stats.items():
        lines.append(
            " | ".join(
                [
                    symbol,
                    str(payload.get("Name", "")),
                    _format_percent(payload.get("Strategy_CAGR")),
                    _format_percent(payload.get("Max_Drawdown")),
                    _format_chart_link(payload.get("Strategy_Chart_Path")),
                ]
            )
        )
    return "\n".join(lines)


def main() -> None:
    print("# A1 Symbol Stats")
    print(_render_table(A1_SYMBOL_STATS))
    print()  # Spacer between tables
    print("# A1g Symbol Stats")
    print(_render_table(A1G_SYMBOL_STATS))


if __name__ == "__main__":
    main()
