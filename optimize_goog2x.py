#!/usr/bin/env python3
"""Backwards-compatible entry point for optimising GOOG2x parameters.

The heavy lifting now lives in :mod:`optimize_leveraged_symbol`.  This shim keeps
existing documentation and workflows functional while delegating to the
generalised optimiser (defaulting to Alphabet with 2x leverage).
"""

from __future__ import annotations

import optimize_leveraged_symbol as optimiser


def main() -> None:
    optimiser.main()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
