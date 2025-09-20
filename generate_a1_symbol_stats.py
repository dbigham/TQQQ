#!/usr/bin/env python3
"""Run A1 strategy for a curated list of symbols and aggregate summary statistics."""

#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class SymbolInfo:
    symbol: str
    name: str
    description: str


SYMBOLS: List[SymbolInfo] = [
    SymbolInfo("AAPL", "Apple Inc.", "Consumer electronics and services giant behind the iPhone"),
    SymbolInfo("MSFT", "Microsoft Corporation", "Enterprise software and cloud platform leader"),
    SymbolInfo("NVDA", "NVIDIA Corporation", "Dominant GPU designer powering AI and accelerated computing"),
    SymbolInfo("GOOGL", "Alphabet Inc.", "Google's parent with leading search, ads, and cloud businesses"),
    SymbolInfo("AMZN", "Amazon.com, Inc.", "E-commerce and cloud infrastructure leader"),
    SymbolInfo("META", "Meta Platforms, Inc.", "Facebook parent building social, messaging, and VR ecosystems"),
    SymbolInfo("TSLA", "Tesla, Inc.", "Electric vehicle and clean energy manufacturer"),
    SymbolInfo("AVGO", "Broadcom Inc.", "Semiconductor and infrastructure software conglomerate"),
    SymbolInfo("TSM", "Taiwan Semiconductor Manufacturing Co.", "Leading pure-play semiconductor foundry"),
    SymbolInfo("LLY", "Eli Lilly and Company", "Innovative pharma company with blockbuster diabetes and obesity drugs"),
    SymbolInfo("JPM", "JPMorgan Chase & Co.", "Largest US bank with global investment and retail operations"),
    SymbolInfo("V", "Visa Inc.", "Global payments network enabling electronic transactions"),
    SymbolInfo("MA", "Mastercard Incorporated", "Global card network powering digital payments"),
    SymbolInfo("UNH", "UnitedHealth Group Incorporated", "Largest US managed healthcare and insurance provider"),
    SymbolInfo("XOM", "Exxon Mobil Corporation", "Integrated oil and gas supermajor"),
    SymbolInfo("JNJ", "Johnson & Johnson", "Diversified healthcare products and pharmaceuticals company"),
    SymbolInfo("WMT", "Walmart Inc.", "Global big-box retailer with growing e-commerce operations"),
    SymbolInfo("PG", "Procter & Gamble Company", "Consumer staples leader with household and personal care brands"),
    SymbolInfo("ORCL", "Oracle Corporation", "Enterprise database and cloud applications provider"),
    SymbolInfo("COST", "Costco Wholesale Corporation", "Membership warehouse club retailer"),
    SymbolInfo("HD", "Home Depot, Inc.", "Leading home improvement retailer"),
    SymbolInfo("BAC", "Bank of America Corporation", "Major US bank with consumer and corporate franchises"),
    SymbolInfo("KO", "The Coca-Cola Company", "Global beverage maker with dominant soft drink brands"),
    SymbolInfo("PEP", "PepsiCo, Inc.", "Snack and beverage company with diversified brands"),
    SymbolInfo("CRM", "Salesforce, Inc.", "Cloud CRM pioneer expanding into enterprise applications"),
    SymbolInfo("ADBE", "Adobe Inc.", "Creative software and digital experience platform leader"),
    SymbolInfo("CSCO", "Cisco Systems, Inc.", "Networking hardware and enterprise security provider"),
    SymbolInfo("NFLX", "Netflix, Inc.", "Global streaming video entertainment platform"),
    SymbolInfo("ACN", "Accenture plc", "Global IT consulting and services firm"),
    SymbolInfo("AMD", "Advanced Micro Devices, Inc.", "CPU and GPU designer challenging incumbents in PCs and data centers"),
    SymbolInfo("NKE", "NIKE, Inc.", "Global athletic footwear and apparel brand"),
    SymbolInfo("TMO", "Thermo Fisher Scientific Inc.", "Scientific instruments and lab services provider"),
    SymbolInfo("MCD", "McDonald's Corporation", "Global quick-service restaurant franchisor"),
    SymbolInfo("DIS", "The Walt Disney Company", "Media, entertainment, and theme park conglomerate"),
    SymbolInfo("ABT", "Abbott Laboratories", "Diversified medical devices and diagnostics company"),
    SymbolInfo("DHR", "Danaher Corporation", "Life sciences and diagnostics conglomerate"),
    SymbolInfo("TXN", "Texas Instruments Incorporated", "Analog and embedded semiconductor supplier"),
    SymbolInfo("LIN", "Linde plc", "Industrial gases leader serving global industries"),
    SymbolInfo("PFE", "Pfizer Inc.", "Pharmaceutical company known for vaccines and therapeutics"),
    SymbolInfo("MRK", "Merck & Co., Inc.", "Global biopharma focused on oncology and vaccines"),
    SymbolInfo("CMCSA", "Comcast Corporation", "Cable, broadband, and media conglomerate"),
    SymbolInfo("VZ", "Verizon Communications Inc.", "US wireless and broadband network operator"),
    SymbolInfo("INTC", "Intel Corporation", "Semiconductor company focused on CPUs and data center chips"),
    SymbolInfo("QCOM", "Qualcomm Incorporated", "Mobile chipset and wireless technology licensor"),
    SymbolInfo("HON", "Honeywell International Inc.", "Industrial technology and aerospace systems provider"),
    SymbolInfo("IBM", "International Business Machines Corp.", "Enterprise IT, hybrid cloud, and services company"),
    SymbolInfo("BA", "The Boeing Company", "Commercial jet and defense aerospace manufacturer"),
    SymbolInfo("CAT", "Caterpillar Inc.", "Heavy equipment maker for construction and mining"),
    SymbolInfo("LMT", "Lockheed Martin Corporation", "Defense contractor specializing in aerospace and security"),
    SymbolInfo("UPS", "United Parcel Service, Inc.", "Global package delivery and logistics provider"),
    SymbolInfo("PM", "Philip Morris International Inc.", "Global tobacco company expanding into reduced-risk products"),
    SymbolInfo("AMAT", "Applied Materials, Inc.", "Semiconductor fabrication equipment leader"),
    SymbolInfo("AMGN", "Amgen Inc.", "Biotechnology firm with biologics for serious illnesses"),
    SymbolInfo("BMY", "Bristol Myers Squibb Company", "Biopharmaceutical company focused on oncology and immunology"),
    SymbolInfo("SBUX", "Starbucks Corporation", "Global specialty coffee retailer"),
    SymbolInfo("GS", "The Goldman Sachs Group, Inc.", "Global investment banking and trading firm"),
    SymbolInfo("RTX", "RTX Corporation", "Aerospace and defense conglomerate born from Raytheon and United Technologies"),
    SymbolInfo("BKNG", "Booking Holdings Inc.", "Online travel agency with Booking.com and Priceline brands"),
    SymbolInfo("SPGI", "S&P Global Inc.", "Financial data, ratings, and index provider"),
    SymbolInfo("ISRG", "Intuitive Surgical, Inc.", "Robotic-assisted surgery systems leader"),
    SymbolInfo("ADP", "Automatic Data Processing, Inc.", "Payroll and human capital management services provider"),
    SymbolInfo("DE", "Deere & Company", "Agricultural and construction machinery manufacturer"),
    SymbolInfo("BLK", "BlackRock, Inc.", "World's largest asset manager"),
    SymbolInfo("GE", "General Electric Company", "Industrial technology company spanning aviation and energy"),
    SymbolInfo("MU", "Micron Technology, Inc.", "Memory and storage semiconductor manufacturer"),
    SymbolInfo("NOW", "ServiceNow, Inc.", "Workflow automation and IT service management cloud platform"),
    SymbolInfo("BK", "The Bank of New York Mellon Corporation", "Custody bank and asset servicing provider"),
    SymbolInfo("GILD", "Gilead Sciences, Inc.", "Biopharma with antiviral and oncology therapies"),
    SymbolInfo("T", "AT&T Inc.", "US telecom operator with wireless and fiber services"),
    SymbolInfo("MS", "Morgan Stanley", "Global investment bank and wealth management firm"),
    SymbolInfo("C", "Citigroup Inc.", "Global diversified banking group"),
    SymbolInfo("USB", "U.S. Bancorp", "Regional bank with consumer and commercial services"),
    SymbolInfo("SCHW", "The Charles Schwab Corporation", "Retail brokerage and wealth management platform"),
    SymbolInfo("ZTS", "Zoetis Inc.", "Animal health pharmaceuticals and vaccines provider"),
    SymbolInfo("MDLZ", "Mondelez International, Inc.", "Global snacking company with Oreo and Cadbury brands"),
    SymbolInfo("SYK", "Stryker Corporation", "Orthopedic implants and medical devices manufacturer"),
    SymbolInfo("CB", "Chubb Limited", "Global property and casualty insurer"),
    SymbolInfo("CHTR", "Charter Communications, Inc.", "Cable broadband and video services provider"),
    SymbolInfo("MMC", "Marsh & McLennan Companies, Inc.", "Professional services firm in risk and insurance"),
    SymbolInfo("MO", "Altria Group, Inc.", "US tobacco company with Marlboro and reduced-risk investments"),
    SymbolInfo("AXP", "American Express Company", "Charge card and payments network serving consumers and businesses"),
    SymbolInfo("SO", "The Southern Company", "Regulated electric utility serving the southeastern United States"),
    SymbolInfo("TTD", "The Trade Desk, Inc.", "Programmatic digital advertising demand-side platform"),
    SymbolInfo("SNOW", "Snowflake Inc.", "Cloud-native data warehouse and analytics platform"),
    SymbolInfo("SHOP", "Shopify Inc.", "E-commerce platform enabling merchants to run online stores"),
    SymbolInfo("INTU", "Intuit Inc.", "Financial software maker with TurboTax, QuickBooks, and Credit Karma"),
    SymbolInfo("PYPL", "PayPal Holdings, Inc.", "Online payments platform for consumers and merchants"),
    SymbolInfo("PANW", "Palo Alto Networks, Inc.", "Cybersecurity platform spanning network, cloud, and SOC automation"),
    SymbolInfo("CRWD", "CrowdStrike Holdings, Inc.", "Cloud-native endpoint security and threat intelligence provider"),
    SymbolInfo("DDOG", "Datadog, Inc.", "Cloud monitoring and observability platform"),
    SymbolInfo("NET", "Cloudflare, Inc.", "Global edge network delivering security and performance services"),
    SymbolInfo("OKTA", "Okta, Inc.", "Identity and access management cloud provider"),
    SymbolInfo("TEAM", "Atlassian Corporation", "Collaboration and developer workflow software company"),
    SymbolInfo("ZM", "Zoom Video Communications, Inc.", "Video conferencing and unified communications platform"),
    SymbolInfo("PLTR", "Palantir Technologies Inc.", "Data analytics and AI software for enterprises and governments"),
    SymbolInfo("SMCI", "Super Micro Computer, Inc.", "High-performance server and storage systems manufacturer"),
    SymbolInfo("ASML", "ASML Holding N.V.", "Dutch lithography leader enabling advanced semiconductor manufacturing"),
    SymbolInfo("ARM", "Arm Holdings plc", "IP provider for ARM architecture powering mobile and edge chips"),
    SymbolInfo("NVO", "Novo Nordisk A/S", "Diabetes and obesity therapeutics leader"),
    SymbolInfo("ENPH", "Enphase Energy, Inc.", "Microinverter and home energy management supplier"),
]


assert len(SYMBOLS) == 100, f"Expected 100 symbols, got {len(SYMBOLS)}"

SUMMARY_RE = re.compile(r"\- \*\*(?P<label>[^*]+)\*\*: (?P<value>.+)%?")
SPAN_RE = re.compile(r"\- \*\*Span\*\*: (?P<start>\d{4}-\d{2}-\d{2}) → (?P<end>\d{4}-\d{2}-\d{2}) \((?P<years>[0-9.]+) years\)")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def run_strategy(symbol: str, summary_path: str) -> None:
    cmd = [
        sys.executable,
        "strategy_tqqq_reserve.py",
        "--base-symbol",
        symbol,
        "--experiment",
        "A1",
        "--no-show",
        "--save-summary",
        summary_path,
    ]
    print(f"Running A1 for {symbol} ...")
    subprocess.run(cmd, check=True)


def parse_summary(summary_path: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    span_info = None
    with open(summary_path, "r", encoding="utf-8") as fh:
        for line in fh:
            span_match = SPAN_RE.search(line)
            if span_match:
                span_info = span_match.groupdict()
                continue
            m = SUMMARY_RE.search(line)
            if not m:
                continue
            label = m.group("label").strip()
            raw_value = m.group("value").strip()
            number_match = NUMBER_RE.search(raw_value)
            if not number_match:
                continue
            value = float(number_match.group(0))
            if "Underlying" in label:
                metrics["underlying_cagr"] = value / 100.0
            elif "Fitted" in label:
                metrics["fitted_cagr"] = value / 100.0
            elif "Strategy CAGR" in label:
                metrics["strategy_cagr"] = value / 100.0
            elif "Max drawdown" in label:
                metrics["max_drawdown"] = abs(value) / 100.0
            elif "Rebalances" in label:
                metrics["rebalances"] = value
    if span_info:
        metrics["span_start"] = span_info["start"]
        metrics["span_end"] = span_info["end"]
        metrics["span_years"] = float(span_info["years"])
    return metrics


def load_price_history(symbol: str) -> pd.DataFrame:
    symbol_upper = symbol.upper()
    if symbol_upper == "QQQ":
        csv_path = "unified_nasdaq.csv"
    else:
        csv_path = os.path.join("symbol_data", f"{symbol_upper}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Price history missing for {symbol_upper} at {csv_path}")
    df = pd.read_csv(csv_path)
    if "date" not in df.columns or "close" not in df.columns:
        raise RuntimeError(f"Unexpected columns in {csv_path}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.dropna(subset=["close"]).copy()
    df = df[df["close"] > 0]
    df = df.set_index("date")
    return df


def compute_symbol_cagr(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    years = (df.index[-1] - df.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    start = float(df["close"].iloc[0])
    end = float(df["close"].iloc[-1])
    if start <= 0:
        return float("nan")
    return (end / start) ** (1.0 / years) - 1.0


def compute_fit_quality(symbol: str, df: pd.DataFrame) -> float:
    cache_path = os.path.join("temperature_cache", f"{symbol.upper()}.json")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Temperature cache missing for {symbol}")
    with open(cache_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    A = float(payload["A"])
    r = float(payload["r"])
    start_ts = pd.to_datetime(payload["start_ts"])
    t_years = (df.index - start_ts).days / 365.25
    pred = A * np.power(1.0 + r, t_years)
    actual = df["close"].to_numpy(dtype=float)
    rel = np.where(pred > 0, (actual - pred) / pred, 0.0)
    rmse_rel = float(np.sqrt(np.mean(rel ** 2)))
    return 1.0 / (1.0 + rmse_rel)


def ensure_assets(symbol: str) -> Dict[str, float]:
    symbol_dir = "qqq" if symbol.upper() == "QQQ" else symbol.lower()
    os.makedirs(symbol_dir, exist_ok=True)
    summary_path = os.path.join(symbol_dir, f"{symbol.lower()}_A1_summary.md")
    symbol_upper = symbol.upper()
    data_path = "unified_nasdaq.csv" if symbol_upper == "QQQ" else os.path.join("symbol_data", f"{symbol_upper}.csv")
    temp_cache_path = os.path.join("temperature_cache", f"{symbol_upper}.json")
    needs_run = not os.path.exists(summary_path) or not os.path.exists(data_path) or not os.path.exists(temp_cache_path)
    if needs_run:
        run_strategy(symbol, summary_path)
    else:
        print(f"Summary already exists for {symbol}, skipping strategy rerun.")
    metrics = parse_summary(summary_path)
    return metrics


def build_stats() -> OrderedDict:
    stats_rows = []
    for info in SYMBOLS:
        metrics = ensure_assets(info.symbol)
        price_df = load_price_history(info.symbol)
        symbol_cagr = compute_symbol_cagr(price_df)
        fit_quality = compute_fit_quality(info.symbol, price_df)
        metrics.setdefault("underlying_cagr", symbol_cagr)
        metrics["symbol_cagr"] = symbol_cagr
        metrics["fit_quality"] = fit_quality
        stats_rows.append((info, metrics))

    def sort_key(item):
        cagr = item[1].get("strategy_cagr")
        if cagr is None:
            return float("-inf")
        if isinstance(cagr, float) and math.isnan(cagr):
            return float("-inf")
        return float(cagr)

    stats_rows.sort(key=sort_key, reverse=True)

    ordered = OrderedDict()
    for info, metrics in stats_rows:
        ordered[info.symbol] = {
            "Symbol": info.symbol,
            "Name": info.name,
            "Description": info.description,
            "Span": {
                "Start": metrics.get("span_start"),
                "End": metrics.get("span_end"),
                "Years": metrics.get("span_years"),
            },
            "Symbol_CAGR": metrics.get("symbol_cagr"),
            "Fitted_CAGR": metrics.get("fitted_cagr"),
            "A1_CAGR": metrics.get("strategy_cagr"),
            "Max_Drawdown": metrics.get("max_drawdown"),
            "Rebalances": metrics.get("rebalances"),
            "Fit_Quality": metrics.get("fit_quality"),
        }
    return ordered


def format_float(val: float | None) -> float | None:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return None
    return float(round(val, 6))


def write_output(stats: OrderedDict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines: List[str] = []
    lines.append("A1_SYMBOL_STATS = {")
    for symbol, data in stats.items():
        lines.append(f"    \"{symbol}\": {{")
        lines.append(f"        \"Symbol\": \"{data['Symbol']}\",")
        lines.append(f"        \"Name\": \"{data['Name']}\",")
        lines.append(f"        \"Description\": \"{data['Description']}\",")
        span = data.get("Span") or {}
        lines.append("        \"Span\": {")
        lines.append(f"            \"Start\": {repr(span.get('Start'))},")
        lines.append(f"            \"End\": {repr(span.get('End'))},")
        years_val = span.get("Years")
        years_repr = "None" if years_val is None else f"{round(years_val, 2)}"
        lines.append(f"            \"Years\": {years_repr},")
        lines.append("        },")
        for key in ["Symbol_CAGR", "Fitted_CAGR", "A1_CAGR", "Max_Drawdown", "Rebalances", "Fit_Quality"]:
            val = data.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                val_fmt = format_float(float(val))
            else:
                val_fmt = val
            if val_fmt is None:
                val_str = "None"
            elif isinstance(val_fmt, float) and key == "Rebalances":
                val_str = str(int(round(val_fmt)))
            elif isinstance(val_fmt, float):
                val_str = f"{val_fmt}"
            else:
                val_str = repr(val_fmt)
            lines.append(f"        \"{key}\": {val_str},")
        lines.append("    },")
    lines.append("}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    stats = build_stats()
    write_output(stats, "a1_symbol_stats.py")


if __name__ == "__main__":
    main()
