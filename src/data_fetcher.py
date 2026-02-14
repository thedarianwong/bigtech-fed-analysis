"""
Data fetching and master dataframe builder.

Fetches stock prices (yfinance), macro indicators (FRED API),
and FOMC event data to produce a unified daily dataframe.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# --- Constants ---

MAG7_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA"]
BENCHMARK_TICKERS = ["^GSPC", "^VIX"]
ALL_TICKERS = MAG7_TICKERS + BENCHMARK_TICKERS
DEFAULT_START = "2015-01-01"

FRED_SERIES = {
    "fed_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "treasury_10y": "DGS10",
    "gdp_growth": "A191RL1Q225SBEA",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


# --- Stock Prices ---


def fetch_stock_prices(
    tickers: list[str] = ALL_TICKERS, start_date: str = DEFAULT_START
) -> pd.DataFrame:
    """Fetch daily Adj Close prices from yfinance.

    Returns a DataFrame with DatetimeIndex and one column per ticker.
    """
    raw = yf.download(tickers, start=start_date, auto_adjust=False)

    # yfinance returns MultiIndex columns for multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Adj Close"]
    else:
        prices = raw[["Adj Close"]]
        prices.columns = tickers

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"

    # Drop rows where all prices are NaN (weekends/holidays already excluded by yfinance)
    prices = prices.dropna(how="all")

    return prices


# --- FRED Macro Data ---


def fetch_macro_data(
    api_key: str | None = None, start_date: str = DEFAULT_START
) -> pd.DataFrame:
    """Fetch macro indicators from FRED API.

    Returns a daily-frequency DataFrame (monthly/quarterly data forward-filled).
    Requires fredapi: pip install fredapi
    """
    from fredapi import Fred

    if api_key is None:
        api_key = os.environ.get("FRED_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise ValueError(
            "FRED API key required. Register free at "
            "https://fred.stlouisfed.org/docs/api/api_key.html "
            "and set FRED_API_KEY in your .env file."
        )

    fred = Fred(api_key=api_key)
    series = {}

    for name, fred_id in FRED_SERIES.items():
        series[name] = fred.get_series(fred_id, observation_start=start_date)

    # Compute CPI YoY on the raw monthly series BEFORE combining
    # (pct_change(12) on monthly data = 12-month lookback = year-over-year)
    if "cpi" in series:
        cpi_monthly = series["cpi"].dropna()
        series["cpi_yoy"] = cpi_monthly.pct_change(periods=12) * 100

    macro = pd.DataFrame(series)
    macro.index = pd.to_datetime(macro.index)
    macro.index.name = "date"

    # GDP is quarterly — interpolate to monthly first
    if "gdp_growth" in macro.columns:
        macro["gdp_growth"] = macro["gdp_growth"].interpolate(method="linear")

    # Resample to daily and forward-fill (FRED data is monthly/daily mix)
    macro = macro.resample("D").asfreq().ffill()

    return macro


# --- FOMC Events ---


def load_fomc_events(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the manually curated FOMC events CSV.

    Expected columns: date, rate_before, rate_after, change_bps, direction
    """
    if path is None:
        path = DATA_DIR / "fomc_events.csv"

    fomc = pd.read_csv(path, parse_dates=["date"])
    fomc = fomc.sort_values("date").reset_index(drop=True)

    expected_cols = {"date", "rate_before", "rate_after", "change_bps", "direction"}
    missing = expected_cols - set(fomc.columns)
    if missing:
        raise ValueError(f"FOMC CSV missing columns: {missing}")

    return fomc


def get_hike_events(fomc: pd.DataFrame) -> pd.DataFrame:
    """Filter FOMC events to only rate hikes."""
    return fomc[fomc["direction"] == "hike"].reset_index(drop=True)


def get_cut_events(fomc: pd.DataFrame) -> pd.DataFrame:
    """Filter FOMC events to only rate cuts."""
    return fomc[fomc["direction"] == "cut"].reset_index(drop=True)


# --- Master Dataframe ---


def build_master_dataframe(
    prices: pd.DataFrame, macro: pd.DataFrame, fomc: pd.DataFrame
) -> pd.DataFrame:
    """Merge prices and macro into a single daily dataframe.

    Adds columns:
    - daily_return: per-ticker daily percentage return
    - rate_regime: 'hiking', 'cutting', or 'holding' based on FOMC direction
    - fomc_event: True on days following an FOMC rate change (T+1)
    """
    # Align macro to price dates via merge
    master = prices.copy()

    # Add macro columns (forward-filled to daily already)
    for col in macro.columns:
        master[col] = macro[col].reindex(master.index, method="ffill")

    # Compute daily returns per ticker
    stock_cols = [c for c in prices.columns if c in MAG7_TICKERS + BENCHMARK_TICKERS]
    for ticker in stock_cols:
        master[f"{ticker}_return"] = master[ticker].pct_change()

    # Add rate regime column
    master["rate_regime"] = _assign_rate_regime(master.index, fomc)

    # Mark FOMC event days (T+1: the trading day after announcement)
    fomc_dates = pd.to_datetime(fomc["date"])
    event_next_days = set()
    for d in fomc_dates:
        # Find the next trading day after the FOMC date
        next_days = master.index[master.index > d]
        if len(next_days) > 0:
            event_next_days.add(next_days[0])
    master["fomc_event"] = master.index.isin(event_next_days)

    return master


def _assign_rate_regime(dates: pd.DatetimeIndex, fomc: pd.DataFrame) -> pd.Series:
    """Assign rate regime (hiking/cutting/holding) based on FOMC history."""
    regime = pd.Series("holding", index=dates, name="rate_regime")

    # Sort FOMC events by date
    sorted_fomc = fomc.sort_values("date")

    for i in range(len(sorted_fomc)):
        row = sorted_fomc.iloc[i]
        start = pd.to_datetime(row["date"])

        # Regime extends until the next FOMC event
        if i + 1 < len(sorted_fomc):
            end = pd.to_datetime(sorted_fomc.iloc[i + 1]["date"])
        else:
            end = dates.max()

        mask = (dates >= start) & (dates < end)
        if row["direction"] == "hike":
            regime[mask] = "hiking"
        elif row["direction"] == "cut":
            regime[mask] = "cutting"
        else:
            regime[mask] = "holding"

    return regime


# --- Export ---


def export_prices_for_tableau(master: pd.DataFrame, output_path: str | Path | None = None):
    """Export the master dataframe as a CSV for Tableau.

    Melts the wide-format prices into long format:
    date, ticker, price, daily_return, fed_rate, cpi, ...
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "prices_daily.csv"

    stock_cols = [c for c in master.columns if c in MAG7_TICKERS + BENCHMARK_TICKERS]
    macro_cols = [c for c in master.columns if c in list(FRED_SERIES.keys()) + ["cpi_yoy", "rate_regime"]]

    rows = []
    for ticker in stock_cols:
        return_col = f"{ticker}_return"
        df_ticker = master[[ticker, return_col] + macro_cols].copy()
        df_ticker = df_ticker.rename(columns={ticker: "price", return_col: "daily_return"})
        df_ticker["ticker"] = ticker
        rows.append(df_ticker)

    long = pd.concat(rows).reset_index()
    long = long[["date", "ticker", "price", "daily_return"] + macro_cols]
    long = long.sort_values(["date", "ticker"]).reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(output_path, index=False)
    print(f"Exported {len(long):,} rows to {output_path}")

    return long
