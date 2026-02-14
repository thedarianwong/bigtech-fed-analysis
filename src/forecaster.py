"""
Prophet forecasting with macro regressors and scenario analysis.

Trains one Prophet model per Magnificent 7 stock using daily prices
and external regressors (Fed rate, CPI YoY, VIX). Generates 90-day
forecasts under "rate hold" and "one more hike" scenarios.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet


# Stocks with volatile pre-AI-era history — cap training to 2020+
VOLATILE_TICKERS = ["NVDA", "TSLA"]
DEFAULT_REGRESSORS = ["fed_rate", "cpi_yoy", "vix"]


def prepare_prophet_data(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    ticker: str,
    regressors: list[str] = DEFAULT_REGRESSORS,
    min_date: str | None = None,
) -> pd.DataFrame:
    """Format price + macro data for Prophet.

    Prophet requires columns: ds (date), y (target), plus regressor columns.
    For NVDA and TSLA, defaults to training from 2020 onward.
    """
    if ticker not in prices.columns:
        raise ValueError(f"Ticker {ticker} not found in prices")

    df = pd.DataFrame({
        "ds": prices.index,
        "y": prices[ticker].values,
    })

    # Add regressors from macro
    for reg in regressors:
        if reg == "vix" and "^VIX" in prices.columns:
            df[reg] = prices["^VIX"].values
        elif reg in macro.columns:
            df[reg] = macro[reg].reindex(prices.index, method="ffill").values
        else:
            raise ValueError(f"Regressor '{reg}' not found in macro or prices")

    # Drop rows where target or any regressor is NaN
    # (cpi_yoy needs 12 months of history, macro may not align perfectly)
    required_cols = ["y"] + [r for r in regressors if r in df.columns]
    df = df.dropna(subset=required_cols)

    # Cap training window for volatile tickers
    if min_date is None and ticker in VOLATILE_TICKERS:
        min_date = "2020-01-01"

    if min_date:
        df = df[df["ds"] >= min_date]

    df = df.reset_index(drop=True)
    return df


def train_prophet_model(
    df: pd.DataFrame,
    regressors: list[str] = DEFAULT_REGRESSORS,
    changepoint_prior_scale: float = 0.05,
) -> Prophet:
    """Train a Prophet model with external regressors.

    Uses conservative changepoint_prior_scale to avoid overfitting.
    Suppresses Prophet's default logging for cleaner output.
    """
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )

    for reg in regressors:
        if reg in df.columns:
            model.add_regressor(reg)

    model.fit(df)
    return model


def _build_future_regressors(
    df: pd.DataFrame,
    future: pd.DataFrame,
    regressors: list[str],
    scenario: str = "hold",
    hike_bps: int = 25,
) -> pd.DataFrame:
    """Fill regressor values for the forecast horizon.

    Scenarios:
    - 'hold': carry forward the last known values
    - 'hike': add hike_bps to fed_rate, keep others at last known values
    """
    for reg in regressors:
        if reg not in df.columns:
            continue

        last_value = df[reg].dropna().iloc[-1]

        if scenario == "hike" and reg == "fed_rate":
            # Simulate a rate hike: add bps converted to percentage points
            future_value = last_value + (hike_bps / 100)
        else:
            future_value = last_value

        # For historical dates, use actual values; for future dates, use scenario value
        historical_values = df.set_index("ds")[reg]
        future[reg] = future["ds"].map(historical_values).fillna(future_value)

    return future


def generate_forecast(
    model: Prophet,
    df: pd.DataFrame,
    periods: int = 90,
    regressors: list[str] = DEFAULT_REGRESSORS,
    scenario: str = "hold",
    hike_bps: int = 25,
) -> pd.DataFrame:
    """Generate a forecast with confidence intervals.

    Returns DataFrame with: ds, yhat, yhat_lower, yhat_upper, and scenario label.
    """
    future = model.make_future_dataframe(periods=periods, freq='B')
    future = _build_future_regressors(df, future, regressors, scenario, hike_bps)

    forecast = model.predict(future)
    forecast["scenario"] = scenario

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "scenario"]]


def run_scenario_comparison(
    model: Prophet,
    df: pd.DataFrame,
    periods: int = 90,
    regressors: list[str] = DEFAULT_REGRESSORS,
    hike_bps: int = 25,
) -> pd.DataFrame:
    """Run both 'hold' and 'hike' scenarios and combine results."""
    hold = generate_forecast(model, df, periods, regressors, scenario="hold")
    hike = generate_forecast(model, df, periods, regressors, scenario="hike", hike_bps=hike_bps)

    return pd.concat([hold, hike], ignore_index=True)


def forecast_all_stocks(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    tickers: list[str],
    periods: int = 90,
    regressors: list[str] = DEFAULT_REGRESSORS,
) -> pd.DataFrame:
    """Train and forecast all stocks. Returns combined DataFrame.

    Columns: ds, ticker, yhat, yhat_lower, yhat_upper, scenario
    """
    all_forecasts = []

    for ticker in tickers:
        print(f"  Training Prophet model for {ticker}...")
        df = prepare_prophet_data(prices, macro, ticker, regressors)
        model = train_prophet_model(df, regressors)
        comparison = run_scenario_comparison(model, df, periods, regressors)
        comparison["ticker"] = ticker
        all_forecasts.append(comparison)

    combined = pd.concat(all_forecasts, ignore_index=True)
    combined = combined[["ds", "ticker", "yhat", "yhat_lower", "yhat_upper", "scenario"]]

    return combined


def export_forecast(
    forecast_df: pd.DataFrame, output_path: str | None = None
):
    """Export forecast data to CSV for Tableau."""
    if output_path is None:
        output_path = Path(__file__).resolve().parent.parent / "output" / "prophet_forecast.csv"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(output_path, index=False)
    print(f"Exported {len(forecast_df):,} rows to {output_path}")
