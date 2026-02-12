"""
Speed of reaction analysis for Fed rate hike events.

Builds event-window dataframes around each rate hike, identifies
peak drop timing per stock, and produces heatmap-ready data.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def build_event_windows(
    prices: pd.DataFrame,
    event_dates: list[pd.Timestamp],
    benchmark_ticker: str = "^GSPC",
    pre: int = 10,
    post: int = 30,
) -> pd.DataFrame:
    """Build event-window returns for [-pre, +post] trading days around each hike.

    Uses T+1 convention: Day 0 is the first trading day after the event.
    Returns are cumulative abnormal returns from the baseline (event day close).

    Returns:
        DataFrame with columns: event_date, ticker, day, cum_return,
        benchmark_return, abnormal_return
    """
    results = []

    for event_date in event_dates:
        # Find the event day (last trading day <= event_date) as baseline
        event_day_candidates = prices.index[prices.index <= event_date]
        if len(event_day_candidates) == 0:
            continue
        baseline_date = event_day_candidates[-1]
        baseline_idx = prices.index.get_loc(baseline_date)

        # Window range: -pre to +post relative to baseline
        for day in range(-pre, post + 1):
            target_idx = baseline_idx + day
            if target_idx < 0 or target_idx >= len(prices.index):
                continue

            target_date = prices.index[target_idx]

            for ticker in prices.columns:
                if ticker == benchmark_ticker:
                    continue

                baseline_price = prices.loc[baseline_date, ticker]
                target_price = prices.loc[target_date, ticker]
                bench_baseline = prices.loc[baseline_date, benchmark_ticker]
                bench_target = prices.loc[target_date, benchmark_ticker]

                if pd.isna(baseline_price) or baseline_price == 0:
                    continue
                if pd.isna(target_price):
                    continue

                cum_return = (target_price - baseline_price) / baseline_price
                bench_return = (bench_target - bench_baseline) / bench_baseline if bench_baseline else 0
                abnormal = cum_return - bench_return

                results.append({
                    "event_date": event_date,
                    "ticker": ticker,
                    "day": day,
                    "cum_return": cum_return,
                    "benchmark_return": bench_return,
                    "abnormal_return": abnormal,
                    "trade_date": target_date,
                })

    return pd.DataFrame(results)


def compute_peak_drop(event_windows: pd.DataFrame) -> pd.DataFrame:
    """Identify the day of maximum drawdown per stock post-hike.

    Only considers days > 0 (after the event). Averages across all events
    to find the typical peak drop day per stock.

    Returns:
        DataFrame with columns: ticker, avg_peak_drop_day, median_peak_drop_day,
        avg_peak_drop_return, n_events
    """
    post_event = event_windows[event_windows["day"] > 0].copy()

    results = []
    for ticker in sorted(post_event["ticker"].unique()):
        ticker_data = post_event[post_event["ticker"] == ticker]
        peak_days = []
        peak_returns = []

        for event_date in ticker_data["event_date"].unique():
            event_data = ticker_data[ticker_data["event_date"] == event_date]
            if event_data.empty:
                continue

            # Day with minimum abnormal return = peak drop
            min_idx = event_data["abnormal_return"].idxmin()
            peak_days.append(event_data.loc[min_idx, "day"])
            peak_returns.append(event_data.loc[min_idx, "abnormal_return"])

        if peak_days:
            results.append({
                "ticker": ticker,
                "avg_peak_drop_day": np.mean(peak_days),
                "median_peak_drop_day": np.median(peak_days),
                "avg_peak_drop_return": np.mean(peak_returns),
                "n_events": len(peak_days),
            })

    return pd.DataFrame(results).sort_values("avg_peak_drop_day")


def build_heatmap_data(event_windows: pd.DataFrame) -> pd.DataFrame:
    """Build a pivot table for heatmap: stocks (rows) x days (columns).

    Values are average cumulative abnormal returns across all events.
    Only includes post-event days (day > 0).

    Returns:
        Pivot DataFrame: index=ticker, columns=day, values=mean abnormal return
    """
    post_event = event_windows[event_windows["day"] > 0].copy()

    heatmap = post_event.pivot_table(
        index="ticker",
        columns="day",
        values="abnormal_return",
        aggfunc="mean",
    )

    return heatmap


def build_full_window_heatmap(event_windows: pd.DataFrame) -> pd.DataFrame:
    """Build heatmap including pre-event days [-pre, +post].

    Useful for seeing whether stocks move before the announcement (anticipation).
    """
    heatmap = event_windows.pivot_table(
        index="ticker",
        columns="day",
        values="abnormal_return",
        aggfunc="mean",
    )

    return heatmap


def export_event_study_data(
    event_windows: pd.DataFrame, output_path: str | None = None
):
    """Export event window data to CSV for Tableau."""
    if output_path is None:
        output_path = Path(__file__).resolve().parent.parent / "output" / "event_study.csv"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    event_windows.to_csv(output_path, index=False)
    print(f"Exported {len(event_windows):,} rows to {output_path}")
