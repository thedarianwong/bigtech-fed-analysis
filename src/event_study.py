"""
Event study analysis for Fed rate hike sensitivity.

Computes post-hike cumulative returns, abnormal returns (vs S&P 500),
statistical significance tests, and sensitivity rankings for the Magnificent 7.
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_post_event_returns(
    prices: pd.DataFrame,
    event_dates: list[pd.Timestamp],
    windows: list[int] = [1, 5, 10, 30, 60, 90],
) -> pd.DataFrame:
    """Compute cumulative returns at each window after each event.

    Uses T+1 convention: Day 0 is the first trading day after the event.

    Returns:
        DataFrame with columns: event_date, ticker, window, cum_return
    """
    results = []

    for event_date in event_dates:
        # Find T+1: first trading day after the event
        future_dates = prices.index[prices.index > event_date]
        if len(future_dates) == 0:
            continue
        t1 = future_dates[0]
        t1_idx = prices.index.get_loc(t1)

        # Price at T+1 open (use T+1 close as baseline since we have daily close)
        # Actually, baseline is the close on event day (last price before reaction)
        event_day_candidates = prices.index[prices.index <= event_date]
        if len(event_day_candidates) == 0:
            continue
        baseline_date = event_day_candidates[-1]

        for ticker in prices.columns:
            baseline_price = prices.loc[baseline_date, ticker]
            if pd.isna(baseline_price) or baseline_price == 0:
                continue

            for window in windows:
                target_idx = t1_idx + window - 1  # -1 because T+1 is day 1
                if target_idx >= len(prices.index):
                    continue

                target_date = prices.index[target_idx]
                target_price = prices.loc[target_date, ticker]

                if pd.isna(target_price):
                    continue

                cum_return = (target_price - baseline_price) / baseline_price

                results.append({
                    "event_date": event_date,
                    "ticker": ticker,
                    "window": window,
                    "cum_return": cum_return,
                    "baseline_date": baseline_date,
                    "target_date": target_date,
                })

    return pd.DataFrame(results)


def compute_abnormal_returns(
    returns_df: pd.DataFrame,
    benchmark_ticker: str = "^GSPC",
) -> pd.DataFrame:
    """Compute abnormal returns: stock return minus benchmark return.

    Isolates company-specific sensitivity by controlling for broad market moves.
    """
    # Separate benchmark and stock returns
    benchmark = returns_df[returns_df["ticker"] == benchmark_ticker][
        ["event_date", "window", "cum_return"]
    ].rename(columns={"cum_return": "benchmark_return"})

    stocks = returns_df[returns_df["ticker"] != benchmark_ticker].copy()

    # Merge benchmark return onto stock returns
    merged = stocks.merge(benchmark, on=["event_date", "window"], how="left")
    merged["abnormal_return"] = merged["cum_return"] - merged["benchmark_return"]

    return merged


def aggregate_sensitivity(abnormal_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate abnormal returns across all events per stock per window.

    Returns mean, median, std, and count per ticker per window.
    """
    agg = (
        abnormal_df.groupby(["ticker", "window"])["abnormal_return"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    agg.columns = ["ticker", "window", "mean_abnormal", "median_abnormal", "std_abnormal", "n_events"]
    return agg


def run_significance_tests(
    abnormal_df: pd.DataFrame, window: int = 30
) -> pd.DataFrame:
    """Run one-sample t-test per stock: is the post-hike abnormal return significantly != 0?

    Tests whether the mean abnormal return at the given window is significantly
    negative (one-sided) or different from zero (two-sided).
    """
    subset = abnormal_df[abnormal_df["window"] == window]
    results = []

    for ticker in sorted(subset["ticker"].unique()):
        values = subset[subset["ticker"] == ticker]["abnormal_return"].dropna()
        if len(values) < 3:
            continue

        t_stat, p_two_sided = stats.ttest_1samp(values, 0)
        # One-sided p-value: testing if mean < 0 (stocks drop after hikes)
        p_one_sided = p_two_sided / 2 if t_stat < 0 else 1 - p_two_sided / 2

        results.append({
            "ticker": ticker,
            "window": window,
            "mean_abnormal": values.mean(),
            "std_abnormal": values.std(),
            "n_events": len(values),
            "t_stat": t_stat,
            "p_value_two_sided": p_two_sided,
            "p_value_one_sided": p_one_sided,
            "significant_5pct": p_one_sided < 0.05,
            "significant_10pct": p_one_sided < 0.10,
        })

    return pd.DataFrame(results).sort_values("mean_abnormal")


def rank_sensitivity(
    abnormal_df: pd.DataFrame, window: int = 30
) -> pd.DataFrame:
    """Rank stocks from most sensitive (largest drop) to most resilient.

    Combines aggregated stats and significance tests into one ranking table.
    """
    agg = aggregate_sensitivity(abnormal_df)
    agg_window = agg[agg["window"] == window].copy()

    sig = run_significance_tests(abnormal_df, window=window)

    ranking = agg_window.merge(
        sig[["ticker", "t_stat", "p_value_one_sided", "significant_5pct"]],
        on="ticker",
        how="left",
    )
    ranking = ranking.sort_values("mean_abnormal").reset_index(drop=True)
    ranking["rank"] = range(1, len(ranking) + 1)

    return ranking


def split_by_cycle(
    abnormal_df: pd.DataFrame,
    cycle_1_end: str = "2019-01-01",
    cycle_2_start: str = "2022-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split abnormal returns into the two hike cycles for consistency check.

    Cycle 1: 2015-2018 hikes
    Cycle 2: 2022-2023 hikes
    """
    cycle_1 = abnormal_df[abnormal_df["event_date"] < cycle_1_end]
    cycle_2 = abnormal_df[abnormal_df["event_date"] >= cycle_2_start]
    return cycle_1, cycle_2


def export_sensitivity_ranking(
    ranking: pd.DataFrame, output_path: str | None = None
):
    """Export sensitivity ranking to CSV for Tableau."""
    from pathlib import Path
    if output_path is None:
        output_path = Path(__file__).resolve().parent.parent / "output" / "sensitivity_ranking.csv"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_path, index=False)
    print(f"Exported sensitivity ranking ({len(ranking)} stocks) to {output_path}")
