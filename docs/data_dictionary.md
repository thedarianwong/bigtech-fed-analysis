# Data Dictionary

Documents every column in every output CSV file used for Tableau and analysis.

---

## `output/prices_daily.csv`

Daily stock prices and macro context in long format (one row per ticker per day).

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Trading date (YYYY-MM-DD) |
| `ticker` | string | Stock ticker symbol (AAPL, MSFT, NVDA, GOOGL, META, AMZN, TSLA, ^GSPC, ^VIX) |
| `price` | float | Adjusted close price in USD (accounts for splits and dividends) |
| `daily_return` | float | Single-day percentage return (0.01 = 1%). NaN on the first day. |
| `fed_rate` | float | Federal Funds effective rate (%). Monthly FRED data forward-filled to daily. |
| `cpi` | float | Consumer Price Index for All Urban Consumers (index level). Monthly, forward-filled. |
| `treasury_10y` | float | 10-Year Treasury Constant Maturity Rate (%). Daily from FRED. |
| `gdp_growth` | float | Real GDP growth rate (% annualized). Quarterly, linearly interpolated then forward-filled. |
| `cpi_yoy` | float | CPI year-over-year change (%). Computed from 12-month CPI pct_change. Available from ~2016. |
| `rate_regime` | string | Current Fed regime: "hiking", "cutting", or "holding". Based on most recent FOMC action. |

---

## `output/sensitivity_ranking.csv`

Sensitivity ranking of Magnificent 7 stocks by average 30-day post-hike abnormal return.

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Stock ticker symbol (Mag 7 only) |
| `window` | int | Event window in trading days (default: 30) |
| `mean_abnormal` | float | Mean abnormal return across all hike events (negative = underperforms market) |
| `median_abnormal` | float | Median abnormal return across all hike events |
| `std_abnormal` | float | Standard deviation of abnormal returns |
| `n_events` | int | Number of hike events included in the average |
| `t_stat` | float | t-statistic from one-sample t-test (H0: mean = 0) |
| `p_value_one_sided` | float | One-sided p-value (testing if mean < 0) |
| `significant_5pct` | bool | True if p_value_one_sided < 0.05 |
| `rank` | int | 1 = most sensitive (largest drop), 7 = most resilient |

---

## `output/event_study.csv`

Event-window data for each stock around each rate hike. Used for the heatmap and reaction trajectory.

| Column | Type | Description |
|--------|------|-------------|
| `event_date` | date | Date of the FOMC rate hike announcement |
| `ticker` | string | Stock ticker symbol (Mag 7 only) |
| `day` | int | Trading day relative to event. 0 = event day close (baseline), negative = pre-event, positive = post-event. |
| `cum_return` | float | Cumulative raw return from baseline (event day close) |
| `benchmark_return` | float | Cumulative S&P 500 return over the same window |
| `abnormal_return` | float | cum_return minus benchmark_return. Isolates stock-specific rate sensitivity. |
| `trade_date` | date | Actual calendar date of this trading day |

---

## `output/prophet_forecast.csv`

90-day Prophet forecasts per stock under two rate scenarios.

| Column | Type | Description |
|--------|------|-------------|
| `ds` | date | Date (historical dates + 90 future forecast days) |
| `ticker` | string | Stock ticker symbol (Mag 7 only) |
| `yhat` | float | Prophet point forecast (predicted price in USD) |
| `yhat_lower` | float | Lower bound of 80% confidence interval |
| `yhat_upper` | float | Upper bound of 80% confidence interval |
| `scenario` | string | "hold" = Fed keeps current rate, "hike" = Fed raises rate +25bps |

---

## `data/fomc_events.csv`

Manually curated FOMC rate decisions (committed to git, not auto-generated).

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | FOMC announcement date (YYYY-MM-DD) |
| `rate_before` | float | Fed Funds target rate upper bound before the decision (%) |
| `rate_after` | float | Fed Funds target rate upper bound after the decision (%) |
| `change_bps` | int | Rate change in basis points. Positive = hike, negative = cut. |
| `direction` | string | "hike" or "cut" |
