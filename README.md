# Big Tech Under Pressure

**How Federal Reserve Rate Decisions Impact the Magnificent 7**

> "When the Fed raises rates, which Big Tech stocks suffer most — and can we predict the next drop using historical macro patterns?"

A data analysis project combining event study methodology, statistical testing, and Prophet time-series forecasting to quantify how Fed rate hikes differentially impact Apple, Microsoft, Nvidia, Alphabet, Meta, Amazon, and Tesla.

---

## Research Questions

1. **Sensitivity (RQ1):** Which Magnificent 7 stocks show the largest price drops in the 30/60/90 days following a Fed rate hike? Is the sensitivity consistent across the 2015–18 and 2022–23 hike cycles?

2. **Speed of Reaction (RQ2):** How quickly does each stock respond to a rate decision — does the market reprice immediately (Day 1–5) or does the full impact unfold over weeks?

3. **Prediction (RQ3):** Can Prophet forecast each stock's 90-day price trajectory given current macro conditions? What does the model predict under a "rate hold" vs. "one more hike" scenario?

---

## Architecture

```
yfinance + FRED API
        |
        v
  Python (Pandas)          data/fomc_events.csv
  fetch + clean     <----  (manually curated)
        |
        v
  src/ modules
  - event_study.py         Abnormal returns, t-tests, sensitivity ranking
  - speed_analysis.py      Event windows, peak drop timing, heatmap
  - forecaster.py          Prophet models with macro regressors
        |
        v
  output/ CSVs  -------->  Tableau Public Dashboard
```

---

## Key Findings

<!-- Update these after running the analysis -->

- **Most rate-sensitive:** _[ticker]_ — avg _[X]%_ abnormal drop in 30 days post-hike
- **Most resilient:** _[ticker]_ — avg _[X]%_ abnormal return in 30 days post-hike
- **Fastest reactor:** _[ticker]_ — peak drop at Day _[X]_
- **Slowest reactor:** _[ticker]_ — peak drop at Day _[X]_
- **Hike penalty (Prophet):** _[ticker]_ most hurt by a +25bps hike scenario

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data Sources | yfinance (stock prices), FRED API (macro indicators), FOMC events (manual CSV) |
| Analysis | Python 3.11, Pandas, NumPy, SciPy, Prophet |
| Visualization | Tableau Public (dashboard), Matplotlib/Seaborn (exploratory) |
| Cloud (planned) | AWS Lambda, S3, EventBridge |

---

## Setup

```bash
# Clone the repo
git clone https://github.com/thedarianwong/bigtech-fed-analysis.git
cd bigtech-fed-analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up FRED API key (free: https://fred.stlouisfed.org/docs/api/api_key.html)
cp .env.example .env
# Edit .env and add your FRED_API_KEY
```

## How to Run

Run the Jupyter notebooks in order:

```bash
cd notebooks
jupyter lab
```

| Notebook | What it does | Output |
|----------|-------------|--------|
| `01_data_ingestion.ipynb` | Fetch prices + macro, validate, export | `output/prices_daily.csv` |
| `02_event_study.ipynb` | Post-hike abnormal returns, t-tests, ranking | `output/sensitivity_ranking.csv` |
| `03_speed_reaction.ipynb` | Event windows, heatmap, peak drop timing | `output/event_study.csv` |
| `04_prophet_forecast.ipynb` | Train Prophet models, scenario forecasts | `output/prophet_forecast.csv` |

---

## Repo Structure

```
bigtech-fed-analysis/
├── src/                     Core Python modules (imported by notebooks)
│   ├── data_fetcher.py      yfinance + FRED fetching, master dataframe
│   ├── event_study.py       Abnormal returns, t-tests, sensitivity ranking
│   ├── speed_analysis.py    Event windows, peak drop, heatmap data
│   └── forecaster.py        Prophet wrapper with scenario analysis
├── notebooks/               Jupyter notebooks (run in order 01→04)
├── data/                    Manually curated data (fomc_events.csv)
├── output/                  Generated CSVs for Tableau (gitignored)
├── docs/                    Data dictionary
├── lambda/                  AWS Lambda automation (planned)
├── CLAUDE.md                Project conventions for AI-assisted development
├── PLAN.md                  Implementation plan
└── BUILD_LOG.md             Step-by-step build log
```

---

## Data Sources

All free, no paid APIs required.

- **yfinance:** AAPL, MSFT, NVDA, GOOGL, META, AMZN, TSLA, ^GSPC, ^VIX — daily adjusted close from 2015
- **FRED API:** Fed Funds Rate, CPI, 10-Year Treasury, GDP Growth — [free API key](https://fred.stlouisfed.org/docs/api/api_key.html)
- **FOMC Events:** 28 rate changes (2015–2024) manually curated from [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/openmarket.htm)

See [`docs/data_dictionary.md`](docs/data_dictionary.md) for full column documentation.

---

## Tableau Dashboard

<!-- Add Tableau Public link once published -->

**Dashboard link:** _[Coming soon]_

Four sheets + one story:
1. **Rate Hike Timeline** — Fed rate overlaid with stock prices
2. **Sensitivity Rankings** — 30-day post-hike abnormal return bar chart
3. **Speed of Reaction** — Heatmap: 7 stocks x 30 days
4. **90-Day Forecast** — Prophet predictions with scenario toggle

---

## Resume Bullets

**Data Analyst focus:**
Engineered event-study analysis of Fed rate hike impacts on Magnificent 7 stocks across 30+ FOMC decisions, quantifying differential price sensitivity using Python (Pandas, SciPy) and publishing interactive findings via Tableau Public.

**Analytics Engineer focus:**
Architected automated data pipeline using AWS Lambda and S3 to fetch weekly stock prices (yfinance) and macro indicators (FRED API), training Prophet time-series models to generate 90-day scenario forecasts under alternate rate conditions.

---

## License

[Apache 2.0](LICENSE)
