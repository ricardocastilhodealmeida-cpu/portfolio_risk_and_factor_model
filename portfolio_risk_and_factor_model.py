#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[11]:


import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


# ============================================================
# FULL MODEL: Portfolio Metrics + FF3 Regression + Rolling Betas
# Portfolio (NYSE): 30% DFUS, 20% DFAI, 20% DFEV, 30% DISV
# Outputs include: betas + R² persisted as objects + optional CSVs
# ============================================================

# -------------------------
# CONFIG
# -------------------------
TICKERS = ["DFUS", "DFAI", "DFEV", "DISV"]
WEIGHTS = np.array([0.30, 0.20, 0.20, 0.30])

START_DATE = "2018-01-01"
ROLLING_WINDOW = 252
ANNUALIZATION = 252

# Set True if you want to save outputs as CSV files
SAVE_OUTPUTS = False


# -------------------------
# VALIDATION
# -------------------------
if len(TICKERS) != len(WEIGHTS):
    raise ValueError("TICKERS and WEIGHTS must have same length")

if not np.isclose(WEIGHTS.sum(), 1.0):
    raise ValueError(f"WEIGHTS must sum to 1, got {WEIGHTS.sum()}")

if ROLLING_WINDOW < 60:
    raise ValueError("ROLLING_WINDOW too small; use at least ~60 trading days")


# ============================================================
# LOAD FAMA-FRENCH DAILY FACTORS (FF3) — no pandas_datareader
# ============================================================
def load_ff3_daily():
    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    )
    for skip in range(10):
        try:
            df = pd.read_csv(url, compression="zip", skiprows=skip)
            df = df.rename(columns={df.columns[0]: "Date"})
            df = df[df["Date"].astype(str).str.match(r"\d{8}")]
            if df.empty:
                continue
            df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
            df = df.set_index("Date")
            df = df[["Mkt-RF", "SMB", "HML", "RF"]].astype(float) / 100.0
            return df
        except Exception:
            continue
    raise RuntimeError("Could not load Fama-French factors")


# ============================================================
# DOWNLOAD PRICES (yfinance) + EXTRACT ADJ CLOSE ROBUSTLY
# ============================================================
raw = yf.download(
    TICKERS,
    start=START_DATE,
    progress=False,
    auto_adjust=False,
    group_by="column"
)

if raw is None or raw.empty:
    raise RuntimeError("yfinance returned empty data")

# Extract Adj Close from MultiIndex (Price, Ticker)
if isinstance(raw.columns, pd.MultiIndex):
    if "Adj Close" in raw.columns.get_level_values(0):
        prices = raw.xs("Adj Close", level=0, axis=1)
    elif "Close" in raw.columns.get_level_values(0):
        prices = raw.xs("Close", level=0, axis=1)
    else:
        raise KeyError(f"Neither Adj Close nor Close present. Top levels: {list(raw.columns.levels[0])}")
else:
    # Single-index fallback (rare for multi-ticker downloads)
    if "Adj Close" in raw.columns:
        prices = raw["Adj Close"].to_frame(name=TICKERS[0])
    elif "Close" in raw.columns:
        prices = raw["Close"].to_frame(name=TICKERS[0])
    else:
        raise KeyError(f"Neither Adj Close nor Close present. Columns: {list(raw.columns)}")

# Align to requested ticker order and keep complete cases
prices = prices[[t for t in TICKERS if t in prices.columns]].dropna(how="any")
if prices.empty:
    raise RuntimeError("No usable price data after cleaning")

if prices.shape[0] < ROLLING_WINDOW + 50:
    raise ValueError(
        f"Too few observations after cleaning: {prices.shape[0]}. "
        f"Need at least {ROLLING_WINDOW + 50}. Reduce ROLLING_WINDOW or adjust START_DATE."
    )

returns = prices.pct_change().dropna()


# ============================================================
# PORTFOLIO RETURNS
# ============================================================
portfolio_returns = returns.dot(WEIGHTS)
portfolio_returns.name = "Portfolio"


# ============================================================
# PORTFOLIO METRICS
# ============================================================
ann_return = portfolio_returns.mean() * ANNUALIZATION
ann_vol = portfolio_returns.std(ddof=1) * np.sqrt(ANNUALIZATION)
sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

equity = (1 + portfolio_returns).cumprod()
drawdown = equity / equity.cummax() - 1
max_dd = drawdown.min()

portfolio_metrics = pd.Series(
    {
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe_rf0": sharpe,
        "max_drawdown": max_dd,
        "start_date": str(returns.index.min().date()),
        "end_date": str(returns.index.max().date()),
        "n_obs": int(len(returns)),
    },
    name="portfolio_metrics"
)

print("\n=== PORTFOLIO METRICS ===")
print(f"Tickers: {TICKERS}")
print(f"Weights: {np.round(WEIGHTS,4)} (sum={WEIGHTS.sum():.2f})")
print(f"Sample:  {portfolio_metrics['start_date']} -> {portfolio_metrics['end_date']} | n={portfolio_metrics['n_obs']}")
print(f"Annual Return:     {portfolio_metrics['annual_return']:.4f}")
print(f"Annual Volatility: {portfolio_metrics['annual_volatility']:.4f}")
print(f"Sharpe Ratio:      {portfolio_metrics['sharpe_rf0']:.2f}")
print(f"Max Drawdown:      {portfolio_metrics['max_drawdown']:.2%}")


# ============================================================
# FACTOR REGRESSION (FF3)
# ============================================================
ff = load_ff3_daily()
df = pd.concat([portfolio_returns, ff], axis=1).dropna()
df["Excess"] = df["Portfolio"] - df["RF"]

X = sm.add_constant(df[["Mkt-RF", "SMB", "HML"]])
y = df["Excess"]
model = sm.OLS(y, X).fit()

alpha_daily = model.params["const"]
alpha_ann_simple = alpha_daily * ANNUALIZATION
alpha_ann_comp = (1 + alpha_daily) ** ANNUALIZATION - 1

# Persist FF3 regression outputs (betas + R²) as a structured object
ff3_results = pd.Series(
    {
        "alpha_daily": alpha_daily,
        "alpha_ann_simple": alpha_ann_simple,
        "alpha_ann_compounded": alpha_ann_comp,
        "beta_mkt": model.params["Mkt-RF"],
        "beta_smb": model.params["SMB"],
        "beta_hml": model.params["HML"],
        "r2": model.rsquared,
        "n_obs": int(model.nobs),
    },
    name="ff3_full_sample_results"
)

print("\n=== FAMA-FRENCH 3-FACTOR REGRESSION (FULL SAMPLE) ===")
print(f"Alpha (daily):                 {ff3_results['alpha_daily']:.6f}")
print(f"Alpha (annualized, simple):    {ff3_results['alpha_ann_simple']:.4f}")
print(f"Alpha (annualized, compounded):{ff3_results['alpha_ann_compounded']:.4%}")
print(f"Beta MKT (Mkt-RF):             {ff3_results['beta_mkt']:.3f}")
print(f"Beta SMB:                      {ff3_results['beta_smb']:.3f}")
print(f"Beta HML:                      {ff3_results['beta_hml']:.3f}")
print(f"R²:                            {ff3_results['r2']:.3f}")
print(f"Obs (aligned):                 {ff3_results['n_obs']}")

# Optional: full statistical table
# print(model.summary())


# ============================================================
# ROLLING FF3 EXPOSURES + ROLLING R²
# ============================================================
if len(df) <= ROLLING_WINDOW:
    raise ValueError(
        f"Not enough overlapping data for rolling window={ROLLING_WINDOW}. "
        f"Overlap rows={len(df)}. Reduce ROLLING_WINDOW or set an earlier START_DATE."
    )

rows = []
idx = df.index[ROLLING_WINDOW:]

for i in range(ROLLING_WINDOW, len(df)):
    y_win = df["Excess"].iloc[i-ROLLING_WINDOW:i]
    X_win = sm.add_constant(df[["Mkt-RF", "SMB", "HML"]].iloc[i-ROLLING_WINDOW:i])
    res = sm.OLS(y_win, X_win).fit()

    rows.append(
        {
            "Alpha": res.params["const"],
            "Mkt-RF": res.params["Mkt-RF"],
            "SMB": res.params["SMB"],
            "HML": res.params["HML"],
            "R2": res.rsquared,
        }
    )

rolling = pd.DataFrame(rows, index=idx)
rolling["Alpha_Ann_Simple"] = rolling["Alpha"] * ANNUALIZATION
rolling["Alpha_Ann_Comp"] = (1 + rolling["Alpha"]) ** ANNUALIZATION - 1


# ============================================================
# OPTIONAL: SAVE OUTPUTS
# ============================================================
if SAVE_OUTPUTS:
    portfolio_metrics.to_csv("portfolio_metrics.csv")
    ff3_results.to_csv("ff3_regression_results.csv")
    rolling.to_csv("rolling_ff3_exposures.csv")


# ============================================================
# PLOTS
# ============================================================
plt.figure()
equity.plot(title="Portfolio Growth of $1")
plt.ylabel("Growth")
plt.show()

plt.figure()
drawdown.plot(title="Portfolio Drawdown")
plt.ylabel("Drawdown")
plt.show()

plt.figure()
rolling[["Mkt-RF", "SMB", "HML"]].plot(title=f"Rolling FF3 Betas ({ROLLING_WINDOW}-day window)")
plt.ylabel("Beta")
plt.show()

plt.figure()
rolling["R2"].plot(title=f"Rolling R² ({ROLLING_WINDOW}-day window)")
plt.ylabel("R²")
plt.show()

plt.figure()
rolling["Alpha_Ann_Simple"].plot(title=f"Rolling Alpha (Annualized, simple, {ROLLING_WINDOW}-day window)")
plt.ylabel("Alpha (annualized)")
plt.show()

plt.figure()
rolling["Alpha_Ann_Comp"].plot(title=f"Rolling Alpha (Annualized, compounded, {ROLLING_WINDOW}-day window)")
plt.ylabel("Alpha (annualized)")
plt.show()


# In[ ]:




