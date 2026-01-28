# portfolio_risk_and_factor_model
Quantitative portfolio analytics framework in Python. Computes performance and risk metrics, estimates Fama–French 3-factor exposures via OLS, and tracks time-varying betas, alpha, and R² using rolling regressions on real market data.
Overview

The framework:

Builds portfolio returns from adjusted price series and fixed asset weights

Computes core performance and risk metrics

Decomposes returns using a Fama–French 3-factor model

Estimates time-varying factor exposures via rolling regressions

Persists model outputs (betas, alpha, R²) as structured data for analysis and reuse

The emphasis is on robust data handling, statistical consistency, and interpretability rather than backtest optimization.

Portfolio Specification

Assets: DFUS, DFAI, DFEV, DISV (NYSE-listed ETFs)

Weights:

DFUS: 30%

DFAI: 20%

DFEV: 20%

DISV: 30%

Data frequency: Daily

Sample period: Determined by data availability after cleaning

Methodology
1. Data Handling

Prices are sourced via yfinance

Adjusted close prices are extracted robustly from MultiIndex structures

Only overlapping dates across all assets are retained

Returns are computed as daily percentage changes

2. Portfolio Metrics

The following metrics are computed:

Annualized return

Annualized volatility

Sharpe ratio (risk-free rate ≈ 0)

Maximum drawdown

3. Factor Model

Model: Fama–French 3-factor (MKT, SMB, HML)

Estimation: Ordinary Least Squares (OLS) on excess returns

Outputs:

Alpha (daily and annualized)

Factor betas

R² (model fit)

Regression results are stored as structured objects rather than only printed.

4. Rolling Analysis

Rolling-window regressions (default: 252 trading days)

Time series of:

Factor betas

Alpha

R²

Used to monitor stability of exposures and regime changes

Outputs

The script produces:

Portfolio performance metrics

Full-sample FF3 regression results (betas, alpha, R²)

Rolling factor exposures and diagnostics

Visualizations:

Portfolio growth

Drawdowns

Rolling betas

Rolling R²

Rolling alpha

Optionally, outputs can be exported to CSV for downstream analysis.

Project Structure
.
├── portfolio_factor_analysis.py
├── README.md
└── requirements.txt

Requirements
python >= 3.10
pandas
numpy
yfinance
statsmodels
matplotlib


Install dependencies with:

pip install -r requirements.txt

How to Run
python portfolio_factor_analysis.py


To enable CSV export of results, set in the script:

SAVE_OUTPUTS = True

Notes and Limitations

The factor model uses U.S. Fama–French daily factors as a baseline specification.

The portfolio includes international equity exposure; therefore, the model is intended as a systematic risk approximation, not a fully global factor decomposition.

Future extensions may include global factor sets, alternative risk models, or Monte Carlo simulations.

Purpose

This project is intended as a demonstration of quantitative portfolio analysis skills, including:

Python-based data handling

Risk and performance measurement

Factor modeling and interpretation

Reproducible research practices
