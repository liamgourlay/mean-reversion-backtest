# Mean Reversion Backtesting Strategy

This project implements a simple mean reversion trading strategy using Z-score signals across multiple equities. It includes a full backtesting pipeline, parameter optimisation, and comparison against buy-and-hold.

---

# Strategy Overview

The strategy:

- Computes rolling mean and standard deviation of asset prices
- Generates Z-scores
- Takes long positions when price is below lower threshold
- Takes short positions when price is above upper threshold
- Uses equal-weight portfolio across assets
- Includes transaction costs and avoids lookahead bias via signal shifting

---

# Benchmark

Performance is compared against an equal-weight buy-and-hold portfolio.

---

# Features

- Multi-asset backtesting
- Parameter grid search
- Transaction cost modelling
- Sharpe ratio optimisation
- Drawdown analysis
- Equity curve visualisation

---

# Results

Best performing strategy:

- Sharpe ratio: 0.506215
- Total return: 0.786775
- Max drawdown: -0.141735

# Findings

- Buy-and-hold significantly outperformed the mean reversion strategy over the test period for most tracked stock portfolios
- The strategy for SPY and QQQ achieved a Sharpe ratio of 0.51, indicating weak risk-adjusted performance
- The best hyperparameters for this included no transaction costs, which reduce returns due to the frequent trading of the portfolio
- Performance was poor during strong trending markets, where mean reversion breaks down

See `results/strategy_vs_buyhold.png`

---

# How to Run

```bash
pip install -r requirements.txt
python src/backtest.py
