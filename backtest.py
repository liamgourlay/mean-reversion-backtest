import pandas as pd
import yfinance as yf
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import os
os.makedirs("results", exist_ok=True)

def load_data(tickers, start, end):
    """1
    Download adjusted close price data from Yahoo Finance.

    Parameters
    ----------
    tickers: str or list
        Single ticker or list of tickers.
    start: str
        Start date in YYYY-MM-DD format.
    end: str
        End date in YYYY-MM-DD format.

    Returns
    ----------
    pd.DataFrame
        Cleaned dataframe with tickers as columns.
    """

    # Ensure input is always a list
    if isinstance(tickers, str):
        tickers = [tickers]
    else:
        tickers = tickers

    data_raw = yf.download(tickers, start=start, end=end, auto_adjust=False)["Adj Close"]

    # Turn single ticker case from series to DataFrame
    if isinstance(data_raw, pd.Series):
        data = data_raw.to_frame(name=tickers[0])
    else:
        data = data_raw.copy()

    # Flatten multiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data.dropna()

def compute_signals(data, rolling = 20, upper = 1, lower = -1):
    """
    Generate mean reversion trading signals using Z-scores.

    Parameters
    ----------
    data: pd.DataFrame
        Price data.
    rolling: int
        Rolling window for mean and std calculations.
    upper: float
        Upper Z-score threshold for short signal.
    lower: float
        Lower Z-score threshold for long signal.

    Returns
    ----------
    pd.Dataframe
        Dataframe indicating the daily position for each ticker.
    """

    signals = pd.DataFrame(index=data.index)

    # Calculating for each ticker in tickers
    for col in data.columns:
        mean = data[col].rolling(rolling).mean()
        std = data[col].rolling(rolling).std(ddof=0)

        # Z-score calculation
        z = (data[col] - mean) / std

        # Mean reversion logic
        pos = np.where(z > upper, -1,
                       np.where(z < lower, 1, 0))

        # Shift dataframe to avoid lookahead
        signals[col] = pd.Series(pos, index=data.index).shift(1).fillna(0)

    return signals

def compute_returns(data, signals, cost=0.001):
    """
    Compute portfolio returns.

    Parameters
    ----------
    data: pd.DataFrame
        Price data for each ticker.
    signals: pd.DataFrame
        Position data for each ticker.
    cost: float
        Transaction cost per unit of transaction.

    Returns
    ----------
    tuple
        strat_returns: pd.Series
        turnover: pd.Series
    """

    returns = data.pct_change()

    # Portfolio return for each day assuming equal weight across all assets.
    strat_returns = (returns * signals).mean(axis=1)

    # Trading activity and cost penalty.
    turnover = signals.diff().abs().sum(axis=1)
    strat_returns -= turnover * cost

    return strat_returns.dropna(), turnover

def compute_metrics(returns):
    """
    Calculate performance metrics for strategy.

    Parameters
    ----------
    returns: pd.Series
        Strategy returns.

    Returns
    ----------
    dict
        Sharpe ratio
        Maximum drawdown
        Total return
        Final equity value
    """

    # Calculates accumulated equity returns.
    equity = (1 + returns).cumprod()

    # Calculate sharpe ratio for strategy.
    sharpe = returns.mean() / returns.std() * np.sqrt(252)

    # Calculate drawdown for strategy.
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1

    return {
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "total_return": equity.iloc[-1] - 1,
        "final_equity": equity.iloc[-1]
    }

def run_backtest(tickers, start, end, rolling, upper, lower, cost, return_equity=False):
    """
    Full backtest pipeline return metrics and an optional equity curve.

    Parameters
    ----------
    tickers: list or str
    start, end: str
    rolling: int
    upper, lower: float
    cost: float
    return_equity: bool

    Returns
    ----------
    dict
        Backtest metrics.
    """

    data = load_data(tickers, start, end)
    signals = compute_signals(data, rolling, upper, lower)
    returns, trades = compute_returns(data, signals, cost)

    metrics = compute_metrics(returns)

    # Calculates return equity for buy and hold if needed.
    if return_equity:
        metrics["equity_curve"] = (1 + returns).cumprod()
        metrics["data"] = data

    return metrics

def parameter_sweep(tickers, start, end):
    """
    Run a grid search for strategy hyperparameters for mean reversion model.

    Parameters
    --------
    tickers: str or list
    start, end: str

    Returns
    ---------
    pd.DataFrame
        Performance metrics for each combination of parameters.
    """

    results = []

    rolling_list = [10, 20, 30, 50]
    thresholds = [1, 1.5, 2]
    costs = [0, 0.0005]

    # Run backtest for each combination of parameters.
    for r, t, c in product(rolling_list, thresholds, costs):
        metrics = run_backtest(
            tickers, start, end,
            rolling=r,
            upper=t,
            lower=-t,
            cost=c
        )

        # Store results and parameters.
        results.append({
            "rolling": r,
            "threshold": t,
            "cost": c,
            **metrics
        })

    return pd.DataFrame(results)

if __name__ == "__main__":

    # Run parameter sweep over chosen tickers and time frame.
    df = parameter_sweep(["AAPL", "MSFT"], "2010-01-01", "2025-01-01")

    # Select best parameter based on Sharpe ratio
    best = df.sort_values("sharpe", ascending=False).iloc[0]

    # Rerun backtest with best parameters for equity curve and raw data.
    best_metrics = run_backtest(
        ["SPY", "QQQ"],
        "2010-01-01",
        "2025-01-01",
        rolling=int(best["rolling"]),
        upper=best["threshold"],
        lower=-best["threshold"],
        cost=best["cost"],
        return_equity=True
    )

    strategy_equity = best_metrics["equity_curve"]
    data = best_metrics["data"]

    # Calculate buy and hold returns
    returns = data.pct_change().dropna()
    buy_hold = (1 + returns.mean(axis=1)).cumprod()

    # Plot and show comparison
    plt.figure()
    strategy_equity.plot(label="Mean Reversion Strategy")
    buy_hold.plot(label="Buy & Hold")

    plt.title("Strategy vs Buy & Hold")
    plt.ylabel("Equity (Growth of $1)")
    plt.legend()

    plt.savefig("results/strategy_vs_buyhold.png")
    plt.show()

    df.to_csv("results/parameter_sweep.csv", index=False)

    # Print best configuration of parameters
    print("Best parameters:")
    print(best)
