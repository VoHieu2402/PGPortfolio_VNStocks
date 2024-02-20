import pandas as pd
import numpy as np

def calc_measures(dataframe, benchmark_dataframe, rf):
    # Calculate daily returns
    dataframe['Returns'] = dataframe['Price'].pct_change()

    # Calculate Sharpe ratio
    sharpe_ratio = (dataframe['Returns'].mean() - rf) / dataframe['Returns'].std()

    # Calculate Sortino ratio
    downside_returns = dataframe['Returns'].copy()
    downside_returns[downside_returns > 0] = 0
    sortino_ratio = (dataframe['Returns'].mean() - rf) / downside_returns.std()

    # Calculate Maximum Drawdown
    cumulative_returns = (1 + dataframe['Returns']).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Calculate Accumulative Portfolio Value
    dataframe['Accumulative Portfolio Value'] = (1 + dataframe['Returns']).cumprod()
    accum_pv = dataframe['Accumulative Portfolio Value'].iloc[-1]

    # Calculate Information Ratio
    benchmark_returns = benchmark_dataframe["Price"].pct_change()
    info_ratio = (dataframe['Returns'] - benchmark_returns).mean() / np.abs((dataframe['Returns'] - benchmark_returns).std())

    return sharpe_ratio, sortino_ratio, max_drawdown, info_ratio, accum_pv
