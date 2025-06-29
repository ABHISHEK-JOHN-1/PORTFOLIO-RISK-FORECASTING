import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_stock_data(stock):
    data = yf.download(stock, period='5y')
    return data['Adj Close']

def cagr_projection(price_series, years=5):
    start_price = price_series.iloc[0]
    end_price = price_series.iloc[-1]
    cagr = (end_price / start_price) ** (1 / years) - 1
    projected_value = end_price * ((1 + cagr) ** years)
    return round(cagr * 100, 2), round(projected_value, 2)

def monte_carlo_simulation(price_series, years=5, simulations=500):
    daily_returns = price_series.pct_change().dropna()
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    S0 = price_series[-1]
    T = years
    N = 252 * T
    dt = 1/252

    paths = np.zeros((N, simulations))
    paths[0] = S0

    for t in range(1, N):
        rand = np.random.normal(0, 1, simulations)
        paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)
    
    return paths

def eps_price_projection(stock, years=5):
    ticker = yf.Ticker(stock)
    eps = ticker.info.get("trailingEps", None)
    pe = ticker.info.get("trailingPE", None)
    growth = 0.12

    if not eps or not pe:
        return None

    future_eps = eps * ((1 + growth) ** years)
    future_price = future_eps * pe

    return round(future_eps, 2), round(future_price, 2)

def sharpe_ratio(price_series):
    returns = price_series.pct_change().dropna()
    annual_return = returns.mean() * 252
    annual_std = returns.std() * np.sqrt(252)
    rf = 0.06  # 6% risk-free rate
    sharpe = (annual_return - rf) / annual_std
    return round(annual_return * 100, 2), round(annual_std * 100, 2), round(sharpe, 2)

def scenario_projection(current_price, cagr):
    pessimistic = current_price * ((1 + (cagr - 0.05)) ** 5)
    base = current_price * ((1 + cagr) ** 5)
    optimistic = current_price * ((1 + (cagr + 0.05)) ** 5)
    return round(pessimistic, 2), round(base, 2), round(optimistic, 2)

def full_projection(stock, years=5, simulations=500):
    data = get_stock_data(stock)
    latest_price = data[-1]

    cagr_pct, cagr_proj = cagr_projection(data, years)
    eps_result = eps_price_projection(stock, years)
    annual_return, vol, sharpe = sharpe_ratio(data)
    pess, base, opt = scenario_projection(latest_price, cagr_pct / 100)
    monte = monte_carlo_simulation(data, years, simulations)

    return {
        "CAGR (%)": cagr_pct,
        "Projected Value (CAGR)": cagr_proj,
        "EPS Projection": eps_result,
        "Sharpe Ratio": sharpe,
        "Annual Return (%)": annual_return,
        "Volatility (%)": vol,
        "Scenario Analysis": {
            "Pessimistic": pess,
            "Base": base,
            "Optimistic": opt
        },
        "Monte Carlo": monte,
        "Current Price": round(latest_price, 2)
    }
