import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# Function to calculate CAGR from recent 5 years of price history (up to today)
def get_cagr(symbol, years=5):
    try:
        ticker = yf.Ticker(symbol + ".NS")
        data = ticker.history(period=f"{years}y", interval="1d")

        if data.empty or len(data) < 2:
            raise Exception("Not enough data")

        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        cagr = ((end_price / start_price) ** (1 / years)) - 1

        return round(cagr, 4)
    except Exception as e:
        print(f"Error fetching CAGR for {symbol}: {e}")
        return None

# Step 1: Take user input for stocks and investment (CAGR auto-calculated)
def get_user_portfolio():
    user_portfolio = {}
    print("Enter stock data (type 'done' when finished):")
    while True:
        stock = input("Stock symbol (e.g., TCS, INFY): ").upper()
        if stock == 'DONE':
            break
        try:
            investment = float(input(f"Investment in {stock} (₹): "))
            cagr = get_cagr(stock)
            if cagr is not None:
                print(f"Estimated 5-Year CAGR for {stock}: {cagr*100:.2f}%")
                user_portfolio[stock] = {'investment': investment, 'cagr': cagr}
            else:
                print(f"Skipping {stock} due to data error.")
        except ValueError:
            print("Invalid input. Please enter numeric values.")
    return user_portfolio

# Step 2: Build projection table
def build_projection(user_portfolio, years=5):
    year_range = np.arange(0, years + 1)
    df = pd.DataFrame({'Year': year_range})

    for stock, data in user_portfolio.items():
        df[stock] = [data['investment'] * ((1 + data['cagr']) ** y) for y in year_range]

    df['Total'] = df.drop(columns='Year').sum(axis=1)
    df['Lower'] = df['Total'] * 0.85
    df['Upper'] = df['Total'] * 1.15
    return df

# Step 3: Plot using Plotly
def plot_projection(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['Year'], y=df['Total'],
                             mode='lines+markers', name='Expected Value',
                             line=dict(color='blue', width=3)))

    fig.add_trace(go.Scatter(x=df['Year'], y=df['Upper'],
                             mode='lines', name='Optimistic Scenario',
                             line=dict(color='green', dash='dot')))

    fig.add_trace(go.Scatter(x=df['Year'], y=df['Lower'],
                             mode='lines', name='Pessimistic Scenario',
                             line=dict(color='red', dash='dot')))

    fig.add_trace(go.Scatter(
        x=list(df['Year']) + list(df['Year'][::-1]),
        y=list(df['Upper']) + list(df['Lower'][::-1]),
        fill='toself',
        fillcolor='rgba(0,100,80,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False
    ))

    fig.update_layout(
        title='Projected Portfolio Growth Over 5 Years (Using Latest CAGR)',
        xaxis_title='Year',
        yaxis_title='Portfolio Value (INR)',
        template='plotly_white'
    )

    fig.show()

# === Main Execution ===
portfolio = get_user_portfolio()
if portfolio:
    projection_df = build_projection(portfolio)
    plot_projection(projection_df)
else:
    print("No stocks entered.")
