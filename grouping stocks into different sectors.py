# stock_data_fetcher.py

import yfinance as yf
from nsepython import nse_eq, nse_fetch
import pandas as pd
import time
from typing import List, Optional, Dict


def get_all_nse_symbols() -> List[str]:
    """Fetches list of all NSE stock symbols from nsepython."""
    try:
        df = pd.read_csv("https://archives.nseindia.com/content/equities/EQUITY_L.csv")
        return df['SYMBOL'].dropna().tolist()
    except Exception as e:
        print("Error fetching NSE symbols:", e)
        return []


def get_historical_data(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Fetch historical data using yfinance (NSE: use '.NS' suffix)."""
    try:
        yf_symbol = symbol.upper() + ".NS"
        df = yf.download(yf_symbol, start=start, end=end, progress=False)
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching historical data for {symbol}:", e)
        return None


def get_realtime_data(symbol: str) -> Optional[dict]:
    """Fetch real-time data using nsepython."""
    try:
        return nse_eq(symbol.upper())
    except Exception as e:
        print(f"Error fetching real-time data for {symbol}:", e)
        return None


def group_stocks_by_sector() -> Dict[str, List[str]]:
    """Group all NSE stocks by their sectors."""
    try:
        df = pd.read_csv("https://archives.nseindia.com/content/equities/EQUITY_L.csv")
        df = df[['SYMBOL', 'NAME OF COMPANY', 'INDUSTRY']].dropna()

        sector_map = {}
        for _, row in df.iterrows():
            sector = row['INDUSTRY'].strip()
            symbol = row['SYMBOL'].strip()
            sector_map.setdefault(sector, []).append(symbol)

        return sector_map
    except Exception as e:
        print("Error grouping stocks by sector:", e)
        return {}


if __name__ == "__main__":
    # Example usage
    symbols = get_all_nse_symbols()[:5]  # Limit to 5 for demo
    print("Sample NSE Symbols:", symbols)

    for symbol in symbols:
        print(f"\n=== {symbol} ===")

        realtime = get_realtime_data(symbol)
        print("Real-Time Data:", realtime["lastPrice"] if realtime else "Unavailable")

        history = get_historical_data(symbol, "2023-01-01", "2023-12-31")
        if history is not None:
            print("Historical Sample:", history.head(1).to_dict("records"))
        time.sleep(1)  # To avoid rate limiting

    print("\n=== Grouped Stocks by Sector ===")
    sector_data = group_stocks_by_sector()
    for sector, syms in list(sector_data.items())[:5]:
        print(f"{sector}: {syms[:5]} ...")
