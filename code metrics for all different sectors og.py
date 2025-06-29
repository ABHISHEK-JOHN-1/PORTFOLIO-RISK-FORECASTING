# stock_data_fetcher.py

import yfinance as yf
from nsepython import nse_eq
import pandas as pd
import time
from typing import List, Optional, Dict

# === Sector Benchmarks (Expanded) ===
SECTOR_BENCHMARKS = {
    "IT": {
        "pe": 25, "pb": 5, "roe": 15, "debtToEquity": 0.3, "eps": 50,
        "revenuePerEmployee": 2000000, "utilizationRate": 80, "attritionRate": 15
    },
    "BANKS": {
        "pe": 15, "pb": 2, "dividendYield": 2.0, "roe": 12, "netNPA": 2,
        "casaRatio": 40, "car": 12, "nim": 3, "roa": 1
    },
    "PHARMA": {
        "pe": 20, "pb": 4, "roe": 14, "debtToEquity": 0.5, "grossMargin": 60,
        "rAndDPercent": 8, "usfdaApprovals": 1
    },
    "AUTO": {
        "pe": 18, "pb": 3, "roe": 12, "debtToEquity": 0.6, "eps": 35,
        "inventoryTurnover": 6, "marketShareGrowth": 1
    },
    "FMCG": {
        "pe": 30, "pb": 8, "roe": 20, "grossMargin": 50,
        "volumeGrowth": 5, "adToSales": 15
    },
    "OIL&GAS": {
        "grm": 5, "netDebtToEBITDA": 2, "roce": 15
    },
    "REAL_ESTATE": {
        "debtToEquity": 1.5, "presalesGrowth": 15, "rentalYield": 6, "occupancyRate": 90
    },
    "INFRA": {
        "orderBookToSales": 3, "ebitdaMargin": 12, "debtToEquity": 1.5,
        "interestCoverage": 2, "workingCapitalDays": 100, "projectDelayRatio": 10
    },
    "POWER_UTILITIES": {
        "plf": 85, "atcLoss": 15, "roce": 12, "dscr": 1.5
    },
    "E_COMMERCE": {
        "sssg": 6, "gmvGrowth": 10, "cac": 300, "ltv": 1000,
        "fulfillmentCost": 20, "inventoryTurnover": 8
    }
}

# === Utilities ===
def get_all_nse_symbols() -> List[str]:
    try:
        df = pd.read_csv("https://archives.nseindia.com/content/equities/EQUITY_L.csv")
        return df['SYMBOL'].dropna().tolist()
    except Exception as e:
        print("Error fetching NSE symbols:", e)
        return []


def get_historical_data(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        yf_symbol = symbol.upper() + ".NS"
        df = yf.download(yf_symbol, start=start, end=end, progress=False)
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching historical data for {symbol}:", e)
        return None


def get_realtime_data(symbol: str) -> Optional[dict]:
    try:
        return nse_eq(symbol.upper())
    except Exception as e:
        print(f"Error fetching real-time data for {symbol}:", e)
        return None

# === Scoring ===
def score_stock(symbol: str, sector: str) -> Dict:
    data = get_realtime_data(symbol)
    if not data:
        return {"symbol": symbol, "score": 0, "issues": ["No data"], "passed": []}

    benchmarks = SECTOR_BENCHMARKS.get(sector.upper(), {})
    if not benchmarks:
        return {"symbol": symbol, "score": 0, "issues": ["No benchmarks for sector"], "passed": []}

    checks = {
        "pe": float(data.get("pE", 0)),
        "pb": float(data.get("pb", 0)),
        "roe": float(data.get("returnOnEquity", 0)),
        "debtToEquity": float(data.get("debtEquity", 1)),
        "eps": float(data.get("eps", 0)),
        "dividendYield": float(data.get("dividendYield", 0)),
        "netNPA": float(data.get("netNPA", 0)),
        "casaRatio": float(data.get("casaRatio", 0)),
        "car": float(data.get("capitalAdequacyRatio", 0)),
        "nim": float(data.get("netInterestMargin", 0)),
        "roa": float(data.get("returnOnAssets", 0)),
        "grossMargin": float(data.get("grossMargin", 0)),
        "rAndDPercent": float(data.get("rAndDPercent", 0)),
        "inventoryTurnover": float(data.get("inventoryTurnover", 0)),
        "volumeGrowth": float(data.get("volumeGrowth", 0)),
        "adToSales": float(data.get("adToSales", 0)),
        "rentalYield": float(data.get("rentalYield", 0)),
        "occupancyRate": float(data.get("occupancyRate", 0)),
        "presalesGrowth": float(data.get("presalesGrowth", 0)),
        "netDebtToEBITDA": float(data.get("netDebtToEBITDA", 1)),
        "grm": float(data.get("grm", 0)),
        "revenuePerEmployee": float(data.get("revenuePerEmployee", 0)),
        "utilizationRate": float(data.get("utilizationRate", 0)),
        "attritionRate": float(data.get("attritionRate", 0)),
        "orderBookToSales": float(data.get("orderBookToSales", 0)),
        "ebitdaMargin": float(data.get("ebitdaMargin", 0)),
        "interestCoverage": float(data.get("interestCoverage", 0)),
        "workingCapitalDays": float(data.get("workingCapitalDays", 0)),
        "projectDelayRatio": float(data.get("projectDelayRatio", 0)),
        "plf": float(data.get("plantLoadFactor", 0)),
        "atcLoss": float(data.get("atcLosses", 0)),
        "dscr": float(data.get("dscr", 0)),
        "sssg": float(data.get("sameStoreSalesGrowth", 0)),
        "gmvGrowth": float(data.get("gmvGrowth", 0)),
        "cac": float(data.get("cac", 0)),
        "ltv": float(data.get("ltv", 0)),
        "fulfillmentCost": float(data.get("fulfillmentCost", 0))
    }

    passed, issues = [], []
    score = 0

    for metric, benchmark in benchmarks.items():
        val = checks.get(metric)
        if val is None:
            issues.append(metric)
        elif metric in ["debtToEquity", "netNPA", "netDebtToEBITDA", "adToSales",
                        "attritionRate", "atcLoss", "workingCapitalDays", "projectDelayRatio",
                        "cac", "fulfillmentCost"]:
            if val <= benchmark:
                score += 1
                passed.append(metric)
            else:
                issues.append(metric)
        else:
            if val >= benchmark:
                score += 1
                passed.append(metric)
            else:
                issues.append(metric)

    return {
        "symbol": symbol,
        "sector": sector,
        "score": score,
        "total": len(benchmarks),
        "passed": passed,
        "issues": issues
    }


def evaluate_portfolio(symbols_with_sectors: Dict[str, str]) -> List[Dict]:
    results = []
    for symbol, sector in symbols_with_sectors.items():
        print(f"Evaluating {symbol} ({sector})...")
        results.append(score_stock(symbol, sector))
        time.sleep(1)
    return results


if __name__ == "__main__":
    portfolio = {
        "INFY": "IT",
        "HDFCBANK": "BANKS",
        "SUNPHARMA": "PHARMA",
        "MARUTI": "AUTO",
        "HINDUNILVR": "FMCG",
        "ONGC": "OIL&GAS",
        "DLF": "REAL_ESTATE",
        "LARSEN": "INFRA",
        "TATAPOWER": "POWER_UTILITIES",
        "NYKAA": "E_COMMERCE"
    }

    results = evaluate_portfolio(portfolio)
    print("\n=== Portfolio Evaluation ===")
    for res in results:
        print(f"{res['symbol']} [{res['sector']}]: Score {res['score']} / {res['total']}\n  Good: {res['passed']}\n  Needs Improvement: {res['issues']}\n")
