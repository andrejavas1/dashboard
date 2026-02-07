#!/usr/bin/env python3
"""Fetch historical XOM data from Yahoo Finance for regime demo."""

import yfinance as yf
import json
from datetime import datetime

def fetch_xom_data():
    """Fetch XOM data from 2010 to present."""
    print("Fetching XOM historical data from Yahoo Finance...")
    print("Date range: 2010-01-01 to 2025-12-31")
    
    # Download data
    ticker = yf.Ticker("XOM")
    df = ticker.history(start="2010-01-01", end="2025-12-31")
    
    print(f"Downloaded {len(df)} trading days")
    
    # Convert to format matching existing ohlcv.json
    data = []
    for index, row in df.iterrows():
        data.append({
            "Date": index.strftime("%Y-%m-%d"),
            "Open": round(float(row["Open"]), 2),
            "High": round(float(row["High"]), 2),
            "Low": round(float(row["Low"]), 2),
            "Close": round(float(row["Close"]), 2),
            "Volume": int(row["Volume"])
        })
    
    # Save to file
    output_path = "data/ohlcv_full_history.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nData saved to: {output_path}")
    print(f"Date range: {data[0]['Date']} to {data[-1]['Date']}")
    print(f"Total records: {len(data)}")
    
    # Show sample
    print("\nFirst 3 records:")
    for record in data[:3]:
        print(f"  {record['Date']}: O={record['Open']}, H={record['High']}, L={record['Low']}, C={record['Close']}")
    
    print("\nLast 3 records:")
    for record in data[-3:]:
        print(f"  {record['Date']}: O={record['Open']}, H={record['High']}, L={record['Low']}, C={record['Close']}")
    
    return output_path

if __name__ == "__main__":
    fetch_xom_data()
