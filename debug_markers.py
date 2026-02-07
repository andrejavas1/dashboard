"""Debug why markers are being filtered."""
import json

# Load occurrence file
with open('data/pattern_31_occurrences.json', 'r') as f:
    occurrences = json.load(f)

# Load OHLCV data
with open('data/ohlcv.json', 'r') as f:
    ohlcv = json.load(f)

# Create set of OHLCV dates
ohlcv_dates = set()
for row in ohlcv:
    date = row['Date'].split(' ')[0] if ' ' in row['Date'] else row['Date']
    ohlcv_dates.add(date)

print(f"Total occurrences in file: {len(occurrences)}")
print(f"OHLCV data range: {ohlcv[0]['Date']} to {ohlcv[-1]['Date']}")
print(f"Total OHLCV records: {len(ohlcv)}")
print()

# Check which occurrence dates are in OHLCV
found = 0
not_found = 0
for occ in occurrences:
    date = occ['Date']
    if date in ohlcv_dates:
        found += 1
    else:
        not_found += 1
        if not_found <= 5:
            print(f"Date not in OHLCV: {date}")

print(f"\nFound in OHLCV: {found}")
print(f"Not found in OHLCV: {not_found}")
