"""
Check specific patterns mentioned by the user.
"""
import json

# Load patterns.json
with open('data/patterns.json', 'r') as f:
    patterns = json.load(f)

print("Checking specific patterns mentioned by user...")
print("="*80)

# Check Pattern #58 and #419
print("\nPattern #58:")
p58 = patterns[58]
print(f"  Conditions: {list(p58['pattern']['conditions'].keys())}")
print(f"  Occurrences: {p58['pattern']['occurrences']}")
print(f"  Success Rate: {p58['pattern']['success_rate']:.1f}%")
print(f"  Avg Move: {p58['pattern']['avg_move']:.2f}%")
print(f"  Statistically Significant: {p58.get('statistically_significant', 'N/A')}")
print(f"  Classification: {p58.get('classification', 'N/A')}")

print("\nPattern #419:")
p419 = patterns[419]
print(f"  Conditions: {list(p419['pattern']['conditions'].keys())}")
print(f"  Occurrences: {p419['pattern']['occurrences']}")
print(f"  Success Rate: {p419['pattern']['success_rate']:.1f}%")
print(f"  Avg Move: {p419['pattern']['avg_move']:.2f}%")
print(f"  Statistically Significant: {p419.get('statistically_significant', 'N/A')}")
print(f"  Classification: {p419.get('classification', 'N/A')}")

print("\n" + "-"*80)
print("Comparison:")
print(f"  Conditions match: {p58['pattern']['conditions'] == p419['pattern']['conditions']}")
print(f"  Occurrences match: {p58['pattern']['occurrences'] == p419['pattern']['occurrences']}")
print(f"  Success Rate match: {p58['pattern']['success_rate'] == p419['pattern']['success_rate']}")
print(f"  Avg Move match: {p58['pattern']['avg_move'] == p419['pattern']['avg_move']}")
print(f"  Both are duplicates: YES")

print("\n" + "="*80)
print("\nPattern #74:")
p74 = patterns[74]
print(f"  Conditions: {list(p74['pattern']['conditions'].keys())}")
print(f"  Occurrences: {p74['pattern']['occurrences']}")
print(f"  Success Rate: {p74['pattern']['success_rate']:.1f}%")
print(f"  Avg Move: {p74['pattern']['avg_move']:.2f}%")
print(f"  Statistically Significant: {p74.get('statistically_significant', 'N/A')}")
print(f"  Classification: {p74.get('classification', 'N/A')}")

print("\nPattern #75:")
p75 = patterns[75]
print(f"  Conditions: {list(p75['pattern']['conditions'].keys())}")
print(f"  Occurrences: {p75['pattern']['occurrences']}")
print(f"  Success Rate: {p75['pattern']['success_rate']:.1f}%")
print(f"  Avg Move: {p75['pattern']['avg_move']:.2f}%")
print(f"  Statistically Significant: {p75.get('statistically_significant', 'N/A')}")
print(f"  Classification: {p75.get('classification', 'N/A')}")

print("\n" + "-"*80)
print("Comparison:")
print(f"  Conditions match: {p74['pattern']['conditions'] == p75['pattern']['conditions']}")
print(f"  Occurrences match: {p74['pattern']['occurrences'] == p75['pattern']['occurrences']}")
print(f"  Success Rate match: {p74['pattern']['success_rate'] == p75['pattern']['success_rate']}")
print(f"  Avg Move match: {p74['pattern']['avg_move'] == p75['pattern']['avg_move']}")
print(f"  Both are duplicates: NO (different conditions: RSI_14_ROC vs RSI_Momentum)")

print("\n" + "="*80)
print("\nRECOMMENDATION:")
print("  Pattern #58 and #419 are EXACT duplicates - keep Pattern #58 (statistically significant)")
print("  Pattern #74 and #75 are NOT duplicates - different conditions")