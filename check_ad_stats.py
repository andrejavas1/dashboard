import pandas as pd

df = pd.read_csv('data/features_matrix.csv')

print('AD_ROC_20d Statistics (AFTER FIX):')
print('  Min:', df['AD_ROC_20d'].min())
print('  Max:', df['AD_ROC_20d'].max())
print('  Mean:', df['AD_ROC_20d'].mean())
print()
print('Chaikin_Oscillator Statistics:')
print('  Min:', df['Chaikin_Oscillator'].min())
print('  Max:', df['Chaikin_Oscillator'].max())
print('  Mean:', df['Chaikin_Oscillator'].mean())
