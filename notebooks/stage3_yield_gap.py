import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv('data/raw/crop_yield.csv')
df.columns = df.columns.str.strip()
df = df[df['Crop'].isin(['Rice', 'Wheat'])].copy()
print(f"Rows after filtering Rice and Wheat: {len(df)}")
df = df.dropna(subset=['Yield'])

df['Potential_Yield'] = df.groupby(['State', 'Crop'])['Yield'].transform(lambda x: x.quantile(0.90))
df['Yield_Gap'] = df['Potential_Yield'] - df['Yield']
df = df[df['Yield_Gap'] >= 0].copy()
print(f"Rows after removing negative gaps: {len(df)}")

# Calculate Gap_Trend without groupby apply
trends = []
for (state, crop), group in df.groupby(['State', 'Crop']):
    if len(group) < 2:
        slope = 0
    else:
        slope = np.polyfit(group['Crop_Year'].values, group['Yield_Gap'].values, 1)[0]
    trends.append({'State': state, 'Crop': crop, 'Gap_Trend': slope})

trends_df = pd.DataFrame(trends)
df = df.merge(trends_df, on=['State', 'Crop'], how='left')

print(f"\nFinal shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df[['State', 'Crop', 'Crop_Year', 'Yield', 'Yield_Gap', 'Gap_Trend']].head())

os.makedirs('outputs', exist_ok=True)
plt.figure(figsize=(8, 4))
plt.hist(df['Yield_Gap'], bins=40, color='steelblue', edgecolor='white')
plt.title('Distribution of Yield Gap (Rice & Wheat)')
plt.xlabel('Yield Gap (kg/ha)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('outputs/yield_gap_histogram.png')
print("Histogram saved")

os.makedirs('data/processed', exist_ok=True)
df.to_csv('data/processed/yield_gap_master.csv', index=False)
print("Saved to data/processed/yield_gap_master.csv")
print("Stage 3 complete!")