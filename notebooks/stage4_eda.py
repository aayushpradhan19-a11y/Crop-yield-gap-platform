import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('outputs', exist_ok=True)

df = pd.read_csv('data/processed/yield_gap_master.csv')
print(f"Loaded: {df.shape}")

# Plot 1 - Bar chart: Average yield gap by state
fig, axes = plt.subplots(2, 1, figsize=(16, 14))
for i, crop in enumerate(['Rice', 'Wheat']):
    crop_df = df[df['Crop'] == crop].groupby('State')['Yield_Gap'].mean().sort_values(ascending=False)
    crop_df.plot(kind='bar', ax=axes[i], color='steelblue', edgecolor='white')
    axes[i].set_title(f'Average Yield Gap by State - {crop}')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Yield Gap (kg/ha)')
    axes[i].tick_params(axis='x', rotation=70, labelsize=8)
plt.tight_layout(pad=3.0)
plt.savefig('outputs/yield_gap_by_state.png')
print("Plot 1 saved: yield_gap_by_state.png")

# Plot 2 - Correlation heatmap
corr_cols = ['Yield_Gap', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']
corr = df[corr_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Heatmap - Yield Gap vs Features')
plt.tight_layout()
plt.savefig('outputs/correlation_heatmap.png')
print("Plot 2 saved: correlation_heatmap.png")

# Plot 3 - High gap vs low gap distributions
median_gap = df['Yield_Gap'].median()
high_gap = df[df['Yield_Gap'] > median_gap]
low_gap = df[df['Yield_Gap'] <= median_gap]
plt.figure(figsize=(10, 4))
plt.hist(high_gap['Annual_Rainfall'], bins=30, alpha=0.6, color='red', label='High Gap')
plt.hist(low_gap['Annual_Rainfall'], bins=30, alpha=0.6, color='green', label='Low Gap')
plt.title('Rainfall Distribution: High Gap vs Low Gap Farms')
plt.xlabel('Annual Rainfall (mm)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/rainfall_gap_distribution.png')
print("Plot 3 saved: rainfall_gap_distribution.png")

print("\nStage 4 complete!")