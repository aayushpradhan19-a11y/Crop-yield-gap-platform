import pandas as pd
import numpy as np

# Load full crop production dataset
df = pd.read_csv('data/raw/crop_production.csv')
print(f"Full dataset loaded: {df.shape}")

# Rename columns to match our pipeline
df.rename(columns={'State_Name': 'State', 'District_Name': 'District'}, inplace=True)

# Drop rows with missing Production
df.dropna(subset=['Production'], inplace=True)
print(f"After dropping nulls: {df.shape}")

# Compute Yield (tonnes/ha)
df['Yield'] = df['Production'] / df['Area']
df = df[df['Yield'] > 0]
df = df[df['Yield'] < df['Yield'].quantile(0.99)]  # remove extreme outliers
print(f"After cleaning yield outliers: {df.shape}")

# Load soil/NPK data
soil = pd.read_csv('data/raw/Crop_recommendation.csv')
print(f"\nSoil dataset loaded: {soil.shape}")
print(f"Soil columns: {soil.columns.tolist()}")

# Compute potential yield per crop from soil data
# Use 90th percentile yield per crop as potential
potential = df.groupby('Crop')['Yield'].quantile(0.90).reset_index()
potential.columns = ['Crop', 'Potential_Yield']
print(f"\nPotential yield computed for {len(potential)} crops")

# Merge potential yield
df = df.merge(potential, on='Crop', how='left')

# Compute Yield Gap
df['Yield_Gap'] = df['Potential_Yield'] - df['Yield']
df = df[df['Yield_Gap'] >= 0]  # remove rows where yield exceeds potential

# Merge soil NPK data (average per crop)
soil_avg = soil.groupby('label').agg({
    'N': 'mean', 'P': 'mean', 'K': 'mean',
    'ph': 'mean', 'rainfall': 'mean',
    'temperature': 'mean', 'humidity': 'mean'
}).reset_index()
soil_avg.rename(columns={'label': 'Crop', 'ph': 'pH'}, inplace=True)

# Normalize crop names for merging
df['Crop_lower'] = df['Crop'].str.lower().str.strip()
soil_avg['Crop_lower'] = soil_avg['Crop'].str.lower().str.strip()
df = df.merge(soil_avg.drop('Crop', axis=1), on='Crop_lower', how='left')
df.drop('Crop_lower', axis=1, inplace=True)

# Compute Gap Trend (rolling mean gap per state-crop)
df = df.sort_values(['State', 'Crop', 'Crop_Year'])
df['Gap_Trend'] = df.groupby(['State', 'Crop'])['Yield_Gap'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)

# Final cleanup
df.dropna(subset=['Yield_Gap', 'Potential_Yield'], inplace=True)
print(f"\nFinal master dataset: {df.shape}")
print(f"Crops: {df['Crop'].nunique()}")
print(f"States: {df['State'].nunique()}")
print(f"Columns: {df.columns.tolist()}")

# Save
df.to_csv('data/processed/yield_gap_master.csv', index=False)
print("\nSaved: data/processed/yield_gap_master.csv")
print("Stage 3 (extended) complete!")