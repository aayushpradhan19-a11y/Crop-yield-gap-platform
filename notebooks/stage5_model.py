import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load data
df = pd.read_csv('data/processed/yield_gap_master.csv')
print(f"Loaded: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Encode all categorical/text columns
print("\nEncoding categorical columns...")
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])
    print(f"  Encoded: {col}")

# Features and target
features = [c for c in df.columns if c != 'Yield_Gap']
print(f"\nFeatures used: {features}")

X = df[features]
y = df['Yield_Gap']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# Train XGBoost model
print("\nTraining XGBoost model...")
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)
print("Training complete!")

# Evaluate on test set
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  R²   : {r2:.4f}")

# Cross validation
print("\nRunning 5-fold cross validation...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"  Cross Validation R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Save model
with open('models/xgb_yield_gap.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved: models/xgb_yield_gap.pkl")

# SHAP explainability
print("\nGenerating SHAP values...")
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.title('SHAP Beeswarm — Feature Impact on Yield Gap', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/shap_beeswarm.png', dpi=300, bbox_inches='tight')
print("SHAP plot saved: outputs/shap_beeswarm.png")

print("\nStage 5 complete!")