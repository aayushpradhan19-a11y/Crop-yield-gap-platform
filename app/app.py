import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Crop Yield Gap Predictor", layout="wide")
st.title("Crop Yield Gap Diagnosis Platform")
st.markdown("Predict and explain yield gaps for Rice & Wheat across Indian states.")

@st.cache_resource
def load_model():
    with open('models/xgb_yield_gap.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv('data/processed/yield_gap_master.csv')

model = load_model()
df = load_data()

encoders = {}
df_enc = df.copy()
for col in df_enc.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])
    encoders[col] = le

st.sidebar.header("Input Parameters")
crop = st.sidebar.selectbox("Crop", df['Crop'].unique())
state = st.sidebar.selectbox("State", sorted(df['State'].unique()))
season = st.sidebar.selectbox("Season", df['Season'].unique())
crop_year = st.sidebar.slider("Crop Year", int(df['Crop_Year'].min()), int(df['Crop_Year'].max()), 2015)
area = st.sidebar.number_input("Area (ha)", min_value=0.1, value=500.0, step=50.0)
production = st.sidebar.number_input("Production (tonnes)", min_value=0.1, value=1500.0, step=100.0)
rainfall = st.sidebar.number_input("Annual Rainfall (mm)", min_value=0.0, value=1100.0, step=50.0)
fertilizer = st.sidebar.number_input("Fertilizer (kg/ha)", min_value=0.0, value=120.0, step=5.0)
pesticide = st.sidebar.number_input("Pesticide (kg/ha)", min_value=0.0, value=1.5, step=0.1)
yield_val = st.sidebar.number_input("Current Yield (kg/ha)", min_value=0.0, value=2.0, step=0.1)
potential_yield = st.sidebar.number_input("Potential Yield (kg/ha)", min_value=0.0, value=3.5, step=0.1)
gap_trend = st.sidebar.number_input("Gap Trend", value=0.0, step=0.1)

def encode(col, val):
    return int(encoders[col].transform([val])[0]) if val in encoders[col].classes_ else 0

input_data = pd.DataFrame([{
    'Crop': encode('Crop', crop),
    'Crop_Year': crop_year,
    'Season': encode('Season', season),
    'State': encode('State', state),
    'Area': area,
    'Production': production,
    'Annual_Rainfall': rainfall,
    'Fertilizer': fertilizer,
    'Pesticide': pesticide,
    'Yield': yield_val,
    'Potential_Yield': potential_yield,
    'Gap_Trend': gap_trend
}])

prediction = model.predict(input_data)[0]

col1, col2, col3 = st.columns(3)
col1.metric("Predicted Yield Gap", f"{prediction:.3f} kg/ha")
col2.metric("Current Yield", f"{yield_val:.2f} kg/ha")
col3.metric("Potential Yield", f"{potential_yield:.2f} kg/ha")

st.markdown("---")
st.subheader("Feature Impact on This Prediction")
explainer = shap.Explainer(model)
shap_values = explainer(input_data)

fig, ax = plt.subplots(figsize=(10, 4))
shap.plots.waterfall(shap_values[0], show=False)
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")
st.subheader("Dataset Overview")
st.dataframe(df.head(20))
