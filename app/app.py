import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Crop Yield Gap Predictor", layout="wide")
st.title("Crop Yield Gap Diagnosis Platform")
st.markdown("Predict and explain yield gaps for 105 crops across all Indian states.")

@st.cache_resource
def load_model():
    with open("models/xgb_yield_gap.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/yield_gap_master.csv")

model = load_model()
df = load_data()

encoders = {}
df_enc = df.copy()
for col in df_enc.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])
    encoders[col] = le

st.sidebar.header("Input Parameters")
crop = st.sidebar.selectbox("Crop", sorted(df["Crop"].unique()))
state = st.sidebar.selectbox("State", sorted(df["State"].unique()))
district = st.sidebar.selectbox("District", sorted(df[df["State"] == state]["District"].unique()))
season = st.sidebar.selectbox("Season", df["Season"].unique())
crop_year = st.sidebar.slider("Crop Year", int(df["Crop_Year"].min()), int(df["Crop_Year"].max()), 2015)
area = st.sidebar.number_input("Area (ha)", min_value=0.1, value=500.0, step=50.0)
production = st.sidebar.number_input("Production (tonnes)", min_value=0.1, value=1500.0, step=100.0)
yield_val = st.sidebar.number_input("Current Yield (tonnes/ha)", min_value=0.0, value=2.0, step=0.1)
potential_yield = st.sidebar.number_input("Potential Yield (tonnes/ha)", min_value=0.0, value=3.5, step=0.1)
nitrogen = st.sidebar.number_input("Nitrogen - N (kg/ha)", min_value=0.0, value=50.0, step=1.0)
phosphorus = st.sidebar.number_input("Phosphorus - P (kg/ha)", min_value=0.0, value=30.0, step=1.0)
potassium = st.sidebar.number_input("Potassium - K (kg/ha)", min_value=0.0, value=40.0, step=1.0)
ph = st.sidebar.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, value=1100.0, step=50.0)
temperature = st.sidebar.number_input("Temperature (C)", min_value=0.0, value=25.0, step=0.5)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
gap_trend = st.sidebar.number_input("Gap Trend", value=0.0, step=0.1)

def encode(col, val):
    return int(encoders[col].transform([val])[0]) if val in encoders[col].classes_ else 0

input_data = pd.DataFrame([{
    "State": encode("State", state),
    "District": encode("District", district),
    "Crop_Year": crop_year,
    "Season": encode("Season", season),
    "Crop": encode("Crop", crop),
    "Area": area,
    "Production": production,
    "Yield": yield_val,
    "Potential_Yield": potential_yield,
    "N": nitrogen,
    "P": phosphorus,
    "K": potassium,
    "pH": ph,
    "rainfall": rainfall,
    "temperature": temperature,
    "humidity": humidity,
    "Gap_Trend": gap_trend
}])

prediction = model.predict(input_data)[0]

col1, col2, col3 = st.columns(3)
col1.metric("Predicted Yield Gap", f"{prediction:.3f} tonnes/ha")
col2.metric("Current Yield", f"{yield_val:.2f} tonnes/ha")
col3.metric("Potential Yield", f"{potential_yield:.2f} tonnes/ha")

st.markdown("---")
st.subheader("Feature Impact on This Prediction")
explainer = shap.Explainer(model)
shap_values = explainer(input_data)

fig, ax = plt.subplots(figsize=(10, 5))
shap.plots.waterfall(shap_values[0], show=False)
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")
st.subheader("Dataset Overview")
st.dataframe(df.head(20))