import streamlit as st
import pandas as pd
import subprocess
import sys

# Check if joblib is installed, if not, install it
try:
    import joblib
except ImportError:
    st.warning("Joblib not found. Attempting to install...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
    import joblib # Import after installation

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['heavy_ship'] = (X['weight_kg'] > 50000).astype(int)
        X['expensive_freight_per_kg'] = X['shipping_cost_usd'] / (X['weight_kg'] + 1)
        return X

@st.cache_resource
def load_model():
    return joblib.load('shipment_delay_pipeline.pkl')

pipeline = load_model()

st.set_page_config(page_title="Shipment Delay Predictor", layout="centered")
st.title("📦 China–Africa Shipment Delay Predictor")
st.markdown("Enter shipment details to get a **delay probability** and **actionable recommendation**.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        transit_days = st.number_input("Transit time (days)", min_value=1, max_value=60, value=14)
        weight = st.number_input("Weight (kg)", min_value=0.0, value=5000.0)
        shipping_cost = st.number_input("Shipping cost (USD)", min_value=0.0, value=5000.0)
        declared_value = st.number_input("Declared value (USD)", min_value=0.0, value=50000.0)
    with col2:
        contract_value = st.number_input("Contract value (USD)", min_value=0.0, value=50000.0)
        exchange_rate = st.number_input("Exchange rate", min_value=0.0, value=50.0)
        volatility = st.number_input("Volatility index", min_value=0.0, max_value=1.0, value=0.2)
        anomaly = st.selectbox("Anomaly flag", [0,1], format_func=lambda x: "Yes" if x else "No")
    transport_mode = st.selectbox("Transport mode", ["Air","Rail","Road","Ship"])
    import_country = st.selectbox("Import country", ["Nigeria","South Africa","Kenya","Egypt","Ethiopia","Ghana","Tanzania"])
    commodity = st.selectbox("Commodity", ["Electronics","Machinery","Agricultural","Textiles","Minerals"])
    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        'transit_time_days': transit_days,
        'weight_kg': weight,
        'shipping_cost_usd': shipping_cost,
        'declared_value_usd': declared_value,
        'contract_value_usd': contract_value,
        'currency_exchange_rate': exchange_rate,
        'price_volatility_index': volatility,
        'anomaly_flag': anomaly,
        'transport_mode': transport_mode,
        'import_country': import_country,
        'commodity': commodity
    }])
    prob = pipeline.predict_proba(input_df)[0,1]
    pred = pipeline.predict(input_df)[0]
    st.subheader("Result")
    if pred == 1:
        st.error(f"⚠️ DELAYED with {prob:.1%} probability")
    else:
        st.success(f"✅ ON TIME with {(1-prob):.1%} probability")
    if prob > 0.7:
        st.warning("🔴 HIGH RISK: Switch to air freight, add buffer days.")
    elif prob > 0.4:
        st.info("🟡 MODERATE RISK: Complete customs docs early, maintain safety stock.")
    else:
        st.success("🟢 LOW RISK: Standard monitoring sufficient.")
    st.caption("Based on historical China–Africa trade patterns.")

    # Resources section
    st.subheader("Resources")
    st.markdown("Additional information and links will be provided here.")

