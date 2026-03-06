
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import json

# Load saved files
with open("scaler.pkl", "rb") as f:
    scale = pickle.load(f)
with open("model_columns.json", "r") as f:
    columns = json.load(f)
with open("top_locations.json", "r") as f:
    top_locations = json.load(f)

# Rebuild model
input_size = len(columns)
model = nn.Sequential(
    nn.Linear(input_size, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

model.load_state_dict(torch.load("housing_model.pth", map_location="cpu"))
model.eval()

def predict_price(location, area_type, availability, total_sqft, bath, balcony, bhk):
    input_df = pd.DataFrame([{"total_sqft": total_sqft, "bath": bath, "balcony": balcony, "bhk": bhk}])
    loc_col   = f"location_{location}" if f"location_{location}" in columns else "location_Other"
    area_col  = f"area_type_{area_type}"
    avail_col = f"availability_{availability}"
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0
    if loc_col in columns:   input_df[loc_col]   = 1
    if area_col in columns:  input_df[area_col]  = 1
    if avail_col in columns: input_df[avail_col] = 1
    input_df     = input_df[columns]
    input_scaled = scale.transform(input_df)
    tensor       = torch.from_numpy(input_scaled).float()
    with torch.no_grad():
        result = model(tensor).item()
    return round(result, 2)

# Streamlit UI
st.title("Bengaluru House Price Predictor")
st.markdown("Fill in the details to get a price prediction.")

col1, col2 = st.columns(2)

with col1:
    location     = st.selectbox("Location", sorted(top_locations) + ["Other"])
    area_type    = st.selectbox("Area Type", ["Super built-up Area", "Built-up Area", "Plot Area", "Carpet Area"])
    availability = st.selectbox("Availability", ["Ready To Move", "Immediate Possession", "6 months", "1 Year"])
    bhk          = st.slider("BHK", 0, 10, 0)

with col2:
    total_sqft = st.number_input("Total SqFt", min_value=200, max_value=10000, value=1200)
    bath       = st.slider("Bathrooms", 0, 10, 0)
    balcony    = st.slider("Balconies", 0, 10, 0)

if st.button("💰 Predict Price"):
    price = predict_price(location, area_type, availability, total_sqft, bath, balcony, bhk)
    st.success(f"Predicted Price: ₹ {price} Lakhs")
    st.info(f"Approx: ₹ {round(price/total_sqft*100000):,} per sq.ft")
