# 🏠 Bengaluru House Price Prediction — ANN

## Overview
My first ANN project built using PyTorch to predict house prices in Bengaluru.
Started with a random dataset where every prediction was ₹254 Lakhs 😅
then switched to real Bengaluru data and built this.

## Results
RMSE : 34.26 Lakhs
MSE  : 1173.71
R²   : 66.32%

## Tech Stack
Python, PyTorch, Pandas, Scikit-learn, Streamlit

## Project Structure
- bengaluru_house_data.csv — Dataset
- bengaluru_housing_model.ipynb — Main notebook
- app.py — Streamlit app
- model_columns.json — Feature columns
- top_locations.json — Top 40 locations

## What I Learned
- Data cleaning for real datasets
- Handling messy columns like size and total_sqft
- One-hot encoding and Label encoding
- Building and training an ANN from scratch
- What BatchNorm and Dropout actually do
- Early stopping to prevent overfitting
- Deploying a model with Streamlit

## How to Run
1. Clone the repo
2. Install dependencies — pip install torch pandas scikit-learn streamlit
3. Run the notebook first to generate model files
4. Launch the app — streamlit run app.py

> Note: housing_model.pth and scaler.pkl are not included.
> Run the notebook first to generate them automatically.

## Dataset
Bengaluru House Price Data — Kaggle
https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data

## Honest Note
This is my first project. I took AI help for some improvements
and explored Streamlit during the project. I made sure I
understood every part before moving forward. Still learning —
especially data cleaning. Will improve with every project.

## Author
Vishnu
Hyderabad, India
