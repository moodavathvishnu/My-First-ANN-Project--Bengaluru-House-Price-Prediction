# 🏠 Bengaluru House Price Prediction — ANN

## Overview
My first ANN project built using PyTorch to predict house prices in Bengaluru.
Started with a random dataset where every prediction was ₹254 Lakhs 😅 
then switched to real Bengaluru data and built this.

## Results
| Metric | Score |
|--------|-------|
| RMSE | 34.26 Lakhs |
| MSE | 1173.71 |
| R² | 66.32% |

## Tech Stack
| Tool | Use |
|------|-----|
| Python | Core language |
| PyTorch | Building and training ANN |
| Pandas | Data cleaning and processing |
| Scikit-learn | Encoding and scaling |
| Streamlit | Model deployment |

## Project Structure
```
├── bengaluru_house_data.csv           # Dataset
├── bengaluru_housing_model.ipynb      # Main notebook
├── app.py                             # Streamlit app
├── housing_model.pth                  # Saved model weights
├── scaler.pkl                         # Saved scaler
├── model_columns.json                 # Feature columns
├── top_locations.json                 # Top 40 locations
└── README.md                          # This file
```

## What I Learned
- Data cleaning for real datasets
- Handling messy columns like size and total_sqft
- One-hot encoding and Label encoding
- Building and training an ANN from scratch
- What BatchNorm and Dropout actually do
- Early stopping to prevent overfitting
- Deploying a model with Streamlit

## How to Run

### 1. Clone the repo
```
git clone your_repo_link
cd your_repo_folder
```

### 2. Install dependencies
```
pip install torch pandas scikit-learn streamlit
```

### 3. Run the notebook
Open `bengaluru_housing_model.ipynb` and run all cells.
This will generate the model files needed for the app.

### 4. Launch the app
```
streamlit run app.py
```

## Dataset
Bengaluru House Price Data — Kaggle  
Link: https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data

## Honest Note
This is my first project. I took AI help for some improvements 
and explored Streamlit during the project. I made sure I 
understood every part before moving forward. Still learning — 
especially data cleaning. Will improve with every project.

## Author
Vishnu
Hyderabad, India
