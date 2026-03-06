# My-First-ANN-Project--Bengaluru-House-Price-Prediction
# Bengaluru House Price Prediction (PyTorch ANN)

This project builds an Artificial Neural Network (ANN) using PyTorch to predict house prices in Bengaluru based on property features such as location, square footage, number of bedrooms, and bathrooms.

The project covers the complete machine learning workflow including data preprocessing, model training, evaluation, and deployment using Streamlit.

---

## Tech Stack

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-learn
* Streamlit

---

## Model Performance

The trained model was evaluated using common regression metrics:

* **RMSE:** 37.82
* **MAE:** 24.67
* **R² Score:** 58.95%

---

## Project Structure

```
bengaluru-house-price-ann/

app.py
bengaluru_housing_model.ipynb

housing_model.pth
scaler.pkl
model_columns.json
top_locations.json

requirements.txt
README.md
```

---

## File Description

**app.py**
Streamlit application that provides a user interface to input property details and receive predicted house prices.

**bengaluru_housing_model.ipynb**
Contains the full machine learning workflow including data preprocessing, feature engineering, model training, and evaluation.

**housing_model.pth**
Saved PyTorch model weights used for inference.

**scaler.pkl**
Saved feature scaler used during training to ensure consistent preprocessing during prediction.

**model_columns.json**
Stores the feature column structure used during training.

**top_locations.json**
Contains the most common locations used in the dataset for encoding and UI dropdown selection.

---

## How to Run the Project

### Clone the repository

```
git clone https://github.com/yourusername/bengaluru-house-price-ann.git
cd bengaluru-house-price-ann
```

### Install dependencies

```
pip install -r requirements.txt
```

### Run the Streamlit application

```
streamlit run app.py
```

The app will open in your browser where you can enter property details and receive predicted prices.

---

## Features Used for Prediction

* Location
* Total Square Footage
* Number of Bedrooms (BHK)
* Number of Bathrooms

---

## What I Learned

* Building neural networks using PyTorch
* Data cleaning and preprocessing for real-world datasets
* Feature encoding and scaling
* Preventing overfitting using Dropout and Early Stopping
* Deploying machine learning models with Streamlit

---

## Future Improvements

* Feature engineering for improved performance
* Comparing ANN with models like Random Forest or XGBoost
* Adding visualization for training diagnostics
* Hyperparameter tuning

---

## Author

Machine Learning beginner project focused on understanding neural networks, model training, and deployment.
