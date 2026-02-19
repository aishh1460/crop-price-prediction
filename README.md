# Crop Price Prediction (Mandi Data)

## Overview
This project predicts the next 7 days of crop prices using historical Mandi data from India. It compares Classical Statistical Models (ARIMA) with Deep Learning approaches (LSTM/MLP).

## Dataset
The project uses Mandi data containing:
- Date
- Crop
- State
- Market
- Modal_Price

*Note: A synthetic dataset generator is included in `data/generate_data.py` for demonstration purposes.*

## Methodology
1. **Data Processing**:
   - Forward fill for missing values.
   - IQR method for outlier handling.
   - Feature Engineering: Rolling means (7, 30 days) and Lags.
2. **Models**:
   - **ARIMA**: Auto-ARIMA to select optimal (p,d,q).
   - **Deep Learning**: LSTM (if TensorFlow available) or MLP Regressor (fallback).
3. **Evaluation**:
   - MAE, RMSE, MAPE.

## Project Structure
```
crop-price-forecast/
├── data/
│   └── generate_data.py  # Script to generate synthetic data
├── models/               # Saved models
├── src/
│   ├── data_processing.py
│   └── models.py
├── app.py                # Streamlit Dashboard
├── train.py              # Model training script
├── requirements.txt
└── README.md
```

## How to Run
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: TensorFlow is optional. The code will fallback to Scikit-Learn if not present.)*

2. **Generate Data**:
   ```bash
   python data/generate_data.py
   ```

3. **Train Models**:
   ```bash
   python train.py
   ```

4. **Run Dashboard**:
   ```bash
   streamlit run app.py
   ```

## Results
The Streamlit dashboard allows you to select a Crop and Market to view:
- Exploratory Data Analysis (Seasonal Decomposition, Stationarity).
- Model Comparison metrics.
- 7-Day Future Forecasts.
