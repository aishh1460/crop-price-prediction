import pandas as pd
import numpy as np
import pickle
import os
from utils.data_processing import load_and_preprocess_data, create_features, prepare_lstm_data
from utils.models import train_arima, train_lstm, evaluate_model

def main():
    # Setup
    data_path = 'data/mandi_data.csv'
    if not os.path.exists(data_path):
        print("Data not found. Please run data/generate_data.py first.")
        return

    # Load parameters (example for one crop/market)
    # In a real scenario, you might loop over all combinations or take args
    crop = 'Wheat'
    market = 'Azadpur'
    
    print(f"Training for {crop} in {market}...")
    
    # 1. Pipeline
    df = load_and_preprocess_data(data_path, crop, market)
    
    # 2. ARIMA
    print("Training ARIMA...")
    train_size = int(len(df) * 0.8)
    train_data = df['Modal_Price'][:train_size]
    test_data = df['Modal_Price'][train_size:]
    
    arima_model, arima_forecast = train_arima(train_data, forecast_horizon=len(test_data))
    
    # 3. LSTM
    print("Training LSTM...")
    df_lstm = create_features(df.copy())
    lstm_input_df = df[['Modal_Price']]
    X, y, scaler = prepare_lstm_data(lstm_input_df, forecast_horizon=1)
    
    train_split = int(len(X) * 0.8)
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]
    
    lstm_model = train_lstm(X_train, y_train, (X.shape[1], 1))
    
    # 4. Save Models
    if not os.path.exists('models'):
        os.makedirs('models')
        
    with open('models/arima_model.pkl', 'wb') as f:
        pickle.dump(arima_model, f)
    
    if hasattr(lstm_model, 'save'):
        lstm_model.save('models/lstm_model.keras')
    else:
        with open('models/lstm_model.pkl', 'wb') as f:
            pickle.dump(lstm_model, f)
    
    print("Models saved in models/")

if __name__ == "__main__":
    main()
