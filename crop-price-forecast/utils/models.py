import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import pickle

# Try importing tensorflow, otherwise handle gracefully
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Using sklearn MLPRegressor as fallback for Deep Learning model.")

def train_arima(train_data, forecast_horizon=7):
    """
    Train ARIMA model using auto_arima to find best parameters.
    """
    # Auto-ARIMA to find best (p,d,q)
    # Using simpler parameters to speed up if pmdarima is slow
    model = pm.auto_arima(train_data, 
                          seasonal=False, 
                          stepwise=True, 
                          suppress_warnings=True, 
                          error_action="ignore", 
                          max_p=5, max_q=5, max_d=2, 
                          trace=False)
    
    # Forecast
    forecast = model.predict(n_periods=forecast_horizon)
    return model, forecast

def build_lstm_model(input_shape):
    """
    Build LSTM model architecture.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available")
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm(X_train, y_train, input_shape):
    """
    Train LSTM model or Fallback MLP.
    """
    if TF_AVAILABLE:
        model = build_lstm_model(input_shape)
        early_stop = EarlyStopping(monitor='loss', patience=5, verbose=0)
        
        model.fit(X_train, y_train, 
                epochs=50, 
                batch_size=32, 
                callbacks=[early_stop], 
                verbose=0)
        return model
    else:
        # Flatten input for MLP (samples, timesteps, features) -> (samples, timesteps*features)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        # y_train is (samples, 1) usually, MLP expects (samples,)
        y_train_flat = y_train.ravel()
        
        # MLPRegressor as a simple Neural Network fallback
        # Hidden layer sizes (50, 50) approximates some depth
        model = MLPRegressor(hidden_layer_sizes=(50, 50), 
                             activation='relu', 
                             solver='adam', 
                             max_iter=500, 
                             random_state=42)
        model.fit(X_train_flat, y_train_flat)
        return model

def predict_lstm(model, X, scaler=None):
    """
    Wrapper for prediction to handle different model types.
    """
    if TF_AVAILABLE:
        pred = model.predict(X)
    else:
        X_flat = X.reshape(X.shape[0], -1)
        pred = model.predict(X_flat)
        pred = pred.reshape(-1, 1) # Ensure shape matches
        
    return pred

def evaluate_model(y_true, y_pred):
    """
    Calculate MAE, RMSE, MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
