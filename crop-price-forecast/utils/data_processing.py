import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

def load_and_preprocess_data(filepath, crop, market):
    """
    Load data, filter by crop/market, handle missing values and outliers.
    """
    df = pd.read_csv(filepath)
    
    # Filter
    df = df[(df['Crop'] == crop) & (df['Market'] == market)].copy()
    
    # Date conversion
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    
    # Handle missing values (Forward Fill, then Backward Fill for leading NaNs)
    df['Modal_Price'] = df['Modal_Price'].ffill().bfill()
    df.dropna(subset=['Modal_Price'], inplace=True)
    
    # Identify and handle outliers (IQR)
    Q1 = df['Modal_Price'].quantile(0.25)
    Q3 = df['Modal_Price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers instead of removing rows to maintain time continuity
    df['Modal_Price'] = np.where(df['Modal_Price'] < lower_bound, lower_bound, df['Modal_Price'])
    df['Modal_Price'] = np.where(df['Modal_Price'] > upper_bound, upper_bound, df['Modal_Price'])
    
    return df

def perform_eda(df):
    """
    Perform EDA: Seasonal Decomposition, ADF Test.
    """
    # Check for sufficient data for decomposition (needs at least 2 cycles of 'period')
    if len(df) < 2 * 30:
        decomposition = None
    else:
        try:
            decomposition = seasonal_decompose(df['Modal_Price'], model='additive', period=30)
        except ValueError:
            decomposition = None
            
    # ADF Test requires some data, handle basic edge case
    if len(df) > 5:
        adf_result = adfuller(df['Modal_Price'])
        adf_stats = {
            'ADF Statistic': adf_result[0],
            'p-value': adf_result[1],
            'Critical Values': adf_result[4]
        }
    else:
        adf_stats = {'ADF Statistic': np.nan, 'p-value': np.nan, 'Critical Values': {}}
    
    return decomposition, adf_stats

def create_features(df):
    """
    Create rolling averages and lag features.
    """
    df['Roll_Mean_7'] = df['Modal_Price'].rolling(window=7).mean()
    df['Roll_Mean_30'] = df['Modal_Price'].rolling(window=30).mean()
    
    df['Lag_1'] = df['Modal_Price'].shift(1)
    df['Lag_7'] = df['Modal_Price'].shift(7)
    
    # Drop NaNs created by rolling/shifting
    df.dropna(inplace=True)
    
    return df

def prepare_lstm_data(df, window_size=60, forecast_horizon=7):
    """
    Prepare data for LSTM model (scaling, sequences).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Modal_Price']])
    
    X, y = [], []
    for i in range(window_size, len(scaled_data) - forecast_horizon + 1):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i:i+forecast_horizon, 0])
        
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler
