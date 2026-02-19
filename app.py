import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_processing import load_and_preprocess_data, perform_eda, create_features, prepare_lstm_data
from utils.models import train_arima, train_lstm, evaluate_model, predict_lstm
from sklearn.preprocessing import MinMaxScaler
import datetime

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Crop Price Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING ---
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #F4F6F8;
    }
    
    /* Metrics Cards */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #E1E8ED;
        text-align: center;
    }
    div[data-testid="metric-container"] label {
        color: #616E7C; /* Muted text */
        font-size: 0.9rem;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #1F2933;
        font-weight: 700;
        font-size: 1.5rem;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #1F2933;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E1E8ED;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2F6F4E; /* Deep Forest Green */
        color: white;
        border-radius: 4px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #26593f;
    }
    
    /* Custom Card Style for other elements */
    .card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.title("Crop Price Prediction System")
st.markdown("Institutional-grade forecasting for agricultural commodities.")
st.markdown("---")

# --- CONFIGURATION SECTION ---
with st.container():
    st.markdown("### ⚙️ Dashboard Configuration")
    
    data_path = 'data/mandi_data.csv'
    
    try:
        raw_df = pd.read_csv(data_path)
        crops = sorted(raw_df['Crop'].unique())
        markets = sorted(raw_df['Market'].unique())
        states = sorted(raw_df['State'].unique())
        
        # Row 1: Filters
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        
        with col_f1:
            selected_state = st.selectbox("Select State", states)
        
        with col_f2:
            # Filter markets by state
            state_markets = sorted(raw_df[raw_df['State'] == selected_state]['Market'].unique())
            selected_market = st.selectbox("Select Market", state_markets)
            
        with col_f3:
            selected_crop = st.selectbox("Select Crop", crops)
            
        with col_f4:
            raw_df['Date'] = pd.to_datetime(raw_df['Date'])
            min_date = raw_df['Date'].min().date()
            max_date = raw_df['Date'].max().date()
            
            date_range = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

        # Row 2: Model Settings & Action
        col_m1, col_m2, col_m3 = st.columns([2, 1, 1])
        
        with col_m1:
            model_selection = st.radio(
                "Prediction Model",
                ["ARIMA", "LSTM", "Compare Both"],
                horizontal=True
            )
            
        with col_m2:
            forecast_horizon = st.slider("Forecast Horizon (Days)", 1, 14, 7)
            
        with col_m3:
            st.write("##") # Spacer to align button
            run_btn = st.button("Generate Forecast", use_container_width=True)
            
    except FileNotFoundError:
        st.error("Data source not found.")
        st.stop()
    
st.markdown("---")

# --- MAIN LOGIC ---
if run_btn or st.session_state.get('run_analysis', False):
    st.session_state['run_analysis'] = True
    
    # Filter Data
    df = load_and_preprocess_data(data_path, selected_crop, selected_market)
    
    # Filter by Date
    if isinstance(date_range, tuple) and len(date_range) == 2:
        mask = (df.index.date >= date_range[0]) & (df.index.date <= date_range[1])
        df_display = df.loc[mask]
    else:
        df_display = df
        
    if df_display.empty:
        st.warning("No data available for the selected range.")
        st.stop()

    # --- 1. KPI CARDS ---
    current_price = df_display['Modal_Price'].iloc[-1]
    avg_price_30 = df_display['Modal_Price'].rolling(30).mean().iloc[-1]
    volatility = df_display['Modal_Price'].std()
    total_records = len(df_display)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"₹{current_price:,.2f}")
    col2.metric("30-Day Average", f"₹{avg_price_30:,.2f}")
    col3.metric("Volatility (Std Dev)", f"₹{volatility:.2f}")
    col4.metric("Total Records", total_records)
    
    st.markdown("---")
    
    # --- 2. HISTORICAL PRICE TREND ---
    st.subheader("Historical Price Trend")
    with st.container():
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_display.index, df_display['Modal_Price'], label='Actual Price', color='#5C6BC0', linewidth=2)
        
        # Calculate rolling average for display
        rolling_mean = df_display['Modal_Price'].rolling(window=30).mean()
        ax.plot(df_display.index, rolling_mean, label='30-Day Rolling Avg', color='#2F6F4E', linestyle='--', linewidth=1.5)
        
        ax.set_ylabel("Price (₹/Quintal)")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        
    # --- 3. SEASONAL ANALYSIS ---
    st.subheader("Seasonal Analysis")
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown("**Monthly Average Prices**")
        monthly_avg = df_display.groupby(df_display.index.month)['Modal_Price'].mean()
        fig_m, ax_m = plt.subplots(figsize=(6, 4))
        monthly_avg.plot(kind='bar', ax=ax_m, color='#5C6BC0')
        ax_m.set_xlabel("Month")
        ax_m.set_ylabel("Avg Price")
        ax_m.grid(axis='y', linestyle=':', alpha=0.5)
        st.pyplot(fig_m)
        
    with col_s2:
        st.markdown("**Seasonal Decomposition**")
        decomposition, _ = perform_eda(df_display)
        if decomposition is not None:
            fig_d = decomposition.plot()
            fig_d.set_size_inches(6, 4)
            st.pyplot(fig_d)
        else:
            st.warning("Insufficient data for seasonal decomposition (requires 60+ days).")
            
    if 'adf_stats' in locals() and 'p-value' in adf_stats and not np.isnan(adf_stats['p-value']):
        st.info(f"Stationarity Test (ADF): p-value = {adf_stats['p-value']:.4f}")
    
    # --- 4. MODEL PERFORMANCE & FORECAST ---
    st.subheader("Model Performance & Forecast")
    
    with st.spinner("Training models and generating forecast..."):
        # Train/Predict logic reused from previous implementation
        # Split
        train_size = int(len(df) * 0.8)
        train_data = df['Modal_Price'][:train_size]
        test_data = df['Modal_Price'][train_size:]
        
        # Result containers
        arima_res = None
        lstm_res = None
        
        # --- ARIMA ---
        if model_selection in ["ARIMA", "Compare Both"]:
            # Train on Train set for Metrics
            arima_model_test, arima_pred_test = train_arima(train_data, forecast_horizon=len(test_data))
            arima_metrics = evaluate_model(test_data, arima_pred_test)
            
            # Train on Full data for Future Forecast
            full_model_arima, arima_future = train_arima(df['Modal_Price'], forecast_horizon=forecast_horizon)
            
            arima_res = {
                "metrics": arima_metrics,
                "test_pred": arima_pred_test,
                "future_pred": arima_future
            }
            
        # --- LSTM ---
        if model_selection in ["LSTM", "Compare Both"]:
            # Logic for LSTM
            df_lstm = create_features(df.copy())
            lstm_input_df = df[['Modal_Price']]
            # Use same window size as before
            window_size = 60
            X, y, scaler = prepare_lstm_data(lstm_input_df, window_size=window_size, forecast_horizon=1)
            
            # Split
            cutoff = int(len(X) * 0.8)
            X_train, X_test_seq = X[:cutoff], X[cutoff:]
            y_train, y_test_seq = y[:cutoff], y[cutoff:]
            
            # Train
            lstm_model = train_lstm(X_train, y_train, (X.shape[1], 1))
            
            # Evaluate on Test
            y_pred_lstm = predict_lstm(lstm_model, X_test_seq)
            y_pred_lstm_inv = scaler.inverse_transform(y_pred_lstm)
            y_test_inv = scaler.inverse_transform(y_test_seq)
            lstm_metrics = evaluate_model(y_test_inv, y_pred_lstm_inv)
            
            # Rolling Forecast for Future
            last_sequence = lstm_input_df['Modal_Price'].values[-window_size:]
            last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
            
            future_forecast_lstm = []
            current_seq = last_sequence_scaled.reshape(1, window_size, 1)
            
            for _ in range(forecast_horizon):
                next_pred = predict_lstm(lstm_model, current_seq) 
                future_forecast_lstm.append(next_pred[0, 0])
                new_val = next_pred.reshape(1, 1, 1)
                current_seq = np.append(current_seq[:, 1:, :], new_val, axis=1)
            
            lstm_future = scaler.inverse_transform(np.array(future_forecast_lstm).reshape(-1, 1)).flatten()
            
            # Align test predictions index (approximate for display)
            # The test set for LSTM is smaller by window_size
            test_indices = df.index[window_size+cutoff : window_size+cutoff+len(y_test_inv)]
            
            lstm_res = {
                "metrics": lstm_metrics,
                "test_pred": pd.Series(y_pred_lstm_inv.flatten(), index=test_indices),
                "future_pred": lstm_future,
                "actual_test": pd.Series(y_test_inv.flatten(), index=test_indices)
            }

    # --- DISPLAY RESULTS ---
    col_p1, col_p2 = st.columns([2, 1])
    
    with col_p1:
        st.markdown("**Actual vs Forecast (Test Data)**")
        fig_p, ax_p = plt.subplots(figsize=(10, 5))
        
        # Plot full actual test data from ARIMA split (it's the clean 20% holdout)
        ax_p.plot(test_data.index, test_data, label='Actual Price', color='#1F2933', linewidth=1.5, alpha=0.7)
        
        if arima_res:
            arima_series = pd.Series(arima_res['test_pred'].values, index=test_data.index)
            ax_p.plot(arima_series.index, arima_series, label='ARIMA Forecast', color='#D9822B', linestyle='-')
            
        if lstm_res:
            ax_p.plot(lstm_res['test_pred'].index, lstm_res['test_pred'], label='LSTM Forecast', color='#2A9D8F', linestyle='-')
            
        ax_p.set_ylabel("Price")
        ax_p.legend()
        ax_p.grid(True, alpha=0.3)
        st.pyplot(fig_p)

    with col_p2:
        st.markdown("**Metric Comparison**")
        metrics_data = []
        if arima_res:
            metrics_data.append(arima_res['metrics'])
        if lstm_res:
            metrics_data.append(lstm_res['metrics'])
            
        indices = []
        if arima_res: indices.append("ARIMA")
        if lstm_res: indices.append("LSTM")
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data, index=indices)
            st.table(metrics_df.style.format("{:.4f}"))
            
            if len(metrics_df) > 1:
                best_model = metrics_df['MAE'].idxmin()
                st.success(f"Best Model: {best_model}")
    
    st.markdown("---")
    
    # --- 5. FUTURE FORECAST ---
    st.subheader(f"{forecast_horizon}-Day Future Travel")
    
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
    forecast_data = {'Date': future_dates}
    
    if arima_res:
        forecast_data['ARIMA Forecast'] = np.array(arima_res['future_pred']).flatten()
    if lstm_res:
        forecast_data['LSTM Forecast'] = np.array(lstm_res['future_pred']).flatten()
        
    forecast_df = pd.DataFrame(forecast_data).set_index('Date')
    
    col_f1, col_f2 = st.columns([1, 2])
    
    with col_f1:
        st.dataframe(forecast_df.style.format("{:.2f}"))
        
        # CSV Download
        csv = forecast_df.to_csv().encode('utf-8')
        st.download_button(
            "Download Forecast CSV",
            csv,
            "forecast.csv",
            "text/csv",
            key='download-csv'
        )
        
    with col_f2:
        fig_f, ax_f = plt.subplots(figsize=(10, 4))
        if 'ARIMA Forecast' in forecast_df.columns:
            ax_f.plot(forecast_df.index, forecast_df['ARIMA Forecast'], label='ARIMA', color='#D9822B', marker='o')
        if 'LSTM Forecast' in forecast_df.columns:
            ax_f.plot(forecast_df.index, forecast_df['LSTM Forecast'], label='LSTM', color='#2A9D8F', marker='s')
            
        ax_f.set_title("Future Price Trajectory")
        ax_f.grid(True, alpha=0.3)
        ax_f.legend()
        st.pyplot(fig_f)

else:
    st.info("Select parameters and click 'Generate Forecast' to begin.")

# --- FOOTER ---
st.markdown("---")
st.markdown("© 2026 Crop Price Prediction System | Institutional Analytics")
