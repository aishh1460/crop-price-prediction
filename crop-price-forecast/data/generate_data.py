import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_data():
    # Parameters
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    
    crops = ['Wheat', 'Rice', 'Onion', 'Tomato', 'Potato']
    markets = ['Azadpur', 'Pune', 'Nasik', 'Indore', 'Jaipur']
    states = ['Delhi', 'Maharashtra', 'Maharashtra', 'Madhya Pradesh', 'Rajasthan']
    
    data = []
    
    for crop in crops:
        for i, market in enumerate(markets):
            state = states[i]
            base_price = np.random.uniform(1500, 5000)
            
            for date in date_list:
                # Add seasonality and trend
                time_idx = (date - start_date).days
                seasonality = 200 * np.sin(2 * np.pi * time_idx / 365)
                trend = 0.5 * time_idx
                noise = np.random.normal(0, 50)
                
                modal_price = base_price + seasonality + trend + noise
                # Ensure price doesn't go negative
                modal_price = max(500, modal_price)
                
                # Introduce some missing values (randomly drop 5% of data)
                if np.random.random() > 0.95:
                    modal_price = np.nan
                
                data.append([date, state, market, crop, modal_price])
                
    df = pd.DataFrame(data, columns=['Date', 'State', 'Market', 'Crop', 'Modal_Price'])
    
    # Sort chronologically
    df.sort_values(by=['Crop', 'Market', 'Date'], inplace=True)
    
    # Save to CSV
    df.to_csv('data/mandi_data.csv', index=False)
    print("Synthetic data generated at data/mandi_data.csv")

if __name__ == "__main__":
    generate_data()
