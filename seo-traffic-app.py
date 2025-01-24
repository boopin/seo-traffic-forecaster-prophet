import streamlit as st
import pandas as pd
from prophet import Prophet

def forecast_traffic(data):
    df = pd.DataFrame({
        'ds': pd.date_range(start=data.index[0], periods=len(data), freq='M'),
        'y': data.values
    })
    
    model = Prophet()
    model.fit(df)
    
    future = model.make_future_dataframe(periods=6, freq='M')
    forecast = model.predict(future)
    
    return forecast

def main():
    st.title('SEO Traffic Forecast')
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
            
            st.write("Original Data:")
            st.dataframe(data)
            
            forecast = forecast_traffic(data)
            
            st.write("Forecast:")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-6:])
        
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
