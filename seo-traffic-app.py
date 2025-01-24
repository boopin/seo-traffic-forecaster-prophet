import streamlit as st
import pandas as pd
from prophet import Prophet

def forecast_seo_traffic(data):
    # Prepare data for Prophet
    df = pd.DataFrame({
        'ds': pd.to_datetime(data.iloc[:, 0]),
        'y': data.iloc[:, 1]
    })
    
    # Fit Prophet model
    model = Prophet()
    model.fit(df)
    
    # Generate future forecast
    future = model.make_future_dataframe(periods=6, freq='M')
    forecast = model.predict(future)
    
    return forecast

def main():
    st.title('SEO Traffic Forecaster')
    
    uploaded_file = st.file_uploader("Upload Traffic Data", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            # Read file based on extension
            data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            
            # Display original data
            st.subheader('Original Data')
            st.dataframe(data)
            
            # Forecast
            forecast = forecast_seo_traffic(data)
            
            # Display forecast results
            st.subheader('Traffic Forecast')
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-6:])
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
