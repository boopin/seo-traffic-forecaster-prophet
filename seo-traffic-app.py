import streamlit as st
import pandas as pd
from prophet import Prophet

def forecast_traffic(data):
    # Convert month-year format to datetime
    data.index = pd.to_datetime(data.index, format='%b-%y')
    
    # Reset index and create Prophet-compatible dataframe
    df = data.reset_index()
    df.columns = ['ds', 'y']
    
    model = Prophet()
    model.fit(df)
    
    future = model.make_future_dataframe(periods=6, freq='M')
    return model.predict(future)

def main():
    st.title('SEO Traffic Forecast')
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file:
        try:
            # Read CSV with first column as index
            data = pd.read_csv(uploaded_file, index_col=0)
            
            st.write("Original Data:")
            st.dataframe(data)
            
            forecast = forecast_traffic(data)
            
            st.write("Forecast:")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-6:])
        
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
