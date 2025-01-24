import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def load_data(uploaded_file):
    try:
        # Support multiple file types
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Ensure date and numeric columns
        df['ds'] = pd.to_datetime(df.iloc[:, 0])
        df['y'] = df.iloc[:, 1]
        
        return df[['ds', 'y']].sort_values('ds')
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return None

def forecast_traffic(df, periods=12):
    model = Prophet(yearly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='M')
    return model.predict(future)

def main():
    st.title('SEO Traffic Forecast')
    
    uploaded_file = st.file_uploader("Upload Data", type=['csv', 'xlsx'])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.dataframe(df)
            
            forecast = forecast_traffic(df)
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['ds'], df['y'], label='Historical')
            ax.plot(forecast['ds'][-12:], forecast['yhat'][-12:], 'r--', label='Forecast')
            ax.fill_between(
                forecast['ds'][-12:], 
                forecast['yhat_lower'][-12:], 
                forecast['yhat_upper'][-12:], 
                color='red', alpha=0.1
            )
            plt.title('Traffic Forecast')
            plt.legend()
            
            st.pyplot(fig)
            
            # Forecast data
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-12:])

if __name__ == "__main__":
    main()
