import streamlit as st
import pandas as pd
from prophet import Prophet

def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df['ds'] = pd.to_datetime(df.iloc[:, 0])
        df['y'] = df.iloc[:, 1]
        return df[['ds', 'y']].sort_values('ds')
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def main():
    st.title('SEO Traffic Forecast')
    
    uploaded_file = st.file_uploader("Upload Data", type=['csv', 'xlsx'])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        
        if df is not None:
            model = Prophet()
            model.fit(df)
            
            future = model.make_future_dataframe(periods=6, freq='M')
            forecast = model.predict(future)
            
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-6:])

if __name__ == "__main__":
    main()
