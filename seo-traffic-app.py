import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from prophet import Prophet
import io

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
        
        # Attempt to automatically detect date and traffic columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        if not date_cols or not numeric_cols:
            st.sidebar.header("Column Selection")
            date_col = st.sidebar.selectbox("Select Date Column", df.columns)
            traffic_col = st.sidebar.selectbox("Select Traffic Column", df.columns)
            
            df['ds'] = pd.to_datetime(df[date_col])
            df['y'] = df[traffic_col]
        else:
            df['ds'] = df[date_cols[0]]
            df['y'] = df[numeric_cols[0]]
        
        return df[['ds', 'y']].sort_values('ds')
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

def forecast_traffic(df, forecast_periods=12):
    model = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=False, 
        daily_seasonality=False
    )
    model.fit(df)
    
    future = model.make_future_dataframe(periods=forecast_periods, freq='M')
    forecast = model.predict(future)
    return model, forecast

def plot_forecast(forecast):
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['y'], 
        mode='lines', 
        name='Historical Traffic'
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'][-12:], 
        y=forecast['yhat'][-12:], 
        mode='lines', 
        name='Forecast',
        line=dict(color='red', dash='dot')
    ))
    
    # Confidence Interval
    fig.add_trace(go.Scatter(
        x=forecast['ds'][-12:],
        y=forecast['yhat_upper'][-12:],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'][-12:],
        y=forecast['yhat_lower'][-12:],
        mode='lines',
        name='Lower Bound',
        fill='tonexty',
        line=dict(width=0),
        showlegend=True
    ))
    
    fig.update_layout(
        title='SEO Traffic Forecast',
        xaxis_title='Date',
        yaxis_title='Traffic'
    )
    
    return fig

def main():
    st.set_page_config(page_title='SEO Traffic Forecast', layout='wide')
    st.title('🚀 SEO Traffic Forecasting Tool')
    
    uploaded_file = st.file_uploader("Upload CSV/Excel file", type=['csv', 'xls', 'xlsx'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.subheader('Data Preview')
            st.dataframe(df)
            
            # Forecast options
            col1, col2 = st.columns(2)
            with col1:
                forecast_months = st.slider('Forecast Periods', 3, 24, 12)
            
            model, forecast = forecast_traffic(df, forecast_periods=forecast_months)
            
            st.subheader('Forecast Visualization')
            fig = plot_forecast(forecast)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader('Forecast Data')
            forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-forecast_months:]
            st.dataframe(forecast_df)
            
            # Download forecast
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')
            
            csv = convert_df(forecast_df)
            st.download_button(
                "Download Forecast",
                csv,
                "seo_traffic_forecast.csv",
                "text/csv"
            )

if __name__ == "__main__":
    main()
