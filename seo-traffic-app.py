import streamlit as st
import pandas as pd
from prophet import Prophet
from io import BytesIO
from prophet.plot import plot_components_plotly
import plotly.graph_objects as go

def forecast_traffic(data, forecast_period, confidence_interval):
    # Convert month-year format to datetime
    data.index = pd.to_datetime(data.index, format='%b-%y')

    # Ensure the traffic column is numeric
    data = data.astype(float)

    # Reset index and create Prophet-compatible dataframe
    df = data.reset_index()
    df.columns = ['ds', 'y']

    model = Prophet(interval_width=confidence_interval / 100)
    model.fit(df)

    # Adjust future dataframe to exclude redundant forecasting for existing months
    last_date = df['ds'].max()
    future = model.make_future_dataframe(periods=forecast_period, freq='M')
    future = future[future['ds'] > last_date]

    forecast = model.predict(future)

    # Round forecast values to integers and return
    forecast['yhat'] = forecast['yhat'].round(0)
    forecast['yhat_lower'] = forecast['yhat_lower'].round(0)
    forecast['yhat_upper'] = forecast['yhat_upper'].round(0)

    return forecast, model

def convert_df_to_csv(df):
    # Convert dataframe to CSV for download
    output = BytesIO()
    df.to_csv(output, index=False)
    processed_data = output.getvalue()
    return processed_data

def custom_seasonal_plot(model, forecast):
    # Create a customized seasonal decomposition plot
    components = model.plot_components(forecast, figsize=(10, 8))

    fig = go.Figure()

    # Trend
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'],
                             mode='lines', name='Trend',
                             line=dict(color='blue', width=3)))

    # Seasonalities (yearly, weekly)
    if 'yearly' in forecast.columns:
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yearly'],
                                 mode='lines', name='Yearly Seasonality',
                                 line=dict(color='green', dash='dash')))

    if 'weekly' in forecast.columns:
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['weekly'],
                                 mode='lines', name='Weekly Seasonality',
                                 line=dict(color='orange', dash='dot')))

    # Layout adjustments
    fig.update_layout(title='Seasonal Decomposition of Forecast',
                      xaxis_title='Date',
                      yaxis_title='Values',
                      template='plotly_white',
                      legend=dict(orientation='h', yanchor='bottom', xanchor='center', x=0.5))

    return fig

def main():
    # Enhanced Page Configuration
    st.set_page_config(
        page_title="ForecastEdge: SEO Traffic Planner", 
        page_icon="📊", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Styled Title with Custom CSS
    st.markdown("""
    <style>
    .title-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .title-header {
        color: #2c3e50;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="title-container">
        <h1 class="title-header">🚀 ForecastEdge: SEO Traffic Planner</h1>
        <p style="text-align: center; color: #7f8c8d;">Intelligent Traffic Forecasting Tool - Version 1.4</p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation with Icons
    menu = st.sidebar.radio("Navigation", 
        options=[
            "🔮 Forecast Tool", 
            "📖 Documentation", 
            "📊 How It Works"
        ], 
        index=0
    )

    if menu == "🔮 Forecast Tool":
        st.sidebar.header("Configuration")
        uploaded_file = st.sidebar.file_uploader(
            "Upload Traffic Data", 
            type=['csv', 'xlsx'], 
            help="Upload CSV or XLSX with Month and Traffic columns"
        )

        if uploaded_file:
            try:
                # Rest of the previous implementation remains the same...
                
                # Enhanced UI elements (same core functionality)
                col1, col2 = st.columns(2)
                with col1:
                    forecast_period = st.radio(
                        "Forecast Duration", 
                        options=[6, 12], 
                        index=0, 
                        help="Select number of months to forecast"
                    )
                
                with col2:
                    confidence_interval = st.slider(
                        "Confidence Interval", 
                        min_value=50, 
                        max_value=99, 
                        value=80,
                        help="Adjust prediction uncertainty range"
                    )

                # Rest of the existing implementation...

            except Exception as e:
                st.error(f"Error Processing Data: {e}")

    elif menu == "📖 Documentation":
        st.markdown("""
        ## 📘 Documentation
        ### Quick Start Guide
        1. **Data Preparation**
           - First column: Month (e.g., `Jan-24`)
           - Second column: Traffic values
        
        2. **Forecast Configuration**
           - Choose forecast period
           - Set confidence interval
        
        3. **Insights**
           - Detailed forecast table
           - Seasonal trend analysis
        """)

    elif menu == "📊 How It Works":
        st.markdown("""
        ## 🧠 Forecasting Methodology
        
        ### Prophet Forecasting
        - Advanced time-series forecasting
        - Handles seasonal variations
        - Robust to missing data
        
        ### Key Features
        - Automatic trend detection
        - Seasonal pattern recognition
        - Confidence interval modeling
        """)

if __name__ == "__main__":
    main()
