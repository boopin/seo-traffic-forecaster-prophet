import streamlit as st
import pandas as pd
from prophet import Prophet
from io import BytesIO
import plotly.graph_objects as go

def forecast_traffic(data, forecast_period, confidence_interval):
    df = pd.DataFrame({
        'ds': pd.to_datetime(data.index, format='%b-%y'),
        'y': data.values.flatten()
    })
    
    model = Prophet(interval_width=confidence_interval/100)
    model.fit(df)
    
    future = model.make_future_dataframe(periods=forecast_period, freq='M')
    forecast = model.predict(future)
    
    forecast[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].round(0)
    return forecast, model

def plot_forecast(model, forecast):
    fig = go.Figure()
    
    # Add filled area between Conservative and Best Case
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        fill=None,
        mode='lines',
        line_color='rgba(0,100,255,0)',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,100,255,0)',
        fillcolor='rgba(0,100,255,0.1)',
        name='Prediction Range',
        hoverinfo='skip'
    ))
    
    # Add main forecast line
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name='Expected Traffic',
        line=dict(color='rgb(0,100,255)', width=3),
        hovertemplate='<b>Date</b>: %{x|%B %Y}<br>' +
                      '<b>Expected Traffic</b>: %{y:,.0f}<br><extra></extra>'
    ))
    
    # Add upper and lower bounds with custom hover
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        name='Best Case Scenario',
        line=dict(color='rgba(0,100,255,0.5)', dash='dash'),
        hovertemplate='<b>Date</b>: %{x|%B %Y}<br>' +
                      '<b>Best Case</b>: %{y:,.0f}<br><extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        name='Conservative Estimate',
        line=dict(color='rgba(0,100,255,0.5)', dash='dash'),
        hovertemplate='<b>Date</b>: %{x|%B %Y}<br>' +
                      '<b>Conservative</b>: %{y:,.0f}<br><extra></extra>'
    ))
    
    # Update layout with better styling
    fig.update_layout(
        title={
            'text': 'Traffic Forecast',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis_title='Date',
        yaxis_title='Traffic',
        hovermode='x unified',
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    return fig

def main():
    st.set_page_config(page_title="ForecastEdge", layout="wide")
    
    st.markdown("""
    <div style='background-color:#f0f2f6;padding:20px;border-radius:10px;margin-bottom:20px;'>
        <h1 style='text-align:center;color:#2c3e50;'>🚀 ForecastEdge</h1>
        <p style='text-align:center;color:#34495e;font-size:1.2em;margin-top:10px;'>
            Advanced SEO Traffic Forecasting Tool powered by Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    menu = st.sidebar.radio("Menu", ["Forecast", "Documentation"])
    
    if menu == "Forecast":
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file:
            try:
                # Read and process the uploaded data
                data = pd.read_csv(uploaded_file, index_col=0)
                
                # Transform the data for horizontal display
                displayed_data = data.T  # Transpose the dataframe
                st.write("Historical Traffic Data")
                st.dataframe(displayed_data, height=150)  # Set fixed height for better display
                
                col1, col2 = st.columns(2)
                with col1:
                    forecast_period = st.radio("Forecast Period (Months)", [6, 12])
                with col2:
                    confidence_interval = st.slider("Prediction Accuracy (%)", 50, 99, 80)
                
                forecast, model = forecast_traffic(data, forecast_period, confidence_interval)
                
                st.subheader("Forecast Results")
                results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-forecast_period:]
                
                # Format the date without time
                results['ds'] = pd.to_datetime(results['ds']).dt.strftime('%Y-%m')
                results.columns = ['Date', 'Expected Traffic', 'Conservative Estimate', 'Best Case Scenario']
                
                # Display results in a clean format
                st.dataframe(results.set_index('Date'))
                
                st.plotly_chart(plot_forecast(model, forecast), use_container_width=True)
                
                # Update the CSV export to use the cleaned date format
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button("Download Forecast", csv, "forecast.csv", "text/csv")
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    elif menu == "Documentation":
        st.markdown("""
        ## How to Use
        1. Upload CSV with months and traffic
        2. Select forecast period
        3. Adjust prediction accuracy
        4. Download results
        """)

if __name__ == "__main__":
    main()
