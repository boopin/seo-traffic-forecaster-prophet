import streamlit as st
import pandas as pd
from prophet import Prophet
from io import BytesIO
from prophet.plot import plot_components_plotly
import plotly.graph_objects as go
import numpy as np

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

def detect_anomalies(data):
    # Detect anomalies using z-score
    data['z_score'] = (data['y'] - data['y'].mean()) / data['y'].std()
    anomalies = data[np.abs(data['z_score']) > 2]
    return anomalies[['ds', 'y', 'z_score']]

def calculate_accuracy_metrics(actual, predicted):
    # Calculate MAPE, RMSE, and MAE
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    return mape, rmse, mae

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

def generate_insights(forecast):
    growth_rate = (forecast['yhat'].iloc[-1] - forecast['yhat'].iloc[0]) / forecast['yhat'].iloc[0] * 100
    return f"The forecast predicts a {growth_rate:.2f}% change in traffic over the selected period."

def main():
    st.set_page_config(page_title="ForecastEdge: SEO Traffic Planner", layout="wide")

    st.title('ForecastEdge: SEO Traffic Planner')
    st.subheader('Version 1.4')

    st.write("Upload your SEO organic traffic data (CSV or XLSX) containing Month and Traffic columns to forecast future traffic.")

    menu = st.sidebar.selectbox("Navigation", options=["Forecast Tool", "Documentation"])

    if menu == "Forecast Tool":
        uploaded_file = st.file_uploader("Upload your file (CSV or XLSX)", type=['csv', 'xlsx'])

        if uploaded_file:
            try:
                # Check file type and read data
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file, dtype=str)
                elif uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file, dtype=str)

                # Ensure there are at least two columns
                if data.shape[1] < 2:
                    raise ValueError("Uploaded file must have at least two columns: Month and Traffic.")

                # Select the first two columns and rename them
                data = data.iloc[:, :2]
                data.columns = ['ds', 'y']

                # Convert dates and ensure numeric traffic values
                data['ds'] = pd.to_datetime(data['ds'], format='%b-%y', errors='coerce')
                data['y'] = pd.to_numeric(data['y'], errors='coerce')

                # Drop rows with invalid dates or traffic values
                data.dropna(inplace=True)

                # Ensure there is data to process
                if data.empty:
                    raise ValueError("No valid data found after processing. Check your file for correct formatting.")

                # Display the cleaned data
                st.write("### Original Data")
                st.dataframe(data, height=200)

                # Detect anomalies in historical data
                anomalies = detect_anomalies(data)
                if not anomalies.empty:
                    st.write("### Anomalies in Historical Data")
                    st.dataframe(anomalies)

                # Forecast period selection
                st.write("### Select Forecast Period")
                forecast_period = st.radio("Choose the forecast duration:", options=[6, 12], index=0)

                # Confidence interval selection
                st.write("### Select Confidence Interval")
                confidence_interval = st.slider("Choose the confidence interval (%):", min_value=50, max_value=99, value=80)

                forecast, model = forecast_traffic(data.set_index('ds'), forecast_period, confidence_interval)

                # Display forecast data
                st.write(f"### Forecasted SEO Traffic for Next {forecast_period} Months")
                forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                forecast_table.columns = ['Date', 'Forecasted Traffic', 'Optimistic Scenario', 'Pessimistic Scenario']
                st.dataframe(forecast_table, height=300)

                # Provide download option for forecast data
                csv_data = convert_df_to_csv(forecast_table)
                st.download_button(label="Download Forecast as CSV",
                                   data=csv_data,
                                   file_name='seo_traffic_forecast.csv',
                                   mime='text/csv')

                # Calculate and display accuracy metrics
                actuals = data['y']
                predictions = forecast['yhat'][:len(actuals)]
                mape, rmse, mae = calculate_accuracy_metrics(actuals, predictions)
                st.write("### Forecast Accuracy Metrics")
                st.write(f"- MAPE: {mape:.2f}%")
                st.write(f"- RMSE: {rmse:.2f}")
                st.write(f"- MAE: {mae:.2f}")

                # Generate and display insights
                insights = generate_insights(forecast)
                st.write("### Insights")
                st.write(insights)

                # Visualization enhancements
                st.write("### Seasonal Decomposition of Forecast")
                seasonal_fig = custom_seasonal_plot(model, forecast)
                st.plotly_chart(seasonal_fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

    elif menu == "Documentation":
        st.write("## Documentation and Education")
        st.markdown(
            """
            ### How to Use the Tool:
            1. **Upload Your Data**: Upload a CSV or Excel file containing two columns:
               - The first column should represent the month (e.g., `Jan-24`).
               - The second column should represent the traffic values.
            2. **Configure Settings**:
               - Select the forecast period (6 or 12 months).
               - Adjust the confidence interval to define uncertainty levels.
            3. **View Results**:
               - The tool provides a forecasted traffic table and seasonal decomposition plots.
               - Download the results as a CSV file.

            ### Key Features:
            - Handles missing values and outliers automatically.
            - Customizable confidence intervals.
            - Interactive seasonal trend analysis.

            ### About Prophet:
            Prophet is an open-source forecasting tool developed by Facebook. It models trends, seasonality, and holidays to make accurate forecasts for time-series data with missing values or outliers.

            """
        )

if __name__ == "__main__":
    main()
