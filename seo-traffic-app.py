import streamlit as st
import pandas as pd
from prophet import Prophet
from io import BytesIO
from prophet.plot import plot_components_plotly
import plotly.graph_objects as go

def forecast_traffic(data, forecast_period, confidence_interval):
    # Convert month-year format to datetime
    data.index = pd.to_datetime(data.index, format='%b-%y')

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

def main():
    st.set_page_config(page_title="ForecastEdge: SEO Traffic Planner", layout="wide")

    st.title('ForecastEdge: SEO Traffic Planner')
    st.subheader('Version 1.3')

    st.write("Upload your SEO organic traffic data (CSV or XLSX) containing Month and Traffic columns to forecast future traffic.")

    menu = st.sidebar.selectbox("Navigation", options=["Forecast Tool", "Documentation"])

    if menu == "Forecast Tool":
        uploaded_file = st.file_uploader("Upload your file (CSV or XLSX)", type=['csv', 'xlsx'])

        if uploaded_file:
            try:
                # Check file type and read data
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file, index_col=0, dtype=str)
                elif uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file, index_col=0, dtype=str)

                # Remove empty rows and rows with all zero traffic values
                data.dropna(how='all', inplace=True)
                data = data[(data != '0').all(axis=1)]

                # Display the data with original month format
                st.write("### Original Data")
                st.dataframe(data.T, height=200)

                # Convert index to datetime for forecasting
                data.index = pd.to_datetime(data.index, format='%b-%y')

                # Forecast period selection
                st.write("### Select Forecast Period")
                forecast_period = st.radio("Choose the forecast duration:", options=[6, 12], index=0)

                # Confidence interval selection
                st.write("### Select Confidence Interval")
                confidence_interval = st.slider("Choose the confidence interval (%):", min_value=50, max_value=99, value=80)

                forecast, model = forecast_traffic(data, forecast_period, confidence_interval)

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

                # Visualization enhancements
                st.write("### Seasonal Decomposition of Forecast")
                seasonal_plot = plot_components_plotly(model, forecast)
                st.plotly_chart(seasonal_plot, use_container_width=True)

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
