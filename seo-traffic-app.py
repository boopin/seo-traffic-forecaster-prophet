import streamlit as st
import pandas as pd
from prophet import Prophet
from io import BytesIO

def forecast_traffic(data, forecast_period):
    # Convert month-year format to datetime
    data.index = pd.to_datetime(data.index, format='%b-%y')

    # Reset index and create Prophet-compatible dataframe
    df = data.reset_index()
    df.columns = ['ds', 'y']

    model = Prophet()
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

    # Remove time from the 'ds' column
    forecast['ds'] = forecast['ds'].dt.strftime('%Y-%m')

    return forecast

def convert_df_to_csv(df):
    # Convert dataframe to CSV for download
    output = BytesIO()
    df.to_csv(output, index=False)
    processed_data = output.getvalue()
    return processed_data

def main():
    st.set_page_config(page_title="SEO Traffic Forecast App", layout="wide")

    st.title('SEO Traffic Forecast App')
    st.subheader('Version 1.1')
    st.write("Upload your SEO organic traffic data (CSV or XLSX) containing Month and Traffic columns to forecast future traffic.")

    uploaded_file = st.file_uploader("Upload your file (CSV or XLSX)", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            # Check file type and read data
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file, index_col=0, dtype=str)
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file, index_col=0, dtype=str)

            # Remove empty rows from the data
            data.dropna(how='all', inplace=True)

            # Display the data with original month format
            st.write("### Original Data")
            st.dataframe(data.T, height=200)

            # Convert index to datetime for forecasting
            data.index = pd.to_datetime(data.index, format='%b-%y')

            # Forecast period selection
            st.write("### Select Forecast Period")
            forecast_period = st.radio("Choose the forecast duration:", options=[6, 12], index=0)

            forecast = forecast_traffic(data, forecast_period)

            # Display forecast data
            st.write(f"### Forecasted SEO Traffic for Next {forecast_period} Months")
            forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            forecast_table.columns = ['Date', 'Forecasted Traffic', 'Lower Bound', 'Upper Bound']
            st.dataframe(forecast_table, height=300)

            # Provide download option for forecast data
            csv_data = convert_df_to_csv(forecast_table)
            st.download_button(label="Download Forecast as CSV",
                               data=csv_data,
                               file_name='seo_traffic_forecast.csv',
                               mime='text/csv')

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
