import streamlit as st
import pandas as pd
from prophet import Prophet
from io import BytesIO

def forecast_traffic(data):
    # Convert month-year format to datetime
    data.index = pd.to_datetime(data.index, format='%b-%y')

    # Reset index and create Prophet-compatible dataframe
    df = data.reset_index()
    df.columns = ['ds', 'y']

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=6, freq='M')
    forecast = model.predict(future)

    # Round forecast values to integers and return
    forecast['yhat'] = forecast['yhat'].round(0)
    forecast['yhat_lower'] = forecast['yhat_lower'].round(0)
    forecast['yhat_upper'] = forecast['yhat_upper'].round(0)

    return forecast

def convert_df_to_csv(df):
    # Convert dataframe to CSV for download
    output = BytesIO()
    df.to_csv(output, index=False)
    processed_data = output.getvalue()
    return processed_data

def main():
    st.title('SEO Traffic Forecast App')
    st.subheader('Version 1.0')
    st.write("Upload your SEO organic traffic data (CSV or XLSX) containing Month and Traffic columns to forecast future traffic.")

    uploaded_file = st.file_uploader("Upload your file (CSV or XLSX)", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            # Check file type and read data
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file, index_col=0)
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file, index_col=0)

            st.write("### Original Data")
            st.dataframe(data)

            forecast = forecast_traffic(data)

            # Display forecast data
            st.write("### Forecasted SEO Traffic for Next 6 Months")
            forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-6:]
            forecast_table.columns = ['Date', 'Forecasted Traffic', 'Lower Bound', 'Upper Bound']
            st.dataframe(forecast_table)

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
