import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title='Stock Market Analyzer', page_icon='📈')

st.title('📊 Stock Market Analyzer')
st.markdown('Welcome! This app allows you to view historical stock prices, visualize trends, and forecast future stock prices.')

# Sidebar inputs
with st.sidebar:
    st.header('Settings')
    ticker = st.text_input('Enter stock ticker symbol', value='AAPL')
    start_date = st.date_input('Start date', value=pd.to_datetime('2010-01-01'))
    end_date = st.date_input('End date', value=pd.to_datetime('2022-12-31'))
    forecast_days = st.number_input('Forecast days', value=30, min_value=1)

# Load data
if ticker:
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        st.success(f'Data loaded for {ticker}')
    except Exception as e:
        st.error(f'Failed to load data for {ticker}: {e}')
else:
    st.error('Please enter a stock ticker symbol')

# Main content
if 'data' in locals():
    st.header('Historical Stock Prices')
    st.write(data['Close'].plot(figsize=(12, 6)))
    st.metric('Current price', data['Close'].iloc[-1])

    # Visualize trends
    st.header('Trend Visualization')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Close'])
    ax.set_title(f'{ticker} Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    st.pyplot(fig)

    # Forecast future stock prices
    st.header('Forecast')
    if st.button('Forecast', type='primary'):
        # Prepare data
        X = np.arange(len(data)).reshape(-1, 1)
        y = data['Close'].values

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Evaluate model
        mse = mean_squared_error(y_test, predictions)
        st.metric('Mean Squared Error', mse)

        # Forecast future prices
        future_X = np.arange(len(data), len(data) + forecast_days).reshape(-1, 1)
        future_predictions = model.predict(future_X)

        # Display forecast
        st.write(f'Forecast for the next {forecast_days} days:')
        st.write(future_predictions)

        # Plot forecast
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['Close'], label='Historical')
        ax.plot(np.arange(len(data), len(data) + forecast_days), future_predictions, label='Forecast', linestyle='--')
        ax.set_title(f'{ticker} Close Price Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        st.pyplot(fig)

# Show example
with st.expander('See example'):
    st.write('Example stock ticker symbols: AAPL, GOOGL, MSFT, AMZN')