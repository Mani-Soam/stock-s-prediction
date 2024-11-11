"""import streamlit as st
import pandas as pd
import numpy as np
import time 

a=[1,2,3,4,5,6,7,8]
n=np.array(a)
nd=n.reshape([2,4])
dic={
    "name":["SOAM","MANI","KUMAR"],
    "age":["18","19","21"],
    "city":["MEDIPALLY","GLOBEL","NIRMAL"]
}
st.dataframe(a)
st.table(nd)
st.json(a)
st.write(a)

@st.cache_resource
def ret_time(a):
    time.sleep(10)
    return time.time()
if st.checkbox("1"):
   st.write(ret_time(1))

if st.checkbox("2"):
   st.write(ret_time(2))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set up Streamlit
st.title("Animated Sine Wave")

# Create a figure for plotting
fig, ax = plt.subplots()
x = np.linspace(0, 2 * np.pi, 100)
line, = ax.plot(x, np.sin(x))

ax.set_ylim(-1.5, 1.5)

# Function to update the line
def update(frame):
    line.set_ydata(np.sin(x + frame / 10))  # Update the data
    return line,

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 100), blit=True)

# Save the animation to a file
ani.save('sine_wave_animation.mp4', writer='ffmpeg')

# Display the animation in Streamlit
st.video('sine_wave_animation.mp4')
"""
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to fetch stock data
def load_data(ticker):
    data = yf.download(ticker, start="2010-01-01", end="2024-01-01")
    data['Date'] = data.index
    return data

# Function to preprocess data (adding moving averages)
def preprocess_data(data):
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    data = data.dropna()  # Drop missing values
    return data

# Function to create features and target for machine learning model
def create_features(data):
    data['Target'] = data['Close'].shift(-1)  # Predict next day's closing price
    features = ['Close', 'MA5', 'MA10', 'MA50', 'MA200']
    X = data[features]
    y = data['Target']
    return X, y

# Machine learning model to predict stock prices
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse

# Streamlit app
def app():
    st.title("Stock Price Prediction")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL")

    if ticker:
        # Load data and preprocess
        st.write(f"Loading data for {ticker}...")
        data = load_data(ticker)
        data = preprocess_data(data)

        # Create features and target
        X, y = create_features(data)

        # Train model
        model, mse = train_model(X, y)
        st.write(f"Model trained with MSE: {mse:.2f}")

        # User inputs to predict future prices
        st.write("## Stock Data")
        st.write(data.tail())

        # Plot the closing prices and moving averages
        st.write("## Stock Closing Prices and Moving Averages")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['Date'], data['Close'], label='Close Price', color='blue')
        ax.plot(data['Date'], data['MA5'], label='5-Day Moving Average', color='red')
        ax.plot(data['Date'], data['MA10'], label='10-Day Moving Average', color='green')
        ax.plot(data['Date'], data['MA50'], label='50-Day Moving Average', color='purple')
        ax.plot(data['Date'], data['MA200'], label='200-Day Moving Average', color='orange')
        plt.legend()
        st.pyplot(fig)

        # Make predictions for the next few days
        st.write("## Predicting Next 5 Days")
        predictions = []
        last_data = data.iloc[-1][['Close', 'MA5', 'MA10', 'MA50', 'MA200']].values.reshape(1, -1)

        for i in range(5):
            next_day = model.predict(last_data)[0]
            predictions.append(next_day)
            # Update the input for the next prediction
            new_data = np.array([next_day, last_data[0][1], last_data[0][2], last_data[0][3], last_data[0][4]]).reshape(1, -1)
            last_data = new_data

        st.write(f"Predicted prices for the next 5 days: {predictions}")

        # Plot the predictions
        future_dates = pd.date_range(start=data['Date'].max(), periods=6, freq='B')[1:]
        st.write("## Predicted Stock Prices")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['Date'], data['Close'], label='Historical Close Price', color='blue')
        ax.plot(future_dates, predictions, label='Predicted Prices', color='red', linestyle='--')
        plt.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    app()   
