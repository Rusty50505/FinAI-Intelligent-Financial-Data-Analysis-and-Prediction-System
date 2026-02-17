import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(layout="wide", page_title="FinAI Pro")

st.title("🚀 FinAI Pro")

# ================= LOAD DATA =================
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, period="2y",
                     interval="1d",
                     auto_adjust=True,
                     progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

stock = st.text_input("Enter Stock Symbol", "AAPL")
df = load_data(stock)

if df.empty:
    st.error("No data found.")
    st.stop()

close = df["Close"].dropna()
current_price = float(close.iloc[-1])

# ================= TABS =================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💰 Paper Trading", "📈 Market Comparison"])

# =========================================================
# ======================= DASHBOARD =======================
# =========================================================
with tab1:

    st.subheader("LSTM Forecast (7-Day)")

    window = 60

    if len(close) < window + 1:
        st.error("Not enough data for LSTM.")
        st.stop()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close.values.reshape(-1,1))

    X, y = [], []

    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], window, 1))

    model = Sequential([
        LSTM(32, input_shape=(window,1)),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=3, batch_size=32, verbose=0)

    # Predict next day only
    last_window = scaled[-window:]
    input_data = last_window.reshape((1, window, 1))
    next_scaled = model.predict(input_data, verbose=0)
    next_price = scaler.inverse_transform(next_scaled)[0][0]

    # Smooth 7-day forecast
    daily_change = (next_price - current_price) / 7
    forecast_prices = []
    prev = current_price

    for _ in range(7):
        prev = prev + daily_change
        forecast_prices.append(prev)

    forecast_index = pd.date_range(close.index[-1],
                                   periods=8,
                                   freq="B")[1:]

    # Chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=close.index,
        y=close.values,
        name="Price"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=forecast_prices,
        name="Forecast",
        line=dict(color="red", dash="dash")
    ))

    fig.update_layout(template="plotly_dark",
                      height=500)

    st.plotly_chart(fig, use_container_width=True)

    st.write("Current Price:", round(current_price,2))

    st.subheader("Prediction Table")
    st.dataframe(pd.DataFrame({
        "Date": forecast_index,
        "Predicted Price": forecast_prices
    }))

# =========================================================
# ==================== PAPER TRADING ======================
# =========================================================
with tab2:

    st.subheader("Paper Trading")

    if "balance" not in st.session_state:
        st.session_state.balance = 100000.0
        st.session_state.shares = 0
        st.session_state.trades = []

    col1, col2 = st.columns(2)

    if col1.button("Buy 1 Share"):
        if st.session_state.balance >= current_price:
            st.session_state.balance -= current_price
            st.session_state.shares += 1
            st.session_state.trades.append(("BUY", current_price))

    if col2.button("Sell 1 Share"):
        if st.session_state.shares > 0:
            st.session_state.balance += current_price
            st.session_state.shares -= 1
            st.session_state.trades.append(("SELL", current_price))

    portfolio_value = st.session_state.shares * current_price
    total_value = portfolio_value + st.session_state.balance

    st.write("Balance:", round(st.session_state.balance,2))
    st.write("Shares Held:", st.session_state.shares)
    st.write("Portfolio Value:", round(portfolio_value,2))
    st.write("Total Account Value:", round(total_value,2))

    if st.session_state.trades:
        st.subheader("Trade History")
        st.dataframe(pd.DataFrame(
            st.session_state.trades,
            columns=["Type","Price"]
        ))

# =========================================================
# ================= MARKET COMPARISON =====================
# =========================================================
with tab3:

    st.subheader("Market Comparison")

    stocks = ["AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA"]
    selected = st.multiselect("Select Stocks",
                              stocks,
                              default=["AAPL","MSFT"])

    comp_fig = go.Figure()

    for s in selected:
        df_s = load_data(s)
        if df_s.empty:
            continue
        close_s = df_s["Close"].dropna()
        normalized = close_s / close_s.iloc[0] * 100
        comp_fig.add_trace(go.Scatter(
            x=close_s.index,
            y=normalized,
            name=s
        ))

    comp_fig.update_layout(template="plotly_dark",
                           height=500)

    st.plotly_chart(comp_fig, use_container_width=True)
