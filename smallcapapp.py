import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ta
import matplotlib.pyplot as plt
from datetime import timedelta

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Small Cap Stock Forecaster", layout="wide")
st.title("📈 Small Cap Stock Forecast & Recommendation App")

st.markdown("""
Enter Indian stock tickers (use `.NS` for NSE stocks, `.BO` for BSE) — example:
`TATAMOTORS.NS, KALYANKJIL.NS, KCP.NS, HSIL.NS, BORORENEW.NS`
""")

tickers_input = st.text_input("Enter Stock Tickers (comma-separated):", "TATAMOTORS.NS, KALYANKJIL.NS, KCP.NS")
start_date = st.date_input("Start Date", pd.to_datetime("2019-01-01"))
forecast_days = st.number_input("Forecast Days", min_value=30, max_value=200, value=100, step=10)

run_btn = st.button("Run Forecast")

# ---------------- MODEL ----------------
def train_lstm_and_forecast(series_values, lookback=60, forecast_days=100, epochs=15, batch_size=32):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series_values.reshape(-1, 1))
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    current_input = scaled[-lookback:].reshape(1, lookback, 1)
    preds_scaled = []
    for _ in range(forecast_days):
        next_scaled = model.predict(current_input, verbose=0)[0][0]
        preds_scaled.append(next_scaled)
        current_input = np.concatenate([current_input[:, 1:, :], np.array(next_scaled).reshape(1, 1, 1)], axis=1)
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).reshape(-1)
    return preds


# ---------------- FIXED SIGNAL FUNCTION ----------------
def compute_signals(last_close, forecast_prices, latest_rsi, latest_macd_hist, threshold_pct=5.0):
    # Ensure scalar numeric values
    def to_scalar(x):
        if isinstance(x, (pd.Series, np.ndarray, list)):
            if len(x) == 0:
                return np.nan
            return float(x.iloc[-1] if hasattr(x, "iloc") else x[-1])
        return float(x)

    latest_rsi = to_scalar(latest_rsi)
    latest_macd_hist = to_scalar(latest_macd_hist)
    pct_change = float((forecast_prices[-1] - last_close) / last_close * 100.0)

    forecast_signal = 1 if pct_change > threshold_pct else -1 if pct_change < -threshold_pct else 0
    rsi_signal = 1 if latest_rsi <= 30 else -1 if latest_rsi >= 70 else 0
    macd_signal = 1 if latest_macd_hist > 0 else -1 if latest_macd_hist < 0 else 0

    return forecast_signal, rsi_signal, macd_signal, pct_change


def combined_recommendation(forecast_signal, rsi_signal, macd_signal, weights, buy_cutoff=0.3, sell_cutoff=-0.3):
    score = (forecast_signal * weights['forecast'] +
             rsi_signal * weights['rsi'] +
             macd_signal * weights['macd'])
    if score >= buy_cutoff:
        return "BUY", score
    elif score <= sell_cutoff:
        return "SELL", score
    else:
        return "HOLD", score


# ---------------- MAIN ----------------
if run_btn:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    results = []
    weights = {'forecast': 0.6, 'rsi': 0.2, 'macd': 0.2}

    progress = st.progress(0)
    for idx, symbol in enumerate(tickers):
        st.subheader(f"📊 {symbol}")
        try:
            df = yf.download(symbol, start=start_date, progress=False)
            if df.empty:
                st.warning(f"No data for {symbol}")
                continue

            # Fix Close column shape
            if isinstance(df['Close'], pd.DataFrame) or len(df['Close'].shape) > 1:
                df['Close'] = df['Close'].squeeze()

            # Indicators
            close_series = pd.Series(df['Close'].values.flatten(), index=df.index)
            df['RSI'] = ta.momentum.RSIIndicator(close_series, window=14, fillna=True).rsi()
            macd = ta.trend.MACD(close_series, fillna=True)
            df['MACD_hist'] = macd.macd_diff()

            last_close = float(df['Close'].iloc[-1])
            latest_rsi = float(df['RSI'].iloc[-1])
            latest_macd_hist = float(df['MACD_hist'].iloc[-1])

            # Forecast
            forecast_prices = train_lstm_and_forecast(df['Close'].values, forecast_days=forecast_days)
            forecast_signal, rsi_signal, macd_signal, pct_change = compute_signals(
                last_close, forecast_prices, latest_rsi, latest_macd_hist
            )
            recommendation, score = combined_recommendation(forecast_signal, rsi_signal, macd_signal, weights)

            # Chart
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df['Close'][-200:], label='Historical Close', color='blue')
            future_dates = pd.bdate_range(df.index[-1] + timedelta(days=1), periods=forecast_days)
            ax.plot(future_dates, forecast_prices, label='Forecast', color='orange')
            ax.set_title(f"{symbol} | Recommendation: {recommendation} (score {score:.2f})")
            ax.legend()
            st.pyplot(fig)

            # Metrics summary
            results.append({
                'Stock': symbol,
                'Last_Close': round(last_close, 2),
                'RSI': round(latest_rsi, 2),
                'MACD_Hist': round(latest_macd_hist, 4),
                'Forecast_Change_%': round(pct_change, 2),
                'Forecast_Signal': forecast_signal,
                'RSI_Signal': rsi_signal,
                'MACD_Signal': macd_signal,
                'Score': round(score, 3),
                'Recommendation': recommendation
            })

        except Exception as e:
            st.error(f"Error for {symbol}: {e}")

        progress.progress((idx + 1) / len(tickers))

    if results:
        st.subheader("📋 Forecast & Recommendation Summary")
        df_results = pd.DataFrame(results)
        st.dataframe(df_results.style.highlight_max(axis=0,
            subset=['Score', 'Forecast_Change_%'], color='lightgreen'))
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "forecast_recommendations.csv", "text/csv")
