import os
import pandas as pd
import numpy as np
import yfinance as yf
import talib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from xgboost import XGBRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import requests
from flask import Flask, request

# Telegram token ve chat id environment variables üzerinden gelecek
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")  # Render ayarlarında ekleyeceğiz

app = Flask(__name__)

# ===== TELEGRAM FONKSİYONLARI =====
def send_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": chat_id, "text": text})

def send_photo(chat_id, photo_path):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as photo:
        requests.post(url, data={"chat_id": chat_id}, files={"photo": photo})

# ===== VERİ VE ANALİZ =====
def get_asset_data(symbol, period="3y"):
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    if "Close" not in df.columns:
        df["Close"] = df["Adj Close"]
    df = df.reset_index()
    return df

def add_indicators(df):
    close = df["Close"].astype(float).values
    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values

    df["EMA50"] = talib.EMA(close, timeperiod=50)
    df["EMA200"] = talib.EMA(close, timeperiod=200)
    macd, macdsignal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df["MACD"] = macd
    df["MACD_signal"] = macdsignal
    upper, middle, lower = talib.BBANDS(close, timeperiod=20)
    df["BB_upper"], df["BB_middle"], df["BB_lower"] = upper, middle, lower
    df["RSI"] = talib.RSI(close, timeperiod=14)
    return df

def lstm_forecast(df, days_ahead=7, look_back=60):
    data = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler((0,1))
    scaled_data = scaler.fit_transform(data)
    X_train, y_train = [], []
    for i in range(look_back, len(scaled_data) - days_ahead):
        X_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i+days_ahead-1, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
    X_test = np.reshape(scaled_data[-look_back:], (1, look_back, 1))
    return scaler.inverse_transform(model.predict(X_test))[0,0]

def gru_forecast(df, days_ahead=7, look_back=60):
    data = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler((0,1))
    scaled_data = scaler.fit_transform(data)
    X_train, y_train = [], []
    for i in range(look_back, len(scaled_data) - days_ahead):
        X_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i+days_ahead-1, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
    model.add(GRU(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
    X_test = np.reshape(scaled_data[-look_back:], (1, look_back, 1))
    return scaler.inverse_transform(model.predict(X_test))[0,0]

def xgb_forecast(df, days_ahead=7, look_back=60):
    data = df["Close"].values
    X, y = [], []
    for i in range(look_back, len(data) - days_ahead):
        X.append(data[i-look_back:i])
        y.append(data[i+days_ahead-1])
    model = XGBRegressor(n_estimators=200, learning_rate=0.05)
    model.fit(np.array(X), np.array(y))
    return model.predict([data[-look_back:]])[0]

def plot_graph(df, symbol, lstm_pred, gru_pred, xgb_pred, days_ahead):
    from plotly.subplots import make_subplots
    future_date = df["Date"].iloc[-1] + pd.Timedelta(days=days_ahead)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Kapanış", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA50"], name="EMA50", line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA200"], name="EMA200", line=dict(color="green")), row=1, col=1)
    fig.add_trace(go.Scatter(x=[future_date], y=[lstm_pred], mode="markers", name="LSTM", marker=dict(color="purple", size=12)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[future_date], y=[gru_pred], mode="markers", name="GRU", marker=dict(color="red", size=12)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[future_date], y=[xgb_pred], mode="markers", name="XGBoost", marker=dict(color="black", size=12)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI", line=dict(color="purple")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    photo_path = f"{symbol}_chart.png"
    pio.write_image(fig, photo_path)
    return photo_path

# ===== TELEGRAM WEBHOOK =====
@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def respond():
    update = request.get_json()
    message = update.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "")

    if text.startswith("/tahmin"):
        try:
            _, symbol, days = text.split()
            days = int(days)
            send_message(chat_id, f"⏳ {symbol} için {days} günlük tahmin hazırlanıyor...")

            df = get_asset_data(symbol)
            df = add_indicators(df)

            last_price = df["Close"].iloc[-1]
            lstm_pred = lstm_forecast(df, days)
            gru_pred = gru_forecast(df, days)
            xgb_pred = xgb_forecast(df, days)
            avg_pred = (lstm_pred + gru_pred + xgb_pred) / 3

            report = f"""
📊 {symbol} RAPORU
Son fiyat: {last_price:.2f}
LSTM: {lstm_pred:.2f}
GRU: {gru_pred:.2f}
XGB: {xgb_pred:.2f}
Ortalama: {avg_pred:.2f}
"""
            send_message(chat_id, report)

            chart_path = plot_graph(df, symbol, lstm_pred, gru_pred, xgb_pred, days)
            send_photo(chat_id, chart_path)

        except Exception as e:
            send_message(chat_id, f"Hata: {e}")

    else:
        send_message(chat_id, "Komut: /tahmin <sembol> <gün>")

    return {"ok": True}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)