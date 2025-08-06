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
from plotly.offline import plot
import plotly.io as pio
import requests

# ==============================
# TELEGRAM AYARLARI
# ==============================
TELEGRAM_TOKEN = "8271487658:AAGExp6N425AoQvy0U94i4kM7q3oQyJQ3LQ"
CHAT_ID = "5640966892"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=payload)

def send_telegram_photo(photo_path):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        files = {'photo': photo}
        data = {'chat_id': CHAT_ID}
        requests.post(url, files=files, data=data)

# ==============================
# VERİ ÇEKME
# ==============================
def get_asset_data(symbol, period="3y"):
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    if "Close" not in df.columns:
        df["Close"] = df["Adj Close"]
    df = df.reset_index()
    return df

# ==============================
# TEKNİK GÖSTERGELER
# ==============================
def add_indicators(df):
    close = df["Close"].astype(float).values
    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    volume = df["Volume"].astype(float).values

    df["EMA50"] = talib.EMA(close, timeperiod=50)
    df["EMA200"] = talib.EMA(close, timeperiod=200)
    macd, macdsignal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df["MACD"] = macd
    df["MACD_signal"] = macdsignal
    upper, middle, lower = talib.BBANDS(close, timeperiod=20)
    df["BB_upper"], df["BB_middle"], df["BB_lower"] = upper, middle, lower
    df["RSI"] = talib.RSI(close, timeperiod=14)
    df["ATR"] = talib.ATR(high, low, close, timeperiod=14)
    return df

# ==============================
# TAHMİN MODELLERİ
# ==============================
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
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
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
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
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

# ==============================
# BACKTEST
# ==============================
def backtest_strategy(df):
    df = df.dropna().copy()
    df["Position"] = 0
    df.loc[df["EMA50"] > df["EMA200"], "Position"] = 1
    df.loc[df["EMA50"] < df["EMA200"], "Position"] = -1
    df["Return"] = df["Close"].pct_change()
    df["Strategy"] = df["Position"].shift(1) * df["Return"]
    total_return = (1 + df["Strategy"]).prod() - 1
    win_rate = (df["Strategy"] > 0).sum() / (df["Strategy"] != 0).sum() * 100
    return win_rate, total_return * 100

# ==============================
# YATIRIM TAVSİYESİ
# ==============================
def investment_advice(avg_pred, last_price, ema50, ema200, rsi):
    change = (avg_pred - last_price) / last_price * 100
    if change > 2 and ema50 > ema200 and rsi < 70:
        return "Yatırım Yap"
    elif change < -2 and ema50 < ema200:
        return "Yatırım Yapma"
    else:
        return "Bekle / Riskli"

# ==============================
# GRAFİK
# ==============================
def plot_graph(df, symbol, lstm_pred, gru_pred, xgb_pred, days_ahead):
    future_date = df["Date"].iloc[-1] + pd.Timedelta(days=days_ahead)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f"{symbol} - Fiyat", f"{symbol} - RSI"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Kapanış", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA50"], name="EMA50", line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA200"], name="EMA200", line=dict(color="green")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_upper"], name="Bollinger Üst", line=dict(color="lightgray")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_lower"], name="Bollinger Alt", line=dict(color="lightgray")), row=1, col=1)
    fig.add_trace(go.Scatter(x=[future_date], y=[lstm_pred], mode="markers", name="LSTM", marker=dict(color="purple", size=12)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[future_date], y=[gru_pred], mode="markers", name="GRU", marker=dict(color="red", size=12)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[future_date], y=[xgb_pred], mode="markers", name="XGBoost", marker=dict(color="black", size=12)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI", line=dict(color="purple")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(height=800, hovermode="x unified")
    
    # PNG olarak kaydet ve Telegram'a gönder
    photo_path = f"{symbol}_chart.png"
    pio.write_image(fig, photo_path)
    send_telegram_photo(photo_path)
    
    # Tarayıcıda aç
    plot(fig)

# ==============================
# ANA AKIŞ
# ==============================
symbol = input("Sembol girin (ör: BTC-USD, AAPL): ")
days = int(input("Tahmin günü: "))

df = get_asset_data(symbol)
df = add_indicators(df)

last_price = df["Close"].iloc[-1]
lstm_pred = lstm_forecast(df, days)
gru_pred = gru_forecast(df, days)
xgb_pred = xgb_forecast(df, days)
avg_pred = (lstm_pred + gru_pred + xgb_pred) / 3

win_rate, total_return = backtest_strategy(df)
advice = investment_advice(avg_pred, last_price, df["EMA50"].iloc[-1], df["EMA200"].iloc[-1], df["RSI"].iloc[-1])

# CMD RAPORU
report = f"""
📊 {symbol} RAPORU
Son fiyat: {last_price:.2f} USD
LSTM: {lstm_pred:.2f} USD ({(lstm_pred-last_price)/last_price*100:+.2f}%)
GRU: {gru_pred:.2f} USD ({(gru_pred-last_price)/last_price*100:+.2f}%)
XGBoost: {xgb_pred:.2f} USD ({(xgb_pred-last_price)/last_price*100:+.2f}%)
Ortalama: {avg_pred:.2f} USD ({(avg_pred-last_price)/last_price*100:+.2f}%)
Trend: {'Yukarı' if df['EMA50'].iloc[-1] > df['EMA200'].iloc[-1] else 'Aşağı'}
MACD: {'Pozitif' if df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1] else 'Negatif'}
RSI: {df['RSI'].iloc[-1]:.2f}
Yatırım Tavsiyesi: {advice}
Backtest Başarı: {win_rate:.2f}%
Backtest Getiri: {total_return:.2f}%
"""

print(report)
send_telegram_message(report)
plot_graph(df, symbol, lstm_pred, gru_pred, xgb_pred, days)