import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Teknik indikatörler ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    return macd_line, signal_line

def calculate_bollinger(series, window=20, num_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower

# --- Veri çekme ---
def get_data(symbol, days=90):
    df = yf.download(symbol, period=f"{days}d", interval="1d")
    df.dropna(inplace=True)
    return df

# --- Analiz ve grafik ---
def analyze_asset(symbol, days=90):
    df = get_data(symbol, days)
    close = df["Close"]

    # Göstergeleri hesapla
    df["RSI"] = calculate_rsi(close)
    df["EMA50"] = calculate_ema(close, 50)
    df["EMA200"] = calculate_ema(close, 200)
    df["MACD"], df["MACD_Signal"] = calculate_macd(close)
    df["BB_Upper"], df["BB_Lower"] = calculate_bollinger(close)

    # Yüzdelik değişim
    price_change = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100

    # Trend yorumu
    trend = "Yukarı" if df["EMA50"].iloc[-1] > df["EMA200"].iloc[-1] else "Aşağı"

    # Yazılı rapor
    report = (
        f"--- {symbol} Analiz ---\n"
        f"Son Fiyat: {close.iloc[-1]:.2f}\n"
        f"RSI: {df['RSI'].iloc[-1]:.2f}\n"
        f"Trend (EMA50 / EMA200): {trend}\n"
        f"MACD: {df['MACD'].iloc[-1]:.2f}, Sinyal: {df['MACD_Signal'].iloc[-1]:.2f}\n"
        f"Bollinger Üst: {df['BB_Upper'].iloc[-1]:.2f}, Alt: {df['BB_Lower'].iloc[-1]:.2f}\n"
        f"Son {days} günde değişim: %{price_change:.2f}\n"
    )

    # Grafik
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Fiyat"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], line=dict(color='blue', width=1), name="EMA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], line=dict(color='orange', width=1), name="EMA200"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], line=dict(color='green', width=1), name="Bollinger Üst"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], line=dict(color='red', width=1), name="Bollinger Alt"))

    fig.update_layout(title=f"{symbol} Teknik Analiz", xaxis_rangeslider_visible=False)
    fig_path = f"{symbol}_chart.png"
    fig.write_image(fig_path)

    return report, fig_path

if __name__ == "__main__":
    rapor, grafik = analyze_asset("BTC-USD", 30)
    print(rapor)
    print(f"Grafik kaydedildi: {grafik}")
