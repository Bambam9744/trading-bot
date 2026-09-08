import os
import time
import threading
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template_string
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ta import add_all_ta_features
from newsapi import NewsApiClient
import requests
import pytz

ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TRADING_MODE = os.getenv("TRADING_MODE", "paper")

if TRADING_MODE == "live":
    ALPACA_BASE = "https://api.alpaca.markets"
else:
    ALPACA_BASE = "https://paper-api.alpaca.markets"

api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE, api_version='v2')
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# ============ SYMBOLS ============
STOCK_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA", "BITO", "GBTC"]
CRYPTO_SYMBOLS = ["BTC/USD"]   # Alpaca crypto

MAX_STOCK_POSITIONS = 3
MAX_CRYPTO_POSITIONS = 1
RISK_PER_TRADE_PCT = 0.01
MAX_DAILY_LOSS_PCT = 0.03
MAX_TRADES_PER_DAY = 20
STOP_LOSS_ATR_MULT = 2.0
TAKE_PROFIT_ATR_MULT = 4.0
ML_CONFIDENCE = 0.6

logging.basicConfig(filename="bot.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

positions = {}
daily_pnl = 0.0
trades_today = 0
model = None
feature_cols = []
bot_running = True
trade_journal = []

US_TZ = pytz.timezone("America/New_York")

def is_market_open_now():
    now = datetime.now(pytz.utc).astimezone(US_TZ)
    if now.weekday() >= 5:
        return False
    if now.hour >= 9 and now.minute >= 30 and now.hour <= 16:
        return True
    return False

def get_stock_data(symbol, timeframe="5m", days=2):
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        bars = api.get_barset(symbol, timeframe, start=start.isoformat(), end=end.isoformat()).df[symbol]
        if bars.empty: return None
        return bars[['open','high','low','close','volume']]
    except:
        return None

def get_crypto_data(symbol, timeframe="1m", limit=100):
    try:
        bars = api.get_crypto_bars(symbol, timeframe, limit=limit).df
        if bars.empty: return None
        return bars[['open','high','low','close','volume']]
    except Exception as e:
        logging.error(f"Crypto data error {symbol}: {e}")
        return None

def add_indicators(df):
    global feature_cols
    df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
    feature_cols = ['trend_sma_fast','trend_sma_slow','momentum_rsi','momentum_macd',
                    'volatility_atr','volume_adi','trend_vwap']
    for c in feature_cols:
        if c not in df.columns: df[c] = 0
    return df

def train_model(df):
    global model
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    df = df.dropna(subset=['target'] + feature_cols)
    if len(df) < 100: return None
    X, y = df[feature_cols].values, df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    logging.info(f"Model acc: {acc:.2f}")
    model = clf

def predict(features):
    if model is None: return 0.5
    return model.predict_proba([features])[0][1]

def equity():
    try:
        return float(api.get_account().equity)
    except:
        return 100000.0

def position_size(price, atr):
    risk = equity() * RISK_PER_TRADE_PCT
    dist = STOP_LOSS_ATR_MULT * atr
    return risk / dist if dist > 0 else 0

def daily_loss_hit(): return daily_pnl < -MAX_DAILY_LOSS_PCT * equity()
def max_trades_hit(): return trades_today >= MAX_TRADES_PER_DAY

def buy_stock(symbol, price, atr):
    global positions, trades_today
    key = f"stock:{symbol}"
    if key in positions: return
    qty = int(position_size(price, atr))
    if qty < 1: return
    sl = price - STOP_LOSS_ATR_MULT * atr
    tp = price + TAKE_PROFIT_ATR_MULT * atr
    order = api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='day')
    if order:
        positions[key] = {'type':'stock','symbol':symbol,'qty':qty,'entry':price,'sl':sl,'tp':tp,'atr':atr}
        trades_today += 1
        send_telegram(f"🟢 BUY STOCK {qty} {symbol} @ {price:.2f}\nSL={sl:.2f} TP={tp:.2f}")

def buy_crypto(symbol, price, atr):
    global positions, trades_today
    key = f"crypto:{symbol}"
    if key in positions: return
    qty = round(position_size(price, atr) / price, 6)
    if qty <= 0: return
    sl = price - STOP_LOSS_ATR_MULT * atr
    tp = price + TAKE_PROFIT_ATR_MULT * atr
    order = api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='gtc')
    if order:
        positions[key] = {'type':'crypto','symbol':symbol,'qty':qty,'entry':price,'sl':sl,'tp':tp,'atr':atr}
        trades_today += 1
        send_telegram(f"🟢 BUY CRYPTO {qty} {symbol} @ {price:.2f}\nSL={sl:.2f} TP={tp:.2f}")

def sell_position(key):
    global positions, daily_pnl
    if key not in positions: return
    pos = positions[key]
    try:
        if pos['type'] == 'stock':
            price = float(api.get_last_trade(pos['symbol']).price)
            api.submit_order(symbol=pos['symbol'], qty=pos['qty'], side='sell', type='market', time_in_force='day')
        else:
            price = float(api.get_last_crypto_trade(pos['symbol']).price)
            api.submit_order(symbol=pos['symbol'], qty=pos['qty'], side='sell', type='market', time_in_force='gtc')
        pnl = (price - pos['entry']) * pos['qty']
        daily_pnl += pnl
        send_telegram(f"🔴 SELL {pos['symbol']} @ {price:.2f}\nPnL: {pnl:.2f}")
        del positions[key]
    except Exception as e:
        logging.error(f"Sell error {key}: {e}")

def monitor_positions():
    while True:
        time.sleep(15)
        for key in list(positions.keys()):
            pos = positions[key]
            try:
                if pos['type'] == 'stock':
                    price = float(api.get_last_trade(pos['symbol']).price)
                else:
                    price = float(api.get_last_crypto_trade(pos['symbol']).price)
                if price <= pos['sl'] or price >= pos['tp']:
                    sell_position(key)
            except Exception as e:
                logging.error(f"Monitor {key}: {e}")

def generate_signal(df):
    if df is None or len(df) < 50: return None
    df = add_indicators(df)
    latest, prev = df.iloc[-1], df.iloc[-2]
    rsi = latest['momentum_rsi']
    macd = latest['momentum_macd']
    macd_sig = latest['trend_macd_signal']
    sma_f, sma_s = latest['trend_sma_fast'], latest['trend_sma_slow']
    close = latest['close']
    bb_l, bb_h = latest['volatility_bbl'], latest['volatility_bbh']
    ml_prob = predict(latest[feature_cols].values)

    buy_cond = sell_cond = 0
    if rsi < 30: buy_cond += 1
    elif rsi > 70: sell_cond += 1
    if macd > macd_sig and prev['momentum_macd'] <= prev['trend_macd_signal']: buy_cond += 1
    elif macd < macd_sig and prev['momentum_macd'] >= prev['trend_macd_signal']: sell_cond += 1
    if close < bb_l: buy_cond += 1
    elif close > bb_h: sell_cond += 1
    if sma_f > sma_s and prev['trend_sma_fast'] <= prev['trend_sma_slow']: buy_cond += 1
    elif sma_f < sma_s and prev['trend_sma_fast'] >= prev['trend_sma_slow']: sell_cond += 1
    if ml_prob > ML_CONFIDENCE: buy_cond += 1
    elif ml_prob < (1 - ML_CONFIDENCE): sell_cond += 1

    if buy_cond >= 3 and sell_cond == 0: return 'buy'
    if sell_cond >= 3 and buy_cond == 0: return 'sell'
    return None

def send_telegram(text):
    if TELEGRAM_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN": return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': text}, timeout=10)
    except:
        pass

def telegram_poll():
    offset = 0
    while bot_running:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates?timeout=30&offset={offset}"
            resp = requests.get(url).json()
            for upd in resp.get('result', []):
                offset = upd['update_id'] + 1
                msg = upd.get('message', {}).get('text', '')
                chat_id = upd.get('message', {}).get('chat', {}).get('id')
                if chat_id:
                    handle_telegram_command(chat_id, msg)
        except:
            pass
        time.sleep(2)

def handle_telegram_command(chat_id, msg):
    global bot_running
    if msg == '/start':
        send_telegram("Bot running. Stocks + Crypto. Commands: /status /pause /resume /close /help")
    elif msg == '/status':
        pos_str = "\n".join([f"{k}: qty {p['qty']}, entry {p['entry']:.2f}" for k,p in positions.items()]) or "No positions"
        send_telegram(f"Positions:\n{pos_str}\nDaily PnL: {daily_pnl:.2f}")
    elif msg == '/pause':
        bot_running = False
        send_telegram("Trading paused.")
    elif msg == '/resume':
        bot_running = True
        send_telegram("Trading resumed.")
    elif msg == '/close':
        for k in list(positions.keys()):
            sell_position(k)
        send_telegram("All positions closed.")
    elif msg == '/help':
        send_telegram("Commands: /start /status /pause /resume /close /help")

@app.route('/')
def dashboard():
    pos = "".join([f"<li>{k}: {p['qty']} @ {p['entry']:.2f}</li>" for k,p in positions.items()]) or "<li>None</li>"
    html = f"<h2>Trading Bot</h2><p>Daily PnL: {daily_pnl:.2f}</p><ul>{pos}</ul>"
    return render_template_string(html)

def main_loop():
    global model
    # Train on SPY initially
    df = get_stock_data("SPY", days=5)
    if df is not None:
        df = add_indicators(df)
        train_model(df)

    while True:
        if not bot_running:
            time.sleep(10)
            continue
        if daily_loss_hit():
            send_telegram("⚠️ Daily loss limit reached. Bot paused.")
            bot_running = False
            continue
        if max_trades_hit():
            time.sleep(3600)
            continue

        # Stocks (during market hours)
        if is_market_open_now():
            for sym in STOCK_SYMBOLS:
                if len([k for k in positions if k.startswith("stock:")]) >= MAX_STOCK_POSITIONS:
                    break
                df = get_stock_data(sym, days=1)
                if df is None: continue
                signal = generate_signal(df)
                if signal == 'buy':
                    latest = df.iloc[-1]
                    atr = latest['volatility_atr'] if latest['volatility_atr'] > 0 else latest['close']*0.01
                    buy_stock(sym, float(api.get_last_trade(sym).price), atr)

        # Crypto (24/7)
        for sym in CRYPTO_SYMBOLS:
            if len([k for k in positions if k.startswith("crypto:")]) >= MAX_CRYPTO_POSITIONS:
                break
            df = get_crypto_data(sym, timeframe="1m", limit=100)
            if df is None: continue
            signal = generate_signal(df)
            if signal == 'buy':
                latest = df.iloc[-1]
                atr = latest['volatility_atr'] if latest['volatility_atr'] > 0 else latest['close']*0.01
                price = float(api.get_last_crypto_trade(sym).price)
                buy_crypto(sym, price, atr)

        # Retrain every hour
        if datetime.now().minute == 0:
            df = get_stock_data("SPY", days=5)
            if df is not None:
                df = add_indicators(df)
                train_model(df)

        # Run every 1 minute
        time.sleep(60)

if __name__ == '__main__':
    logging.info("Starting bot with BTC/USD")
    threading.Thread(target=main_loop, daemon=True).start()
    threading.Thread(target=monitor_positions, daemon=True).start()
    threading.Thread(target=telegram_poll, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
