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

SYMBOLS = ["AAPL", "MSFT", "GOOG"]
TIMEFRAME = "5m"
RISK_PER_TRADE_PCT = 0.01
MAX_DAILY_LOSS_PCT = 0.03
MAX_TRADES_PER_DAY = 10
COOLDOWN_MINUTES = 5
STOP_LOSS_ATR_MULT = 2.0
TAKE_PROFIT_ATR_MULT = 4.0
TRAILING_STOP_ATR_MULT = 1.5
ML_CONFIDENCE = 0.6
SENTIMENT_THRESHOLD = 0.15

logging.basicConfig(filename="ultimate_bot.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

positions = {}
daily_pnl = 0.0
trades_today = 0
last_trade_times = {}
model = None
feature_cols = []
bot_running = True

def get_historical_data(symbol, timeframe="5m", days=5):
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        bars = api.get_barset(symbol, timeframe, start=start.isoformat(), end=end.isoformat()).df[symbol]
        if bars.empty: return None
        return bars[['open','high','low','close','volume']]
    except Exception as e:
        logging.error(f"Data error {symbol}: {e}")
        return None

def add_indicators(df):
    global feature_cols
    df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
    feature_cols = ['trend_sma_fast','trend_sma_slow','momentum_rsi','momentum_macd',
                    'volatility_atr','volume_adi','trend_vwap']
    for c in feature_cols:
        if c not in df.columns: df[c] = 0
    return df

def get_news_sentiment(symbol):
    try:
        articles = newsapi.get_everything(q=f"{symbol} stock", language='en', sort_by='publishedAt', page_size=10)
        if articles['status'] != 'ok' or articles['totalResults'] == 0: return 0.0
        pos = ['up','gain','rise','bull','beat','profit','growth','positive','outperform','strong']
        neg = ['down','fall','drop','bear','miss','loss','decline','negative','underperform','weak','lawsuit']
        p = n = 0
        for a in articles['articles']:
            text = (a['title'] + ' ' + (a.get('description') or '')).lower()
            p += sum(text.count(w) for w in pos)
            n += sum(text.count(w) for w in neg)
        total = p + n
        return (p - n) / total if total else 0.0
    except Exception as e:
        logging.error(f"News error {symbol}: {e}")
        return 0.0

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

def predict(latest_features):
    if model is None: return 0.5
    return model.predict_proba([latest_features])[0][1]

def equity(): return float(api.get_account().equity)
def position_size(price, atr):
    risk = equity() * RISK_PER_TRADE_PCT
    dist = STOP_LOSS_ATR_MULT * atr
    return int(risk / dist) if dist > 0 else 0

def daily_loss_hit(): return daily_pnl < -MAX_DAILY_LOSS_PCT * equity()
def max_trades_hit(): return trades_today >= MAX_TRADES_PER_DAY

def buy(symbol, price, atr):
    global positions, trades_today
    if symbol in positions: return
    qty = position_size(price, atr)
    if qty < 1: return
    sl = price - STOP_LOSS_ATR_MULT * atr
    tp = price + TAKE_PROFIT_ATR_MULT * atr
    tr = price - TRAILING_STOP_ATR_MULT * atr
    order = api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='day')
    if order:
        positions[symbol] = {'qty':qty,'entry':price,'sl':sl,'tp':tp,'trail':tr,'atr':atr}
        trades_today += 1
        msg = f"🟢 BUY {qty} {symbol} @ {price:.2f}\nSL={sl:.2f} TP={tp:.2f}"
        logging.info(msg); send_telegram(msg)
        threading.Thread(target=monitor, args=(symbol,), daemon=True).start()

def sell(symbol):
    global positions, daily_pnl
    if symbol not in positions: return
    pos = positions[symbol]
    qty = pos['qty']
    price = float(api.get_last_trade(symbol).price)
    order = api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
    if order:
        pnl = (price - pos['entry']) * qty
        daily_pnl += pnl
        msg = f"🔴 SELL {qty} {symbol} @ {price:.2f}\nPnL: {pnl:.2f}"
        logging.info(msg); send_telegram(msg)
        del positions[symbol]

def monitor(symbol):
    while symbol in positions:
        time.sleep(30)
        try:
            price = float(api.get_last_trade(symbol).price)
            p = positions[symbol]
            if price <= p['sl'] or price >= p['tp']:
                sell(symbol); break
            if price > p['entry']:
                new_trail = price - TRAILING_STOP_ATR_MULT * p['atr']
                if new_trail > p['trail']:
                    p['trail'] = new_trail
                    p['sl'] = max(p['sl'], new_trail)
        except Exception as e:
            logging.error(f"Monitor {symbol}: {e}")

def generate_signal(symbol):
    df = get_historical_data(symbol, days=1)
    if df is None or len(df) < 50: return None, "No data"
    df = add_indicators(df)
    latest, prev = df.iloc[-1], df.iloc[-2]
    rsi = latest['momentum_rsi']
    macd = latest['momentum_macd']
    macd_sig = latest['trend_macd_signal']
    sma_f, sma_s = latest['trend_sma_fast'], latest['trend_sma_slow']
    close = latest['close']
    bb_l, bb_h = latest['volatility_bbl'], latest['volatility_bbh']
    ml_prob = predict(latest[feature_cols].values)
    senti = get_news_sentiment(symbol)

    buy_cond = sell_cond = 0
    expl = []
    if rsi < 30: buy_cond +=1; expl.append(f"RSI oversold {rsi:.1f}")
    elif rsi > 70: sell_cond +=1; expl.append(f"RSI overbought {rsi:.1f}")
    if macd > macd_sig and prev['momentum_macd'] <= prev['trend_macd_signal']:
        buy_cond +=1; expl.append("MACD bull cross")
    elif macd < macd_sig and prev['momentum_macd'] >= prev['trend_macd_signal']:
        sell_cond +=1; expl.append("MACD bear cross")
    if close < bb_l: buy_cond +=1; expl.append("Below lower BB")
    elif close > bb_h: sell_cond +=1; expl.append("Above upper BB")
    if sma_f > sma_s and prev['trend_sma_fast'] <= prev['trend_sma_slow']:
        buy_cond +=1; expl.append("SMA bull cross")
    elif sma_f < sma_s and prev['trend_sma_fast'] >= prev['trend_sma_slow']:
        sell_cond +=1; expl.append("SMA bear cross")
    if ml_prob > ML_CONFIDENCE: buy_cond +=1; expl.append(f"ML up {ml_prob:.2f}")
    elif ml_prob < (1 - ML_CONFIDENCE): sell_cond +=1; expl.append(f"ML down {1-ml_prob:.2f}")
    if senti > SENTIMENT_THRESHOLD: buy_cond +=1; expl.append(f"News pos {senti:.2f}")
    elif senti < -SENTIMENT_THRESHOLD: sell_cond +=1; expl.append(f"News neg {senti:.2f}")

    if buy_cond >= 3 and sell_cond == 0: return 'buy', "; ".join(expl)
    if sell_cond >= 3 and buy_cond == 0: return 'sell', "; ".join(expl)
    return None, "No signal"

def send_telegram(text):
    if TELEGRAM_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN": return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': text}, timeout=10)
    except Exception as e:
        logging.error(f"Telegram error: {e}")

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
        except Exception as e:
            logging.error(f"Poll error: {e}")
        time.sleep(2)

def handle_telegram_command(chat_id, msg):
    global bot_running
    if msg == '/start':
        send_telegram("Bot running. Commands: /status, /pause, /resume, /close, /help")
    elif msg == '/status':
        pos_str = "\n".join([f"{s}: qty {p['qty']}, entry {p['entry']:.2f}" for s,p in positions.items()]) or "No positions"
        send_telegram(f"Positions:\n{pos_str}\nDaily PnL: {daily_pnl:.2f}")
    elif msg == '/pause':
        bot_running = False
        send_telegram("Trading paused.")
    elif msg == '/resume':
        bot_running = True
        send_telegram("Trading resumed.")
    elif msg == '/close':
        for s in list(positions.keys()):
            sell(s)
        send_telegram("All positions closed.")
    elif msg == '/help':
        send_telegram("Commands: /start /status /pause /resume /close")

@app.route('/')
def dashboard():
    pos = "".join([f"<li>{s}: {p['qty']} @ {p['entry']:.2f}</li>" for s,p in positions.items()]) or "<li>None</li>"
    html = f"""
    <h2>Trading Bot</h2>
    <p>Daily PnL: {daily_pnl:.2f}</p>
    <ul>{pos}</ul>
    <a href="/pause">Pause</a> | <a href="/resume">Resume</a> | <a href="/close">Close All</a>
    """
    return render_template_string(html)

@app.route('/pause')
def pause():
    global bot_running
    bot_running = False
    return "Paused"

@app.route('/resume')
def resume():
    global bot_running
    bot_running = True
    return "Resumed"

@app.route('/close')
def close_all():
    for s in list(positions.keys()):
        sell(s)
    return "Closed all"

def main_loop():
    global model
    df = get_historical_data(SYMBOLS[0], days=5)
    if df is not None:
        df = add_indicators(df)
        train_model(df)

    while True:
        if not bot_running:
            time.sleep(10)
            continue
        if daily_loss_hit():
            logging.warning("Daily loss hit. Pausing.")
            send_telegram("⚠️ Daily loss limit reached. Bot paused.")
            bot_running = False
            continue
        if max_trades_hit():
            logging.warning("Max trades reached.")
            time.sleep(3600)
            continue

        for sym in SYMBOLS:
            try:
                signal, explanation = generate_signal(sym)
                logging.info(f"{sym}: {signal} - {explanation}")
                if signal == 'buy' and sym not in positions:
                    price = float(api.get_last_trade(sym).price)
                    df2 = get_historical_data(sym, days=1)
                    atr = df2.iloc[-1]['volatility_atr'] if df2 is not None and len(df2)>0 else price*0.01
                    if atr <= 0: atr = price*0.01
                    buy(sym, price, atr)
                elif signal == 'sell' and sym in positions:
                    sell(sym)
            except Exception as e:
                logging.error(f"Loop error {sym}: {e}")

        if datetime.now().minute == 0:
            df = get_historical_data(SYMBOLS[0], days=5)
            if df is not None:
                df = add_indicators(df)
                train_model(df)

        now = datetime.now()
        next_run = now + timedelta(minutes=5 - (now.minute % 5), seconds=0)
        time.sleep(max(1, (next_run - datetime.now()).total_seconds()))

if __name__ == '__main__':
    logging.info("Starting trading bot")
    threading.Thread(target=main_loop, daemon=True).start()
    threading.Thread(target=telegram_poll, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
