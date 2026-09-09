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

STOCK_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AMD", "BABA", "BITO", "GBTC"]
CRYPTO_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]

CRYPTO_TRADE_AMOUNT = 10.0
STOCK_TRADE_AMOUNT = 100.0

STOP_LOSS_PCT = 0.01
TAKE_PROFIT_PCT = 0.02
TRAILING_STOP_PCT = 0.005

ML_CONFIDENCE = 0.45

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
last_check_time = "Never"
last_error = "None"

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

def get_crypto_data(symbol, timeframe="1Min", limit=100):
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

def buy_stock(symbol, price, reason):
    global positions, trades_today
    key = f"stock:{symbol}"
    if key in positions: return
    qty = int(STOCK_TRADE_AMOUNT / price)
    if qty < 1:
        qty = 1
    sl = price * (1 - STOP_LOSS_PCT)
    tp = price * (1 + TAKE_PROFIT_PCT)
    order = api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='day')
    if order:
        positions[key] = {'type':'stock','symbol':symbol,'qty':qty,'entry':price,'sl':sl,'tp':tp,'trail':sl}
        trades_today += 1
        msg = f"🟢 BUY STOCK {qty} {symbol} @ {price:.2f}\nReason: {reason}\nSL={sl:.2f} TP={tp:.2f}"
        send_telegram(msg)
        trade_journal.append(msg)
    else:
        send_telegram(f"❌ Stock buy failed for {symbol}")

def buy_crypto(symbol, price, reason):
    global positions, trades_today
    key = f"crypto:{symbol}"
    if key in positions: return
    qty = round(CRYPTO_TRADE_AMOUNT / price, 6)
    if qty * price < 10:
        qty = round(10.0 / price, 6)
    sl = price * (1 - STOP_LOSS_PCT)
    tp = price * (1 + TAKE_PROFIT_PCT)
    order = api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='gtc')
    if order:
        positions[key] = {'type':'crypto','symbol':symbol,'qty':qty,'entry':price,'sl':sl,'tp':tp,'trail':sl}
        trades_today += 1
        msg = f"🟢 BUY CRYPTO {qty} {symbol} @ {price:.2f}\nReason: {reason}\nSL={sl:.2f} TP={tp:.2f}"
        send_telegram(msg)
        trade_journal.append(msg)
    else:
        send_telegram(f"❌ Crypto buy failed for {symbol}")

def sell_short_crypto(symbol, price, reason):
    global positions, trades_today
    key = f"crypto_short:{symbol}"
    if key in positions: return
    qty = round(CRYPTO_TRADE_AMOUNT / price, 6)
    if qty * price < 10:
        qty = round(10.0 / price, 6)
    sl = price * (1 + STOP_LOSS_PCT)
    tp = price * (1 - TAKE_PROFIT_PCT)
    order = api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='gtc')
    if order:
        positions[key] = {'type':'crypto_short','symbol':symbol,'qty':qty,'entry':price,'sl':sl,'tp':tp,'trail':sl}
        trades_today += 1
        msg = f"🔻 SHORT CRYPTO {qty} {symbol} @ {price:.2f}\nReason: {reason}\nSL={sl:.2f} TP={tp:.2f}"
        send_telegram(msg)
        trade_journal.append(msg)
    else:
        send_telegram(f"❌ Crypto short failed for {symbol}")

def sell_position(key, reason="Manual"):
    global positions, daily_pnl
    if key not in positions: return
    pos = positions[key]
    try:
        if pos['type'] == 'stock':
            price = float(api.get_last_trade(pos['symbol']).price)
            api.submit_order(symbol=pos['symbol'], qty=pos['qty'], side='sell', type='market', time_in_force='day')
        elif pos['type'] == 'crypto':
            bars = api.get_crypto_bars(pos['symbol'], "1Min", limit=1).df
            price = float(bars['close'].iloc[-1])
            api.submit_order(symbol=pos['symbol'], qty=pos['qty'], side='sell', type='market', time_in_force='gtc')
        elif pos['type'] == 'crypto_short':
            bars = api.get_crypto_bars(pos['symbol'], "1Min", limit=1).df
            price = float(bars['close'].iloc[-1])
            api.submit_order(symbol=pos['symbol'], qty=pos['qty'], side='buy', type='market', time_in_force='gtc')
        pnl = (price - pos['entry']) * pos['qty']
        daily_pnl += pnl
        msg = f"🔴 CLOSE {pos['symbol']} @ {price:.2f}\nReason: {reason}\nPnL: {pnl:.2f}"
        send_telegram(msg)
        trade_journal.append(msg)
        del positions[key]
    except Exception as e:
        logging.error(f"Sell error {key}: {e}")
        send_telegram(f"❌ Sell error {key}: {e}")

def monitor_positions():
    global last_check_time, last_error
    while True:
        time.sleep(15)
        last_check_time = datetime.now().strftime("%H:%M:%S")
        for key in list(positions.keys()):
            pos = positions[key]
            try:
                if pos['type'] == 'stock':
                    price = float(api.get_last_trade(pos['symbol']).price)
                else:
                    bars = api.get_crypto_bars(pos['symbol'], "1Min", limit=1).df
                    price = float(bars['close'].iloc[-1])

                if pos['type'] == 'crypto_short':
                    new_trail = price * (1 + TRAILING_STOP_PCT)
                    if new_trail < pos.get('trail', 999999):
                        pos['trail'] = new_trail
                        pos['sl'] = min(pos['sl'], new_trail)
                else:
                    new_trail = price * (1 - TRAILING_STOP_PCT)
                    if new_trail > pos.get('trail', 0):
                        pos['trail'] = new_trail
                        pos['sl'] = max(pos['sl'], new_trail)

                if pos['type'] == 'crypto_short':
                    if price >= pos['sl'] or price <= pos['tp']:
                        sell_position(key, "TP/SL hit")
                else:
                    if price <= pos['sl'] or price >= pos['tp']:
                        sell_position(key, "TP/SL hit")
            except Exception as e:
                last_error = str(e)
                logging.error(f"Monitor {key}: {e}")

def generate_signal(df):
    if df is None or len(df) < 30: return None, "No data", "WAIT"
    df = add_indicators(df)
    latest, prev = df.iloc[-1], df.iloc[-2]
    rsi = latest['momentum_rsi']
    macd = latest['momentum_macd']
    macd_sig = latest['trend_macd_signal']
    sma_f, sma_s = latest['trend_sma_fast'], latest['trend_sma_slow']
    close = latest['close']
    ml_prob = predict(latest[feature_cols].values)

    buy_cond = 0
    sell_cond = 0
    reasons = []
    trend = "WAIT"

    if sma_f > sma_s:
        trend = "UPTREND"
        buy_cond += 1
        reasons.append("SMA uptrend")
    elif sma_f < sma_s:
        trend = "DOWNTREND"
        sell_cond += 1
        reasons.append("SMA downtrend")

    if rsi > 50:
        buy_cond += 1
        reasons.append("RSI bullish")
    elif rsi < 50:
        sell_cond += 1
        reasons.append("RSI bearish")

    if macd > macd_sig:
        buy_cond += 1
        reasons.append("MACD bullish")
    elif macd < macd_sig:
        sell_cond += 1
        reasons.append("MACD bearish")

    if ml_prob > 0.45:
        buy_cond += 1
        reasons.append(f"ML up {ml_prob:.2f}")
    elif ml_prob < 0.55:
        sell_cond += 1
        reasons.append(f"ML down {ml_prob:.2f}")

    if buy_cond >= 2 and buy_cond > sell_cond:
        return 'buy', "; ".join(reasons), trend
    if sell_cond >= 2 and sell_cond > buy_cond:
        return 'sell', "; ".join(reasons), trend
    return None, "No strong signal", trend

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
    global bot_running, positions, trades_today
    if msg == '/start':
        send_telegram("Bot running. Commands: /status /journal /calc /checksignal /autostatus /forcebuy /pause /resume /close /testbuy /help")
    elif msg == '/status':
        pos_str = "\n".join([f"{k}: qty {p['qty']}, entry {p['entry']:.2f}, TP {p['tp']:.2f}, SL {p['sl']:.2f}" for k,p in positions.items()]) or "No positions"
        send_telegram(f"Positions:\n{pos_str}\nDaily PnL: {daily_pnl:.2f}")
    elif msg == '/autostatus':
        auto_buy = "✅ ON" if bot_running else "❌ OFF"
        auto_sell = "✅ ON" if bot_running else "❌ OFF"
        monitoring = "✅ Yes" if bot_running else "❌ No"
        send_telegram(f"🤖 Auto Status\nAuto Buy: {auto_buy}\nAuto Sell: {auto_sell}\nBot Running: {monitoring}\nLast Check: {last_check_time}\nLast Error: {last_error}\nTrades Today: {trades_today}\nDaily PnL: {daily_pnl:.2f}")
    elif msg == '/journal':
        j = "\n".join(trade_journal) if trade_journal else "No trades yet."
        send_telegram(f"📒 Journal:\n{j}")
    elif msg == '/calc':
        try:
            bars = api.get_crypto_bars("BTC/USD", "1Min", limit=1).df
            price = float(bars['close'].iloc[-1])
            atr = price * 0.005
            sl = price * (1 - STOP_LOSS_PCT)
            tp = price * (1 + TAKE_PROFIT_PCT)
            risk = price - sl
            reward = tp - price
            send_telegram(f"📊 BTC/USD @ {price:.2f}\nSL: {sl:.2f}\nTP: {tp:.2f}\nRisk: {risk:.2f}\nReward: {reward:.2f}")
        except Exception as e:
            send_telegram(f"Calc failed: {e}")
    elif msg == '/checksignal':
        try:
            df = get_crypto_data("BTC/USD", timeframe="1Min", limit=100)
            if df is None:
                send_telegram("No data")
            else:
                df = add_indicators(df)
                latest = df.iloc[-1]
                rsi = latest['momentum_rsi']
                macd = latest['momentum_macd']
                macd_sig = latest['trend_macd_signal']
                sma_f = latest['trend_sma_fast']
                sma_s = latest['trend_sma_slow']
                close = latest['close']
                ml_prob = predict(latest[feature_cols].values)
                signal, reason, trend = generate_signal(df)
                send_telegram(f"📊 BTC/USD @ {close:.2f}\nTrend: {trend}\nRSI: {rsi:.1f}\nMACD: {'Bull' if macd > macd_sig else 'Bear'}\nSMA: {'Bull' if sma_f > sma_s else 'Bear'}\nML: {ml_prob:.2f}\nSignal: {signal}\nReason: {reason}")
        except Exception as e:
            send_telegram(f"Check signal failed: {e}")
    elif msg == '/forcebuy':
        try:
            bars = api.get_crypto_bars("BTC/USD", "1Min", limit=1).df
            price = float(bars['close'].iloc[-1])
            buy_crypto("BTC/USD", price, "Forced buy")
        except Exception as e:
            send_telegram(f"Force buy failed: {e}")
    elif msg == '/pause':
        bot_running = False
        send_telegram("Trading paused.")
    elif msg == '/resume':
        bot_running = True
        send_telegram("Trading resumed.")
    elif msg == '/close':
        for k in list(positions.keys()):
            sell_position(k, "Manual close")
        send_telegram("All positions closed.")
    elif msg == '/testbuy':
        try:
            bars = api.get_crypto_bars("BTC/USD", "1Min", limit=1).df
            price = float(bars['close'].iloc[-1])
            buy_crypto("BTC/USD", price, "Test buy")
        except Exception as e:
            send_telegram(f"Test buy failed: {e}")
    elif msg == '/help':
        send_telegram("Commands: /start /status /journal /calc /checksignal /autostatus /forcebuy /pause /resume /close /testbuy /help")

@app.route('/')
def dashboard():
    pos = "".join([f"<li>{k}: {p['qty']} @ {p['entry']:.2f}</li>" for k,p in positions.items()]) or "<li>None</li>"
    html = f"<h2>Trading Bot</h2><p>Daily PnL: {daily_pnl:.2f}</p><ul>{pos}</ul>"
    return render_template_string(html)

def main_loop():
    global model, last_check_time, last_error, bot_running
    df = get_stock_data("SPY", days=5)
    if df is not None:
        df = add_indicators(df)
        train_model(df)

    while True:
        if not bot_running:
            time.sleep(10)
            continue

        last_check_time = datetime.now().strftime("%H:%M:%S")

        # Stocks
        if is_market_open_now():
            for sym in STOCK_SYMBOLS:
                df = get_stock_data(sym, days=1)
                if df is None: continue
                df = add_indicators(df)
                signal, reason, trend = generate_signal(df)
                if signal == 'buy':
                    price = float(api.get_last_trade(sym).price)
                    buy_stock(sym, price, reason)

        # Crypto
        for sym in CRYPTO_SYMBOLS:
            df = get_crypto_data(sym, timeframe="1Min", limit=100)
            if df is None: continue
            df = add_indicators(df)
            signal, reason, trend = generate_signal(df)
            if signal == 'buy':
                bars = api.get_crypto_bars(sym, "1Min", limit=1).df
                price = float(bars['close'].iloc[-1])
                buy_crypto(sym, price, reason)
            elif signal == 'sell':
                bars = api.get_crypto_bars(sym, "1Min", limit=1).df
                price = float(bars['close'].iloc[-1])
                sell_short_crypto(sym, price, reason)

        if datetime.now().minute == 0:
            df = get_stock_data("SPY", days=5)
            if df is not None:
                df = add_indicators(df)
                train_model(df)

        time.sleep(60)

if __name__ == '__main__':
    logging.info("Starting final working bot")
    threading.Thread(target=main_loop, daemon=True).start()
    threading.Thread(target=monitor_positions, daemon=True).start()
    threading.Thread(target=telegram_poll, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
