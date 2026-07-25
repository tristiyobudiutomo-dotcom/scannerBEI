import pandas as pd
import yfinance as yf
import requests
import os
import csv
import time
import schedule
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CSV_FILE = "signals.csv"
ALERT_FILE = "alerts.csv"
BATCH_SIZE = 3
REQUEST_DELAY = 3


def send(msg, parse_mode="Markdown"):
    if not TOKEN or not CHAT_ID:
        print("Telegram token/chat id belum diset.")
        return

    for i in range(0, len(msg), 4000):
        chunk = msg[i:i + 4000]
        for attempt in range(3):
            try:
                requests.post(
                    f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                    json={
                        "chat_id": CHAT_ID,
                        "text": chunk,
                        "disable_web_page_preview": False,
                        "parse_mode": parse_mode,
                    },
                    timeout=10,
                )
                break
            except Exception as e:
                if attempt == 2:
                    print(f"Gagal kirim Telegram: {e}")
                time.sleep(2)


def fetch_data(symbol, period="6mo", interval="1d"):
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            timeout=15
        )
        if df.empty or len(df) < 30:
            return None
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df
    except Exception as e:
        print(f"fetch_data error {symbol}: {e}")
        return None


def fetch_intraday(symbol):
    try:
        df = yf.download(
            symbol,
            period="5d",
            interval="15m",
            auto_adjust=True,
            progress=False,
            timeout=15
        )
        if df.empty or len(df) < 10:
            return None, None, None, None, None

        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        daily = df.resample("D").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna()

        if len(daily) < 2:
            return None, None, None, None, None

        open_today = daily["Open"].iloc[-1]
        close_yest = daily["Close"].iloc[-2]
        gap_pct = ((open_today - close_yest) / close_yest) * 100
        recent_15m = df.tail(16)
        support = float(recent_15m["Low"].min())
        current_price = float(df["Close"].iloc[-1])

        return open_today, close_yest, gap_pct, support, current_price
    except Exception as e:
        print(f"fetch_intraday error {symbol}: {e}")
        return None, None, None, None, None


def level_entry(item, price, open_today, support, rvol, atr_pct):
    if item["tipe"] == "breakout":
        entry_limit = min(open_today, price * 0.995)
        stop = min(support, price * 0.98)
        tp1 = entry_limit * 1.05
        tp2 = entry_limit * 1.10
        tp3 = entry_limit * 1.15
        return round(entry_limit, 0), round(stop, 0), round(tp1, 0), round(tp2, 0), round(tp3, 0)
    else:
        entry_limit = price * 0.99
        stop = price * 0.96
        tp1 = price * 1.03
        tp2 = price * 1.06
        return round(entry_limit, 0), round(stop, 0), round(tp1, 0), round(tp2, 0), None


def trend_regime(symbol, label):
    df = fetch_data(symbol, period="6mo")
    if df is None:
        return f"{label}: DATA ERR"

    close = df["Close"].squeeze()
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    price = close.iloc[-1]

    if price > ema20 > ema50:
        return f"{label}: 🟢 BULLISH"
    elif price < ema20 < ema50:
        return f"{label}: 🔴 BEARISH"
    return f"{label}: 🟡 NEUTRAL"


def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker + ".JK")
        info = stock.info or {}
        pbv = info.get("priceToBook", None)
        per = info.get("trailingPE", None)
        mcap = info.get("marketCap", None)
        return pbv, per, mcap
    except Exception as e:
        print(f"get_fundamentals error {ticker}: {e}")
        return None, None, None


# ── SCAN ──
def morning_scan():
    print(f"\n=== SCAN {datetime.now().strftime('%H:%M')} ===")

    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "date", "time", "ticker", "score",
                "rsi", "price", "rvol", "gap_pct",
                "breakout", "atr_pct", "entry_limit",
                "stop_loss", "tp1", "tp2", "tp3"
            ])

    if not os.path.exists("stocks.txt"):
        print("stocks.txt tidak ditemukan.")
        return

    with open("stocks.txt", "r", encoding="utf-8") as f:
        STOCKS = [x.strip() for x in f if x.strip()]
    STOCKS = [s if s.endswith(".JK") else s + ".JK" for s in STOCKS]

    sector_map = {}
    if os.path.exists("sectors.csv"):
        with open("sectors.csv", "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sector_map[row["ticker"].upper().replace(".JK", "")] = row["sector"].upper()

    regime_lines = []
    for ticker, label in [("^JKSE", "IHSG"), ("USDIDR=X", "USDIDR"), ("GC=F", "GOLD"), ("CL=F", "OIL")]:
        result = trend_regime(ticker, label)
        print(f"  {result}")
        regime_lines.append(result)
        time.sleep(REQUEST_DELAY)

    IHSG_line = regime_lines[0] if regime_lines else ""

    results_momentum = []
    results_reversal = []
    failed_tickers = []
    all_setups = []

    for i, stock in enumerate(STOCKS):
        print(f"Scanning {i + 1}/{len(STOCKS)}: {stock}")
        df = fetch_data(stock, period="6mo")
        if df is None:
            failed_tickers.append(stock)
            continue

        try:
            close = df["Close"].squeeze()
            high = df["High"].squeeze()
            low = df["Low"].squeeze()
            volume = df["Volume"].squeeze()

            price = float(close.iloc[-1])
            rsi = RSIIndicator(close, window=4).rsi()
            rsi_now = float(rsi.iloc[-1])
            rsi_prev = float(rsi.iloc[-2])

            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

            avg_vol = float(volume.tail(20).mean())
            daily_value = price * float(volume.iloc[-1])
            if avg_vol <= 0 or price <= 0:
                continue
            rvol = float(volume.iloc[-1]) / avg_vol

            atr = AverageTrueRange(high, low, close, window=14).average_true_range()
            atr_val = float(atr.iloc[-1])
            atr_pct = atr_val / price * 100

            prev_20_high = float(close.shift(1).tail(20).max())
            prev_20_low = float(close.shift(1).tail(20).min())
            breakout_high = float(close.iloc[-2]) > prev_20_high

            open_today, close_yest, gap_pct, support, _ = fetch_intraday(stock)

            range_pct = ((prev_20_high - prev_20_low) / prev_20_low) * 100

            pbv, per, mcap = get_fundamentals(stock)
            fundament_flag = ""
            if pbv and per and mcap:
                if pbv < 2 and per < 25 and 500_000_000_000 < mcap < 50_000_000_000_000:
                    fundament_flag = "FUNDAMENTAL OK"

            ticker_clean = stock.replace(".JK", "")
            sector = sector_map.get(ticker_clean, "OTHER")

            if atr_pct < 3 or daily_value < 5_000_000_000 or price < 50:
                continue

            item = {
                "ticker": ticker_clean,
                "rsi": round(rsi_now, 1),
                "price": round(price, 0),
                "rvol": round(rvol, 2),
                "gap_pct": round(gap_pct, 2) if gap_pct is not None else 0,
                "breakout": breakout_high,
                "range_pct": round(range_pct, 1),
                "atr_pct": round(atr_pct, 2),
                "fundamental": fundament_flag,
                "sector": sector,
                "daily_value": round(daily_value, 0),
                "open_today": round(open_today, 0) if open_today else 0,
                "support": round(support, 0) if support else 0
            }

            score_m = 0
            if rvol > 5:
                score_m += 35
            elif rvol > 3:
                score_m += 20
            elif rvol > 2:
                score_m += 10

            if gap_pct is not None and gap_pct > 3:
                score_m += min(30, gap_pct * 4)
            elif gap_pct is not None and gap_pct > 0:
                score_m += 8

            if breakout_high:
                score_m += 15

            if rsi_now > 50 and rsi_now > rsi_prev:
                score_m += 10
            elif rsi_now > 40 and rsi_now > rsi_prev:
                score_m += 5

            if 3 < range_pct < 25:
                score_m += 10
            elif range_pct >= 25:
                score_m += 5

            if "🟢 BULLISH" in IHSG_line:
                score_m += 10
            if price > ema20:
                score_m += 5
            if ema20 > ema50:
                score_m += 5

            item["tipe"] = "breakout"
            item["score"] = round(score_m, 1)

            if score_m >= 40:
                el, sl, tp1, tp2, tp3 = level_entry(item, price, item["open_today"], item["support"], rvol, atr_pct)
                item["entry_limit"] = el
                item["stop_loss"] = sl
                item["tp1"] = tp1
                item["tp2"] = tp2
                item["tp3"] = tp3
                results_momentum.append(item)
                all_setups.append(item)

            score_r = 0
            if rsi_now < 22:
                score_r += 30
            elif rsi_now < 28:
                score_r += 20
            elif rsi_now < 35:
                score_r += 10

            score_r += max(0, min(20, (rsi_now - rsi_prev) * 5))
            score_r += min(15, rvol * 3)

            if price < ema20:
                score_r += 10
            if not breakout_high:
                score_r += 10
            if fundament_flag:
                score_r += 5
            if "🟢 BULLISH" in IHSG_line:
                score_r += 5

            if score_r >= 25:
                item_rev = item.copy()
                item_rev["tipe"] = "reversal"
                item_rev["score"] = round(score_r, 1)
                el, sl, tp1, tp2, tp3 = level_entry(item_rev, price, item_rev["open_today"], item_rev["support"], rvol, atr_pct)
                item_rev["entry_limit"] = el
                item_rev["stop_loss"] = sl
                item_rev["tp1"] = tp1
                item_rev["tp2"] = tp2
                item_rev["tp3"] = tp3
                results_reversal.append(item_rev)
                all_setups.append(item_rev)

        except Exception as e:
            failed_tickers.append(f"{stock}: {e}")
            continue

        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(REQUEST_DELAY)

    results_momentum = sorted(results_momentum, key=lambda x: x["score"], reverse=True)
    results_reversal = sorted(results_reversal, key=lambda x: x["score"], reverse=True)

    with open(ALERT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "entry_limit", "tp1", "tp2", "tp3", "stop_loss", "score", "tipe"])
        for item in all_setups:
            w.writerow([
                item["ticker"], item["entry_limit"], item["tp1"],
                item["tp2"], item["tp3"], item["stop_loss"],
                item["score"], item["tipe"]
            ])

    now = datetime.now()
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for item in results_momentum[:10]:
            w.writerow([
                now.strftime("%Y-%m-%d"), now.strftime("%H:%M"),
                item["ticker"], item["score"], item["rsi"], item["price"],
                item["rvol"], item["gap_pct"], item["breakout"],
                item["atr_pct"], item["entry_limit"], item["stop_loss"],
                item["tp1"], item["tp2"], item["tp3"]
            ])

    regime_header = "🌏 **MARKET REGIME**\n"
    for line in regime_lines:
        regime_header += f"  {line}\n"
    regime_header += f"\n  Lolos filter: {len(results_momentum)} momentum + {len(results_reversal)} reversal\n"
    regime_header += f"  Gagal: {len(failed_tickers)}\n"
    send(regime_header)

    msg1 = "🔥 **BREAKOUT MOMENTUM – Daily 5-20%**\n_Entry limit + support intraday_\n\n"
    for item in results_momentum[:8]:
        if item["score"] < 40:
            continue
        tv = f"https://www.tradingview.com/chart/?symbol=IDX:{item['ticker']}"
        gap_emoji = "🚀" if item["gap_pct"] > 3 else "📈" if item["gap_pct"] > 0 else "➖"
        vol_emoji = "🔥" if item["rvol"] > 5 else "⚡" if item["rvol"] > 3 else "📊"
        funda = f" ({item['fundamental']})" if item["fundamental"] else ""
        rr1 = round((item["tp1"] - item["entry_limit"]) / (item["entry_limit"] - item["stop_loss"]), 2)
        msg1 += (
            f"{gap_emoji} **#{item['ticker']}**{funda} | Score: {item['score']}\n"
            f"┃ Entry: Rp {item['entry_limit']:,.0f}\n"
            f"┃ Stop:  Rp {item['stop_loss']:,.0f} (-{round((1 - item['stop_loss'] / item['entry_limit']) * 100, 1)}%)\n"
            f"┃ TP1: Rp {item['tp1']:,.0f} (+5%)  TP2: Rp {item['tp2']:,.0f} (+10%)\n"
            f"┃ TP3: Rp {item['tp3']:,.0f} (+15%)  R:R = {rr1}\n"
            f"{vol_emoji} RVOL: {item['rvol']}x | Gap: {item['gap_pct']}% | ATR: {item['atr_pct']}%\n"
            f"RSI: {item['rsi']} | Support: Rp {item['support']:,.0f}\n"
            f"[TradingView]({tv})\n\n"
        )
        if len(msg1) > 3800:
            break
    if len(msg1) > 200:
        send(msg1)

    msg2 = "🔄 **REVERSAL EXTREME – Scalping 3-5%**\n_Entry di area oversold_\n\n"
    for item in results_reversal[:6]:
        if item["score"] < 25:
            continue
        tv = f"https://www.tradingview.com/chart/?symbol=IDX:{item['ticker']}"
        rr1 = round((item["tp1"] - item["entry_limit"]) / (item["entry_limit"] - item["stop_loss"]), 2)
        msg2 += (
            f"📉 **#{item['ticker']}** | Score: {item['score']}\n"
            f"┃ Entry: Rp {item['entry_limit']:,.0f}\n"
            f"┃ Stop:  Rp {item['stop_loss']:,.0f} (-{round((1 - item['stop_loss'] / item['entry_limit']) * 100, 1)}%)\n"
            f"┃ TP1: Rp {item['tp1']:,.0f} (+3%)  TP2: Rp {item['tp2']:,.0f} (+6%)\n"
            f"┃ R:R = {rr1}  | RSI: {item['rsi']} | RVOL: {item['rvol']}x\n"
            f"Harga: Rp {item['price']:,.0f}\n"
            f"[TradingView]({tv})\n\n"
        )
        if len(msg2) > 3800:
            break
    if len(msg2) > 200:
        send(msg2)

    if failed_tickers:
        fail_msg = "⚠️ **Gagal di-load**\n\n"
        for ft in failed_tickers[:10]:
            fail_msg += f"  {ft}\n"
        if len(failed_tickers) > 10:
            fail_msg += f"  ...dan {len(failed_tickers) - 10} lainnya"
        send(fail_msg)

    print(f"\n✅ SCAN SELESAI — {datetime.now().strftime('%H:%M')}")
    print(f"  Momentum: {len(results_momentum)} | Reversal: {len(results_reversal)}")


def check_alerts():
    pass


# ── JADWAL WIB ──
schedule.every().day.at("12:00").do(morning_scan)
schedule.every().day.at("18:00").do(morning_scan)

print("=" * 50)
print("NEUROBRO SCANNER – INTRADAY ENTRY ALERT")
print(f"Mulai: {datetime.now().strftime('%H:%M')} WIB")
print("Jadwal:")
print("  - Scan: 12:00 dan 18:00 WIB")
print("  - Alert check: mati")
print("=" * 50)

morning_scan()

while True:
    schedule.run_pending()
    time.sleep(60)
