import os
import csv
import time
import schedule
import requests
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_IDS = [
    6262086905,
    6003751935,
]

CSV_FILE = "signals.csv"
ALERT_FILE = "alerts.csv"
STOCKS_FILE = "stocks.txt"
SECTORS_FILE = "sectors.csv"
ERROR_LOG_FILE = "error.log"

BATCH_SIZE = 3
REQUEST_DELAY = 3
ALERT_EXPIRY_HOURS = 3

MIN_GAP_PCT = 2.0
MAX_GAP_PCT = 5.0

MIN_ATR_PCT = 3.0
MIN_DAILY_VALUE = 5_000_000_000
MIN_PRICE = 50

def log_error(text):
    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {text}\n")
    except Exception:
        pass

def send(msg, parse_mode="Markdown"):
    if not TOKEN:
        print("Telegram token belum diset.")
        return

    chunks = [msg[i:i + 4000] for i in range(0, len(msg), 4000)]
    for chunk in chunks:
        for chat_id in CHAT_IDS:
            for attempt in range(3):
                try:
                    requests.post(
                        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                        json={
                            "chat_id": chat_id,
                            "text": chunk,
                            "disable_web_page_preview": False,
                            "parse_mode": parse_mode,
                        },
                        timeout=10
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        log_error(f"Telegram error chat_id={chat_id} | {e}")
                    time.sleep(2)

def safe_float(value, default=None):
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default

def clean_columns(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

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
        df = clean_columns(df)
        if df is None or df.empty or len(df) < 30:
            return None
        return df
    except Exception as e:
        log_error(f"fetch_data {symbol} | {e}")
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
        df = clean_columns(df)
        if df is None or df.empty or len(df) < 10:
            return None, None, None, None, None

        daily = df.resample("D").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna()

        if len(daily) < 2:
            return None, None, None, None, None

        open_today = safe_float(daily["Open"].iloc[-1])
        close_yest = safe_float(daily["Close"].iloc[-2])

        if open_today is None or close_yest is None or close_yest == 0:
            gap_pct = None
        else:
            gap_pct = ((open_today - close_yest) / close_yest) * 100

        recent_15m = df.tail(16)
        support = safe_float(recent_15m["Low"].min())
        current_price = safe_float(df["Close"].iloc[-1])

        return open_today, close_yest, gap_pct, support, current_price
    except Exception as e:
        log_error(f"fetch_intraday {symbol} | {e}")
        return None, None, None, None, None

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
        log_error(f"get_fundamentals {ticker} | {e}")
        return None, None, None

def score_to_label(score):
    if score >= 60:
        return "🟢 BUY"
    elif score >= 40:
        return "🟡 HOLD"
    return "🔴 SKIP"

def compute_stop_loss(df, entry_price, gap_pct):
    try:
        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()

        atr = AverageTrueRange(high, low, close, window=14).average_true_range()
        atr_val = safe_float(atr.iloc[-1])
        swing_low_10 = safe_float(low.tail(10).min())

        if atr_val is None or swing_low_10 is None:
            return None, None

        buffer = 0.5 * atr_val
        stop_loss = swing_low_10 - buffer

        if stop_loss >= entry_price:
            stop_loss = entry_price - (1.0 * atr_val)

        return safe_float(atr_val), safe_float(stop_loss)
    except Exception as e:
        log_error(f"compute_stop_loss | {e}")
        return None, None

def compute_targets(entry_price, stop_loss, gap_pct):
    risk = entry_price - stop_loss
    if risk <= 0:
        return None, None, None

    tp1 = entry_price + (1.5 * risk)
    tp2 = entry_price + (2.5 * risk)
    tp3 = entry_price + (3.5 * risk)

    return tp1, tp2, tp3

def level_entry_breakout(item, price, open_today, support, df):
    gap = safe_float(item.get("gap_pct"), 0) or 0

    if not (MIN_GAP_PCT <= gap <= MAX_GAP_PCT):
        return None

    if support and support < price:
        entry_limit = max(support, price * 0.99)
    else:
        entry_limit = min(open_today if open_today else price, price * 0.995)

    atr_val, stop_loss = compute_stop_loss(df, entry_limit, gap)
    if atr_val is None or stop_loss is None:
        return None

    tp1, tp2, tp3 = compute_targets(entry_limit, stop_loss, gap)
    if tp1 is None:
        return None

    return (
        round(entry_limit, 0),
        round(stop_loss, 0),
        round(tp1, 0),
        round(tp2, 0),
        round(tp3, 0),
        round(atr_val, 2)
    )

def level_entry_reversal(price, support, df):
    if support and support < price:
        entry_limit = support
    else:
        entry_limit = price * 0.99

    atr_val, stop_loss = compute_stop_loss(df, entry_limit, 0)
    if atr_val is None or stop_loss is None:
        return None

    tp1, tp2, tp3 = compute_targets(entry_limit, stop_loss, 0)
    if tp1 is None:
        return None

    return (
        round(entry_limit, 0),
        round(stop_loss, 0),
        round(tp1, 0),
        round(tp2, 0),
        round(tp3, 0),
        round(atr_val, 2)
    )

def read_stocks():
    if not os.path.exists(STOCKS_FILE):
        print(f"{STOCKS_FILE} tidak ditemukan.")
        return []

    with open(STOCKS_FILE, "r", encoding="utf-8") as f:
        stocks = [x.strip() for x in f if x.strip()]

    return [s if s.endswith(".JK") else s + ".JK" for s in stocks]

def read_sector_map():
    sector_map = {}
    if not os.path.exists(SECTORS_FILE):
        return sector_map

    with open(SECTORS_FILE, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ticker = row.get("ticker", "").upper().replace(".JK", "")
            sector = row.get("sector", "").upper()
            if ticker:
                sector_map[ticker] = sector
    return sector_map

def init_csv_files():
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "date", "time", "ticker", "score", "signal",
            "rsi", "price", "rvol", "gap_pct",
            "atr_pct", "entry_limit", "stop_loss",
            "tp1", "tp2", "tp3"
        ])

    with open(ALERT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ticker", "entry_limit", "tp1", "tp2", "tp3",
            "stop_loss", "score", "signal", "scan_time", "gap_pct"
        ])

def append_signals_csv(rows):
    now = datetime.now()
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for item in rows:
            writer.writerow([
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M"),
                item["ticker"],
                item["score"],
                item["signal"],
                item["rsi"],
                item["price"],
                item["rvol"],
                item["gap_pct"],
                item["atr_pct"],
                item["entry_limit"],
                item["stop_loss"],
                item["tp1"],
                item["tp2"],
                item["tp3"],
            ])

def morning_scan():
    print(f"\n=== MORNING SCAN {datetime.now().strftime('%H:%M')} ===")
    init_csv_files()

    stocks = read_stocks()
    if not stocks:
        print("Tidak ada stock list.")
        return

    sector_map = read_sector_map()

    regime_lines = []
    for ticker, label in [
        ("^JKSE", "IHSG"),
        ("USDIDR=X", "USDIDR"),
        ("GC=F", "GOLD"),
        ("CL=F", "OIL"),
    ]:
        result = trend_regime(ticker, label)
        print(f"  {result}")
        regime_lines.append(result)
        time.sleep(REQUEST_DELAY)

    ihsg_line = regime_lines[0]
    scan_time = datetime.now()

    results_momentum = []
    results_reversal = []
    failed_tickers = []
    all_setups = []

    for i, stock in enumerate(stocks):
        print(f"Scanning {i + 1}/{len(stocks)}: {stock}")
        df = fetch_data(stock, period="6mo")
        if df is None:
            failed_tickers.append(stock)
            continue

        try:
            close = df["Close"].squeeze()
            high = df["High"].squeeze()
            low = df["Low"].squeeze()
            volume = df["Volume"].squeeze()

            price = safe_float(close.iloc[-1])
            if price is None:
                failed_tickers.append(stock)
                continue

            rsi = RSIIndicator(close, window=4).rsi()
            rsi_now = safe_float(rsi.iloc[-1])
            rsi_prev = safe_float(rsi.iloc[-2])

            ema20 = safe_float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = safe_float(close.ewm(span=50, adjust=False).mean().iloc[-1])

            avg_vol = safe_float(volume.tail(20).mean())
            daily_value = price * safe_float(volume.iloc[-1], 0)

            if avg_vol is None or avg_vol <= 0 or price <= 0:
                continue

            rvol = safe_float(volume.iloc[-1], 0) / avg_vol

            atr = AverageTrueRange(high, low, close, window=14).average_true_range()
            atr_val = safe_float(atr.iloc[-1])
            atr_pct = (atr_val / price * 100) if atr_val is not None and price else None
            if atr_pct is None:
                continue

            prev_20_high = safe_float(close.shift(1).tail(20).max())
            prev_20_low = safe_float(close.shift(1).tail(20).min())
            breakout_high = safe_float(close.iloc[-2]) > prev_20_high if prev_20_high is not None else False

            open_today, close_yest, gap_pct, support, _ = fetch_intraday(stock)
            if gap_pct is None:
                gap_pct = 0

            range_pct = ((prev_20_high - prev_20_low) / prev_20_low * 100) if prev_20_high and prev_20_low else 0

            pbv, per, mcap = get_fundamentals(stock.replace(".JK", ""))
            fundament_flag = ""
            if pbv is not None and per is not None and mcap is not None:
                if pbv < 2 and per < 25 and 500_000_000_000 < mcap < 50_000_000_000_000:
                    fundament_flag = "FUNDAMENTAL OK"

            ticker_clean = stock.replace(".JK", "")
            sector = sector_map.get(ticker_clean, "OTHER")

            if atr_pct < MIN_ATR_PCT or daily_value < MIN_DAILY_VALUE or price < MIN_PRICE:
                continue

            item = {
                "ticker": ticker_clean,
                "rsi": round(rsi_now, 1) if rsi_now is not None else 0,
                "price": round(price, 0),
                "rvol": round(rvol, 2),
                "gap_pct": round(gap_pct, 2),
                "breakout": breakout_high,
                "range_pct": round(range_pct, 1),
                "atr_pct": round(atr_pct, 2),
                "fundamental": fundament_flag,
                "sector": sector,
                "daily_value": round(daily_value, 0),
                "open_today": round(open_today, 0) if open_today else 0,
                "support": round(support, 0) if support else 0,
                "scan_time": scan_time.strftime("%Y-%m-%d %H:%M"),
            }

            score_m = 0
            if MIN_GAP_PCT <= gap_pct <= MAX_GAP_PCT:
                score_m += 30
            if rvol > 5:
                score_m += 25
            elif rvol > 3:
                score_m += 15
            elif rvol > 2:
                score_m += 8

            if breakout_high:
                score_m += 10

            if rsi_now is not None and rsi_prev is not None:
                if rsi_now > 50 and rsi_now > rsi_prev:
                    score_m += 10
                elif rsi_now > 40 and rsi_now > rsi_prev:
                    score_m += 5

            if 3 < range_pct < 25:
                score_m += 10

            if "🟢 BULLISH" in ihsg_line:
                score_m += 10

            if ema20 is not None and price > ema20:
                score_m += 5
            if ema20 is not None and ema50 is not None and ema20 > ema50:
                score_m += 5

            signal_m = score_to_label(score_m)

            if score_m >= 40 and MIN_GAP_PCT <= gap_pct <= MAX_GAP_PCT:
                levels = level_entry_breakout(item, price, item["open_today"], item["support"], df)
                if levels:
                    el, sl, tp1, tp2, tp3, atr_used = levels
                    item["entry_limit"] = el
                    item["stop_loss"] = sl
                    item["tp1"] = tp1
                    item["tp2"] = tp2
                    item["tp3"] = tp3
                    item["atr_used"] = atr_used
                    item["signal"] = "BREAKOUT"
                    item["score"] = round(score_m, 1)
                    results_momentum.append(item)
                    all_setups.append(item)

            score_r = 0
            if rsi_now is not None:
                if rsi_now < 22:
                    score_r += 30
                elif rsi_now < 28:
                    score_r += 20
                elif rsi_now < 35:
                    score_r += 10

            if rsi_now is not None and rsi_prev is not None:
                score_r += max(0, min(20, (rsi_now - rsi_prev) * 5))

            score_r += min(15, rvol * 3)

            if ema20 is not None and price < ema20:
                score_r += 10
            if not breakout_high:
                score_r += 10
            if fundament_flag:
                score_r += 5
            if "🟢 BULLISH" in ihsg_line:
                score_r += 5

            if score_r >= 25:
                levels = level_entry_reversal(price, item["support"], df)
                if levels:
                    el, sl, tp1, tp2, tp3, atr_used = levels
                    item_rev = item.copy()
                    item_rev["entry_limit"] = el
                    item_rev["stop_loss"] = sl
                    item_rev["tp1"] = tp1
                    item_rev["tp2"] = tp2
                    item_rev["tp3"] = tp3
                    item_rev["atr_used"] = atr_used
                    item_rev["signal"] = "REVERSAL"
                    item_rev["score"] = round(score_r, 1)
                    results_reversal.append(item_rev)
                    all_setups.append(item_rev)

        except Exception as e:
            failed_tickers.append(f"{stock}: {e}")
            log_error(f"{stock} | {e}")
            continue

        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(REQUEST_DELAY)

    results_momentum = sorted(results_momentum, key=lambda x: x["score"], reverse=True)
    results_reversal = sorted(results_reversal, key=lambda x: x["score"], reverse=True)

    with open(ALERT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "ticker", "entry_limit", "tp1", "tp2", "tp3",
            "stop_loss", "score", "signal", "scan_time", "gap_pct"
        ])
        for item in all_setups:
            w.writerow([
                item["ticker"],
                item["entry_limit"],
                item["tp1"],
                item["tp2"],
                item["tp3"],
                item["stop_loss"],
                item["score"],
                item["signal"],
                item["scan_time"],
                item["gap_pct"],
            ])

    append_signals_csv(results_momentum[:10])

    # Kirim market regime
    regime_header = "🌏 **MARKET REGIME**\n"
    for line in regime_lines:
        regime_header += f"  {line}\n"
    regime_header += f"\n  Momentum: {len(results_momentum)} | Reversal: {len(results_reversal)}\n"
    regime_header += f"  Gagal: {len(failed_tickers)}\n"
    send(regime_header)

    # Kirim breakout signals
    msg1 = "🔥 **BREAKOUT MOMENTUM - GAP 2-5%**\n_ATR + swing low stop loss_\n\n"
    for item in results_momentum[:8]:
        tv = f"https://www.tradingview.com/chart/?symbol=IDX:{item['ticker']}"
        rr1 = round((item["tp1"] - item["entry_limit"]) / (item["entry_limit"] - item["stop_loss"]), 2) if item["entry_limit"] > item["stop_loss"] else 0

        msg1 += (
            f"🟢 **#{item['ticker']}** | {item['signal']} | Score: {item['score']}\n"
            f"┃ Entry: Rp {item['entry_limit']:,.0f}\n"
            f"┃ Stop:  Rp {item['stop_loss']:,.0f}\n"
            f"┃ TP1: Rp {item['tp1']:,.0f}  TP2: Rp {item['tp2']:,.0f}\n"
            f"┃ TP3: Rp {item['tp3']:,.0f}  R:R = {rr1}\n"
            f"RVOL: {item['rvol']}x | Gap: {item['gap_pct']}% | ATR: {item['atr_pct']}%\n"
            f"RSI: {item['rsi']} | Support: Rp {item['support']:,.0f}\n"
            f"[TradingView]({tv})\n\n"
        )
        if len(msg1) > 3800:
            break

    if len(msg1) > 200:
        send(msg1)

    # Kirim reversal signals
    msg2 = "🔄 **REVERSAL EXTREME - ATR STOP**\n_Entry di support, stop di bawah swing low_\n\n"
    for item in results_reversal[:6]:
        tv = f"https://www.tradingview.com/chart/?symbol=IDX:{item['ticker']}"
        rr1 = round((item["tp1"] - item["entry_limit"]) / (item["entry_limit"] - item["stop_loss"]), 2) if item["entry_limit"] > item["stop_loss"] else 0

        msg2 += (
            f"📉 **#{item['ticker']}** | {item['signal']} | Score: {item['score']}\n"
            f"┃ Entry: Rp {item['entry_limit']:,.0f}\n"
            f"┃ Stop:  Rp {item['stop_loss']:,.0f}\n"
            f"┃ TP1: Rp {item['tp1']:,.0f}  TP2: Rp {item['tp2']:,.0f}\n"
            f"┃ TP3: Rp {item['tp3']:,.0f}  R:R = {rr1}\n"
            f"RSI: {item['rsi']} | RVOL: {item['rvol']}x | ATR: {item['atr_pct']}%\n"
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

    print(f"\n✅ MORNING SCAN SELESAI — {datetime.now().strftime('%H:%M')}")
    print(f"  Momentum: {len(results_momentum)} | Reversal: {len(results_reversal)}")

def check_alerts():
    """
    Alert checker: hanya maintain file CSV dengan cleanup setup expired.
    TIDAK mengirim notifikasi Telegram.
    """
    print(f"  ⌛ Alert check {datetime.now().strftime('%H:%M')}...")

    if not os.path.exists(ALERT_FILE):
        return

    with open(ALERT_FILE, "r", encoding="utf-8") as f:
        setups = list(csv.DictReader(f))

    if not setups:
        return

    now = datetime.now()
    active_setups = []

    for setup in setups:
        ticker = setup.get("ticker", "") + ".JK"
        entry = safe_float(setup.get("entry_limit"))
        stop = safe_float(setup.get("stop_loss"))
        scan_time_str = setup.get("scan_time", "")
        ticker_clean = setup.get("ticker", "")

        if entry is None:
            continue

        # Cek expired
        if scan_time_str:
            try:
                scan_dt = datetime.strptime(scan_time_str, "%Y-%m-%d %H:%M")
                if (now - scan_dt) > timedelta(hours=ALERT_EXPIRY_HOURS):
                    print(f"  ⏰ {ticker_clean} expired")
                    continue
            except Exception:
                pass

        try:
            df = yf.download(
                ticker,
                period="1d",
                interval="15m",
                auto_adjust=True,
                progress=False,
                timeout=10
            )
            df = clean_columns(df)
            if df is None or df.empty:
                active_setups.append(setup)
                continue

            current = safe_float(df["Close"].iloc[-1])
            low_today = safe_float(df["Low"].min())

            if current is None or low_today is None:
                active_setups.append(setup)
                continue

            # Setup yang entry-nya sudah kena: tetap dipertahankan di file
            if low_today <= entry <= current:
                active_setups.append(setup)
                continue

            # Harga udah jauh di atas entry + 2x risk? Expire
            risk = (stop and entry - stop) or entry * 0.02
            if current > entry + risk * 2:
                print(f"  ⏰ {ticker_clean} harga sudah jauh ({current:,.0f}), expire")
                continue

            active_setups.append(setup)

        except Exception as e:
            log_error(f"check_alerts {ticker} | {e}")
            active_setups.append(setup)
            continue

    # Tulis ulang file CSV dengan setup yang masih aktif
    with open(ALERT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "ticker", "entry_limit", "tp1", "tp2", "tp3",
            "stop_loss", "score", "signal", "scan_time", "gap_pct"
        ])
        for s in active_setups:
            w.writerow([
                s.get("ticker", ""),
                s.get("entry_limit", ""),
                s.get("tp1", ""),
                s.get("tp2", ""),
                s.get("tp3", ""),
                s.get("stop_loss", ""),
                s.get("score", ""),
                s.get("signal", ""),
                s.get("scan_time", ""),
                s.get("gap_pct", ""),
            ])

    print(f"  ✅ Alert check selesai - {len(active_setups)} aktif, {len(setups) - len(active_setups)} expired/dihapus")

def schedule_jobs():
    schedule.clear()
    schedule.every().day.at("08:30").do(morning_scan)
    schedule.every().day.at("09:15").do(morning_scan)
    schedule.every().day.at("10:30").do(morning_scan)

    for hour in range(9, 17):
        for minute in [0, 15, 30, 45]:
            schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(check_alerts)

def main():
    print("=" * 50)
    print("NEUROBRO SCANNER - INTRADAY ENTRY ALERT")
    print(f"Mulai: {datetime.now().strftime('%H:%M')} WIB")
    print("Jadwal:")
    print("  - Morning scan: 08:30, 09:15, 10:30")
    print("  - Alert check: tiap 15 menit (09:00-16:00)")
    print("  - NOTIF ENTRY: OFF (hanya simpan setup di CSV)")
    print("=" * 50)

    schedule_jobs()
    morning_scan()

    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
