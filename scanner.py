import os
import csv
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

CSV_FILE = "signals.csv"
ALERT_FILE = "alerts.csv"
STOCK_FILE = "stocks.txt"
SECTOR_FILE = "sectors.csv"

BATCH_SIZE = 3
REQUEST_DELAY = 3

# Approx flow thresholds in IDR value terms
STRONG_ACCUMULATION = 1_000_000_000     # >= Rp 1B net buy pressure
MEDIUM_ACCUMULATION = 250_000_000       # >= Rp 250M
MEDIUM_DISTRIBUTION = -250_000_000      # <= -Rp 250M
STRONG_DISTRIBUTION = -1_000_000_000    # <= -Rp 1B


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


def normalize_columns(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
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
        df = normalize_columns(df)
        if df is None or df.empty or len(df) < 30:
            return None
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
        df = normalize_columns(df)
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

        open_today = float(daily["Open"].iloc[-1])
        close_yest = float(daily["Close"].iloc[-2])
        gap_pct = ((open_today - close_yest) / close_yest) * 100 if close_yest else 0
        recent = df.tail(16)
        support = float(recent["Low"].min())
        current_price = float(df["Close"].iloc[-1])

        return open_today, close_yest, gap_pct, support, current_price
    except Exception as e:
        print(f"fetch_intraday error {symbol}: {e}")
        return None, None, None, None, None


def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        pbv = info.get("priceToBook", None)
        per = info.get("trailingPE", None)
        mcap = info.get("marketCap", None)
        return pbv, per, mcap
    except Exception as e:
        print(f"get_fundamentals error {ticker}: {e}")
        return None, None, None


def flow_score(net_flow_idr):
    """
    net_flow_idr:
    positif  = accumulation / net buy / bullish
    negatif  = distribution / net sell / bearish
    """
    score = 0
    label = "NEUTRAL"

    if net_flow_idr >= STRONG_ACCUMULATION:
        score += 35
        label = "STRONG ACCUMULATION"
    elif net_flow_idr >= MEDIUM_ACCUMULATION:
        score += 20
        label = "ACCUMULATION"
    elif net_flow_idr <= STRONG_DISTRIBUTION:
        score -= 30
        label = "STRONG DISTRIBUTION"
    elif net_flow_idr <= MEDIUM_DISTRIBUTION:
        score -= 15
        label = "DISTRIBUTION"

    return score, label


def get_flow_proxy(df):
    """
    Proxy flow untuk saham Indonesia:
    - positif = akumulasi / tekanan beli
    - negatif = distribusi / tekanan jual
    """
    try:
        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

        price = float(close.iloc[-1])
        vol_now = float(volume.iloc[-1])
        vol_avg20 = float(volume.tail(20).mean())

        if vol_avg20 <= 0 or price <= 0:
            return 0

        rvol = vol_now / vol_avg20
        traded_value_today = price * vol_now
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])

        # Breakout + volume tinggi + harga di atas EMA20 => akumulasi
        if rvol >= 5 and price > ema20:
            return traded_value_today
        elif rvol >= 3 and price > ema20:
            return traded_value_today * 0.5

        # Weakness + volume tinggi + harga di bawah EMA20 => distribusi
        elif rvol >= 4 and price < ema20:
            return -traded_value_today * 0.5
        elif rvol >= 2 and price < ema20:
            return -traded_value_today

        return 0
    except Exception as e:
        print(f"get_flow_proxy error: {e}")
        return 0


def level_entry(item, price, support):
    if item["tipe"] == "breakout":
        entry_limit = max(support, price * 0.995) if support else price * 0.995
        stop = min(support * 0.995 if support else price * 0.97, price * 0.97)
        tp1 = entry_limit * 1.05
        tp2 = entry_limit * 1.10
        tp3 = entry_limit * 1.15
    else:
        entry_limit = price * 0.995
        stop = price * 0.96
        tp1 = price * 1.03
        tp2 = price * 1.06
        tp3 = None

    return round(entry_limit, 0), round(stop, 0), round(tp1, 0), round(tp2, 0), (round(tp3, 0) if tp3 else None)


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


def load_stock_list():
    if not os.path.exists(STOCK_FILE):
        print("stocks.txt tidak ditemukan.")
        return []

    with open(STOCK_FILE, "r", encoding="utf-8") as f:
        stocks = [x.strip() for x in f if x.strip()]

    return [s if s.endswith(".JK") else s + ".JK" for s in stocks]


def load_sector_map():
    sector_map = {}
    if os.path.exists(SECTOR_FILE):
        with open(SECTOR_FILE, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ticker = row.get("ticker", "").upper().replace(".JK", "")
                sector = row.get("sector", "OTHER").upper()
                if ticker:
                    sector_map[ticker] = sector
    return sector_map


def send_chunked(items, kind="momentum"):
    if kind == "momentum":
        msg = "🔥 *BREAKOUT MOMENTUM - ACCUMULATION + BREAKOUT + RVOL*\n\n"
    else:
        msg = "🔄 *REVERSAL EXTREME - SCALPING 3-5%*\n\n"

    for item in items:
        tv = f"https://www.tradingview.com/chart/?symbol=IDX:{item['ticker']}"
        rr1 = round(
            (item["tp1"] - item["entry_limit"]) / (item["entry_limit"] - item["stop_loss"]),
            2
        ) if item["entry_limit"] > item["stop_loss"] else 0

        if kind == "momentum":
            msg += (
                f"• *#{item['ticker']}* | Score {item['score']}\n"
                f"  Flow: {item['flow_label']} | Net Rp {item['flow_idr']:,.0f}\n"
                f"  Entry Rp {item['entry_limit']:,.0f} | SL Rp {item['stop_loss']:,.0f}\n"
                f"  TP1 Rp {item['tp1']:,.0f} | TP2 Rp {item['tp2']:,.0f} | TP3 Rp {item['tp3']:,.0f}\n"
                f"  RVOL {item['rvol']}x | Gap {item['gap_pct']}% | RSI {item['rsi']} | R:R {rr1}\n"
                f"  Support Rp {item['support']:,.0f} | [TradingView]({tv})\n\n"
            )
        else:
            msg += (
                f"• *#{item['ticker']}* | Score {item['score']}\n"
                f"  Flow: {item['flow_label']} | Net Rp {item['flow_idr']:,.0f}\n"
                f"  Entry Rp {item['entry_limit']:,.0f} | SL Rp {item['stop_loss']:,.0f}\n"
                f"  TP1 Rp {item['tp1']:,.0f} | TP2 Rp {item['tp2']:,.0f}\n"
                f"  RVOL {item['rvol']}x | RSI {item['rsi']} | R:R {rr1}\n"
                f"  Harga Rp {item['price']:,.0f} | [TradingView]({tv})\n\n"
            )

        if len(msg) > 3800:
            break

    if len(msg) > 100:
        send(msg)


def morning_scan():
    print(f"\n=== SCAN {datetime.now().strftime('%H:%M')} ===")

    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "date", "time", "ticker", "score",
                "rsi", "price", "rvol", "gap_pct",
                "breakout", "atr_pct", "flow_label",
                "flow_idr", "entry_limit", "stop_loss",
                "tp1", "tp2", "tp3", "signal_type"
            ])

    stocks = load_stock_list()
    if not stocks:
        return

    sector_map = load_sector_map()

    regimes = []
    for ticker, label in [("^JKSE", "IHSG"), ("USDIDR=X", "USDIDR"), ("GC=F", "GOLD"), ("CL=F", "OIL")]:
        result = trend_regime(ticker, label)
        print(f"  {result}")
        regimes.append(result)
        time.sleep(REQUEST_DELAY)

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

            price = float(close.iloc[-1])
            if price <= 0:
                continue

            rsi = RSIIndicator(close, window=4).rsi()
            rsi_now = float(rsi.iloc[-1])
            rsi_prev = float(rsi.iloc[-2])

            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

            avg_vol = float(volume.tail(20).mean())
            daily_value = price * float(volume.iloc[-1])

            if avg_vol <= 0 or daily_value <= 0:
                continue

            rvol = float(volume.iloc[-1]) / avg_vol

            atr = AverageTrueRange(high, low, close, window=14).average_true_range()
            atr_val = float(atr.iloc[-1])
            atr_pct = (atr_val / price) * 100 if price else 0

            prev_20_high = float(close.shift(1).tail(20).max())
            prev_20_low = float(close.shift(1).tail(20).min())
            breakout_high = price > prev_20_high

            open_today, close_yest, gap_pct, support, _ = fetch_intraday(stock)

            range_pct = ((prev_20_high - prev_20_low) / prev_20_low) * 100 if prev_20_low else 0

            pbv, per, mcap = get_fundamentals(stock)
            fundamental_flag = ""
            if pbv and per and mcap:
                if pbv < 2 and per < 25 and 500_000_000_000 < mcap < 50_000_000_000_000:
                    fundamental_flag = "FUNDAMENTAL OK"

            ticker_clean = stock.replace(".JK", "")
            sector = sector_map.get(ticker_clean, "OTHER")

            if atr_pct < 3 or daily_value < 5_000_000_000 or price < 50:
                continue

            proxy_flow_idr = get_flow_proxy(df)
            flow_bonus, flow_label = flow_score(proxy_flow_idr)

            item = {
                "ticker": ticker_clean,
                "rsi": round(rsi_now, 1),
                "price": round(price, 0),
                "rvol": round(rvol, 2),
                "gap_pct": round(gap_pct, 2) if gap_pct is not None else 0,
                "breakout": breakout_high,
                "range_pct": round(range_pct, 1),
                "atr_pct": round(atr_pct, 2),
                "fundamental": fundamental_flag,
                "sector": sector,
                "daily_value": round(daily_value, 0),
                "open_today": round(open_today, 0) if open_today else 0,
                "support": round(support, 0) if support else 0,
                "flow_idr": round(proxy_flow_idr, 0),
                "flow_label": flow_label,
            }

            # BREAKOUT MOMENTUM SCORE - lebih cocok buat saham Indonesia
            score_m = 0

            # Volume must be real, but not too extreme only
            if rvol >= 5:
                score_m += 25
            elif rvol >= 3:
                score_m += 18
            elif rvol >= 2:
                score_m += 10

            # Price structure
            if breakout_high:
                score_m += 18
            if price > ema20:
                score_m += 8
            if ema20 > ema50:
                score_m += 6

            # Flow logic - accumulation supports upside
            score_m += flow_bonus

            # Daily movement quality
            if gap_pct is not None and gap_pct > 0:
                score_m += min(12, gap_pct * 2)
            if 3 < range_pct < 25:
                score_m += 8
            elif range_pct >= 25:
                score_m += 4

            # RSI for momentum continuation, not overbought chase
            if 45 <= rsi_now <= 70 and rsi_now > rsi_prev:
                score_m += 8
            elif rsi_now > 70:
                score_m += 2
            elif rsi_now < 40:
                score_m -= 5

            # Basic fundamental support, not mandatory
            if fundamental_flag:
                score_m += 4

            item["score"] = round(score_m, 1)
            item["tipe"] = "breakout"

            # Jangan terlalu ketat - accumulation hanya penguat, bukan penjegal
            strong_flow_breakout = (
                breakout_high
                and rvol >= 2.5
                and price > ema20
                and score_m >= 30
                and flow_label not in ["STRONG DISTRIBUTION"]
            )

            if strong_flow_breakout:
                el, sl, tp1, tp2, tp3 = level_entry(item, price, support)
                item["entry_limit"] = el
                item["stop_loss"] = sl
                item["tp1"] = tp1
                item["tp2"] = tp2
                item["tp3"] = tp3
                results_momentum.append(item)
                all_setups.append(item)

            # REVERSAL SCORE - oversold bounce
            score_r = 0

            if rsi_now < 22:
                score_r += 30
            elif rsi_now < 28:
                score_r += 20
            elif rsi_now < 35:
                score_r += 10

            if rsi_now > rsi_prev:
                score_r += 10

            if rvol >= 3:
                score_r += 10
            elif rvol >= 2:
                score_r += 6

            if price < ema20:
                score_r += 10
            if not breakout_high:
                score_r += 8

            if fundamental_flag:
                score_r += 4

            # Untuk reversal, distribution berat justru bisa jadi tanda capitulation
            if flow_label in ["DISTRIBUTION", "STRONG DISTRIBUTION"]:
                score_r += 5
            elif flow_label in ["ACCUMULATION", "STRONG ACCUMULATION"]:
                score_r -= 8

            if score_r >= 25:
                item_rev = item.copy()
                item_rev["tipe"] = "reversal"
                item_rev["score"] = round(score_r, 1)
                el, sl, tp1, tp2, tp3 = level_entry(item_rev, price, support)
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
        w.writerow(["ticker", "entry_limit", "tp1", "tp2", "tp3", "stop_loss", "score", "tipe", "flow_label", "flow_idr"])
        for item in all_setups:
            w.writerow([
                item["ticker"], item["entry_limit"], item["tp1"],
                item["tp2"], item["tp3"], item["stop_loss"],
                item["score"], item["tipe"], item["flow_label"], item["flow_idr"]
            ])

    now = datetime.now()
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for item in results_momentum[:10]:
            w.writerow([
                now.strftime("%Y-%m-%d"), now.strftime("%H:%M"),
                item["ticker"], item["score"], item["rsi"], item["price"],
                item["rvol"], item["gap_pct"], item["breakout"],
                item["atr_pct"], item["flow_label"], item["flow_idr"],
                item["entry_limit"], item["stop_loss"],
                item["tp1"], item["tp2"], item["tp3"], item["tipe"]
            ])

    regime_header = "🌏 *MARKET REGIME*\n"
    for line in regimes:
        regime_header += f"• {line}\n"
    regime_header += f"\n• Momentum lolos: {len(results_momentum)}\n"
    regime_header += f"• Reversal lolos: {len(results_reversal)}\n"
    regime_header += f"• Gagal load: {len(failed_tickers)}\n"
    send(regime_header)

    if results_momentum:
        send_chunked(results_momentum[:8], kind="momentum")

    if results_reversal:
        send_chunked(results_reversal[:6], kind="reversal")

    if failed_tickers:
        fail_msg = "⚠️ *GAGAL DI-LOAD*\n\n"
        for ft in failed_tickers[:10]:
            fail_msg += f"• {ft}\n"
        if len(failed_tickers) > 10:
            fail_msg += f"• ...dan {len(failed_tickers) - 10} lainnya"
        send(fail_msg)

    print(f"\n✅ SCAN SELESAI — {datetime.now().strftime('%H:%M')}")
    print(f"Momentum: {len(results_momentum)} | Reversal: {len(results_reversal)}")


if __name__ == "__main__":
    print("=" * 50)
    print("NEUROBRO SCANNER - ACCUMULATION + BREAKOUT + RVOL")
    print(f"Mulai: {datetime.now().strftime('%H:%M')}")
    print("=" * 50)
    morning_scan()
