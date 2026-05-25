import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from flask import jsonify
from dotenv import load_dotenv
from flask import Flask, render_template, request

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


load_dotenv()

app = Flask(__name__)

USE_ALPACA = True
USE_POLYGON_FALLBACK = True

AI_ANALYSIS_CACHE = {}
AI_ANALYSIS_TTL = 4 * 60 * 60  # 4小时

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY) if OpenAI and OPENAI_API_KEY else None

# 先用一批高流动性股票，避免太多垃圾信号
UNIVERSE = [
    # ===== Mega Cap AI =====
    "NVDA","MSFT","META","AMZN","GOOGL","GOOG","AAPL",

    # ===== AI / Software =====
    "PLTR","SNOW","NOW","CRM","ORCL","ADBE","INTU",

    # ===== Semiconductor =====
    "AVGO","AMD","MU","QCOM","LRCX","AMAT","KLAC",
    "ASML","TSM","ARM","INTC",

    # ===== Momentum / SPMO =====
    "JPM","GE","CAT","XOM","NFLX","CSCO",

    # ===== AI Infra / Cloud =====
    "ANET","PANW","CRWD","DDOG","MDB","NET",
    "GLW","OKLO","POET","IREN","NBIS","LITE",

    # ===== Robotics / Future AI =====
    "ISRG","TSLA","RXRX",

    # 医药 / 医疗
    "LLY","NVO","VRTX","REGN",
    "UNH","ABBV","MRK","TMO","DHR",

    # 太空 / 国防
    "RKLB","LUNR","ASTS","PL","SPIR",
    "KTOS","AVAV","RTX","LMT","NOC"

    # ===== ETF / Market =====
    "QQQ","SPY","SMH","IGV","SPMO"
]

MARKET_TICKERS = ["SPY", "QQQ"]

CACHE = {}
CACHE_TTL = 60 

MARKET_STATUS_CACHE = {
    "data": None,
    "time": 0
}

MARKET_CACHE_TTL = 600  # 10分鐘

NEWS_CACHE = {}
NEWS_CACHE_TTL = 1800


def summarize_news_with_ai(title, summary):

    try:

        prompt = f"""
请用繁体中文总结以下股票新闻。

要求：
1. 用投资者风格
2. 简短
3. 开头加 emoji
4. 用 3 个重点
5. 不要超过 80 字

标题：
{title}

内容：
{summary}
"""

        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_completion_tokens=120
        )

        return response.choices[0].message.content

    except Exception as e:
        print("AI 新闻总结失败:", e)
        return summary


def get_insider_data(ticker):   

    try:

        url = "https://finnhub.io/api/v1/stock/insider-transactions"

        params = {
            "symbol": ticker.upper(),
            "token": FINNHUB_API_KEY
        }

        r = requests.get(url, params=params, timeout=10)
        js = r.json()

        raw_data = js.get("data", [])

        grouped = {}

        from datetime import datetime, timedelta

        # 只保留最近180天
        cutoff = datetime.now() - timedelta(days=180)

        hist = get_history(ticker, days=220)

        close_map = {}

        if hist is not None and not hist.empty:
            hist = hist.copy()

            hist.index = hist.index.strftime("%Y-%m-%d")

            for idx, row in hist.iterrows():
                close_map[idx] = float(row["Close"])

        buy_count = 0
        sell_count = 0
        buy_value = 0
        sell_value = 0

        for d in raw_data:

            date_str = d.get("transactionDate", "")

            try:
                tx_date = datetime.strptime(date_str, "%Y-%m-%d")
                if tx_date < cutoff:
                    continue
            except:
                continue

            name = d.get("name", "")
            title = (d.get("title") or "").upper()

            change = d.get("change", 0) or 0
            share = d.get("share", 0) or 0
            price = (
                d.get("transactionPrice")
                or d.get("price")
                or d.get("transactionPricePerShare")
                or None
            )

            is_estimated = False

            if price:
                price = round(float(price), 2)
            else:
                price = close_map.get(date_str)

                if price:
                    price = round(float(price), 2)
                    is_estimated = True
                else:
                    price = None

            key = f"{name}_{date_str}"

            if key not in grouped:
                grouped[key] = {
                    "name": name,
                    "title": title,
                    "transactionDate": date_str,
                    "buy": 0,
                    "sell": 0,
                    "net": 0,
                    "share": share,
                    "price": price,
                    "estimated": is_estimated
                }

           # ===== summary =====

            if change > 0:
                grouped[key]["buy"] += int(change)

                buy_count += 1

                try:
                    price_value = float(price)
                except:
                    price_value = 0

                if price_value > 0:
                    buy_value += abs(change) * price_value

            elif change < 0:
                grouped[key]["sell"] += int(abs(change))

                sell_count += 1

                try:
                    price_value = float(price)
                except:
                    price_value = 0

                if price_value > 0:
                    sell_value += abs(change) * price_value

            grouped[key]["share"] = share
            grouped[key]["net"] = grouped[key]["buy"] - grouped[key]["sell"]

        summary = {
            "buy_count": buy_count,
            "sell_count": sell_count,
            "buy_value": buy_value,
            "sell_value": sell_value,
            "total_value": buy_value + sell_value
        }

        return {
            "data": list(grouped.values())[:20],
            "summary": summary
        }

    except Exception as e:

        print("Insider error:", e)

        return []


def get_institution_data(ticker):

    try:
        import pandas as pd

        path = "data/13f_latest.csv"
        df = pd.read_csv(path)

        print(df.columns)

        ticker_col = "Ticker" if "Ticker" in df.columns else "ticker"
        name_col = "NameOfIssuer" if "NameOfIssuer" in df.columns else "nameofissuer"
        shares_col = "sshPrnamt" if "sshPrnamt" in df.columns else "shares"

        rows = df[df[ticker_col].astype(str).str.upper() == ticker.upper()]

        result = []

        for _, r in rows.head(10).iterrows():

            result.append({
                "name": str(r.get(name_col, "Institution")),
                "shares": int(float(r.get(shares_col, 0))),
                "change": 0,
                "date": "13F 最新季度"
            })

        return result

    except Exception as e:
        print("SEC 13F error:", e)
        return []


def get_company_news(ticker, limit=5):

    cache_key = f"{ticker}_news_{limit}"

    cached = NEWS_CACHE.get(cache_key)

    if cached and time.time() - cached["time"] < NEWS_CACHE_TTL:
        print(f"📰 Using cached news for {ticker}")
        return cached["data"]

    try:
        today = datetime.now().strftime("%Y-%m-%d")

        url = "https://finnhub.io/api/v1/company-news"

        params = {
            "symbol": ticker.upper(),
            "from": today,
            "to": today,
            "token": FINNHUB_API_KEY
        }

        r = requests.get(url, params=params, timeout=10)

        js = r.json()

        news = []

        for item in js[:limit]:

            raw_summary = item.get("summary", "")

            ai_summary = summarize_news_with_ai(
                item.get("headline", ""),
                raw_summary
            )

            news.append({
                "title": item.get("headline", ""),
                "publisher": item.get("source", ""),
                "url": item.get("url", ""),
                "time": datetime.fromtimestamp(
                    item.get("datetime", 0)
                ).strftime("%Y-%m-%d"),
                "summary": ai_summary
            })

        NEWS_CACHE[cache_key] = {
            "time": time.time(),
            "data": news
        }

        return news

    except Exception as e:
        print("Finnhub news error:", e)
        return []


def get_real_time_price(ticker):
    try:
        data = get_history(ticker, days=2)

        if data is None or data.empty:
            return None

        return float(data["Close"].iloc[-1])

    except Exception as e:
        print("Real price error:", e)
        return None
    

def cache_get(key):
    item = CACHE.get(key)
    if not item:
        return None
    ts, value = item
    if time.time() - ts > CACHE_TTL:
        return None
    return value


def cache_set(key, value):
    CACHE[key] = (time.time(), value)


def get_polygon_daily_bars(ticker: str, days: int = 520):
    ticker = ticker.upper().strip()

    if not POLYGON_API_KEY:
        return None, "缺少 POLYGON_API_KEY，请先在 .env 里加入你的 Polygon API Key"

    cache_key = f"daily:{ticker}:{days}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached, None

    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 5000,
        "apiKey": POLYGON_API_KEY
    }

    try:
        r = requests.get(url, params=params, timeout=60)

        if r.status_code != 200:
            return None, f"Polygon API 错误 {r.status_code}: {r.text[:200]}"

        js = r.json()
        results = js.get("results", [])

        if not results:
            return None, f"{ticker} 没有返回K线数据"

        df = pd.DataFrame(results)
        df["Date"] = pd.to_datetime(df["t"], unit="ms")

        df = df.rename(columns={
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume"
        })

        df = df.set_index("Date")
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        df = df.dropna()

        cache_set(cache_key, df)
        return df, None
    
    except requests.exceptions.Timeout:
        return None, "Polygon 连接超时，请等30秒再试，或检查网络。"

    except Exception as e:
        return None, f"读取 Polygon 数据失败：{e}"

def calc_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    return round(float(atr.iloc[-1]), 2)


def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist


def add_indicators(df):
    df = df.copy()

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA120"] = df["Close"].rolling(120).mean()

    df["RSI"] = calc_rsi(df["Close"])
    df["MACD_DIF"], df["MACD_DEA"], df["MACD_HIST"] = calc_macd(df["Close"])

    df["BB_MID"] = df["Close"].rolling(25).mean()
    std = df["Close"].rolling(25).std()

    df["BB_UPPER"] = df["BB_MID"] + 2.2 * std
    df["BB_LOWER"] = df["BB_MID"] - 2.2 * std

    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["HIGH_20"] = df["High"].rolling(20).max()
    df["LOW_20"] = df["Low"].rolling(20).min()
    df["HIGH_60"] = df["High"].rolling(60).max()
    df["LOW_60"] = df["Low"].rolling(60).min()

    return df


def fmt_money(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "-"


def fmt_pct(x):
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return "-"


def fmt_num(x):
    try:
        x = float(x)
        if x >= 1_000_000_000:
            return f"{x / 1_000_000_000:.2f}B"
        if x >= 1_000_000:
            return f"{x / 1_000_000:.2f}M"
        if x >= 1_000:
            return f"{x / 1_000:.1f}K"
        return f"{x:.0f}"
    except Exception:
        return "-"


def get_market_filter():
    """
    大盘过滤：
    SPY > MA50 且 QQQ > MA50，做多环境较好。
    否则减少交易或只观察。
    """
    spy_df, spy_err = get_polygon_daily_bars("SPY")
    qqq_df, qqq_err = get_polygon_daily_bars("QQQ")

    if spy_err or qqq_err:
        return {
            "status": "unknown",
            "label": "无法判断",
            "message": "大盘数据读取失败，暂时不建议自动交易。",
            "allow_long": False
        }

    spy = add_indicators(spy_df).dropna()
    qqq = add_indicators(qqq_df).dropna()

    spy_last = spy.iloc[-1]
    qqq_last = qqq.iloc[-1]

    spy_ok = spy_last["Close"] > spy_last["MA50"] and spy_last["MACD_HIST"] > 0
    qqq_ok = qqq_last["Close"] > qqq_last["MA50"] and qqq_last["MACD_HIST"] > 0

    if spy_ok and qqq_ok:
        return {
            "status": "green",
            "label": "绿色：可以偏进攻",
            "message": "SPY 和 QQQ 都在 MA50 上方，且 MACD 动能偏正，做多环境较好。",
            "allow_long": True
        }

    if spy_last["Close"] > spy_last["MA50"] or qqq_last["Close"] > qqq_last["MA50"]:
        return {
            "status": "yellow",
            "label": "黄色：只做高质量机会",
            "message": "大盘部分转强，但未完全确认。建议只做高分股票，控制仓位。",
            "allow_long": True
        }

    return {
        "status": "red",
        "label": "红色：防守为主",
        "message": "SPY / QQQ 趋势偏弱，自动选股信号降级，建议少做或不做。",
        "allow_long": False
    }


alpaca_client = StockHistoricalDataClient(
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY
)

def get_history(ticker, days=180, interval="1d"):
    try:
        # ===== Alpaca =====
        end = datetime.now() - timedelta(minutes=16)
        start = end - timedelta(days=days)

        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute if interval == "5m" else TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )

        bars = alpaca_client.get_stock_bars(request).df

        if bars.empty:
            raise Exception("Alpaca empty")

        if "symbol" in bars.index.names:
            df = bars.xs(ticker, level="symbol")
        else:
            df = bars

        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })

        print(f"✅ Using ALPACA for {ticker}")
        return df

    except Exception as e:
        print(f"⚠️ Alpaca failed: {ticker}", e)

        # ===== Polygon fallback =====
        if USE_POLYGON_FALLBACK:
            try:
                df, err = get_polygon_daily_bars(ticker, days)
                if err:
                    raise Exception(err)
                print(f"🟡 Using POLYGON for {ticker}")
                return df
            except Exception as e2:
                print(f"❌ Polygon failed: {ticker}", e2)

        return None


def get_money_flow(ticker):

    try:

        bars = get_history(ticker, days=1, interval="5m")

        if bars is None or bars.empty:
            return {
                "buy_flow": "$0",
                "sell_flow": "$0",
                "signal": "暂无数据"
            }

        buy_flow = 0
        sell_flow = 0

        for idx, row in bars.iterrows():

            open_price = float(row["Open"])
            close_price = float(row["Close"])
            volume = float(row["Volume"])

            avg_price = (open_price + close_price) / 2

            money = avg_price * volume

            # 上涨K线 → 买盘
            if close_price > open_price:
                buy_flow += money

            # 下跌K线 → 卖盘
            elif close_price < open_price:
                sell_flow += money

        # format
        def fmt(v):

            if v >= 1_000_000_000:
                return f"${v/1_000_000_000:.2f}B"

            elif v >= 1_000_000:
                return f"${v/1_000_000:.2f}M"

            return f"${v:,.0f}"

        # signal
        if buy_flow > sell_flow * 1.2:
            signal = "🔥 资金偏多"

        elif sell_flow > buy_flow * 1.2:
            signal = "⚠️ 资金流出"

        else:
            signal = "🤝 多空平衡"

        return {
            "buy_flow": fmt(buy_flow),
            "sell_flow": fmt(sell_flow),
            "signal": signal
        }

    except Exception as e:

        print("资金流错误:", e)

        return {
            "buy_flow": "$0",
            "sell_flow": "$0",
            "signal": "暂无数据"
        }


def analyze_ticker(ticker):
    display_ticker = "VIX" if ticker == "I:VIX" else ticker
    df, err = get_polygon_daily_bars(ticker)
    if err:
        return None, None, None, err

    if df is None or len(df) < 150:
        return None, None, None, "数据不足，至少需要150根日K"

    df = add_indicators(df).dropna()

    if df.empty:
        return None, None, None, "技术指标计算失败"

    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(last["Close"])
    if ticker == "I:VIX":
        real_price = None
    else:
        real_price = get_real_time_price(ticker)

    price = real_price if real_price else close
    prev_close = float(prev["Close"])
    change_pct = (price / prev_close - 1) * 100 if price and prev_close else 0
    change = price - prev_close
    change_val = f"{change:+.2f}"
    change_val = price - prev_close
    try:
        info = yf.Ticker(ticker).info
    except:
        info = {}

    pre_price = None
    post_price = None
    pre_change = None
    pre_pct = None
    session = None
    session_price = None
    diff = None
    pct = None
    open_signal = "暂无盘前数据"

    pre_price = info.get("preMarketPrice", None)
    post_price = info.get("postMarketPrice")

    if pre_price:
        session = "盘前"
        session_price = pre_price
    elif post_price:
        session = "盘后"
        session_price = post_price

    diff = None
    pct = None

    if session_price and price:
        diff = session_price - price
        pct = round((diff / price) * 100, 2)
    else:
        diff = None
        pct = None    

    try:
        vix_card = get_vix_card()
        vix_price = float(vix_card["price"])
    except:
        vix_price = 16    

    # === 开盘预判（用盘前数据）===
    open_signal = "暂无盘前数据"

    if pre_pct is not None:
        if pre_pct <= -2 and vix_price >= 20:
            open_signal = "🔴 低开风险高，可能继续走弱"
        elif pre_pct <= -2:
            open_signal = "⚠️ 低开概率高，观察是否反弹"
        elif pre_pct >= 2 and vix_price < 18:
            open_signal = "🟢 高开偏强，可能延续上涨"
        elif pre_pct >= 2:
            open_signal = "⚠️ 高开但偏热，小心回落"
        elif -1 <= pre_pct <= 1:
            open_signal = "😐 平开概率大，等方向选择"
        else:
            open_signal = "👀 有波动，开盘再确认"    

    rsi = float(last["RSI"])
    macd_hist = float(last["MACD_HIST"])
    ma20 = float(last["MA20"])
    ma50 = float(last["MA50"])
    ma60 = float(last["MA60"])
    ma120 = float(last["MA120"])
    bb_upper = float(last["BB_UPPER"])
    bb_mid = float(last["BB_MID"])
    bb_lower = float(last["BB_LOWER"])
    vol = float(last["Volume"])
    vol20 = float(last["VOL20"])
    money_flow = get_money_flow(ticker)
    atr = calc_atr(df)

    high_52w = float(df["High"].tail(252).max())
    low_52w = float(df["Low"].tail(252).min())

    future_low = round(price - atr * 5, 2)
    future_high = round(price + atr * 5, 2)

    safe_low = round(price - atr * 2, 2)
    safe_high = round(price + atr * 2, 2)

    extreme_low = round(price - atr * 10, 2)
    extreme_high = round(price + atr * 10, 2)

    atr_pct = round((atr / price) * 100, 2)

    if atr_pct < 2:
        volatility_level = "低波动"

    elif atr_pct < 5:
        volatility_level = "中等波动"

    else:
        volatility_level = "高波动"

    score = 0
    notes = []

    # 趋势
    if close > ma20:
        score += 1
        notes.append("价格站上 MA20，短线趋势较好")
    else:
        notes.append("价格未站稳 MA20，短线仍需观察")

    if close > ma60:
        score += 1
        notes.append("价格站上 MA60，中期趋势未破坏")
    else:
        notes.append("价格低于 MA60，中期趋势偏弱")

    if ma20 > ma60:
        score += 1
        notes.append("MA20 > MA60，均线结构偏多")
    else:
        notes.append("MA20 未高于 MA60，均线结构未完全转强")

    # 动能
    if 40 <= rsi <= 65:
        score += 1
        notes.append("RSI 在健康区间，未明显过热")

    elif rsi >= 80:
        notes.append("🔴 RSI 超买（>80），短线风险较高")

    elif rsi <= 20:
        notes.append("🟢 RSI 超卖（<20），可能出现反弹机会")

    elif rsi > 65:
        notes.append("RSI 偏高，接近过热区")

    elif rsi < 35:
        notes.append("RSI 偏低，但未到极端低位")

    # 成交量
    volume_ratio = vol / vol20 if vol20 else 0
    if volume_ratio >= 1.2 and close > ma20:
        score += 1
        notes.append("成交量高于20日均量，配合价格转强")
    else:
        notes.append("成交量未明显放大")

    # Setup 判定
    setup = "观察"
    if close > ma20 > ma60 and 40 <= rsi <= 65 and macd_hist > 0:
        setup = "趋势突破型"
    elif close > ma60 and close <= ma20 * 1.03 and 30 <= rsi <= 60:
        setup = "回调低吸型"
    elif close > ma120 and rsi < 40:
        setup = "中线回调观察型"

    # 买卖计划
    support = max(float(last["BB_LOWER"]), float(last["LOW_20"]), ma60 * 0.98)
    buy_low = min(close * 0.97, ma20 * 0.99)
    buy_high = close * 0.995
    stop = min(support * 0.98, close * 0.93)
    target1 = close * 1.08
    target2 = close * 1.15

    if score >= 5:
        decision = "Buy Watch"
    elif score >= 4:
        decision = "Wait Pullback"
    else:
        decision = "Skip"
    if price > ma60 and price <= bb_lower:
        decision = "🔥 低吸机会（顺势）"
    elif price > ma60 and price >= bb_upper:
        decision = "⚠️ 不追高"
    elif price < ma60:
        decision = "Skip"

    boll_pct = (price - bb_lower) / (bb_upper - bb_lower)
    boll_pct = round(boll_pct * 100, 0)

    # 风险评分（独立）
    risk_score = 0

    # RSI
    if rsi > 75:
        risk_score += 2
    elif rsi > 65:
        risk_score += 1

    # BOLL
    if boll_pct > 90:
        risk_score += 2
    elif boll_pct > 75:
        risk_score += 1

    # 趋势
    if price < ma60:
        risk_score += 1

    # 最终风险
    if risk_score >= 3:
        risk = "较高"
    elif risk_score == 2:
        risk = "中等偏高"
    elif risk_score == 1:
        risk = "中等"
    else:
        risk = "较低"

    if boll_pct > 80:
        boll_pos = f"接近上轨（{boll_pct}%）：⚠️ 不追高"
    elif boll_pct > 65:
        boll_pos = f"偏高（{boll_pct}%）：注意回调"
    elif boll_pct < 20:
        boll_pos = f"接近下轨（{boll_pct}%）：🔥 低吸观察"
    elif boll_pct < 35:
        boll_pos = f"偏低（{boll_pct}%）：可留意"
    else:
        boll_pos = f"中轨区（{boll_pct}%）：正常"
    if boll_pct > 80 and rsi > 70:
        tech_signal = "⚠️ 高位过热：谨慎追高"
    elif boll_pct < 20 and rsi < 30:
        tech_signal = "🔥 低位机会：关注反弹"
    elif 40 < boll_pct < 70 and 40 < rsi < 65:
        tech_signal = "✅ 健康区间：趋势正常"
    else:
        tech_signal = "🤔 中性：等待方向"
    if rsi >= 70 and boll_pct >= 80:
        tech_signal = "⚠️ 短线过热：不追高"
    elif rsi <= 30 and boll_pct <= 20:
        tech_signal = "🔥 低位反弹区：可留意"
    elif 45 <= rsi <= 65 and 35 <= boll_pct <= 70:
        tech_signal = "✅ 健康区间：趋势正常"
    else:
        tech_signal = "观察"   

    if boll_pct >= 80:
        boll_color = "#ff3b30"   # 🔥 强烈过热
    elif boll_pct >= 70:
        boll_color = "#ff6b6b"
    elif boll_pct <= 20:
        boll_color = "#00c853"   # 🔥 强低位
    elif boll_pct <= 30:
        boll_color = "#4cd964"
    else:
        boll_color = "#aaaaaa"  

    if "高" in risk:
        risk_color = "#ff6b6b"
    elif "低" in risk:
        risk_color = "#4cd964"
    else:
        risk_color = "#aaaaaa"     

    analysis = {
        "ticker": display_ticker.upper(),
        "price": price,
        "price_fmt": fmt_money(price),
        "change_val": f"{change_val:+.2f}",
        "change_pct": f"{change_pct:+.2f}%",
        "change_color": "#4ade80" if change_pct >= 0 else "#f87171",
        "change_class": "up" if change_pct >= 0 else "down",
        "score": score,
        "decision": decision,
        "setup": setup,
        "risk": risk,
        "risk_color": risk_color,
        "rsi": round(rsi, 2),
        "future_low": future_low,
        "future_high": future_high,
        "safe_low": safe_low,
        "safe_high": safe_high,
        "extreme_low": extreme_low,
        "extreme_high": extreme_high,
        "volatility_level": volatility_level,
        "atr_pct": atr_pct,
        "atr": atr,
        "macd_hist": round(macd_hist, 4),
        "buy_flow": money_flow["buy_flow"],
        "sell_flow": money_flow["sell_flow"],
        "flow_signal": money_flow["signal"],
        "ma20": fmt_money(ma20),
        "ma50": fmt_money(ma50),
        "ma60": fmt_money(ma60),
        "ma120": fmt_money(ma120),
        "bb_upper": fmt_money(bb_upper),
        "bb_mid": fmt_money(bb_mid),
        "bb_lower": fmt_money(bb_lower),
        "boll_pos": boll_pos,
        "tech_signal": tech_signal,
        "boll_pct": boll_pct,
        "volume": fmt_num(vol),
        "vol20": fmt_num(vol20),
        "volume_ratio": round(volume_ratio, 2),
        "bb_upper": fmt_money(last["BB_UPPER"]),
        "bb_lower": fmt_money(last["BB_LOWER"]),
        "high_52w": fmt_money(high_52w),
        "low_52w": fmt_money(low_52w),
        "buy_low": fmt_money(buy_low),
        "buy_high": fmt_money(buy_high),
        "buy_zone": f"{fmt_money(buy_low)} - {fmt_money(buy_high)}",
        "stop": fmt_money(stop),
        "target1": fmt_money(target1),
        "target2": fmt_money(target2),
        "notes": notes,
        "price_date": datetime.now().strftime("%Y-%m-%d"),
        "pre_price": pre_price,
        "pre_pct": pre_pct,
        "open_signal": open_signal,
        "session": session,
        "session_price": session_price,
        "session_diff": diff,
        "session_pct": pct,
        "boll_pos": boll_pos,
        "boll_text": boll_pos,
        "boll_color": boll_color,
        "trend_arrow": "↑" if close > ma20 else "↓",
        "trend_label": "短线向上" if close > ma20 else "短线偏弱",
    }

    chart = build_chart(df)

    return analysis, chart, price, None


def build_chart(df):
    recent = df.tail(160)
    return {
        "labels": [d.strftime("%Y-%m-%d") for d in recent.index],
        "close": [round(float(x), 2) for x in recent["Close"]],
        "ma20": [round(float(x), 2) if pd.notna(x) else None for x in recent["MA20"]],
        "ma60": [round(float(x), 2) if pd.notna(x) else None for x in recent["MA60"]],
        "bb_upper": [round(float(x), 2) if pd.notna(x) else None for x in recent["BB_UPPER"]],
        "bb_lower": [round(float(x), 2) if pd.notna(x) else None for x in recent["BB_LOWER"]],
    }


def get_ai_advice(analysis, market_filter):
    fallback = (
        f"{analysis['ticker']} 当前信号：{analysis['decision']}，Setup：{analysis['setup']}。\n"
        f"技术分数 {analysis['score']}/6，RSI {analysis['rsi']}，MACD柱体 {analysis['macd_hist']}。\n"
        f"参考买入区：{analysis['buy_zone']}；止损：{analysis['stop']}；目标：{analysis['target1']} / {analysis['target2']}。\n"
        f"大盘状态：{market_filter['label']}。"
    )

    if not client:
        return fallback

    prompt = f"""
你是一个偏稳健的股票交易系统分析助手。请用繁体中文/粤语口吻，简洁输出，不要保证收益，不要夸张。

大盘状态：{market_filter['label']}
大盘说明：{market_filter['message']}

股票：{analysis['ticker']}
现价：{analysis['price_fmt']}
涨跌：{analysis['change_pct']}
Setup：{analysis['setup']}
系统决定：{analysis['decision']}
技术分：{analysis['score']}/6
风险：{analysis['risk']}
RSI：{analysis['rsi']}
MACD柱体：{analysis['macd_hist']}
MA20：{analysis['ma20']}
MA60：{analysis['ma60']}
成交量倍率：{analysis['volume_ratio']}
买入区：{analysis['buy_zone']}
止损：{analysis['stop']}
目标1：{analysis['target1']}
目标2：{analysis['target2']}

请输出：
1. 一句话结论
2. 为什么可以买/等/跳过
3. 买入策略
4. 风险提醒
"""

    try:
        res = client.chat.completions.create(
            model="gpt-5.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.35
        )
        return res.choices[0].message.content
    except Exception as e:
        return fallback + f"\n\nAI 暂时不可用：{e}"
    

VIX_CACHE = {"data": None, "time": 0}
VIX_TTL = 1800  # 30分钟

def get_vix_card():
    now = time.time()

    if VIX_CACHE["data"] and now - VIX_CACHE["time"] < VIX_TTL:
        return VIX_CACHE["data"]

    try:
        df = yf.download("^VIX", period="6mo", interval="1d", progress=False)

        if df is None or df.empty:
            raise Exception("VIX no data")

        close = df["Close"].dropna()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
            close = close.dropna()

        price = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        change_pct = round((price / prev - 1) * 100, 2)

        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        rs = gain.rolling(14).mean() / loss.rolling(14).mean()
        rsi = 100 - (100 / (1 + rs))
        rsi = float(rsi.iloc[-1])

        data = {
            "ticker": "VIX",
            "price": f"{price:.2f}",
            "change_pct": f"{change_pct:+.2f}%",
            "change_class": "up" if change_pct >= 0 else "down",
            "rsi": round(rsi, 2),
            "ma20": f"{ma20:.2f}" if ma20 else "--",
            "ma60": f"{ma60:.2f}" if ma60 else "--",
            "decision": "风险观察",
            "setup": "波动指标",
            "score": "—",
            "boll_pos": "--",
            "high_52w": "--",
            "low_52w": "--",
            "volume_ratio": "--",
            "session": None,
            "session_price": None,
            "session_diff": None,
            "session_pct": None,
        }

        VIX_CACHE["data"] = data
        VIX_CACHE["time"] = now

        print("🧠 Using YFINANCE for VIX")
        return data

    except Exception as e:
        print("❌ VIX error:", e)
        return {
            "ticker": "VIX",
            "price": "--",
            "change_pct": "--",
            "change_class": "",
            "rsi": "--",
            "ma20": "--",
            "ma60": "--",
            "decision": "暂无",
            "setup": "暂无",
            "score": "--",
            "boll_pos": "--",
        }


def get_market_cards():
    cards = []

    for t in MARKET_TICKERS:
        analysis, _, _, err = analyze_ticker(t)

        if err or not analysis:
            cards.append({
                "ticker": t,
                "price": "--",
                "change_pct": "--",
                "change_class": "",
                "decision": "读取失败",
                "score": "--",
                "rsi": "--",
                "setup": "--",
                "ma20": "--",
                "ma60": "--",
                "boll_pos": "--",
            })
        else:
            cards.append({
                "ticker": t,

                # 价格（已经带 $）
                "price": analysis.get("price_fmt"),

                # 涨跌 %
                "change_pct": analysis.get("change_pct"),

                # 红绿
                "change_class": "up" if (analysis.get("change_num") or 0) >= 0 else "down",

                "decision": analysis.get("decision"),
                "score": analysis.get("score"),
                "rsi": analysis.get("rsi"),
                "setup": analysis.get("setup"),

                "ma20": analysis.get("ma20"),
                "ma60": analysis.get("ma60"),
                "boll_pos": analysis.get("boll_pos"),
                "high_52w": analysis.get("high_52w"),
                "low_52w": analysis.get("low_52w"),
                "volume_ratio": analysis.get("volume_ratio"),
                "session": analysis.get("session"),
                "session_price": analysis.get("session_price"),
                "session_diff": analysis.get("session_diff"),
                "session_pct": analysis.get("session_pct"),
                "change_val": analysis.get("change_val"),
            })

    cards.append(get_vix_card())
    return cards

def get_market_status():
    now = time.time()
    if MARKET_STATUS_CACHE["data"] and now - MARKET_STATUS_CACHE["time"] < MARKET_CACHE_TTL:
        return MARKET_STATUS_CACHE["data"]
    
    market_temp = 50
    temp_zone = "中性"
    vix_text = "暂无"
    open_signal = "暂无数据"
    label = "⚪ 数据不足"
    desc = "暂时无法判断市场状态"
    color = "gray"

    try:
        # === SPY / QQQ ===
        spy = analyze_ticker("SPY")[0]
        qqq = analyze_ticker("QQQ")[0]

        spy_up = spy and float(spy["price"]) > float(spy["ma20"].replace("$",""))
        qqq_up = qqq and float(qqq["price"]) > float(qqq["ma20"].replace("$",""))

        vix = get_vix_card()
        vix_price = float(vix["price"]) if vix["price"] != "--" else 20
        vix_text = f"{vix['price']} ({vix['change_pct']})"

        # === 盘前数据（用SPY）===
        spy_pre = None
        spy_pre_pct = None

        try:
            spy_info = yf.Ticker("SPY").info
            spy_pre = spy_info.get("preMarketPrice")
            spy_prev_close = spy_info.get("previousClose")

            if spy_pre and spy_prev_close:
                spy_pre_pct = round((spy_pre / spy_prev_close - 1) * 100, 2)
        except:
            pass

        # === 大盘开盘预判 ===
        open_signal = "暂无盘前数据"

        if spy_pre_pct is not None:
            if spy_pre_pct <= -1.5 and vix_price >= 20:
                open_signal = "🔴 低开风险高，可能继续走弱"
            elif spy_pre_pct <= -1.5:
                open_signal = "⚠️ 低开，观察是否反弹"
            elif spy_pre_pct >= 1.5 and vix_price <= 18:
                open_signal = "🟢 高开偏强，可能继续上涨"
            elif spy_pre_pct >= 1.5:
                open_signal = "⚠️ 高开偏热，小心回落"
            else:
                open_signal = "🟡 平开震荡，等待方向"

        # === VIX ===
        vix_card = get_vix_card()

        vix_price = float(vix_card["price"])
        vix_rsi = float(vix_card["rsi"]) if vix_card["rsi"] != "--" else 50

        # === VIX 情绪判断 ===
        if vix_price >= 30:
            vix_signal = "🩸 极度恐慌：抄底窗口"
            vix_status = "green"

        elif vix_price >= 25:
            vix_signal = "😰 恐惧区：开始有机会"
            vix_status = "yellow"

        elif vix_price >= 18:
            vix_signal = "😐 中性区：震荡"
            vix_status = "neutral"

        elif vix_price >= 14:
            vix_signal = "🙂 偏乐观：趋势正常"
            vix_status = "green"

        else:
            vix_signal = "😈 贪婪区：注意见顶风险"
            vix_status = "red"  
        
        # === BOLL 数值（从 analysis 拿） ===
        spy_boll_pct = spy.get("boll_pct", 50) if spy else 50
        qqq_boll_pct = qqq.get("boll_pct", 50) if qqq else 50

        market_temp = int((spy_boll_pct + qqq_boll_pct) / 2)
        if market_temp < 30:
            temp_label = "❄️ 冰点（低吸区）"
        elif market_temp < 70:
            temp_label = "🌤 正常区"
        else:
            temp_label = "🔥 过热区"  

        # === 判断逻辑（最终版） ===

        # 🔥 1. 抄底窗口
        if spy_boll_pct < 25 and vix_price >= 25:
            label = "🔥 抄底窗口"
            desc = "SPY 低位 + VIX 恐慌，市场恐惧，适合分批低吸"
            color = "green"

        # ⚠️ 2. 高位风险
        elif spy_boll_pct > 75 and vix_price < 14:
            label = "⚠️ 高位区：不追高"
            desc = "SPY 接近上轨 + 市场贪婪，注意回调风险"
            color = "yellow"

        # 🟢 3. 趋势健康
        elif spy_up and qqq_up and 14 <= vix_price <= 18:
            label = "🟢 趋势健康"
            desc = "趋势向上 + 情绪正常，可持有或顺势"
            color = "green"

        # 🔴 4. 极度恐慌
        elif vix_price >= 30:
            label = "🔴 极度恐慌"
            desc = "市场剧烈波动，谨慎操作或等待确认"
            color = "red"
             
        # 🟡 5. 默认
        else:
            label = "🟡 震荡区"
            desc = "方向不明，等待机会"
            color = "yellow"

    except Exception as e:
        print("Market Status Error:", e)
        label = "⚪ 数据不足"
        desc = "暂时无法判断市场状态"
        temp_zone = "中性"
        vix_text = "暂无"
        open_signal = "暂无数据"
        color = "gray"
    
    result = {
        "label": label,
        "desc": desc,
        "temp": market_temp,
        "temp_zone": temp_zone,
        "vix_text": vix_text,
        "open_signal": open_signal,
        "color": color,
    }

    MARKET_STATUS_CACHE["data"] = result
    MARKET_STATUS_CACHE["time"] = time.time()

    return result


def get_rebound_signals():
    picks = []

    for ticker in UNIVERSE:
        analysis, _, _, err = analyze_ticker(ticker)
        if err or not analysis:
            continue

        try:
            price = float(analysis["price"])
            rsi = float(analysis["rsi"])
            macd_hist = float(analysis["macd_hist"])
            ma20 = float(str(analysis["ma20"]).replace("$", "").replace(",", ""))

            near_ma20 = abs(price - ma20) / ma20 <= 0.08 # 距离MA20 8%内

            if 20 <= rsi <= 55 and macd_hist > -1.5 and near_ma20:
                picks.append({
                    "ticker": analysis["ticker"],
                    "price": analysis["price_fmt"],
                    "rsi": analysis["rsi"],
                    "macd": analysis["macd_hist"],
                    "ma20": analysis["ma20"],
                    "decision": analysis["decision"],
                    "setup": "低位反弹信号",
                    "buy_low": analysis["buy_low"],
                    "buy_high": analysis["buy_high"],
                    "stop": analysis["stop"],
                })

        except Exception:
            continue

    return picks[:3]


@app.route("/", methods=["GET", "POST"])
def index():
    ticker = request.form.get("ticker", "").upper().strip()

    analysis, chart, error = None, None, None
    price = None
    news = []

    market_filter = {
        "status": "manual",
        "label": "手动分析模式",
        "message": "当前为手动模式，不自动抓大盘数据",
        "allow_long": True
    }

    if ticker:
        analysis, chart, price, error = analyze_ticker(ticker)
        news = get_company_news(ticker)

    ai_advice = None

    market_status = None
    top_picks = []
    market_cards = []

    return render_template(
        "index.html",
        ticker=ticker,
        analysis=analysis,
        chart=chart,
        price=price,
        error=error,
        news=news,
        ai_advice=ai_advice,
        market_filter=market_filter,
        top_picks=top_picks,
        market_cards=market_cards,
        market_status=market_status,
    )


@app.route("/api/ai_analysis/<ticker>")
def api_ai_analysis(ticker):
    ticker = ticker.upper()
    now = time.time()

    cached = AI_ANALYSIS_CACHE.get(ticker)
    if cached and now - cached["time"] < AI_ANALYSIS_TTL:
        return jsonify({
            "ticker": ticker,
            "ai_analysis": cached["text"],
            "cached": True
        })

    analysis, chart, price, error = analyze_ticker(ticker)

    if not analysis:
        return jsonify({
            "ticker": ticker,
            "ai_analysis": error or "暂时无法取得股票分析数据。",
            "cached": False
        })

    market_filter = {
        "status": "manual",
        "label": "手动分析模式",
        "message": "当前为手动模式，不自动抓大盘数据",
        "allow_long": True
    }

    ai_text = get_ai_advice(analysis, market_filter)

    AI_ANALYSIS_CACHE[ticker] = {
        "time": now,
        "text": ai_text
    }

    return jsonify({
        "ticker": ticker,
        "ai_analysis": ai_text,
        "cached": False
    })

@app.route("/health")
def health():
    return {"status": "ok", "has_polygon_key": bool(POLYGON_API_KEY)}

@app.route("/market")
def market():
    market_filter = get_market_filter()
    market_cards = get_market_cards()
    market_status= get_market_status()

    return render_template(
        "index.html",
        ticker="",
        analysis=None,
        chart=None,
        error=None,
        ai_advice=None,
        market_filter=market_filter,
        top_picks=[],
        market_cards=market_cards,
        market_status=market_status,
    )

@app.route("/scan_rebound")
def scan_rebound():
    picks = get_rebound_signals()
    return render_template("rebound.html", picks=picks)

@app.route("/news/<ticker>")
def stock_news(ticker):
    news = get_company_news(ticker)

    return render_template(
        "news.html",
        ticker=ticker.upper(),
        news=news
    )

@app.route("/insider/<ticker>")
def insider_page(ticker):
    insider = get_insider_data(ticker)
    institution = []

    data = insider["data"]
    summary = insider["summary"]

    return render_template(
        "insider.html",
        ticker=ticker.upper(),
        data=data,
        summary=summary,
        institution=institution
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)