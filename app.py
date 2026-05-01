import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


load_dotenv()

app = Flask(__name__)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY) if OpenAI and OPENAI_API_KEY else None

# 先用一批高流动性股票，避免太多垃圾信号
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "AVGO", "AMD", "NFLX", "PLTR", "ORCL", "CRM", "NOW",
    "LLY", "ABBV", "ISRG", "COST", "JPM", "V", "MA",
    "QQQ", "SPY", "DIA", "SMH", "IGV", "GLD", "IBIT"
]

MARKET_TICKERS = ["SPY", "QQQ"]

CACHE = {}
CACHE_TTL = 60 * 20  # 20分钟缓存，减少API次数


def get_real_time_price(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info

        price = info.get("last_price") or info.get("regular_market_price")
        if price:
            return float(price)

        hist = t.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])

        return None
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

    high_52w = float(df["High"].tail(252).max())
    low_52w = float(df["Low"].tail(252).min())

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
        risk = "中等"
    elif score >= 4:
        decision = "Wait Pullback"
        risk = "中等偏高"
    else:
        decision = "Skip"
        risk = "较高"
    if price > ma60 and price <= bb_lower:
        decision = "🔥 低吸机会（顺势）"
    elif price > ma60 and price >= bb_upper:
        decision = "⚠️ 不追高"
    elif price < ma60:
        decision = "Skip"

    boll_pct = (price - bb_lower) / (bb_upper - bb_lower)
    boll_pct = round(boll_pct * 100, 0)

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

    analysis = {
        "ticker": display_ticker.upper(),
        "price": price,
        "price_fmt": fmt_money(price),
        "change_pct": fmt_pct(change_pct),
        "change_num": round(change_pct, 2),
        "change_color": "#4ade80" if change_pct >= 0 else "#f87171",
        "change_class": "up" if change_pct >= 0 else "down",
        "score": score,
        "decision": decision,
        "setup": setup,
        "risk": risk,
        "rsi": round(rsi, 2),
        "macd_hist": round(macd_hist, 4),
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
        "price_date": datetime.now().strftime("%Y-%m-%d")
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
    

def get_vix_card():
    try:
        df = yf.download("^VIX", period="6mo", interval="1d", progress=False, auto_adjust=False)

        if df is None or df.empty:
            raise Exception("VIX no data")

        # ✅ 防止 yfinance MultiIndex 问题
        close_series = df["Close"]
        if hasattr(close_series, "columns"):
            close_series = close_series.iloc[:, 0]

        close_series = close_series.dropna()

        # ✅ RSI 计算
        delta = close_series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss
        rsi_series = 100 - (100 / (1 + rs))

        rsi = float(rsi_series.iloc[-1])

        if len(close_series) < 60:
            raise Exception("VIX data not enough")

        close = float(close_series.iloc[-1])
        prev_close = float(close_series.iloc[-2])
        change_pct = (close / prev_close - 1) * 100

        ma20 = float(close_series.rolling(20).mean().iloc[-1])
        ma60 = float(close_series.rolling(60).mean().iloc[-1])

        return {
            "ticker": "VIX",
            "price": f"{close:.2f}",
            "change_pct": f"{change_pct:.2f}%",
            "change_class": "up" if change_pct >= 0 else "down",
            "decision": "风险观察",
            "score": 5 if close < 18 else 3 if close < 22 else 1,
            "rsi": f"{rsi:.2f}",
            "setup": "低风险区间" if close < 18 else "正常区间" if close < 22 else "高风险区间",
            "ma20": f"{ma20:.2f}",
            "ma60": f"{ma60:.2f}",
        }

    except Exception as e:
        print("VIX ERROR:", e)  # ✅ 方便你在 Terminal 看到原因
        return {
            "ticker": "VIX",
            "price": "--",
            "change_pct": "--",
            "change_class": "",
            "decision": "读取失败",
            "score": "--",
            "rsi": "--",
            "setup": "--",
            "ma20": "--",
            "ma60": "--",
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
            })

    cards.append(get_vix_card())
    return cards

def get_market_status():
    try:
        # === SPY / QQQ ===
        spy = analyze_ticker("SPY")[0]
        qqq = analyze_ticker("QQQ")[0]

        spy_up = spy and float(spy["price"]) > float(spy["ma20"].replace("$",""))
        qqq_up = qqq and float(qqq["price"]) > float(qqq["ma20"].replace("$",""))

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

        # === 判断逻辑（新版本） ===

        # 🔥 1. 低吸窗口（优先）
        if spy_boll_pct < 25 and vix_price >= 25:
            return {
                "label": "🔥 抄底窗口",
                "status": "green",
                "message": "SPY 低位 + VIX 恐慌，市场恐惧，适合分批低吸",
                "temp": f"{market_temp} / 100",
                "temp_label": temp_label,
                "vix_sentiment": vix_signal,
                "vix_status": vix_status,
            }

        # ⚠️ 2. 高位风险
        elif spy_boll_pct > 75 and vix_price < 14:
            return {
                "label": "⚠️ 高位区：不追高",
                "status": "yellow",
                "message": "SPY 接近上轨 + 市场贪婪，注意回调风险",
                "temp": f"{market_temp} / 100",
                "temp_label": temp_label,
                "vix_sentiment": vix_signal,
                "vix_status": vix_status,
            }

        # 🟢 3. 正常上涨趋势
        elif spy_up and qqq_up and 14 <= vix_price <= 18:
            return {
                "label": "🟢 趋势健康",
                "status": "green",
                "message": "趋势向上 + 情绪正常，可持有或顺势",
                "temp": f"{market_temp} / 100",
                "temp_label": temp_label,
                "vix_sentiment": vix_signal,
                "vix_status": vix_status,
            }

        # 🔴 4. 风险高
        elif vix_price >= 30:
            return {
                "label": "🔴 极度恐慌",
                "status": "red",
                "message": "市场剧烈波动，谨慎操作或等待确认",
                "temp": f"{market_temp} / 100",
                "temp_label": temp_label,
                "vix_sentiment": vix_signal,
                "vix_status": vix_status,
            }

        # 🟡 5. 默认
        else:
            return {
                "label": "🟡 震荡区",
                "status": "yellow",
                "message": "方向不明，等待机会",
                "temp": f"{market_temp} / 100",
                "temp_label": temp_label,
                "vix_sentiment": vix_signal,
                "vix_status": vix_status,
            }

    except Exception as e:
        print("Market Status Error:", e)
        return {
            "label": "⚪ 数据不足",
            "status": "gray",
            "message": "暂时无法判断市场状态",
            "temp": f"{market_temp} / 100",
            "temp_label": temp_label,
            "vix_sentiment": vix_signal,
            "vix_status": vix_status,
        }


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

            near_ma20 = abs(price - ma20) / ma20 <= 0.04  # 距离MA20 4%内

            if 20 <= rsi <= 30 and macd_hist > 0 and near_ma20:
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

    market_filter = {
        "status": "manual",
        "label": "手动分析模式",
        "message": "当前为手动模式，不自动抓大盘数据",
        "allow_long": True
    }

    if ticker:
        analysis, chart, price, error = analyze_ticker(ticker)

    ai_advice = None
    if analysis:
        ai_advice = get_ai_advice(analysis, market_filter)

    market_status = get_market_status()
    top_picks = []
    market_cards = []

    return render_template(
        "index.html",
        ticker=ticker,
        analysis=analysis,
        chart=chart,
        price=price,
        error=error,
        ai_advice=ai_advice,
        market_filter=market_filter,
        top_picks=top_picks,
        market_cards=market_cards,
        market_status=market_status,
    )


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


if __name__ == "__main__":
    app.run(debug=True, port=5001)