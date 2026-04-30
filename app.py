import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
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
        r = requests.get(url, params=params, timeout=20)

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

    df["BB_MID"] = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_UPPER"] = df["BB_MID"] + 2 * std
    df["BB_LOWER"] = df["BB_MID"] - 2 * std

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
    df, err = get_polygon_daily_bars(ticker)
    if err:
        return None, None, err

    if df is None or len(df) < 150:
        return None, None, "数据不足，至少需要150根日K"

    df = add_indicators(df).dropna()

    if df.empty:
        return None, None, "技术指标计算失败"

    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(last["Close"])
    prev_close = float(prev["Close"])
    change_pct = (close / prev_close - 1) * 100

    rsi = float(last["RSI"])
    macd_hist = float(last["MACD_HIST"])
    ma20 = float(last["MA20"])
    ma50 = float(last["MA50"])
    ma60 = float(last["MA60"])
    ma120 = float(last["MA120"])
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
    elif rsi > 70:
        notes.append("RSI 超过70，短线过热")
    elif rsi < 35:
        notes.append("RSI 偏低，可能反弹但要小心趋势未确认")

    if macd_hist > 0:
        score += 1
        notes.append("MACD柱体为正，动能偏多")
    else:
        notes.append("MACD柱体为负，动能未确认")

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
    elif close > ma60 and close <= ma20 * 1.03 and 35 <= rsi <= 55:
        setup = "回调低吸型"
    elif close > ma120 and rsi < 45:
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

    analysis = {
        "ticker": ticker.upper(),
        "price": close,
        "price_fmt": fmt_money(close),
        "change_pct": fmt_pct(change_pct),
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
        "volume": fmt_num(vol),
        "vol20": fmt_num(vol20),
        "volume_ratio": round(volume_ratio, 2),
        "bb_upper": fmt_money(last["BB_UPPER"]),
        "bb_lower": fmt_money(last["BB_LOWER"]),
        "high_52w": fmt_money(high_52w),
        "low_52w": fmt_money(low_52w),
        "buy_zone": f"{fmt_money(buy_low)} - {fmt_money(buy_high)}",
        "stop": fmt_money(stop),
        "target1": fmt_money(target1),
        "target2": fmt_money(target2),
        "notes": notes
    }

    chart = build_chart(df)

    return analysis, chart, None


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


def get_market_cards():
    cards = []
    for t in MARKET_TICKERS:
        analysis, _, err = analyze_ticker(t)
        if err:
            cards.append({
                "ticker": t,
                "price": "-",
                "decision": "读取失败",
                "score": "-",
                "rsi": "-",
                "setup": "-",
            })
        else:
            cards.append({
                "ticker": t,
                "price": analysis["price_fmt"],
                "decision": analysis["decision"],
                "score": analysis["score"],
                "rsi": analysis["rsi"],
                "setup": analysis["setup"],
            })
    return cards


def get_top_picks():
    market_filter = get_market_filter()
    picks = []

    for ticker in UNIVERSE:
        analysis, _, err = analyze_ticker(ticker)
        if err:
            continue

        # 大盘红灯时，降低全部信号
        if not market_filter["allow_long"] and analysis["score"] < 6:
            continue

        # 只选高质量信号
        if analysis["score"] < 5:
            continue

        # 第二步：只做趋势突破型
        if analysis["setup"] != "趋势突破型":
            continue

        # 第三步：不要追太接近52周高位
        high_52w = float(analysis["high_52w"].replace("$", "").replace(",", ""))
        if analysis["price"] > high_52w * 0.97:
            continue

        # RSI 不要太热
        if analysis["rsi"] >= 68:
            continue

            picks.append({
                "ticker": analysis["ticker"],
                "price": analysis["price_fmt"],
                "change_pct": analysis["change_pct"],
                "score": analysis["score"],
                "setup": analysis["setup"],
                "decision": analysis["decision"],
                "rsi": analysis["rsi"],
                "buy_zone": analysis["buy_zone"],
                "stop": analysis["stop"],
                "target1": analysis["target1"],
                "risk": analysis["risk"],
            })

    picks = sorted(picks, key=lambda x: (x["score"], x["rsi"] < 70), reverse=True)
    return picks[:5]


@app.route("/", methods=["GET", "POST"])
def index():
    ticker = request.form.get("ticker", "").upper().strip()

    market_filter = {
        "status": "manual",
        "label": "手动分析模式",
        "message": "当前为手动模式，不自动抓大盘数据",
        "allow_long": True
    }
    analysis, chart, error = None, None, None

    if ticker:
        analysis, chart, error = analyze_ticker(ticker)

    ai_advice = None
    if analysis:
        ai_advice = get_ai_advice(analysis, market_filter)

    top_picks = []
    market_cards = []

    return render_template(
        "index.html",
        ticker=ticker,
        analysis=analysis,
        chart=chart,
        error=error,
        ai_advice=ai_advice,
        market_filter=market_filter,
        top_picks=top_picks,
        market_cards=market_cards,
    )


@app.route("/health")
def health():
    return {"status": "ok", "has_polygon_key": bool(POLYGON_API_KEY)}


if __name__ == "__main__":
    app.run(debug=True, port=5001)