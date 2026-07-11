import os
import time
import hmac
import secrets
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from flask import jsonify, session, redirect, url_for, flash
from dotenv import load_dotenv
from flask import Flask, render_template, request

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET_KEY") or secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("RENDER", "").lower() == "true",
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
)


def load_login_users():
    """Load up to 4 login accounts from Render environment variables."""
    users = {}
    for i in range(1, 5):
        username = os.getenv(f"LOGIN_USER_{i}", "").strip()
        password = os.getenv(f"LOGIN_PASSWORD_{i}", "")
        if username and password:
            users[username.lower()] = {
                "username": username,
                "password": password,
            }
    return users


def is_safe_next_url(target):
    return bool(target) and target.startswith("/") and not target.startswith("//")


@app.before_request
def require_login():
    public_endpoints = {"login", "health", "static"}
    if request.endpoint in public_endpoints:
        return None
    if not session.get("logged_in"):
        next_url = request.full_path if request.query_string else request.path
        return redirect(url_for("login", next=next_url))
    return None


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        users = load_login_users()
        account = users.get(username.lower())

        valid_user = account is not None
        valid_password = valid_user and hmac.compare_digest(password, account["password"])

        if valid_password:
            session.clear()
            session.permanent = True
            session["logged_in"] = True
            session["username"] = account["username"]
            next_url = request.form.get("next", "")
            return redirect(next_url if is_safe_next_url(next_url) else url_for("index"))

        error = "用户名或密码不正确"

    return render_template("login.html", error=error, next=request.args.get("next", ""))


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))

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

# 今日机会股票池：优先读取 data/spy.txt，避免把股票写死在 app.py
def load_universe():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    universe_path = os.path.join(base_dir, "data", "spy.txt")

    try:
        with open(universe_path, "r", encoding="utf-8") as f:
            symbols = []
            seen = set()
            for raw_line in f:
                line = raw_line.strip().upper()
                if not line or line.startswith("#"):
                    continue
                # 容许 txt 内有逗号或空格分隔，但建议一行一只股票
                for symbol in line.replace(",", " ").split():
                    if symbol and symbol not in seen:
                        seen.add(symbol)
                        symbols.append(symbol)

        if symbols:
            print(f"✅ Loaded {len(symbols)} symbols from data/spy.txt")
            return symbols
    except Exception as e:
        print(f"⚠️ Unable to load data/spy.txt: {e}")

    # 文件不存在或为空时的安全备用股票池，避免网站直接报错
    return [
        "NVDA", "MSFT", "META", "AMZN", "GOOGL", "AAPL", "TSLA",
        "AVGO", "AMD", "PLTR", "NOW", "CRM", "ORCL", "TSM",
        "ANET", "PANW", "CRWD", "ISRG", "LLY", "RKLB",
        "SPY", "QQQ", "SOXX", "SMH", "IGV", "SPMO",
    ]


UNIVERSE = load_universe()


MARKET_TICKERS = ["SPY", "QQQ", "SOXX", "DIA"]

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
    

HUNTER_CACHE = {"time": 0, "data": []}
HUNTER_CACHE_TTL = 60 * 30


def load_tickers(filename):
    try:
        with open(f"data/{filename}", "r") as f:
            return [x.strip().upper() for x in f.readlines() if x.strip()]
    except Exception as e:
        print("load ticker error:", e)
        return []


def clean_num(x):
    try:
        return float(str(x).replace("$", "").replace("%", "").replace("x", "").replace(",", "").strip())
    except:
        return 0


OPPORTUNITY_SECTORS = {
    # Mega-cap / software
    "NVDA":"AI / 半导体", "MSFT":"软件 / 云计算", "META":"互联网 / AI",
    "AMZN":"电商 / 云计算", "GOOGL":"互联网 / AI", "GOOG":"互联网 / AI",
    "AAPL":"消费电子", "PLTR":"AI 软件", "SNOW":"云数据", "NOW":"企业软件",
    "CRM":"企业软件", "ORCL":"数据库 / 云计算", "ADBE":"软件", "INTU":"金融软件",
    # Semiconductors
    "AVGO":"半导体 / AI", "AMD":"半导体 / AI", "MU":"存储芯片",
    "QCOM":"半导体", "LRCX":"半导体设备", "AMAT":"半导体设备",
    "KLAC":"半导体设备", "ASML":"半导体设备", "TSM":"晶圆代工",
    "ARM":"芯片架构", "INTC":"半导体",
    # Infra / cloud
    "ANET":"AI 网络", "PANW":"网络安全", "CRWD":"网络安全",
    "DDOG":"云监控", "MDB":"数据库", "NET":"云网络", "GLW":"光通信",
    "OKLO":"核能 / AI 电力", "POET":"光通信", "IREN":"AI 数据中心",
    "NBIS":"AI 基建", "LITE":"光通信",
    # Other growth
    "ISRG":"医疗机器人", "TSLA":"电动车 / 机器人", "RXRX":"AI 制药",
    "LLY":"医药", "NVO":"医药", "VRTX":"生物科技", "REGN":"生物科技",
    "UNH":"医疗保险", "ABBV":"医药", "MRK":"医药", "TMO":"生命科学", "DHR":"生命科学",
    "RKLB":"太空", "LUNR":"太空", "ASTS":"卫星通信", "PL":"卫星数据",
    "SPIR":"卫星数据", "KTOS":"国防科技", "AVAV":"无人机", "RTX":"国防",
    "LMT":"国防", "NOC":"国防",
    "JPM":"金融", "GE":"工业", "CAT":"工业", "XOM":"能源",
    "NFLX":"流媒体", "CSCO":"网络设备",
    "QQQ":"科技 ETF", "SPY":"大盘 ETF", "SMH":"半导体 ETF",
    "IGV":"软件 ETF", "SPMO":"动量 ETF",
}


def opportunity_details(raw_score, risk_level, entry_status, reasons):
    """Convert the compact scanner score into a clearer 0-100 display.

    This is a model confidence score, not a historical win rate.
    """
    score_100 = max(0, min(100, round(45 + raw_score * 5)))

    if score_100 >= 90:
        grade, stars = "S级机会", 5
    elif score_100 >= 82:
        grade, stars = "A级机会", 4
    elif score_100 >= 72:
        grade, stars = "B级机会", 3
    elif score_100 >= 62:
        grade, stars = "C级观察", 2
    else:
        grade, stars = "D级观察", 1

    if "不追高" in entry_status:
        action = "🔴 暂不追高"
    elif "等更好位置" in entry_status:
        action = "🟡 等待回踩"
    elif "等放量" in entry_status:
        action = "🟡 等待放量确认"
    elif risk_level == "较低" and score_100 >= 82:
        action = "🟢 可分批关注"
    else:
        action = "🟡 继续观察"

    reasons_list = [r.strip() for r in str(reasons).split("、") if r.strip()]
    return score_100, grade, stars, action, reasons_list


def scan_market_hunter():
    now = time.time()
    if HUNTER_CACHE["data"] and now - HUNTER_CACHE["time"] < HUNTER_CACHE_TTL:
        return HUNTER_CACHE["data"]

    tickers = load_tickers("spy.txt")
    results = []

    # QQQ 参考强弱
    qqq_df = get_history("QQQ", days=180, interval="1d")
    qqq_change = 0

    if qqq_df is not None and not qqq_df.empty and len(qqq_df) >= 2:
        qqq_df.columns = [str(c).capitalize() for c in qqq_df.columns]
        qqq_close = float(qqq_df["Close"].iloc[-1])
        qqq_prev = float(qqq_df["Close"].iloc[-2])
        qqq_change = round((qqq_close / qqq_prev - 1) * 100, 2)

    for ticker in tickers:
        try:
            df = get_history(ticker, days=180, interval="1d")

            if df is None or df.empty or len(df) < 120:
                continue

            df.columns = [str(c).capitalize() for c in df.columns]
            close = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2])
            high = float(df["High"].iloc[-1])
            low = float(df["Low"].iloc[-1])
            ma20 = float(df["Close"].rolling(20).mean().iloc[-1])
            ma60 = float(df["Close"].rolling(60).mean().iloc[-1])
            ma120 = float(df["Close"].rolling(120).mean().iloc[-1])
            high20 = float(df["High"].rolling(20).max().iloc[-2])
            low20 = float(df["Low"].rolling(20).min().iloc[-2])
            change_pct = round((close / prev_close - 1) * 100, 2)
            rs_strength = round(change_pct - qqq_change, 2)
            vol = float(df["Volume"].iloc[-1])
            vol20 = float(df["Volume"].rolling(20).mean().iloc[-1])
            volume_ratio = round(vol / vol20, 2) if vol20 else 0
            dollar_volume = close * vol

            # RSI
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = round(float(100 - (100 / (1 + rs.iloc[-1]))), 2)

            # BOLL
            mid = df["Close"].rolling(20).mean()
            std = df["Close"].rolling(20).std()
            bb_upper = float((mid + 2 * std).iloc[-1])
            bb_lower = float((mid - 2 * std).iloc[-1])
            boll_pct = round(((close - bb_lower) / (bb_upper - bb_lower)) * 100, 1) if bb_upper != bb_lower else 50

            # ATR 波动
            tr1 = df["High"] - df["Low"]
            tr2 = abs(df["High"] - df["Close"].shift())
            tr3 = abs(df["Low"] - df["Close"].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr14 = float(tr.rolling(14).mean().iloc[-1])
            atr_pct = round((atr14 / close) * 100, 2)

            # 过滤垃圾 / 低波动 / 无资金
            if close < 20:
                continue
            if atr_pct < 2.0:
                continue
            if dollar_volume < 50_000_000:
                continue

            score = 0
            reasons = []
            setup_type = "👀 观察"

            # 强势回调：优先
            if (
                close > ma20 > ma60
                and abs(close - ma20) / ma20 <= 0.035
                and 45 <= rsi <= 62
                and boll_pct < 80
            ):
                setup_type = "🟡 强势回调"
                score += 7
                reasons += ["趋势回踩MA20", "低风险位置"]

            # 主升浪
            elif (
                close > ma20 > ma60 > ma120
                and 55 <= rsi <= 72
                and rs_strength > 0
                and boll_pct < 90
            ):
                setup_type = "🚀 主升浪"
                score += 6
                reasons += ["多头结构", "中期趋势强", f"强过QQQ {rs_strength}%"]

            # 真突破
            elif (
                close > high20
                and volume_ratio >= 1.2
                and 55 <= rsi <= 72
                and rs_strength > 0
                and boll_pct < 92
            ):
                setup_type = "🔥 真突破"
                score += 6
                reasons += ["突破20日高", "放量确认"]

            # 低位启动
            elif (
                close > ma20
                and prev_close < ma20
                and 45 <= rsi <= 60
                and 20 <= boll_pct <= 65
            ):
                setup_type = "💎 低位启动"
                score += 4
                reasons += ["重新站回MA20", "低位转强"]

            # 加分
            if rs_strength > 1:
                score += 2
                reasons.append("明显强过QQQ")
            elif rs_strength > 0.5:
                score += 1
                reasons.append("略强QQQ")
            if volume_ratio >= 1.5:
                score += 2
                reasons.append("明显放量")
            elif volume_ratio >= 1.0:
                score += 1
                reasons.append("量能正常")
            if close >= high * 0.98:
                score += 1
                reasons.append("收盘靠近高位")

            # 风险扣分
            if rsi >= 75:
                score -= 3
                reasons.append("RSI过热")
            if boll_pct >= 95:
                score -= 4
                reasons.append("严重接近上轨")
            elif boll_pct >= 88:
                score -= 2
                reasons.append("防追高")
            if volume_ratio < 0.8:
                score -= 1
                reasons.append("等待放量确认")
            if score < 5:
                continue

            # 买区 / 止损 / 目标
            buy_low = round(close * 0.985, 2)
            buy_high = round(close * 1.01, 2)
            stop = round(min(ma20 * 0.97, low20 * 0.98, close * 0.94), 2)
            target1 = round(close * 1.06, 2)
            target2 = round(close * 1.12, 2)
            stop_pct = round((close - stop) / close * 100, 1)

            if boll_pct > 90 or rsi > 74:
                risk_level = "高"
                entry_status = "⚠️ 不追高"
            elif stop_pct > 8:
                risk_level = "中"
                entry_status = "👀 等更好位置"
            elif volume_ratio < 0.8:
                risk_level = "中"
                entry_status = "👀 等放量确认"
            else:
                risk_level = "较低"
                entry_status = "✅ 可观察"
            display_score, opportunity_grade, opportunity_stars, action_text, reasons_list = opportunity_details(
                score, risk_level, entry_status, reasons
            )

            results.append({
                "ticker": ticker,
                "price": round(close, 2),
                "score": score,
                "display_score": display_score,
                "opportunity_grade": opportunity_grade,
                "opportunity_stars": opportunity_stars,
                "action_text": action_text,
                "sector": OPPORTUNITY_SECTORS.get(ticker, "成长股"),
                "historical_rate_text": "数据累积中",
                "change_pct": change_pct,
                "setup_type": setup_type,
                "entry_status": entry_status,
                "risk_level": risk_level,
                "rsi": rsi,
                "boll_pct": boll_pct,
                "volume_ratio": volume_ratio,
                "rs_strength": rs_strength,
                "atr_pct": atr_pct,
                "dollar_volume": round(dollar_volume / 1_000_000, 1),
                "high20": round(high20, 2),
                "reasons": "、".join(reasons),
                "reasons_list": reasons_list,
                "buy_low": buy_low,
                "buy_high": buy_high,
                "stop": stop,
                "stop_pct": stop_pct,
                "target1": target1,
                "target2": target2
            })

        except Exception as e:
            print("Hunter error:", ticker, e)

    results = sorted(results, key=lambda x: x.get("display_score", 0), reverse=True)[:20]

    HUNTER_CACHE["time"] = now
    HUNTER_CACHE["data"] = results

    return results

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
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        url = "https://finnhub.io/api/v1/company-news"

        params = {
            "symbol": ticker.upper(),
            "from": from_date,
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


def get_pe_label(ticker):
    try:
        url = "https://finnhub.io/api/v1/stock/metric"
        params = {
            "symbol": ticker,
            "metric": "all",
            "token": FINNHUB_API_KEY
        }

        r = requests.get(url, params=params, timeout=8)
        data = r.json().get("metric", {})

        pe = data.get("peNormalizedAnnual") or data.get("peTTM")

        if not pe:
            return {
                "pe_text": "",
                "pe_label": "",
                "pe_color": "#ffffff"
            }

        pe = round(float(pe), 1)

        if pe < 15:
            label = "最便宜"
            pe_color = "#22c55e"
        elif pe < 30:
            label = "合理"
            pe_color = "#facc15"
        else:
            label = "偏贵"
            pe_color = "#ef4444"

        return {
            "pe_text": f"{pe}x",
            "pe_label": label,
            "pe_color": pe_color
        }

    except Exception as e:
        print(f"PE data error for {ticker}: {e}")

        return {
            "pe_text": "",
            "pe_label": "",
            "pe_color": "#ffffff"
        }


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
    

def calc_trend_stage(price, ma20, ma60, ma120, rsi, boll_pct, volume_ratio):
    if not price or not ma20 or not ma60:
        return "数据不足", "#94a3b8"

    # 多头排列
    if ma120 and price > ma20 > ma60 > ma120:
        if rsi and rsi > 72:
            return "高位过热", "#fb7185"
        elif volume_ratio and volume_ratio >= 1.3:
            return "主升阶段", "#22c55e"
        else:
            return "健康上升", "#4ade80"

    # 突破/启动
    if price > ma20 and ma20 > ma60:
        return "启动阶段", "#38bdf8"

    # 高位震荡
    if price > ma60 and rsi and 55 <= rsi <= 72 and boll_pct and boll_pct > 70:
        return "高位震荡", "#facc15"

    # 回调但未破坏
    if price < ma20 and price > ma60:
        return "上升回调", "#facc15"

    # 趋势转弱
    if price < ma60:
        return "趋势转弱", "#fb923c"

    # 空头
    if ma120 and price < ma20 < ma60 < ma120:
        return "下跌阶段", "#ef4444"

    return "震荡阶段", "#94a3b8"    


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
    

def get_sector_rotation():
    sector_data = []

    SECTOR_ETFS = {
        "QQQ": "科技",
        "SMH": "半导体",
        "IGV": "软件",
        "XLF": "金融",
        "XLV": "医疗",
        "XLE": "能源",
        "XLY": "消费",
    }

    for ticker, name in SECTOR_ETFS.items():
        try:
            df = get_history(ticker, days=5, interval="5m")

            if df is None or df.empty or len(df) < 2:
                print("Sector no intraday data:", ticker)
                continue

            close = float(df["Close"].iloc[-1])

            # 找昨日/前一交易日参考价
            daily = get_history(ticker, days=5, interval="1d")
            if daily is None or daily.empty or len(daily) < 2:
                print("Sector no daily data:", ticker)
                continue

            prev_close = float(daily["Close"].iloc[-2])

            change_pct = round(((close - prev_close) / prev_close) * 100, 2)

            if change_pct >= 2:
                status = "🔥 强势"
            elif change_pct >= 0.5:
                status = "🟢 偏强"
            elif change_pct <= -2:
                status = "🔴 弱势"
            elif change_pct <= -0.5:
                status = "⚠️ 偏弱"
            else:
                status = "⚪ 中性"

            sector_data.append({
                "ticker": ticker,
                "name": name,
                "change": change_pct,
                "status": status
            })

        except Exception as e:
            print("Sector error:", ticker, e)

    return sorted(sector_data, key=lambda x: x["change"], reverse=True)


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
                "buy_flow": "暂无数据",
                "sell_flow": "暂无数据"
            }

        buy_flow = 0
        sell_flow = 0

        for idx, row in bars.iterrows():
            open_price = float(row["Open"])
            close_price = float(row["Close"])
            volume = float(row["Volume"])

            avg_price = (open_price + close_price) / 2
            money = avg_price * volume

            # 上涨K线 → 买盘估算
            if close_price > open_price:
                buy_flow += money

            # 下跌K线 → 卖盘估算
            elif close_price < open_price:
                sell_flow += money

        def fmt(v):
            if v >= 1_000_000_000:
                return f"${v / 1_000_000_000:.2f}B"
            elif v >= 1_000_000:
                return f"${v / 1_000_000:.2f}M"
            else:
                return f"${v:,.0f}"

        return {
            "buy_flow": fmt(buy_flow),
            "sell_flow": fmt(sell_flow)
        }

    except Exception as e:
        print("资金流错误:", e)

        return {
            "buy_flow": "暂无数据",
            "sell_flow": "暂无数据"
        }
    

def get_intraday_signal(ticker):
    """Return stock-side intraday confirmation signals.

    This deliberately does not invent options-flow, dealer gamma, or dark-pool
    direction. Those fields remain unavailable until a real options-flow data
    source is connected.
    """
    bars = get_history(ticker, days=10, interval="5m")

    if bars is None or bars.empty or len(bars) < 3:
        print("No intraday:", ticker)
        return None

    bars = bars.copy()
    bars.columns = [str(c).capitalize() for c in bars.columns]
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(bars.columns):
        return None

    # Normalize timestamps without assuming the feed timezone.
    idx = pd.to_datetime(bars.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("America/New_York")
    bars.index = idx
    bars["session_date"] = bars.index.date

    sessions = []
    for _, session_df in bars.groupby("session_date"):
        session_df = session_df.sort_index().copy()
        if len(session_df) >= 3:
            sessions.append(session_df)

    if not sessions:
        return None

    today = sessions[-1]
    current_bar_count = len(today)
    total_volume = float(today["Volume"].sum())
    if total_volume <= 0:
        return None

    typical = (today["High"] + today["Low"] + today["Close"]) / 3
    cumulative_volume = today["Volume"].cumsum()
    today["VWAP"] = (typical * today["Volume"]).cumsum() / cumulative_volume.replace(0, np.nan)

    vwap = float(today["VWAP"].iloc[-1])
    first_price = float(today["Close"].iloc[0])
    last_price = float(today["Close"].iloc[-1])
    momentum = ((last_price - first_price) / first_price) * 100 if first_price else 0

    consecutive_bars = 0
    for _, row in today.iloc[::-1].iterrows():
        if float(row["Close"]) > float(row["VWAP"]):
            consecutive_bars += 1
        else:
            break
    # A regular U.S. session has at most 390 minutes. Cap the value to prevent
    # stale/misaligned intraday feeds from showing impossible durations.
    above_vwap_minutes = min(consecutive_bars * 5, 390)

    # Same-time relative volume: today's cumulative volume versus prior
    # sessions at the same 5-minute bar count.
    prior_same_time_volumes = []
    for session_df in sessions[:-1][-20:]:
        n = min(current_bar_count, len(session_df))
        if n >= 3:
            prior_same_time_volumes.append(float(session_df["Volume"].iloc[:n].sum()))

    if prior_same_time_volumes:
        baseline = float(np.mean(prior_same_time_volumes))
        relative_volume = total_volume / baseline if baseline > 0 else None
    else:
        relative_volume = None

    return {
        "vwap": vwap,
        "momentum": momentum,
        "above_vwap_minutes": above_vwap_minutes,
        "relative_volume": round(relative_volume, 2) if relative_volume is not None else None,
        "current_price": last_price,
        "bar_count": current_bar_count,
    }


def build_short_options_status(
    price,
    macd_hist,
    intraday,
    rsi,
    ma20,
    ma60,
    ma120,
    boll_pct,
    relative_strength,
    change_pct,
):
    """Build a higher-quality stock-side Smart Money confirmation score.

    This is a rules-based confirmation model using only existing stock data.
    It does not claim to be options flow, dealer positioning, or a measured
    historical win rate. The score represents current signal agreement.
    """
    signals = []
    risks = []
    earned = 0.0
    available = 0.0

    def add_signal(weight, result, positive_text, neutral_text=None, negative_text=None, partial=0.5):
        nonlocal earned, available
        available += weight
        if result == "positive":
            earned += weight
            signals.append({"type": "positive", "text": positive_text})
        elif result == "neutral":
            earned += weight * partial
            signals.append({"type": "neutral", "text": neutral_text or positive_text})
        else:
            signals.append({"type": "negative", "text": negative_text or neutral_text or positive_text})

    # 1) VWAP persistence — strongest intraday confirmation.
    if intraday:
        minutes = min(int(intraday.get("above_vwap_minutes") or 0), 390)
        vwap_value = float(intraday.get("vwap") or price)
        if price > vwap_value and minutes >= 20:
            add_signal(22, "positive", f"股价连续{minutes}分钟站稳VWAP")
        elif price > vwap_value and minutes >= 5:
            add_signal(22, "neutral", "", f"股价位于VWAP上方，已确认{minutes}分钟", "")
        else:
            add_signal(22, "negative", "", "", "股价尚未站稳VWAP")

        # 2) Same-time relative volume, not full-day volume ratio.
        rel_vol = intraday.get("relative_volume")
        if rel_vol is not None:
            rel_vol = float(rel_vol)
            if rel_vol >= 1.35:
                add_signal(18, "positive", f"同时段相对成交量{rel_vol:.2f}倍")
            elif rel_vol >= 0.90:
                add_signal(18, "neutral", "", f"同时段相对成交量{rel_vol:.2f}倍", "")
            else:
                add_signal(18, "negative", "", "", f"同时段相对成交量仅{rel_vol:.2f}倍")

        # 3) Intraday momentum. Avoid rewarding an already overextended move.
        momentum = float(intraday.get("momentum") or 0)
        if 0.6 <= momentum <= 4.0:
            add_signal(12, "positive", f"盘中动能偏强（{momentum:+.2f}%）")
        elif -0.5 < momentum < 0.6 or 4.0 < momentum <= 6.0:
            add_signal(12, "neutral", "", f"盘中动能中性（{momentum:+.2f}%）", "")
        else:
            add_signal(12, "negative", "", "", f"盘中动能偏弱或过热（{momentum:+.2f}%）")
    else:
        risks.append("盘中VWAP及同时段成交量数据不足")

    # 4) MACD confirmation.
    if macd_hist > 0:
        add_signal(12, "positive", "MACD柱状值为正")
    elif macd_hist > -0.15:
        add_signal(12, "neutral", "", "MACD接近翻正", "")
    else:
        add_signal(12, "negative", "", "", "MACD柱状值仍为负")

    # 5) Moving-average structure. This reduces false positives from one-day pops.
    if price > ma20 > ma60 and price > ma120:
        add_signal(14, "positive", "价格与均线结构偏多")
    elif price > ma20 and price >= ma60:
        add_signal(14, "neutral", "", "价格站上短中期均线", "")
    else:
        add_signal(14, "negative", "", "", "均线结构尚未形成多头确认")

    # 6) RSI sweet spot. Do not reward overbought conditions.
    if 48 <= rsi <= 68:
        add_signal(9, "positive", f"RSI处于健康强势区（{rsi:.1f}）")
    elif 40 <= rsi < 48 or 68 < rsi <= 75:
        add_signal(9, "neutral", "", f"RSI处于观察区（{rsi:.1f}）", "")
    else:
        add_signal(9, "negative", "", "", f"RSI偏弱或过热（{rsi:.1f}）")

    # 7) Relative strength versus benchmark.
    if relative_strength >= 0.8:
        add_signal(8, "positive", f"相对大盘强度领先（{relative_strength:+.2f}%）")
    elif relative_strength > -0.5:
        add_signal(8, "neutral", "", f"相对大盘表现中性（{relative_strength:+.2f}%）", "")
    else:
        add_signal(8, "negative", "", "", f"相对大盘表现偏弱（{relative_strength:+.2f}%）")

    # 8) Bollinger location / extension risk.
    if 35 <= boll_pct <= 78:
        add_signal(5, "positive", "价格位置未明显过热")
    elif 20 <= boll_pct < 35 or 78 < boll_pct <= 90:
        add_signal(5, "neutral", "", "价格接近波动区边缘", "")
    else:
        add_signal(5, "negative", "", "", "价格位置偏极端，追价风险较高")

    raw_score = round(earned / available * 100) if available else None

    # Hard confirmation guardrails: do not call it strong when key evidence is absent.
    if raw_score is not None and intraday:
        rel_vol = intraday.get("relative_volume")
        above_vwap = price > float(intraday.get("vwap") or price)
        if not above_vwap or (rel_vol is not None and float(rel_vol) < 0.75):
            raw_score = min(raw_score, 54)
        if macd_hist <= 0 and price < ma20:
            raw_score = min(raw_score, 39)

    score = raw_score
    if score is None:
        status = "数据不足"
        status_class = "neutral"
    elif score >= 80:
        status = "股票端资金确认较强"
        status_class = "positive"
    elif score >= 65:
        status = "资金回流迹象增强"
        status_class = "positive"
    elif score >= 45:
        status = "多空信号分歧"
        status_class = "neutral"
    else:
        status = "资金确认偏弱"
        status_class = "negative"

    # Show only the most useful signals: negatives first, then positives/neutral.
    order = {"negative": 0, "positive": 1, "neutral": 2}
    signals = sorted(signals, key=lambda x: order.get(x["type"], 9))[:6]

    risks.append("当前分数是多因子信号一致度，不是历史胜率")
    risks.append("未包含真实Call/Put Flow、Gamma及Dark Pool")

    return {
        "status": status,
        "status_class": status_class,
        "score": score,
        "history_rate": None,
        "history_sample": 0,
        "signals": signals,
        "risks": risks[:2],
        "updated_at": datetime.now().strftime("%H:%M"),
        "options_connected": False,
        "model_version": "Stock Multi-Factor V2",
    }

def estimate_smart_money(price, vwap, volume_ratio, momentum):
    score = 0
    reasons = []

    if price > vwap:
        score += 1
        reasons.append("📈 价格高于VWAP")
    else:
        score -= 1
        reasons.append("📉 价格低于VWAP")

    if volume_ratio > 1.5:
        score += 1
        reasons.append("🔥 成交量明显放大")
    elif volume_ratio < 0.8:
        reasons.append("⚪ 成交量偏低")

    if momentum > 1:
        score += 1
        reasons.append("⚡ Momentum偏强")
    elif momentum < -1:
        score -= 1
        reasons.append("Momentum偏弱")

    if score >= 1:
        status = "🟢 主力偏吸筹"
    elif score <= -1:
        status = "🔴 主力偏流出"
    else:
        status = "🟡 主力观望"

    # 主力强度评分
    strength = 50
    strength += score * 20

    # 成交量确认
    if volume_ratio > 1.5:
        strength += 15

    elif volume_ratio < 0.8:
        strength -= 10

    # Momentum加强
    if momentum > 2:
        strength += 10

    elif momentum < -2:
        strength -= 10

    # 限制范围
    strength = max(10, min(100, strength))  

    if volume_ratio > 1.5:
        confidence = "高"

    elif volume_ratio > 1:
        confidence = "中"

    else:
        confidence = "低" 

    # 主力行为判断
    if volume_ratio > 1.2 and momentum > 1 and price > vwap:
        behavior = "🔥 放量吸筹"
    elif momentum > 1 and price > vwap:
        behavior = "🟢 趋势吸筹"
    elif volume_ratio > 1.2 and momentum < -1 and price < vwap:
        behavior = "🔴 放量派发"
    elif volume_ratio < 0.8 and momentum < 0 and price < vwap:
        behavior = "🟡 缩量洗盘"
    else:
        behavior = "⚪ 资金观望"    

    # 主力异动
    alert = ""

    if volume_ratio > 2 and momentum > 2:
        alert = "🚨 主力异动：放量拉升"

    elif volume_ratio > 2 and momentum < -2:
        alert = "⚠️ 主力异常撤退：放量下跌"

    elif volume_ratio > 1.5 and price > vwap:
        alert = "🔥 主力资金活跃"

    elif volume_ratio < 0.7:
        alert = "⚪ 市场交投清淡"    

    # 风险提示
    risk_warning = ""

    if momentum > 3 and volume_ratio < 1:
        risk_warning = "⚠️ 短线过热，谨慎追高"

    elif momentum < -3 and volume_ratio > 1.5:
        risk_warning = "⚠️ 放量下跌，注意风险"

    elif price < vwap and momentum < -2:
        risk_warning = "⚠️ 跌破VWAP，短线偏弱"

    elif volume_ratio < 0.6:
        risk_warning = "⚪ 成交量极低，观望为主" 

    # 主力转向
    rotation = ""

    if score >= 1 and momentum > 1:
        rotation = "🔄 主力转强：资金开始回流"

    elif score <= -1 and momentum < -1:
        rotation = "⚠️ 主力转弱：资金持续流出"

    elif score > 0 and momentum < 0:
        rotation = "🟡 主力分歧：趋势转弱"

    elif score < 0 and momentum > 0:
        rotation = "🟢 主力试盘：可能止跌"       

    return {
        "status": status,
        "reasons": reasons,
        "strength": strength,
        "confidence": confidence,
        "behavior": behavior,
        "alert": alert,
        "risk_warning": risk_warning,
        "rotation": rotation
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

    # Relative Strength（相对大盘强弱）
    spy_df = get_history("SPY", days=10, interval="1d")

    if spy_df is not None and not spy_df.empty and len(spy_df) >= 2:
        spy_last = float(spy_df["Close"].iloc[-1])
        spy_prev = float(spy_df["Close"].iloc[-2])

        spy_change_pct = ((spy_last - spy_prev) / spy_prev) * 100
    else:
        spy_change_pct = 0

    relative_strength = change_pct - spy_change_pct

    if relative_strength >= 2:
        relative_signal = "🔥 明显强于大盘"

    elif relative_strength >= 0.8:
        relative_signal = "🟢 强于大盘"

    elif relative_strength <= -2:
        relative_signal = "⚠️ 明显弱于大盘"

    elif relative_strength <= -0.8:
        relative_signal = "🔴 弱于大盘"

    else:
        relative_signal = "⚪ 跟随大盘"

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

    pe_info = get_pe_label(ticker)

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
    boll_pct = ((price - bb_lower) / (bb_upper - bb_lower)) * 100
    vol = float(last["Volume"])
    vol20 = float(last["VOL20"])

    intraday = get_intraday_signal(ticker)
    if intraday:
        vwap = intraday["vwap"]
        momentum = intraday["momentum"]
    else:
        vwap = (
            (df["Close"].tail(20) * df["Volume"].tail(20)).sum()
            / df["Volume"].tail(20).sum()
        )

    momentum = ((price - ma20) / ma20) * 100
    volume_ratio = vol / vol20 if vol20 > 0 else 1
    # 放量行为判断
    if volume_ratio >= 1.8 and price > ma20 and momentum > 2:
        volume_signal = "🔥 放量突破"

    elif volume_ratio >= 1.5 and change_pct < 0:
        volume_signal = "⚠️ 放量出货"

    elif volume_ratio < 0.8 and momentum > 0:
        volume_signal = "🟡 缩量洗盘"

    elif volume_ratio < 0.8:
        volume_signal = "⚪ 缩量观望"

    else:
        volume_signal = "正常"

    momentum = ((price - ma20) / ma20) * 100
    short_options_status = build_short_options_status(
        price=price,
        macd_hist=macd_hist,
        intraday=intraday,
        rsi=rsi,
        ma20=ma20,
        ma60=ma60,
        ma120=ma120,
        boll_pct=boll_pct,
        relative_strength=relative_strength,
        change_pct=change_pct,
    )

    # ===== 综合评分 =====
    final_score = 0

    # 趋势
    if price > ma20:
        final_score += 15

    if price > ma60:
        final_score += 15

    # RSI
    if 40 <= rsi <= 65:
        final_score += 15

    elif rsi > 75:
        final_score -= 10

    # BOLL
    if 30 <= boll_pct <= 70:
        final_score += 15

    elif boll_pct > 85:
        final_score -= 10

    # 相对强弱
    if relative_strength > 0.8:
        final_score += 15

    elif relative_strength < -0.8:
        final_score -= 10

    # 盘中确认（只使用可验证的股价/VWAP/同时段量比）
    if short_options_status["score"] is not None:
        if short_options_status["score"] >= 70:
            final_score += 15
        elif short_options_status["score"] < 35:
            final_score -= 5

    final_score = max(0, min(100, final_score))

    if final_score >= 75:
        final_decision = "🟢 可关注买入"

    elif final_score >= 55:
        final_decision = "🟡 观察等回调"

    else:
        final_decision = "🔴 暂时观望"

    atr = calc_atr(df)

    high_52w = float(df["High"].tail(252).max())
    low_52w = float(df["Low"].tail(252).min())

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

    # ===== 趋势因子 =====
    trend_factor = 1.0

    # 站上 MA20
    if close > ma20:
        trend_factor += 0.2

    # 站上 MA60
    if close > ma60:
        trend_factor += 0.2

    # RSI 偏强
    if rsi >= 60:
        trend_factor += 0.2

    # MACD 转强
    if macd_hist > 0:
        trend_factor += 0.2

    # RSI 偏弱
    if rsi < 40:
        trend_factor -= 0.2

    # 跌破 MA20
    if close < ma20:
        trend_factor -= 0.2

    # 最低保护
    trend_factor = max(trend_factor, 0.5)

    future_low = round(price - atr * 5 / trend_factor, 2)
    future_high = round(price + atr * 5 * trend_factor, 2)

    # ===== 趋势强度评分 =====
    trend_score = 50

    if close > ma20:
        trend_score += 10
    else:
        trend_score -= 10

    if close > ma60:
        trend_score += 10
    else:
        trend_score -= 10

    if rsi >= 60:
        trend_score += 10
    elif rsi < 40:
        trend_score -= 10

    if macd_hist > 0:
        trend_score += 10
    else:
        trend_score -= 5

    if vol20 and vol > vol20:
        trend_score += 5

    trend_score = max(0, min(100, trend_score))

    if trend_score >= 75:
        trend_label = "强势"
    elif trend_score >= 55:
        trend_label = "偏强"
    elif trend_score >= 40:
        trend_label = "中性"
    else:
        trend_label = "偏弱"

    # ===== 主力状态 =====
    if trend_score >= 70 and close > ma20 and vol > vol20:
        main_force = "🟢 主力吸筹"

    elif trend_score >= 55 and close > ma20:
        main_force = "🟡 温和流入"

    elif trend_score < 40 and close < ma20:
       main_force = "🔴 主力流出"

    elif atr_pct >= 5:
        main_force = "⚠️ 高波动洗盘"

    else:
        main_force = "⚪ 观望"    

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
    setup = f"⚪ 中性｜观察（{trend_label}）"

    if close > ma20 > ma60 and 40 <= rsi <= 65 and macd_hist > 0:
        setup = f"🟢 多头｜趋势突破（{trend_label}）"

    elif close > ma60 and close <= ma20 * 1.03 and 30 <= rsi <= 60:
        setup = f"🟢 多头｜上升回调（{trend_label}）"

    elif close > ma120 and rsi < 40:
        setup = f"🟡 中线｜回调观察（{trend_label}）"

    elif close < ma20 < ma60:
        setup = f"🔴 空头｜下降趋势（{trend_label}）"

    # 买卖计划
    support = max(float(last["BB_LOWER"]), float(last["LOW_20"]), ma60 * 0.98)
    buy_low = min(close * 0.97, ma20 * 0.99)
    buy_high = close * 0.995
    stop = min(support * 0.98, close * 0.93)
    target1 = close * 1.08
    target2 = close * 1.15

    # ===== 支撑 / 压力 =====
    support = round(
        min(ma20, ma60, bb_lower),
        2
    )

    resistance = round(
        max(bb_upper, float(last["HIGH_20"])),
        2
    )

    # ===== 风险回报 =====
    risk = buy_low - stop
    reward = target1 - buy_low
    rr_ratio = round(
        reward / risk,
        2
    ) if risk > 0 else 0


    # ===== 支撑距离 =====
    support_pct = round(((price - support) / support) * 100, 1)
    if support_pct >= 0:
        support_text = f"+{support_pct}%"
    else:
        support_text = f"{support_pct}%"

    # ===== 压力距离 =====
    resistance_pct = round(((resistance - price) / price) * 100, 1)
    if resistance_pct >= 0:
        resistance_text = f"+{resistance_pct}%"
    else:
        resistance_text = f"{resistance_pct}%"

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

    trend_stage, trend_stage_color = calc_trend_stage(
        price=price,
        ma20=ma20,
        ma60=ma60,
        ma120=ma120,
        rsi=rsi,
        boll_pct=boll_pct,
        volume_ratio=volume_ratio
    )     

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
        "atr_pct": atr_pct,
        "short_options": short_options_status,
        "atr": atr,
        "trend_score": trend_score,
        "trend_label": trend_label,
        "main_force": main_force,
        "macd_hist": round(macd_hist, 4),
        "relative_strength": round(relative_strength, 2),
        "relative_signal": relative_signal,
        "volume_signal": volume_signal,
        "final_score": final_score,
        "final_decision": final_decision,
        "technical_score": trend_score,
        "intraday_score": short_options_status.get("score"),
        "stock_flow_score": short_options_status.get("score"),
        "options_score": None,
        "options_connected": short_options_status.get("options_connected", False),
        "tech_signal": tech_signal,
        "final_score": final_score,
        "trend_stage": trend_stage,
        "trend_stage_color": trend_stage_color,
        "support": support,
        "resistance": resistance,
        "support_text": support_text,
        "resistance_text": resistance_text,
        "pe_text": pe_info["pe_text"],
        "pe_label": pe_info["pe_label"],
        "pe_color": pe_info["pe_color"],
        "flow_signal": "",
        "ma20": fmt_money(ma20),
        "ma50": fmt_money(ma50),
        "ma60": fmt_money(ma60),
        "ma120": fmt_money(ma120),
        "bb_upper": fmt_money(bb_upper),
        "bb_mid": fmt_money(bb_mid),
        "bb_lower": fmt_money(bb_lower),
        "boll_pos": boll_pos,
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



# ===== Market Technical V3 helpers =====
MARKET_TECH_CACHE = {"data": None, "time": 0}
MARKET_TECH_TTL = 300  # 5 minutes
MARKET_BREADTH_CACHE = {"data": None, "time": 0}
MARKET_BREADTH_TTL = 900  # 15 minutes
MARKET_BREADTH_LIMIT = int(os.getenv("MARKET_BREADTH_LIMIT", "30"))

def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "").replace("%", "").replace("x", "").strip()
        return float(value)
    except Exception:
        return default

def calculate_adx(df, period=14):
    """Return ADX, +DI and -DI from daily OHLC data."""
    if df is None or df.empty or len(df) < period + 5:
        return 0.0, 0.0, 0.0
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return (
        round(float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else 0.0, 2),
        round(float(plus_di.iloc[-1]) if pd.notna(plus_di.iloc[-1]) else 0.0, 2),
        round(float(minus_di.iloc[-1]) if pd.notna(minus_di.iloc[-1]) else 0.0, 2),
    )

def _pct_return(series, periods):
    if series is None or len(series) <= periods:
        return 0.0
    old = float(series.iloc[-periods-1])
    new = float(series.iloc[-1])
    return ((new / old) - 1) * 100 if old else 0.0

def build_index_technical(ticker, spy_returns=None):
    """Technical structure for one market ETF. Uses daily bars only."""
    try:
        df = get_history(ticker, days=220, interval="1d")
        if df is None or df.empty or len(df) < 130:
            return {}
        df = df.copy()
        df.columns = [str(c).capitalize() for c in df.columns]
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        volume = df["Volume"].astype(float)
        price = float(close.iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma60 = float(close.rolling(60).mean().iloc[-1])
        ma120 = float(close.rolling(120).mean().iloc[-1])
        ma20_5 = float(close.rolling(20).mean().iloc[-6])
        ma60_10 = float(close.rolling(60).mean().iloc[-11])
        ma20_slope = ((ma20 / ma20_5) - 1) * 100 if ma20_5 else 0
        ma60_slope = ((ma60 / ma60_10) - 1) * 100 if ma60_10 else 0
        _, _, hist = calc_macd(close)
        macd_hist = float(hist.iloc[-1])
        rsi = float(calc_rsi(close).iloc[-1])
        adx, plus_di, minus_di = calculate_adx(df)
        tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1])
        atr_pct = (atr14 / price * 100) if price else 0
        high20_prev = float(high.rolling(20).max().iloc[-2])
        low20_prev = float(low.rolling(20).min().iloc[-2])
        vol20 = float(volume.rolling(20).mean().iloc[-1])
        volume_ratio = float(volume.iloc[-1] / vol20) if vol20 else 0

        trend_score = 0
        reasons = []
        if price > ma20: trend_score += 15; reasons.append("价格站上MA20")
        else: reasons.append("价格低于MA20")
        if price > ma60: trend_score += 15
        if price > ma120: trend_score += 10
        if ma20 > ma60: trend_score += 15
        if ma60 > ma120: trend_score += 10
        if ma20_slope > 0.3: trend_score += 10
        elif ma20_slope > 0: trend_score += 5
        if macd_hist > 0: trend_score += 10
        if adx >= 25 and plus_di > minus_di: trend_score += 15
        elif adx >= 18 and plus_di > minus_di: trend_score += 8
        trend_score = max(0, min(100, round(trend_score)))

        if trend_score >= 75 and price > high20_prev: stage = "主升趋势"
        elif trend_score >= 70: stage = "上升趋势"
        elif price > ma20 and price < max(ma60, ma120): stage = "反弹，尚未反转"
        elif trend_score >= 50: stage = "震荡修复"
        elif price < ma20 and price < ma60: stage = "下跌趋势"
        else: stage = "中性震荡"

        returns = {"1d": _pct_return(close, 1), "5d": _pct_return(close, 5), "20d": _pct_return(close, 20)}
        rs = {}
        if spy_returns and ticker != "SPY":
            rs = {k: round(returns[k] - spy_returns.get(k, 0), 2) for k in returns}
        else:
            rs = {"1d": 0.0, "5d": 0.0, "20d": 0.0}

        breakout = price > high20_prev and volume_ratio >= 1.2
        return {
            "trend_score_v3": trend_score, "market_stage": stage,
            "adx": adx, "plus_di": plus_di, "minus_di": minus_di,
            "ma20_slope": round(ma20_slope, 2), "ma60_slope": round(ma60_slope, 2),
            "macd_hist_v3": round(macd_hist, 4), "atr_pct_v3": round(atr_pct, 2),
            "rs_1d": rs["1d"], "rs_5d": rs["5d"], "rs_20d": rs["20d"],
            "return_1d": round(returns["1d"], 2), "return_5d": round(returns["5d"], 2), "return_20d": round(returns["20d"], 2),
            "breakout_confirmed": breakout, "volume_ratio_daily": round(volume_ratio, 2),
            "technical_reasons_v3": reasons,
        }
    except Exception as exc:
        print(f"Market technical error {ticker}: {exc}")
        return {}


def enrich_index_card_readability(card):
    """Translate technical market fields into concise Chinese labels."""
    if not card or card.get("ticker") == "VIX":
        return card

    adx = _safe_float(card.get("adx"))
    plus_di = _safe_float(card.get("plus_di"))
    minus_di = _safe_float(card.get("minus_di"))
    ma20_slope = _safe_float(card.get("ma20_slope"))
    ma60_slope = _safe_float(card.get("ma60_slope"))

    if adx >= 30:
        card["adx_label"] = "强趋势"
        card["adx_class"] = "positive"
    elif adx >= 20:
        card["adx_label"] = "趋势形成中"
        card["adx_class"] = "neutral"
    else:
        card["adx_label"] = "震荡整理"
        card["adx_class"] = "muted"

    di_gap = plus_di - minus_di
    if di_gap >= 5:
        card["di_label"] = "多头占优"
        card["di_class"] = "positive"
    elif di_gap <= -5:
        card["di_label"] = "空头占优"
        card["di_class"] = "negative"
    else:
        card["di_label"] = "多空接近"
        card["di_class"] = "neutral"

    if ma20_slope >= 1:
        card["ma20_label"] = "短线加速上涨"
        card["ma20_class"] = "positive"
    elif ma20_slope > 0:
        card["ma20_label"] = "短线上行"
        card["ma20_class"] = "positive"
    elif ma20_slope > -0.5:
        card["ma20_label"] = "短线走平"
        card["ma20_class"] = "neutral"
    else:
        card["ma20_label"] = "短线走弱"
        card["ma20_class"] = "negative"

    if ma60_slope >= 1:
        card["ma60_label"] = "中期明显转强"
        card["ma60_class"] = "positive"
    elif ma60_slope > 0:
        card["ma60_label"] = "中期向上"
        card["ma60_class"] = "positive"
    elif ma60_slope > -0.5:
        card["ma60_label"] = "中期横盘"
        card["ma60_class"] = "neutral"
    else:
        card["ma60_label"] = "中期下降"
        card["ma60_class"] = "negative"

    if card.get("breakout_confirmed"):
        card["breakout_label"] = "✓ 已突破"
        card["breakout_class"] = "positive"
    elif _safe_float(card.get("return_1d")) > 0 and _safe_float(card.get("trend_score_v3")) >= 60:
        card["breakout_label"] = "△ 接近突破"
        card["breakout_class"] = "neutral"
    else:
        card["breakout_label"] = "未突破"
        card["breakout_class"] = "muted"

    for key, label_key in (("rs_1d", "rs1"), ("rs_5d", "rs5"), ("rs_20d", "rs20")):
        value = _safe_float(card.get(key))
        if card.get("ticker") == "SPY":
            card[f"{label_key}_text"] = "基准指数"
            card[f"{label_key}_class"] = "muted"
        elif value >= 1:
            card[f"{label_key}_text"] = f"+{value:.2f}% 明显跑赢"
            card[f"{label_key}_class"] = "positive"
        elif value > 0:
            card[f"{label_key}_text"] = f"+{value:.2f}% 跑赢"
            card[f"{label_key}_class"] = "positive"
        elif value <= -1:
            card[f"{label_key}_text"] = f"{value:.2f}% 明显落后"
            card[f"{label_key}_class"] = "negative"
        elif value < 0:
            card[f"{label_key}_text"] = f"{value:.2f}% 落后"
            card[f"{label_key}_class"] = "negative"
        else:
            card[f"{label_key}_text"] = "持平"
            card[f"{label_key}_class"] = "neutral"

    summary = []
    summary.append(f"{card.get('adx_label', '趋势不明')}，{card.get('di_label', '多空接近')}")
    summary.append(card.get("ma20_label", "短线状态不明"))
    summary.append(card.get("breakout_label", "突破状态不明"))
    card["readable_summary"] = "；".join(summary)
    return card


def get_market_breadth_v3(spy_return_1d=0.0):
    """Breadth from the configured universe, cached and capped for Render stability."""
    now = time.time()
    if MARKET_BREADTH_CACHE["data"] and now - MARKET_BREADTH_CACHE["time"] < MARKET_BREADTH_TTL:
        return MARKET_BREADTH_CACHE["data"]
    try:
        symbols = list(dict.fromkeys(UNIVERSE))[:max(10, MARKET_BREADTH_LIMIT)]
    except Exception:
        symbols = []
    stats = {"valid": 0, "up": 0, "ma20": 0, "ma60": 0, "ma120": 0, "macd": 0, "rsi50": 0, "outperform": 0}
    for ticker in symbols:
        try:
            df = get_history(ticker, days=220, interval="1d")

            if df is None or df.empty:
                print(f"⚠️ Breadth no data: {ticker}")
                continue

            if len(df) < 125:
                print(f"⚠️ Breadth insufficient history: {ticker}, rows={len(df)}")
                continue
            df = df.copy(); df.columns = [str(c).capitalize() for c in df.columns]
            close = df["Close"].astype(float)
            price = float(close.iloc[-1]); prev = float(close.iloc[-2])
            ma20 = float(close.rolling(20).mean().iloc[-1]); ma60 = float(close.rolling(60).mean().iloc[-1]); ma120 = float(close.rolling(120).mean().iloc[-1])
            _, _, hist = calc_macd(close); rsi = float(calc_rsi(close).iloc[-1])
            ret = ((price / prev) - 1) * 100 if prev else 0
            stats["valid"] += 1
            stats["up"] += ret > 0
            stats["ma20"] += price > ma20
            stats["ma60"] += price > ma60
            stats["ma120"] += price > ma120
            stats["macd"] += float(hist.iloc[-1]) > 0
            stats["rsi50"] += rsi > 50
            stats["outperform"] += ret > spy_return_1d
        except Exception as e:
            print(f"⚠️ Breadth failed for {ticker}: {e}")
            continue
    n = stats["valid"]
    pct = lambda key: round(stats[key] / n * 100) if n else 0
    data = {
        "sample_size": n, "requested_size": len(symbols),
        "up_pct": pct("up"), "above_ma20_pct": pct("ma20"), "above_ma60_pct": pct("ma60"),
        "above_ma120_pct": pct("ma120"), "macd_positive_pct": pct("macd"),
        "rsi_above_50_pct": pct("rsi50"), "outperform_spy_pct": pct("outperform"),
    }
    data["score"] = round(
        data["above_ma20_pct"] * .25 + data["above_ma60_pct"] * .20 + data["above_ma120_pct"] * .15 +
        data["macd_positive_pct"] * .15 + data["rsi_above_50_pct"] * .10 + data["outperform_spy_pct"] * .15
    )
    if data["up_pct"] >= 60 and data["above_ma20_pct"] >= 55: data["label"] = "广度健康"
    elif data["up_pct"] >= 50: data["label"] = "广度一般"
    else: data["label"] = "广度偏弱"
    MARKET_BREADTH_CACHE["data"] = data; MARKET_BREADTH_CACHE["time"] = now
    return data

def get_market_cards():
    cards = []

    # Daily multi-period relative strength uses SPY as the common baseline.
    spy_tech = build_index_technical("SPY")
    spy_returns = {
        "1d": spy_tech.get("return_1d", 0),
        "5d": spy_tech.get("return_5d", 0),
        "20d": spy_tech.get("return_20d", 0),
    }

    for t in MARKET_TICKERS:
        analysis, _, _, err = analyze_ticker(t)
        technical_v3 = spy_tech if t == "SPY" else build_index_technical(t, spy_returns)

        if err or not analysis:
            card = {
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
            }
        else:
            card = {
                "ticker": t,
                "price": analysis.get("price_fmt"),
                "change_pct": analysis.get("change_pct"),
                "change_class": "up" if (analysis.get("change_num") or 0) >= 0 else "down",
                "decision": analysis.get("decision"),
                "score": analysis.get("final_score") or analysis.get("score") or 0,
                "technical_score": analysis.get("technical_score") or 0,
                "intraday_score": analysis.get("intraday_score") or 0,
                "stock_flow_score": analysis.get("stock_flow_score") or 0,
                "rsi": analysis.get("rsi"),
                "setup": analysis.get("setup"),
                "trend_label": analysis.get("trend_label"),
                "trend_arrow": analysis.get("trend_arrow"),
                "vwap_status": (analysis.get("short_options") or {}).get("vwap_status", "--"),
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
            }
        card.update(technical_v3)
        enrich_index_card_readability(card)
        cards.append(card)

    cards.append(get_vix_card())
    return cards

def build_market_dashboard(cards, market_status):
    """Market Regime V3: trend, breadth, intraday, cross-index and volatility."""
    index_cards = [c for c in cards if c.get("ticker") in MARKET_TICKERS]
    vix_card = next((c for c in cards if c.get("ticker") == "VIX"), {})
    by_ticker = {c.get("ticker"): c for c in index_cards}

    # 1) Daily trend structure (ADX, MA slopes and multi-period structure)
    trend_values = [_safe_float(c.get("trend_score_v3")) for c in index_cards if c.get("trend_score_v3") is not None]
    trend_score = round(sum(trend_values) / len(trend_values)) if trend_values else 50

    # 2) Breadth from the configured universe. Cached 15 minutes and capped by env setting.
    spy_return_1d = _safe_float((by_ticker.get("SPY") or {}).get("return_1d"))
    breadth = get_market_breadth_v3(spy_return_1d)
    breadth_score = int(breadth.get("score") or 0)

    # 3) Intraday confirmation from already-computed stock analysis fields.
    intraday_values = [_safe_float(c.get("intraday_score")) for c in index_cards if _safe_float(c.get("intraday_score")) >= 0]
    intraday_score = round(sum(intraday_values) / len(intraday_values)) if intraday_values else 50

    # 4) Cross-index confirmation.
    confirming = sum(1 for c in index_cards if _safe_float(c.get("trend_score_v3")) >= 60)
    cross_index_score = round(confirming / len(index_cards) * 100) if index_cards else 0
    if confirming == 4: cross_label = "4/4 全面确认"
    elif confirming == 3: cross_label = "3/4 多数确认"
    elif confirming == 2: cross_label = "2/4 结构分化"
    elif confirming == 1: cross_label = "1/4 上涨集中"
    else: cross_label = "0/4 全面偏弱"

    # 5) Relative strength persistence, not only today's change.
    rs_points = []
    for c in index_cards:
        if c.get("ticker") == "SPY":
            continue
        vals = [_safe_float(c.get("rs_1d")), _safe_float(c.get("rs_5d")), _safe_float(c.get("rs_20d"))]
        rs_points.append(sum(1 for x in vals if x > 0) / 3 * 100)
    relative_strength_score = round(sum(rs_points) / len(rs_points)) if rs_points else 50

    # 6) Volatility regime combines VIX level/change and SPY ATR.
    vix_price = _safe_float(vix_card.get("price"), 20)
    vix_change = _safe_float(vix_card.get("change_pct"), 0)
    spy_atr = _safe_float((by_ticker.get("SPY") or {}).get("atr_pct_v3"), 1.5)
    if vix_price < 16 and vix_change <= 3 and spy_atr < 1.8:
        volatility_regime, volatility_score = "低波动多头", 85
    elif vix_price < 22 and vix_change < 8:
        volatility_regime, volatility_score = "正常波动", 65
    elif vix_change >= 8 or vix_price >= 25:
        volatility_regime, volatility_score = "风险升高", 25
    else:
        volatility_regime, volatility_score = "高波动环境", 40

    market_score = round(
        trend_score * .30 + breadth_score * .25 + intraday_score * .20 +
        cross_index_score * .10 + relative_strength_score * .10 + volatility_score * .05
    )
    # Hard caps reduce false bullish readings when participation or core indices are weak.
    if breadth_score < 35:
        market_score = min(market_score, 60)
    if _safe_float((by_ticker.get("SPY") or {}).get("trend_score_v3")) < 40 and _safe_float((by_ticker.get("QQQ") or {}).get("trend_score_v3")) < 40:
        market_score = min(market_score, 45)

    positive = sum(1 for c in index_cards if _safe_float(c.get("change_pct")) > 0)
    participation = round(positive / len(index_cards) * 100) if index_cards else 0
    ranked = sorted(index_cards, key=lambda c: _safe_float(c.get("change_pct")), reverse=True)
    strongest = ranked[0] if ranked else {}
    weakest = ranked[-1] if ranked else {}

    if market_score >= 75 and breadth_score >= 60:
        bias, bias_class, strategy = "偏多", "positive", "顺势持有，优先强势回调与放量突破，避免追弱势反弹"
        regime = "主升趋势" if trend_score >= 75 else "上涨趋势"
    elif market_score >= 60:
        bias, bias_class, strategy = "震荡偏多", "neutral", "控制仓位，优先强于SPY且站稳VWAP的股票"
        regime = "指数偏强，仍需广度确认" if breadth_score < 55 else "上涨初期"
    elif market_score >= 45:
        bias, bias_class, strategy = "震荡", "neutral", "轻仓观察，等待成交量与指数同步确认"
        regime = "结构分化"
    else:
        bias, bias_class, strategy = "偏弱", "negative", "减少追涨，等待SPY和QQQ重新站稳关键均线"
        regime = "下跌或弱势修复"

    if vix_price < 16: risk_label, risk_class = "中等偏低", "positive"
    elif vix_price < 22: risk_label, risk_class = "中等", "neutral"
    else: risk_label, risk_class = "偏高", "negative"

    style_text = "市场风格暂不明确"
    if strongest:
        style_text = f"{strongest.get('ticker')}领涨"
        if strongest.get("ticker") == "SOXX": style_text += "，半导体与AI方向占优"
        elif strongest.get("ticker") == "QQQ": style_text += "，科技成长占优"
        elif strongest.get("ticker") == "DIA": style_text += "，传统蓝筹占优"
        else: style_text += "，大盘整体较强"
    if weakest:
        style_text += f"；{weakest.get('ticker')}相对落后"

    reasons = []
    if trend_score >= 65: reasons.append("✓ 日线趋势结构偏强")
    else: reasons.append("△ 日线趋势尚未全面确认")
    if breadth_score >= 60: reasons.append(f"✓ 市场广度健康（MA20上方 {breadth.get('above_ma20_pct', 0)}%）")
    elif breadth_score >= 45: reasons.append(f"△ 市场广度一般（MA20上方 {breadth.get('above_ma20_pct', 0)}%）")
    else: reasons.append(f"✕ 市场广度偏弱（MA20上方 {breadth.get('above_ma20_pct', 0)}%）")
    reasons.append(f"{'✓' if intraday_score >= 60 else '△' if intraday_score >= 45 else '✕'} 盘中确认 {intraday_score}/100")
    reasons.append(f"{'✓' if confirming >= 3 else '△' if confirming == 2 else '✕'} 指数确认：{cross_label}")
    if volatility_score >= 65: reasons.append(f"✓ 波动环境：{volatility_regime}")
    else: reasons.append(f"⚠ 波动环境：{volatility_regime}")

    return {
        "score": market_score,
        "bias": bias,
        "bias_class": bias_class,
        "participation": participation,
        "risk_label": risk_label,
        "risk_class": risk_class,
        "strategy": strategy,
        "style_text": style_text,
        "strongest": strongest,
        "weakest": weakest,
        "vix": vix_card,
        # V3 technical fields (safe for future template use)
        "regime": regime,
        "trend_score": trend_score,
        "breadth_score": breadth_score,
        "intraday_score": intraday_score,
        "cross_index_score": cross_index_score,
        "cross_index_label": cross_label,
        "relative_strength_score": relative_strength_score,
        "volatility_score": volatility_score,
        "volatility_regime": volatility_regime,
        "breadth": breadth,
        "reasons": reasons,
    }

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
    sector_rotation = []

    try:
        # === SPY / QQQ ===
        spy = analyze_ticker("SPY")[0]
        qqq = analyze_ticker("QQQ")[0]

        spy_up = spy and float(spy["price"]) > float(spy["ma20"].replace("$",""))
        qqq_up = qqq and float(qqq["price"]) > float(qqq["ma20"].replace("$",""))

        vix = get_vix_card()
        vix_price = float(vix["price"]) if vix["price"] != "--" else 20
        vix_text = f"{vix['price']} ({vix['change_pct']})"
        sector_rotation = get_sector_rotation()

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
        "sector_rotation": sector_rotation
    }

    MARKET_STATUS_CACHE["data"] = result
    MARKET_STATUS_CACHE["time"] = time.time()

    return result

SPY_TICKERS = load_tickers("spy.txt")
def get_rebound_signals():
    picks = []
    tickers = load_tickers("spy.txt")

    for ticker in tickers:
        analysis, _, _, err = analyze_ticker(ticker)
        if err or not analysis:
            continue
        try:
            price = clean_num(analysis["price"])
            rsi = clean_num(analysis["rsi"])
            macd_hist = clean_num(analysis["macd_hist"])
            ma20 = clean_num(analysis["ma20"])
            ma60 = clean_num(analysis.get("ma60", 0))
            boll_pct = clean_num(analysis.get("boll_pct", 50))
            volume_ratio = clean_num(analysis.get("volume_ratio", 1))
        
            if price <= 0 or ma20 <= 0:
                continue

            # 距离 MA20
            near_ma20 = abs(price - ma20) / ma20 <= 0.06
            score = 0
            reasons = []

            # 低位反弹核心
            if 35 <= rsi <= 55:
                score += 2
                reasons.append("RSI低位回升")
            if macd_hist > -1.5:
                score += 1
                reasons.append("MACD未继续恶化")
            if near_ma20:
                score += 2
                reasons.append("靠近MA20")
            if ma60 > 0 and price > ma60:
                score += 2
                reasons.append("中期趋势未破")
            if 20 <= boll_pct <= 65:
                score += 2
                reasons.append("BOLL低位/中低位")
            if volume_ratio >= 0.8:
                score += 1
                reasons.append("量能正常")

            # 太高不要
            if rsi > 60 or boll_pct > 85:
                continue
            if score < 5:
                continue
            support = max(ma20 * 0.97, price * 0.94)
            stop = round(support * 0.98, 2)
            stop_pct = round((price - stop) / price * 100, 1)
            if stop_pct > 10:
                continue

            buy_low = round(price * 0.985, 2)
            buy_high = round(price * 1.005, 2)
            target1 = round(price * 1.06, 2)
            target2 = round(price * 1.12, 2)
            risk_pct = round((price - stop) / price * 100, 1)
            reward_pct = round((target1 - price) / price * 100, 1)
            rr = round(reward_pct / risk_pct, 2) if risk_pct > 0 else 0

            if rr < 1.2:
                continue
            if rr >= 3:
                rr_grade = "🔥 极佳"
            elif rr >= 2:
                rr_grade = "🟢 优秀"
            else:
                rr_grade = "🟡 合格"

            picks.append({
                "ticker": analysis["ticker"],
                "price": analysis["price_fmt"],
                "rsi": analysis["rsi"],
                "macd": analysis["macd_hist"],
                "ma20": analysis["ma20"],
                "decision": "💎 低位反弹",
                "setup": "低位反弹信号",
                "buy_low": buy_low,
                "buy_high": buy_high,
                "stop": stop,
                "stop_pct": stop_pct,
                "target1": target1,
                "target2": target2,
                "score": score,
                "risk_pct": risk_pct,
                "rr": rr,
                "reasons": "、".join(reasons)
            })

        except Exception as e:
            print("rebound error:", ticker, e)
            continue

    return sorted(
        picks,
        key=lambda x: (x["rr"], x["score"]),
        reverse=True
    )[:3]


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
        market_dashboard=None,
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
    market_status = get_market_status()
    market_dashboard = build_market_dashboard(market_cards, market_status)

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
        market_dashboard=market_dashboard,
    )

@app.route("/opportunities", methods=["GET", "POST"])
def opportunities():
    """Unified on-demand scanner for today's opportunities.

    GET only renders the page. The expensive market scan runs only after the
    user presses the scan button (POST), which helps keep Render stable.
    """
    scanned = request.method == "POST"
    results = scan_market_hunter() if scanned else []

    categories = {
        "breakout": [],
        "pullback": [],
        "early": [],
        "trend": [],
    }

    for item in results:
        setup = str(item.get("setup_type", ""))
        if "突破" in setup:
            categories["breakout"].append(item)
        elif "回调" in setup:
            categories["pullback"].append(item)
        elif "低位" in setup or "启动" in setup:
            categories["early"].append(item)
        else:
            categories["trend"].append(item)

    return render_template(
        "opportunities.html",
        scanned=scanned,
        results=results,
        categories=categories,
        cache_minutes=round(HUNTER_CACHE_TTL / 60),
    )


# Keep old bookmarks working without running the old duplicate scanners.
@app.route("/scan_rebound")
def scan_rebound():
    return redirect(url_for("opportunities"))


@app.route("/market_hunter")
def market_hunter():
    return redirect(url_for("opportunities"))

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