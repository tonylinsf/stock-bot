import os
from datetime import datetime as _dt
from zoneinfo import ZoneInfo

from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# 載入 .env（如有）
load_dotenv()

client = OpenAI()  # 用環境變數 OPENAI_API_KEY
app = Flask(__name__)

ET = ZoneInfo("America/New_York")

# ====== 大盤概覽缓存（SPY/QQQ/DIA）======
MARKET_CACHE: list[dict] | None = None
MARKET_CACHE_DATE: str | None = None  # ET 日期字串，例如 "2025-12-15"


# =========================================================
# 技術指標
# =========================================================
def build_today_status(last_close, ma20, ma60, rsi14, macd=None, macd_signal=None):
    if last_close is None or ma20 is None or ma60 is None or rsi14 is None:
        return None, None

    # 趨勢基礎
    above20 = last_close >= ma20
    above60 = last_close >= ma60
    ma_bull = ma20 >= ma60

    # RSI 狀態
    if rsi14 >= 70:
        rsi_tag = "RSI偏高(≥70)"
    elif rsi14 <= 30:
        rsi_tag = "RSI偏低(≤30)"
    else:
        rsi_tag = f"RSI中性({rsi14:.0f})"

    # MACD（可選）
    macd_tag = None
    if macd is not None and macd_signal is not None:
        macd_tag = "MACD偏多" if macd >= macd_signal else "MACD偏空"

    # 一句總結
    if ma_bull and above20 and above60 and rsi14 >= 50:
        text = f"✅ 偏多：站上MA20/60，{rsi_tag}"
        cls = "up"
    elif (not ma_bull) and (not above20) and (not above60) and rsi14 <= 50:
        text = f"⚠️ 偏空：跌破MA20/60，{rsi_tag}"
        cls = "down"
    else:
        text = f"⏸️ 盤整：MA糾纏/方向未明，{rsi_tag}"
        cls = "mid"

    if macd_tag:
        text += f"，{macd_tag}"

    return text, cls


def calc_atr_pct(df: pd.DataFrame, period: int = 14) -> float | None:
    """
    用 High/Low/Close 計 ATR，最後回傳 ATR%（ATR/Price*100）
    """
    try:
        if df is None or df.empty:
            return None
        if not all(c in df.columns for c in ["High", "Low", "Close"]):
            return None

        high = df["High"].dropna()
        low = df["Low"].dropna()
        close = df["Close"].dropna()
        if len(close) < period + 2:
            return None

        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        last_atr = float(atr.iloc[-1])
        last_price = float(close.iloc[-1])
        if last_price == 0:
            return None
        return round(last_atr / last_price * 100, 2)
    except Exception:
        return None


def rolling_levels(df: pd.DataFrame, window: int = 20) -> tuple[float | None, float | None]:
    """
    近 window 日 支撐/阻力：Low rolling min / High rolling max
    """
    try:
        if df is None or df.empty:
            return None, None
        if not all(c in df.columns for c in ["High", "Low"]):
            return None, None
        if len(df) < window:
            return None, None
        support = float(df["Low"].dropna().rolling(window).min().iloc[-1])
        resist = float(df["High"].dropna().rolling(window).max().iloc[-1])
        return round(support, 2), round(resist, 2)
    except Exception:
        return None, None
    
def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss.replace(0, pd.NA))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist


def calc_atr(df: pd.DataFrame, period: int = 14):
    if df is None or df.empty:
        return None

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    return atr


def calc_boll(close: pd.Series, days: int = 20):
    mid = close.rolling(days).mean()
    std = close.rolling(days).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return upper, mid, lower


def format_volume(v: float) -> str:
    try:
        v = float(v)
    except Exception:
        return "-"
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v/1_000:.1f}K"
    return str(int(v))


def compute_trend_score(ind: dict) -> int:
    score = 50

    rsi = ind.get("rsi")
    macd = ind.get("macd")
    ma5 = ind.get("ma5")
    ma20 = ind.get("ma20")
    ma60 = ind.get("ma60")
    last_price = ind.get("last_price")
    boll_up = ind.get("boll_upper")
    boll_low = ind.get("boll_lower")

    if rsi is not None:
        if 45 <= rsi <= 60:
            score += 10
        elif 35 <= rsi < 45 or 60 < rsi <= 70:
            score += 5
        elif rsi < 30 or rsi > 75:
            score -= 10

    if last_price and ma60:
        dist = (last_price - ma60) / ma60 * 100
        if -5 <= dist <= 5:
            score += 10
        elif -10 <= dist < -5:
            score += 5
        elif dist < -15 or dist > 15:
            score -= 10

    if ma5 and ma20 and ma60:
        if ma5 > ma20 > ma60:
            score += 10
        elif ma5 < ma20 < ma60:
            score -= 10

    if macd is not None:
        score += 5 if macd > 0 else -5

    if boll_up and boll_low and last_price:
        width = (boll_up - boll_low) / last_price * 100
        if width > 25:
            score -= 5

    return max(0, min(100, int(round(score))))


# =========================================================
# AI（個股分析用）
# =========================================================
def build_ai_prompt(ticker: str, ind: dict) -> str:
    return f"""
你係一位專業股票技術分析顧問，請用簡單中文（廣東話風格）、條理清晰，幫我分析以下股票：

股票代號：{ticker}
最新收市價：{ind.get("last_price")}（日期：{ind.get("last_date")}）
最近三個月走勢概要：{ind.get("trend_text")}

技術指標（最近一日）：
- RSI14：{ind.get("rsi")}
- MA5 / MA20 / MA60：{ind.get("ma5")} / {ind.get("ma20")} / {ind.get("ma60")}
- MACD：{ind.get("macd")}
- 布林帶：上軌 {ind.get("boll_upper")}，中軌 {ind.get("boll_mid")}，下軌 {ind.get("boll_lower")}
- 成交量：{ind.get("volume_str")}，20 日平均量：{ind.get("avg20_volume_str")}
- 52 週區間：{ind.get("low_52w")} ~ {ind.get("high_52w")}
- 現價距離 52 週高位：約 {ind.get("from_high_pct")}%

系統計算趨勢分數（0–100）：{ind.get("trend_score")}

請你：
1) 用 2–3 句描述最近三個月走勢（偏強、偏弱、定係橫行），可引用 RSI/均線/MACD。
2) 估計「可能買入區間」同「大約止蝕位」（風險太大可建議觀望）。
3) 提醒 1–2 點要留意嘅風險（例如跌穿支持、消息風險、大市）。
4) 最後加一句：以上只係參考，唔係任何投資建議。

用分段輸出，換行排版清晰。
"""


def get_ai_advice(prompt: str) -> str:
    try:
        resp = client.responses.create(
            model="gpt-5-mini",   # 你話用 gpt-5-mini
            input=prompt,
        )
        return resp.output_text
    except Exception as e:
        return f"AI 分析出錯：{e}"


def build_ai_short_prompt(ticker: str, indicators: dict) -> str:
    return f"""
你係一個專業股票短線交易顧問。

根據以下最新技術指標，請用 **一句話** 總結呢隻股票短線情況：
必須包含：
1）偏強 / 偏弱 / 橫行
2）觀望 / 分段吸納 / 減持
3）一句風險提示

只輸出一句話，不要分點。

股票：{ticker}
最新收市價：{indicators.get("last_price")}
當日漲跌：{indicators.get("change_pct")}%
RSI：{indicators.get("rsi")}
MACD：{indicators.get("macd")}
MA5：{indicators.get("ma5")}
MA20：{indicators.get("ma20")}
MA60：{indicators.get("ma60")}
BOLL 上軌：{indicators.get("boll_upper")}
BOLL 中軌：{indicators.get("boll_mid")}
BOLL 下軌：{indicators.get("boll_lower")}
成交量：{indicators.get("volume_str")}（20 日平均：{indicators.get("avg20_volume_str")}）
52 週高低：{indicators.get("low_52w")} ~ {indicators.get("high_52w")}
趨勢分數：{indicators.get("trend_score")}
"""


def get_ai_short_summary(prompt: str) -> str:
    try:
        resp = client.responses.create(
            model="gpt-5-mini",
            input=prompt,
        )
        return resp.output_text
    except Exception as e:
        return f"短評生成失敗：{e}"


# =========================================================
# Market Overview（SPY/QQQ/DIA）升級版
# =========================================================
def _et_now():
    return _dt.now(ET)


def _download_daily(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            ticker,
            period="3mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty or "Close" not in df.columns:
            return None
        return df
    except Exception:
        return None


def _download_weekly(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            ticker,
            period="3mo",
            interval="1wk",
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty or "Close" not in df.columns:
            return None
        return df
    except Exception:
        return None


def _market_signals(
    rsi14: float | None,
    price: float | None,
    ma20: float | None,
    ma60: float | None
):
    s = []

    # ===== RSI 新分界：80 / 50 / 20 =====
    if rsi14 is not None:
        if rsi14 >= 80:
            s.append("RSI 極度超買（>80）")
        elif rsi14 >= 50:
            s.append("RSI 強勢區（50–80）")
        elif rsi14 <= 20:
            s.append("RSI 極度超賣（<20）")
        else:
            s.append("RSI 中性偏弱（20–50）")

    # ===== 均線 + 價格趨勢 =====
    if price is not None and ma20 is not None and ma60 is not None:
        if ma20 > ma60 and price > ma20:
            s.append("偏強（價在 MA20 上，MA20 > MA60）")
        elif ma20 < ma60 and price < ma20:
            s.append("偏弱（價在 MA20 下，MA20 < MA60）")
        else:
            s.append("橫行／拉鋸")

    return s

def _rsi_badge(rsi14: float | None):
    """
    回傳 (label, css_class)
    4 級：極度超買 / 強勢 / 中性偏弱 / 極度超賣
    """
    if rsi14 is None:
        return ("RSI --", "badge-neutral")

    if rsi14 >= 80:
        return (f"RSI {rsi14:.1f} 極度超買", "badge-hot")
    if rsi14 >= 50:
        return (f"RSI {rsi14:.1f} 強勢", "badge-strong")
    if rsi14 <= 20:
        return (f"RSI {rsi14:.1f} 極度超賣", "badge-cold")
    return (f"RSI {rsi14:.1f} 中性偏弱", "badge-weak")


def _market_grade(rsi14: float | None, price: float | None, ma20: float | None, ma60: float | None):
    """
    A / B / C（簡單但實用）
    - A：趨勢強（價 > MA20 且 MA20 > MA60）+ RSI 在強勢區 (>=50 且 <80)
    - B：中性/拉鋸（其他非極端情況）
    - C：偏弱（價 < MA20 且 MA20 < MA60）或 RSI 極端（>80 / <20 風險偏高）
    """
    if rsi14 is None or price is None or ma20 is None or ma60 is None:
        return "B"

    strong_trend = (price > ma20) and (ma20 > ma60)
    weak_trend = (price < ma20) and (ma20 < ma60)

    extreme = (rsi14 >= 80) or (rsi14 <= 20)

    if strong_trend and (50 <= rsi14 < 80) and not extreme:
        return "A"
    if weak_trend or extreme:
        return "C"
    return "B"


def _market_conclusion(ticker: str, grade: str, rsi14: float | None, price: float | None, ma20: float | None, ma60: float | None):
    """
    一句總評（ETF 更實用）
    """
    if grade == "A":
        return f"{ticker}：偏強趨勢市，可順勢操作，但留意回調。"
    if grade == "C":
        return f"{ticker}：偏弱或風險偏高，宜保守觀望／等待確認。"
    return f"{ticker}：拉鋸/橫行，適合等突破或做區間策略。"


def get_market_overview(force_refresh: bool = False, auto_refresh_945: bool = True):
    global MARKET_CACHE, MARKET_CACHE_DATE

    now_et = _et_now()
    today_et = now_et.strftime("%Y-%m-%d")

    # 有 cache 而且係今日：直接用
    if (not force_refresh) and MARKET_CACHE_DATE == today_et and MARKET_CACHE:
        return MARKET_CACHE

    # 9:45 前：如果有舊 cache 就先用（避免朝早狂刷新）
    if (not force_refresh) and auto_refresh_945:
        if now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 45):
            if MARKET_CACHE:
                return MARKET_CACHE

    tickers = ["SPY", "QQQ", "DIA"]
    results: list[dict] = []
    closes_map = {}   # 用嚟後面計 RS
    dates: list[str] = []

    for t in tickers:
        ddf = _download_daily(t)
        if ddf is None or ddf.empty or "Close" not in ddf.columns:
            results.append({"ticker": t, "error": "no data"})
            continue

        close = ddf["Close"].dropna()
        if close.empty:
            results.append({"ticker": t, "error": "no close"})
            continue

        price = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) >= 2 else price
        change_pct = ((price - prev) / prev * 100) if prev else 0.0
        updated_date = close.index[-1].strftime("%Y-%m-%d")

        rsi_series = calc_rsi(close, 14)
        rsi14 = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else None

        ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
        ma60 = float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else None

        high_52w = float(close.max()) if len(close) >= 200 else float(close.max())
        low_52w = float(close.min()) if len(close) >= 200 else float(close.min())

        dist_ma60 = ((price - ma60) / ma60 * 100) if (ma60 and ma60 != 0) else None
        from_high = ((price - high_52w) / high_52w * 100) if high_52w else None

        # ===== 1M / 3M 涨跌%（用同一份 close）=====
        ret_1m = None
        ret_3m = None
        try:
            if close is not None and len(close) >= 22:
                ret_1m = (float(close.iloc[-1]) / float(close.iloc[-22]) - 1) * 100
            if close is not None and len(close) >= 64:
                ret_3m = (float(close.iloc[-1]) / float(close.iloc[-64]) - 1) * 100
        except Exception:
            pass

        # ===== 距离 MA20 / MA60 % =====
        dist_ma20 = None
        dist_ma60 = None
        try:
            if price is not None and ma20:
                dist_ma20 = (float(price) / float(ma20) - 1) * 100
            if price is not None and ma60:
                dist_ma60 = (float(price) / float(ma60) - 1) * 100
        except Exception:
            pass

        # ===== ATR%（可选）=====
        atr_pct = None
        try:
            # df 需要有 High/Low/Close；你用 yf.download(interval="1d") 通常都有
            if df is not None and not df.empty and all(c in df.columns for c in ["High", "Low", "Close"]):
                high = df["High"].dropna()
                low = df["Low"].dropna()
                c = df["Close"].dropna()
                if len(c) >= 15:
                    prev_close = c.shift(1)
                    tr = (high - low).abs()
                    tr2 = (high - prev_close).abs()
                    tr3 = (low - prev_close).abs()
                    true_range = pd.concat([tr, tr2, tr3], axis=1).max(axis=1)
                    atr14 = true_range.rolling(14).mean().iloc[-1]
                    if pd.notna(atr14) and price:
                        atr_pct = float(atr14) / float(price) * 100
        except Exception:
            pass

        # ===== 1周 支撑 / 压力（最近5个交易日）=====
        week_support = None
        week_resistance = None

        try:
            if df is not None and not df.empty and len(df) >= 5:
                recent = df.tail(5)
                week_support = float(recent["Low"].min())
                week_resistance = float(recent["High"].max())
        except Exception:
            pass

        # ===== 趋势标签（偏强/整理/偏弱）=====
        trend_label = "整理"
        trend_class = "tag-mid"
        try:
            if ma20 and ma60 and rsi14 is not None:
                if float(ma20) > float(ma60) and float(rsi14) >= 55:
                    trend_label = "偏强"
                    trend_class = "tag-strong"
                elif float(ma20) < float(ma60) and float(rsi14) <= 45:
                    trend_label = "偏弱"
                    trend_class = "tag-weak"
        except Exception:
            pass

        # ===== 今日关键状态（一行总结）=====
        key_status = None
        try:
            if trend_label == "偏强":
                key_status = "多头结构，占优但留意回调"
            elif trend_label == "偏弱":
                key_status = "空方主导，反弹压力较大"
            else:
                key_status = "区间整理，等待方向确认"
        except Exception:
            pass

        # ===== 今日关键状态（一行总结）=====
        status_text = ""
        status_class = "neutral"

        try:
            if price and ma20 and ma60:
                if price > ma20 > ma60:
                    status_text = "强势上行｜站稳 MA20 / MA60"
                    status_class = "up"
                elif price > ma60 and price < ma20:
                    status_text = "回调中｜仍高于 MA60"
                    status_class = "neutral"
                elif price < ma20 < ma60:
                    status_text = "偏弱｜跌破 MA20 / MA60"
                    status_class = "down"
                else:
                    status_text = "区间震荡｜均线缠绕"
                    status_class = "neutral"

                if rsi14:
                    if rsi14 >= 70:
                        status_text += "｜偏热"
                    elif rsi14 <= 30:
                        status_text += "｜偏冷"

        except Exception:
            pass

        dist_to_support = None
        dist_to_resistance = None

        try:
            if price and week_support and week_resistance:
                dist_to_support = (price / week_support - 1) * 100
                dist_to_resistance = (price / week_resistance - 1) * 100
        except Exception:
            pass

        # ===== 20 日趋势（百分比）=====
        trend20_pct = None
        trend20_dir = None

        if len(close) >= 21:
            price_20d_ago = float(close.iloc[-21])
            if price_20d_ago != 0:
                trend20_pct = round((price / price_20d_ago - 1) * 100, 2)
                if trend20_pct > 0.3:
                    trend20_dir = "up"
                elif trend20_pct < -0.3:
                    trend20_dir = "down"
                else:
                    trend20_dir = "flat"

        # ===== MA20 / MA60 金叉死叉 =====
        ma_cross = None

        if len(close) >= 61:
            ma20_prev = float(close.rolling(20).mean().iloc[-2])
            ma60_prev = float(close.rolling(60).mean().iloc[-2])

            if ma20_prev <= ma60_prev and ma20 > ma60:
                ma_cross = "golden"   # 金叉
            elif ma20_prev >= ma60_prev and ma20 < ma60:
                ma_cross = "death"    # 死叉            

        # === 支撑 / 阻力（20日，容错版）===
        support20 = None
        resist20 = None

        # 距离支撑 / 阻力（百分比）
        dist_support_pct = None
        dist_resist_pct = None
        sr_zone = None   # support / resist / mid

        if support20 and price:
            dist_support_pct = round((price - support20) / support20 * 100, 2)

        if resist20 and price:
            dist_resist_pct = round((resist20 - price) / resist20 * 100, 2)

        # 位置判断（给前端上色）
        if dist_support_pct is not None and dist_support_pct <= 2:
            sr_zone = "support"   # 靠近支撑
        elif dist_resist_pct is not None and dist_resist_pct <= 2:
            sr_zone = "resist"    # 靠近阻力
        else:
            sr_zone = "mid"

        # === 支撑 / 阻力 区间判断 ===
        sr_zone = None
        sr_text = None

        if support20 and resist20 and price:
            dist_support = (price - support20) / price
            dist_resist = (resist20 - price) / price

            if abs(dist_support) <= 0.01:
                sr_zone = "support"
                sr_text = "靠近支撑区（低风险）"
            elif abs(dist_resist) <= 0.01:
                sr_zone = "resist"
                sr_text = "接近阻力位（追价风险）"
            else:
                sr_zone = "middle"
                sr_text = "区间中段（等方向）"    

        # 优先用 High/Low（较准确）
        if {"High", "Low"}.issubset(ddf.columns):
            lowN = ddf["Low"].dropna().tail(20)
            highN = ddf["High"].dropna().tail(20)
            if len(lowN) >= 10 and len(highN) >= 10:
                support20 = float(lowN.min())
                resist20 = float(highN.max())

        # 如果 High/Low 冇数据：退而求其次用 Close（保证有数）
        if (support20 is None or resist20 is None) and "Close" in ddf.columns:
            closeN = ddf["Close"].dropna().tail(20)
            if len(closeN) >= 10:
                support20 = float(closeN.min())
                resist20 = float(closeN.max())

        atr_series = calc_atr(ddf, 14)
        atr14 = None
        if atr_series is not None and not atr_series.dropna().empty:
            atr14 = float(atr_series.dropna().iloc[-1])

        # ATR 转成百分比（更直观）
        atr_pct = round((atr14 / price) * 100, 2) if atr14 and price else None

        # weekly：睇大方向（可選）
        wdf = _download_weekly(t)
        weekly_text = None
        if wdf is not None and not wdf.empty and "Close" in wdf.columns:
            wclose = wdf["Close"].dropna()
            if len(wclose) >= 12:
                w_first = float(wclose.iloc[-12])
                w_last = float(wclose.iloc[-1])
                if w_last > w_first * 1.05:
                    weekly_text = "近3個月偏向上"
                elif w_last < w_first * 0.95:
                    weekly_text = "近3個月偏向下"
                else:
                    weekly_text = "近3個月偏橫行"

        signals = _market_signals(rsi14, price, ma20, ma60)

        rsi_label, rsi_class = _rsi_badge(rsi14)
        grade = _market_grade(rsi14, price, ma20, ma60)
        conclusion = None
        if grade == "A":
            conclusion = "整體屬於偏強結構，可留意回調機會。"
        elif grade == "B":
            conclusion = "目前屬於整理區，方向未算明確。"
        elif grade == "C":
            conclusion = "走勢偏弱，短線風險較高。"

        results.append({
            "ticker": t,
            "last_price": round(price, 2),
            "change_pct": round(change_pct, 2),
            "updated_date": updated_date,

            "rsi14": round(rsi14, 2) if rsi14 is not None else None,
            "ma20": round(ma20, 2) if ma20 is not None else None,
            "ma60": round(ma60, 2) if ma60 is not None else None,
            "dist_ma60": round(dist_ma60, 2) if dist_ma60 is not None else None,

            "high_52w": round(high_52w, 2) if high_52w is not None else None,
            "low_52w": round(low_52w, 2) if low_52w is not None else None,
            "from_high_pct": round(from_high, 2) if from_high is not None else None,

            "weekly_text": weekly_text,
            "signals": _market_signals(rsi14, price, ma20, ma60),

            # 🔥 B）新增
            "rsi_label": rsi_label,
            "rsi_class": rsi_class,
            "grade": grade,
            "conclusion": conclusion,
            "atr_pct": atr_pct,
            "support20": round(support20, 2) if support20 is not None else None,
            "resist20": round(resist20, 2) if resist20 is not None else None,
            "support20": round(support20, 2) if support20 else None,
            "resist20": round(resist20, 2) if resist20 else None,
            "dist_support_pct": dist_support_pct,
            "dist_resist_pct": dist_resist_pct,
            "sr_zone": sr_zone,
            "trend20_pct": trend20_pct,
            "trend20_dir": trend20_dir,
            "ma_cross": ma_cross,
            "sr_zone": sr_zone,
            "sr_text": sr_text,
            "ret_1m": None if ret_1m is None else round(ret_1m, 2),
            "ret_3m": None if ret_3m is None else round(ret_3m, 2),
            "dist_ma20": None if dist_ma20 is None else round(dist_ma20, 2),
            "dist_ma60": None if dist_ma60 is None else round(dist_ma60, 2),  
            "atr_pct": None if atr_pct is None else round(atr_pct, 2),
            "trend_label": trend_label,
            "trend_class": trend_class,
            "status_text": status_text,
            "status_class": status_class,
            "week_support": week_support,
            "week_resistance": week_resistance,
            "dist_to_support": dist_to_support,
            "dist_to_resistance": dist_to_resistance,
            "key_status": key_status,
        })

        # 👉 市場最新交易日（以三隻 ETF 入面最新為準）
        market_latest_date = max(dates) if dates else None

        # 👉 標記邊啲 ETF 資料落後
        for r in results:
            r["is_stale"] = (
                market_latest_date is not None
                and r.get("updated_date") != market_latest_date
            )
            r["market_latest_date"] = market_latest_date  # 可留可唔留

    MARKET_CACHE = results
    MARKET_CACHE_DATE = today_et
    return results


# =========================================================
# Flask Routes
# =========================================================
@app.route("/", methods=["GET", "POST"])
def index():
    ticker = ""
    indicators = {}
    ai_advice = None
    ai_summary = None
    error = None

    market_overview = get_market_overview()

    if request.method == "POST":
        ticker = request.form.get("ticker", "").strip().upper()

        if not ticker:
            error = "請輸入股票代號，例如 NVDA、AAPL、QQQ。"
        else:
            try:
                df = yf.download(
                    ticker,
                    period="4mo",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                )

                if df is None or df.empty:
                    raise ValueError(f"找不到代號 {ticker} 的數據，可能打錯或者冇上市。")
                if "Close" not in df.columns:
                    raise ValueError("數據缺少 Close 價。")

                close = df["Close"].dropna()
                if len(close) < 2:
                    raise ValueError("數據太少，冇辦法計算漲跌。")

                # volume
                if "Volume" in df.columns and not df["Volume"].dropna().empty:
                    vol_series = df["Volume"].dropna()
                    last_vol = float(vol_series.iloc[-1])
                    avg20_vol = float(vol_series.tail(20).mean())
                else:
                    last_vol = 0.0
                    avg20_vol = 0.0

                volume_str = format_volume(last_vol)
                avg20_volume_str = format_volume(avg20_vol)

                # indicators
                rsi_series = calc_rsi(close)
                dif, dea, macd_hist = calc_macd(close)
                ma5 = close.rolling(5).mean()
                ma20 = close.rolling(20).mean()
                ma60 = close.rolling(60).mean()
                boll_upper, boll_mid, boll_lower = calc_boll(close, 20)

                last_close = float(close.iloc[-1])
                last_date = close.index[-1].strftime("%Y-%m-%d")
                prev_close = float(close.iloc[-2])
                change_pct = (last_close - prev_close) / prev_close * 100

                # trend text（近 3 個月，用 df）
                trend_text = ""

                try:
                    recent = close.tail(60)  # 約 3 個月交易日
                    first_90 = float(recent.iloc[0])
                    last_90 = float(recent.iloc[-1])

                    if last_90 > first_90 * 1.05:
                        trend_text = "最近三個月整體屬於向上走勢。"
                    elif last_90 < first_90 * 0.95:
                        trend_text = "最近三個月整體偏向下跌或調整。"
                    else:
                        trend_text = "最近三個月大致屬於橫行或區間震盪。"
                except Exception:
                    trend_text = ""

                # =========================
                # 今日关键状态（一行总结）
                # =========================
                try:
                    status_text, status_class = build_today_status(
                        last_close,
                        float(ma20.iloc[-1]) if ma20 is not None else None,
                        float(ma60.iloc[-1]) if ma60 is not None else None,
                        float(rsi_series.iloc[-1]) if rsi_series is not None else None,
                        float(dif.iloc[-1]) if dif is not None else None,
                        float(dea.iloc[-1]) if dea is not None else None,
                    )

                    indicators["status_text"] = status_text
                    indicators["status_class"] = status_class  
                except Exception:
                    indicators["status_text"] = None
                    indicators["status_class"] = None 

                # 52w
                high_52w = None
                low_52w = None
                try:
                    tkr = yf.Ticker(ticker)
                    hist_4m = tkr.history(period="4m", interval="1d", auto_adjust=True)
                    if hist_4m is not None and not hist_4m.empty and "Close" in hist_4m.columns:
                        high_52w = float(hist_4m["Close"].max())
                        low_52w = float(hist_4m["Close"].min())
                except Exception:
                    pass

                from_high_pct = ((last_close - high_52w) / high_52w * 100) if high_52w else None                                        

                rsi_val = round(float(rsi_series.iloc[-1]), 2) if (rsi_series is not None and not rsi_series.empty) else None

                # RSI 颜色 class + 文案
                if rsi_val is None:
                    rsi_label, rsi_class = None, None
                elif rsi_val < 30:
                    rsi_label, rsi_class = f"RSI {rsi_val} 超卖", "rsi-low"
                elif rsi_val < 45:
                    rsi_label, rsi_class = f"RSI {rsi_val} 偏弱", "rsi-weak"
                elif rsi_val <= 60:
                    rsi_label, rsi_class = f"RSI {rsi_val} 中性", "rsi-mid"
                elif rsi_val <= 70:
                    rsi_label, rsi_class = f"RSI {rsi_val} 偏强", "rsi-strong"
                else:
                    rsi_label, rsi_class = f"RSI {rsi_val} 超买", "rsi-high"

                indicators = {
                    "ticker": ticker,
                    "last_price": round(last_close, 2),
                    "last_date": last_date,
                    "change_pct": round(change_pct, 2),
                    "trend_text": trend_text,
                    "rsi": rsi_val,
                    "rsi_label": rsi_label,
                    "rsi_class": rsi_class,
                    "ma5": round(float(ma5.iloc[-1]), 2) if len(ma5.dropna()) else None,
                    "ma20": round(float(ma20.iloc[-1]), 2) if len(ma20.dropna()) else None,
                    "ma60": round(float(ma60.iloc[-1]), 2) if len(ma60.dropna()) else None,
                    "macd": round(float(macd_hist.iloc[-1]), 4) if macd_hist is not None and not macd_hist.empty else None,
                    "boll_upper": round(float(boll_upper.iloc[-1]), 2) if len(boll_upper.dropna()) else None,
                    "boll_mid": round(float(boll_mid.iloc[-1]), 2) if len(boll_mid.dropna()) else None,
                    "boll_lower": round(float(boll_lower.iloc[-1]), 2) if len(boll_lower.dropna()) else None,
                    "volume_str": volume_str,
                    "avg20_volume_str": avg20_volume_str,
                    "high_52w": round(high_52w, 2) if high_52w else None,
                    "low_52w": round(low_52w, 2) if low_52w else None,
                    "from_high_pct": round(from_high_pct, 2) if from_high_pct is not None else None,
                }

                indicators["trend_score"] = compute_trend_score(indicators)

                # tech signals
                tech_signals: list[dict] = []
                rsi = indicators.get("rsi")

                if rsi is not None:
                    if rsi >= 70:
                        tech_signals.append({"level": "risk", "label": "RSI 超買", "text": "RSI 高於 70，短線有回調風險。"})
                    elif rsi <= 30:
                        tech_signals.append({"level": "opportunity", "label": "RSI 超賣", "text": "RSI 低於 30，屬於超賣區，或有技術反彈機會。"})

                if indicators.get("ma5") and indicators.get("ma20") and indicators.get("ma60"):
                    if indicators["ma5"] > indicators["ma20"] > indicators["ma60"]:
                        tech_signals.append({"level": "bull", "label": "多頭排列", "text": "短中長期均線呈多頭排列，整體趨勢偏強。"})
                    elif indicators["ma5"] < indicators["ma20"] < indicators["ma60"]:
                        tech_signals.append({"level": "bear", "label": "空頭排列", "text": "短中長期均線呈空頭排列，整體趨勢偏弱。"})

                if indicators.get("macd") is not None:
                    if indicators["macd"] > 0:
                        tech_signals.append({"level": "bull", "label": "MACD 正柱", "text": "MACD 柱狀圖在零軸以上，動能偏多。"})
                    else:
                        tech_signals.append({"level": "risk", "label": "MACD 負柱", "text": "MACD 在零軸以下，需留意下跌動能。"})

                indicators["tech_signals"] = tech_signals

                ai_advice = get_ai_advice(build_ai_prompt(ticker, indicators))
                ai_summary = get_ai_short_summary(build_ai_short_prompt(ticker, indicators))

            except Exception as e:
                error = f"後端錯誤：{e}"

    return render_template(
        "index.html",
        ticker=ticker,
        indicators=indicators,
        ai_advice=ai_advice,
        ai_summary=ai_summary,
        error=error,
        market_overview=market_overview,
    )


@app.post("/refresh_market")
def refresh_market():
    get_market_overview(force_refresh=True, auto_refresh_945=False)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)