import os
import requests
from datetime import datetime as _dt
from zoneinfo import ZoneInfo

from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# 載入 .env（如有）
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

client = OpenAI()  # 用環境變數 OPENAI_API_KEY
app = Flask(__name__)

ET = ZoneInfo("America/New_York")

# ====== 大盤概覽缓存（SPY/QQQ/DIA）======
MARKET_CACHE: list[dict] | None = None
MARKET_CACHE_DATE: str | None = None  # ET 日期字串，例如 "2025-12-15"


# =========================================================
# 技術指標
# =========================================================
def short_term_levels(last, support, resistance, risk_pct=0.02):
    # fallback：冇支撑/阻力 → 用最近价估
    if support is None or resistance is None:
        buy_low  = round(last * 0.97, 2)
        buy_high = round(last * 0.99, 2)

        stop = round(buy_low * (1 - risk_pct), 2)   # ✅ 用 buy_low 算止损
        pressure = round(last * 1.03, 2)

        # ✅ 保险：止损必须低过 buy_low
        if stop >= buy_low:
            stop = round(buy_low * 0.98, 2)

        return buy_low, buy_high, stop, pressure

    # 有支撑/阻力 → 用支撑位为核心
    buy_low  = round(support * 0.995, 2)
    buy_high = round(support * 1.01, 2)

    stop = round(buy_low * (1 - risk_pct), 2)       # ✅ 同样用 buy_low 算止损
    pressure = round(resistance, 2)

    # ✅ 保险：止损必须低过 buy_low
    if stop >= buy_low:
        stop = round(buy_low * 0.98, 2)

    return buy_low, buy_high, stop, pressure


def build_today_status(last: float|None, ma20: float|None, ma60: float|None,
                       rsi14: float|None, macd_hist: float|None,
                       wk_sup: float|None, wk_res: float|None) -> tuple[str|None, str|None, str|None, str|None, str|None]:
    """
    回传：loc_tag, risk_tag, action_tag, status_text, status_class
    """
    if last is None:
        return None, None, None, None, None

    # ---- 位置（Location）----
    loc_tag = None
    if wk_sup is not None and abs((last - wk_sup) / wk_sup) <= 0.02:
        loc_tag = "🟢 接近支撑"
    elif wk_res is not None and abs((last - wk_res) / wk_res) <= 0.02:
        loc_tag = "🔴 接近阻力"
    elif wk_sup is not None and wk_res is not None and wk_sup < last < wk_res:
        loc_tag = "⚪ 区间中部"

    # ---- 风险（Risk）----
    risk_tag = None
    if rsi14 is None:
        risk_tag = "🟡 风险中性"
    elif rsi14 < 40 and loc_tag == "🟢 接近支撑":
        risk_tag = "🟢 风险偏低"
    elif 40 <= rsi14 <= 60:
        risk_tag = "🟡 风险中性"
    else:
        risk_tag = "🔴 风险偏高"

    # ---- 趋势（Trend）----
    trend_tag = tag_trend(ma20, ma60, macd_hist)  # 直接用上面函数

    # ---- 行动（Action）----
    action_tag = "观望"
    if trend_tag and "多头" in trend_tag and loc_tag == "🟢 接近支撑" and "偏低" in risk_tag:
        action_tag = "可小仓尝试"
    elif trend_tag and "多头" in trend_tag and loc_tag == "🔴 接近阻力":
        action_tag = "勿追"
    elif trend_tag and "空头" in trend_tag and loc_tag == "🟢 接近支撑":
        action_tag = "等确认"
    elif "偏高" in risk_tag:
        action_tag = "观望"

    # ---- 一句话 ----
    status_text = f"{trend_tag or '—'} ｜ {loc_tag or '—'} ｜ {risk_tag or '—'}（{action_tag}）"

    # CSS class
    status_class = "status-mid"
    if "偏低" in risk_tag:
        status_class = "status-good"
    elif "偏高" in risk_tag:
        status_class = "status-bad"

    return loc_tag, risk_tag, action_tag, status_text, status_class


def get_fundamentals_yf(ticker: str):
    t = yf.Ticker(ticker)

    # 1) price：用 fast_info -> history
    price = None
    try:
        fi = getattr(t, "fast_info", {}) or {}
        price = fi.get("last_price") or fi.get("lastPrice")
    except Exception:
        fi = {}
    if not price:
        try:
            h = t.history(period="5d")
            if not h.empty:
                price = float(h["Close"].iloc[-1])
        except Exception:
            price = None

    # 2) info（yfinance 新版用 get_info() 會穩啲）
    try:
        info = t.get_info() or {}
    except Exception:
        info = {}

    # 3) shares
    shares = (
        (fi.get("shares") if isinstance(fi, dict) else None)
        or info.get("sharesOutstanding")
    )

    # 4) EPS(TTM)
    eps_ttm = (
        info.get("trailingEps")
        or info.get("epsTrailingTwelveMonths")
        or info.get("trailingEPS")
    )

    # 5) PE(TTM)（如果 trailingPE 冇，就自己用 price/eps 算）
    pe_ttm = info.get("trailingPE")
    if (pe_ttm is None) and price and eps_ttm and eps_ttm > 0:
        pe_ttm = float(price) / float(eps_ttm)

    # 6) Revenue/Share(TTM)
    rps_ttm = info.get("revenuePerShare")

    # 7) Forward
    forward_eps = info.get("forwardEps")
    forward_pe = info.get("forwardPE")

    # 若 revenuePerShare 冇：用 totalRevenue / shares
    if (rps_ttm is None) and shares:
        total_rev = info.get("totalRevenue")
        if total_rev:
            try:
                rps_ttm = float(total_rev) / float(shares)
            except Exception:
                rps_ttm = None

    # 再唔得：試 quarterly revenue *4（估算）
    if (rps_ttm is None) and shares:
        try:
            q = t.quarterly_financials
            # yfinance 有時係 "Total Revenue" 或 "TotalRevenue"
            for key in ["Total Revenue", "TotalRevenue", "totalRevenue"]:
                if q is not None and (key in q.index):
                    series = q.loc[key].dropna()
                    if len(series) >= 1:
                        # 用最近一季 *4 當作 rough TTM
                        approx_ttm = float(series.iloc[0]) * 4.0
                        rps_ttm = approx_ttm / float(shares)
                        break
        except Exception:
            pass

    # ✅ forward（加喺 funda = { } 之前）
    eps_fwd = info.get("forwardEps")
    pe_fwd  = info.get("forwardPE")

    # ✅ 用 forward 优先（冇就用 trailing）
    eps_use = eps_fwd if eps_fwd not in (None, 0, "0") else eps_ttm
    pe_use  = pe_fwd  if pe_fwd  not in (None, 0, "0") else pe_ttm  

    # ===== PS（新，optional）
    ps_ttm = info.get("priceToSalesTrailing12Months")
    revenue_ttm = info.get("totalRevenue")  # optiona 

    # ====== extra fundamentals (valuation add-ons) ======
    market_cap = info.get("marketCap")
    enterprise_value = info.get("enterpriseValue")

    free_cashflow = info.get("freeCashflow")
    total_revenue = info.get("totalRevenue")

    price_to_book = info.get("priceToBook")
    roe = info.get("returnOnEquity")  # 通常係 0.25 = 25%

    earnings_growth = info.get("earningsGrowth")  # 0.3 = 30%
    revenue_growth = info.get("revenueGrowth")
    peg_ratio = info.get("pegRatio")  # yfinance 有時直接有
    forward_pe = info.get("forwardPE")

    funda = {
        "source": "yfinance",
        "price": float(price) if price else None,
        "shares": float(shares) if shares else None,
        "eps_ttm": float(eps_ttm) if eps_ttm is not None else None,
        "pe_ttm": float(pe_ttm) if pe_ttm is not None else None,
        "forward_eps": float(forward_eps) if forward_eps is not None else None,
        "forward_pe": float(forward_pe) if forward_pe is not None else None,
        "revenue_per_share_ttm": float(rps_ttm) if rps_ttm is not None else None,
        "ps_ttm": float(ps_ttm) if ps_ttm is not None else None,
        "revenue_ttm": float(revenue_ttm) if revenue_ttm is not None else None,
        "market_cap": float(market_cap) if market_cap else None,
        "enterprise_value": float(enterprise_value) if enterprise_value else None,
        "free_cashflow": float(free_cashflow) if free_cashflow else None,
        "total_revenue": float(total_revenue) if total_revenue else None,
        "peg_ratio": float(peg_ratio) if peg_ratio else None,
        "earnings_growth": float(earnings_growth) if earnings_growth else None,
        "revenue_growth": float(revenue_growth) if revenue_growth else None,
        "price_to_book": float(price_to_book) if price_to_book else None,
        "roe": float(roe) if roe is not None else None,
    }

    # ✅ 你想 debug 就放呢度（最有用）
    print("YF FUNDA:", funda)

    return funda
    

def _clamp(x, lo, hi):
    try:
        x = float(x)
        return max(lo, min(hi, x))
    except Exception:
        return None


def _range_from_eps_pe(eps, pe_low, pe_high):
    if eps is None:
        return None
    try:
        eps = float(eps)
        if eps <= 0:
            return None
        return (eps * float(pe_low), eps * float(pe_high))
    except Exception:
        return None


def _range_from_rps_ps(rps, ps_low, ps_high):
    if rps is None:
        return None
    try:
        rps = float(rps)
        if rps <= 0:
            return None
        return (rps * float(ps_low), rps * float(ps_high))
    except Exception:
        return None


def calc_fair_value(funda: dict, ma20: float = None, ma60: float = None):
    price = funda.get("price")
    eps_ttm = funda.get("eps_ttm")
    pe_ttm = funda.get("pe_ttm")

    fwd_eps = funda.get("forward_eps")
    fwd_pe = funda.get("forward_pe")

    rps = funda.get("revenue_per_share_ttm")
    ps_ttm = funda.get("ps_ttm")

    # ===== 1) TTM PE range（你原本：18-26）
    pe_low, pe_high = 18, 26

    # 如果 yfinance 有 trailingPE，用嚟微調（但唔好誇張）
    if pe_ttm is not None:
        pe_ttm_c = _clamp(pe_ttm, 8, 60)  # <- 重要：避免你見到 53.03765759 呢啲亂飛
        if pe_ttm_c:
            pe_low = max(12, min(pe_low, pe_ttm_c * 0.85))
            pe_high = max(pe_high, pe_ttm_c * 1.15)

    ttm_range = _range_from_eps_pe(eps_ttm, pe_low, pe_high)

    # ===== 2) Forward range（新）
    # 用 forwardPE 做 anchor：但一樣 clamp，避免離地
    f_pe_low, f_pe_high = 18, 30
    if fwd_pe is not None:
        fwd_pe_c = _clamp(fwd_pe, 8, 80)
        if fwd_pe_c:
            f_pe_low = max(12, fwd_pe_c * 0.85)
            f_pe_high = min(80, fwd_pe_c * 1.15)
            # 如果 range 太窄就拉返少少
            if (f_pe_high - f_pe_low) < 6:
                f_pe_low = max(10, f_pe_low - 3)
                f_pe_high = min(80, f_pe_high + 3)

    forward_range = _range_from_eps_pe(fwd_eps, f_pe_low, f_pe_high)

    # ===== 3) PS range（新）
    # PS range 建議：大盤科技 4-10；成熟股 3-7；成長股 6-12
    # 我哋用 ps_ttm 作 anchor + clamp
    ps_low, ps_high = 4, 10
    if ps_ttm is not None:
        ps_c = _clamp(ps_ttm, 1, 30)
        if ps_c:
            ps_low = max(2, ps_c * 0.75)
            ps_high = min(25, ps_c * 1.25)
            if (ps_high - ps_low) < 2:
                ps_low = max(1.5, ps_low - 1)
                ps_high = min(25, ps_high + 1)

    ps_range = _range_from_rps_ps(rps, ps_low, ps_high)

    # ===== 主顯示：優先 forward，其次 ttm，其次 ps
    main = forward_range or ttm_range or ps_range
    fair_low = fair_high = None
    if main:
        fair_low, fair_high = main

    # gap%
    gap_pct = None
    if price and fair_low and fair_high:
        mid = (fair_low + fair_high) / 2
        try:
            gap_pct = (mid - float(price)) / float(price) * 100
        except Exception:
            gap_pct = None

    return {
        "fair_low": fair_low,
        "fair_high": fair_high,
        "gap_pct": gap_pct,

        # ✅ 三個模型都回傳（模板顯示用）
        "ttm_fair_low": ttm_range[0] if ttm_range else None,
        "ttm_fair_high": ttm_range[1] if ttm_range else None,
        "pe_low": pe_low,
        "pe_high": pe_high,

        "fwd_fair_low": forward_range[0] if forward_range else None,
        "fwd_fair_high": forward_range[1] if forward_range else None,
        "fwd_pe_low": f_pe_low,
        "fwd_pe_high": f_pe_high,

        "ps_fair_low": ps_range[0] if ps_range else None,
        "ps_fair_high": ps_range[1] if ps_range else None,
        "ps_low": ps_low,
        "ps_high": ps_high,

        # debug/顯示
        "ps_ttm": ps_ttm,
        "pe_ttm": pe_ttm,
        "fwd_pe": fwd_pe,
    }


def calc_extra_valuation(funda: dict):
    if not funda:
        return {}

    mc = funda.get("market_cap")
    ev = funda.get("enterprise_value")
    fcf = funda.get("free_cashflow")
    rev = funda.get("total_revenue")

    pe = funda.get("pe_ttm")
    fwd_pe = funda.get("forward_pe")

    peg = funda.get("peg_ratio")
    eg = funda.get("earnings_growth")
    rg = funda.get("revenue_growth")

    pb = funda.get("price_to_book")
    roe = funda.get("roe")

    # --- PEG ---
    # 1) 优先用 yfinance pegRatio
    # 2) 如果没有，用 PE / earningsGrowth（growth 用 % 即 0.25）
    peg_calc = None
    if peg:
        peg_calc = peg
    elif pe and eg and eg > 0:
        peg_calc = pe / (eg * 100)  # eg=0.25 => 25%

    # --- FCF Yield ---
    fcf_yield = None
    if fcf and mc and mc > 0:
        fcf_yield = (fcf / mc) * 100

    # --- EV/Sales ---
    ev_sales = None
    if ev and rev and rev > 0:
        ev_sales = ev / rev

    # --- PB & ROE ---
    roe_pct = None
    if roe is not None:
        roe_pct = roe * 100

    return {
        "pe_ttm": pe,
        "forward_pe": fwd_pe,
        "peg": peg_calc,
        "fcf_yield": fcf_yield,
        "ev_sales": ev_sales,
        "pb": pb,
        "roe": roe_pct,
        "growth_earnings": (eg * 100) if eg else None,
        "growth_revenue": (rg * 100) if rg else None,
    }


def calc_week_levels(df: pd.DataFrame) -> tuple[float|None, float|None, float|None]:
    """
    用最近 5 个交易日（约1周）High/Low 估支撑/阻力
    """
    try:
        if df is None or df.empty:
            return None, None, None
        need_cols = {"High", "Low"}
        if not need_cols.issubset(set(df.columns)):
            return None, None, None

        w = df.tail(5)
        if w.empty:
            return None, None, None

        support = float(w["Low"].min())
        resistance = float(w["High"].max())
        mid = (support + resistance) / 2.0
        return support, resistance, mid
    except Exception:
        return None, None, None


def tag_trend(ma20: float|None, ma60: float|None, macd_hist: float|None) -> str|None:
    if ma20 is None or ma60 is None:
        return None
    if ma20 > ma60 and (macd_hist is None or macd_hist >= 0):
        return "📈 多头结构"
    if ma20 < ma60 and (macd_hist is None or macd_hist <= 0):
        return "📉 空头结构"
    return "↔️ 区间结构"


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

    rsi = ind.get("rsi14") or ind.get("rsi")
    macd_hist = ind.get("macd_hist")
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

    # MA20 vs MA60 结构
    if ma20 and ma60:
        score += 10 if ma20 > ma60 else -10

    # MACD hist
    if macd_hist is not None:
        score += 5 if macd_hist >= 0 else -5

    # Boll 宽度
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
    indicators = {
        # 基础（个股）
        "ticker": None,
        "last_price": None,
        "last_date": None,
        "change_pct": None,
        # 均线/指标（用于趋势+风险）
        "rsi14": None,
        "ma20": None,
        "ma60": None,
        "macd_hist": None,      # 你已经有 macd_hist
        "macd_signal": None,    # ✅ 新增：给“关键状态”更稳
        # 位置（周支撑/阻力）
        "week_support": None,
        "week_resistance": None,
        "week_mid": None,
        # 关键状态（最终一句话）
        "trend_tag": None,      # 📈/📉/↔️
        "loc_tag": None,        # 🟢/🔴/⚪/🚀
        "risk_tag": None,       # 🟢/🟡/🔴
        "action_tag": None,     # 可小仓/观望/勿追/等确认
        "status_text": None,    # 一句话
        "status_class": None,   # CSS class
        "st_buy_low": None,
        "st_buy_high": None,
        "st_stop": None,
        "st_resistance": None,
    }
    result = None
    valuation = None
    funda = None
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
                    period="6mo",
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

                # ✅ 关键数值（if/then & status 用）
                ma20_val = float(ma20.iloc[-1]) if ma20 is not None and not ma20.dropna().empty else None
                ma60_val = float(ma60.iloc[-1]) if ma60 is not None and not ma60.dropna().empty else None
                rsi14_val = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.dropna().empty else None
                macd_hist_val = float(macd_hist.iloc[-1]) if macd_hist is not None and not macd_hist.dropna().empty else None
                macd_signal_val = float(dea.iloc[-1]) if dea is not None and not dea.dropna().empty else None

                # ===== 估值 =====
                funda = get_fundamentals_yf(ticker)
                valuation = calc_fair_value(funda, ma20=ma20_val, ma60=ma60_val) if funda else None

                # ✅ 新增：extra valuation
                extra_val = calc_extra_valuation(funda) if funda else {}
                if valuation is None:
                    valuation = {}
                valuation["extra"] = extra_val

                # ✅ 週線位（功能2）
                try:
                    wk_sup, wk_res, wk_mid = calc_week_levels(df)
                except Exception:
                    wk_sup, wk_res, wk_mid = None, None, None

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

                # 52w
                high_52w = None
                low_52w = None
                try:
                    tkr = yf.Ticker(ticker)
                    hist_6mo = tkr.history(period="6mo", interval="1d", auto_adjust=True)
                    if hist_6mo is not None and not hist_6mo.empty and "Close" in hist_6mo.columns:
                        high_52w = float(hist_6mo["Close"].max())
                        low_52w = float(hist_6mo["Close"].min())
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

                indicators.update({
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
                     "week_support": indicators.get("week_support"),
                     "week_resistance": indicators.get("week_resistance"),
                     "week_mid": indicators.get("week_mid"),
                     "fundamentals": funda,
                     "valuation": valuation,
                })

                # ✅ 今日關鍵狀態（一句）
                loc_tag, risk_tag, action_tag, status_text, status_class = build_today_status(
                    last_close, ma20_val, ma60_val, rsi14_val, macd_hist_val, wk_sup, wk_res
                )
                indicators.update({
                    "trend_tag": tag_trend(ma20_val, ma60_val, macd_hist_val),
                    "loc_tag": loc_tag,
                    "risk_tag": risk_tag,
                    "action_tag": action_tag,
                    "status_text": status_text,
                    "status_class": status_class,
                })

                # ✅ 短線買入/止蝕/壓力（用 1 周支撐阻力）
                buy_low, buy_high, stop, pressure = short_term_levels(
                    last_close, wk_sup, wk_res, risk_pct=0.02
                )
                indicators.update({
                    "st_buy_low": buy_low,
                    "st_buy_high": buy_high,
                    "st_stop": stop,
                    "st_resistance": pressure,
                })

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
        result=result,
        valuation=valuation,
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