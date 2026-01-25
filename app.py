import os
import requests
from datetime import datetime as _dt
from zoneinfo import ZoneInfo
from pathlib import Path
import re
import time
import json
import xml.etree.ElementTree as ET

from sec13f_moat import get_institutional_moat_sec13f

from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv


# =========================
# 基本設定
# =========================
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

app = Flask(__name__)

ET = ZoneInfo("America/New_York")
FMP_API_KEY = os.getenv("FMP_API_KEY", "").strip()

# ====== 大盤概覽缓存（SPY/QQQ/DIA）======
MARKET_CACHE: list[dict] | None = None
MARKET_CACHE_DATE: str | None = None  # ET 日期字串，例如 "2026-01-21"

# ====== SPY/QQQ/VIX/TNX 跑馬燈 cache ======
MARKET_SNAPSHOT_CACHE = None
MARKET_SNAPSHOT_TS = 0

# ===== 個股分析 cache（避免同一隻 ticker 重覆 download）======
IND_CACHE: dict[str, dict] = {}
IND_CACHE_TS: dict[str, float] = {}
IND_CACHE_TTL = 600  # 10分鐘

# =========================
# 工具：安全取值 / 避免 warnings
# =========================
def to_float(x):
    """安全轉 float：支援 scalar / numpy / pandas Series"""
    try:
        if x is None:
            return None
        if isinstance(x, pd.Series):
            if x.empty:
                return None
            return float(x.iloc[-1])
        return float(x)
    except Exception:
        return None


def pct(a, b):
    """a 相对 b 的百分比差"""
    if a is None or b is None or b == 0:
        return None
    try:
        return (float(a) / float(b) - 1) * 100
    except Exception:
        return None


# =========================
# 1) 大盤跑馬燈：SPY/QQQ/VIX/TNX 快照
# =========================
def get_market_snapshot(ttl_sec: int = 300):
    """SPY/QQQ/VIX/TNX 當日價 + 漲跌幅（5分鐘 cache）"""
    global MARKET_SNAPSHOT_CACHE, MARKET_SNAPSHOT_TS
    now = _dt.now().timestamp()

    if MARKET_SNAPSHOT_CACHE and (now - MARKET_SNAPSHOT_TS) < ttl_sec:
        return MARKET_SNAPSHOT_CACHE

    tickers = {"SPY": "SPY", "QQQ": "QQQ", "VIX": "^VIX", "TNX": "^TNX"}
    out = {}

    for k, t in tickers.items():
        try:
            df = yf.download(t, period="5d", interval="1d", auto_adjust=True, progress=False)
            if df is None or df.empty or "Close" not in df.columns:
                out[k] = {"price": None, "chg_pct": None}
                continue

            close = df["Close"].dropna()
            price = to_float(close.iloc[-1])
            prev = to_float(close.iloc[-2]) if len(close) >= 2 else price
            chg_pct = ((price / prev) - 1) * 100 if (price is not None and prev) else None

            out[k] = {
                "price": price,
                "chg_pct": round(chg_pct, 2) if chg_pct is not None else None,
            }
        except Exception:
            out[k] = {"price": None, "chg_pct": None}

    # ^TNX 通常係「收益率*10」→ 轉回 %
    out["TNX"]["yield_pct"] = round(out["TNX"]["price"] / 10.0, 2) if out["TNX"]["price"] is not None else None

    MARKET_SNAPSHOT_CACHE = out
    MARKET_SNAPSHOT_TS = now
    return out


# =========================
# 2) FMP：新聞
# =========================
def fmp_stock_news(ticker: str, limit: int = 6):
    if not FMP_API_KEY or not ticker:
        return []
    try:
        url = "https://financialmodelingprep.com/api/v3/stock_news"
        params = {"tickers": ticker.upper(), "limit": limit, "apikey": FMP_API_KEY}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json() or []
        out = []
        for it in data:
            out.append({
                "title": it.get("title", ""),
                "site": it.get("site", ""),
                "publishedDate": it.get("publishedDate", ""),
                "url": it.get("url", ""),
                "text": it.get("text", ""),
            })
        return out
    except Exception:
        return []
    

# =========================
# 技術指標 / 計算
# =========================
def short_term_levels(last, support, resistance, risk_pct=0.02):
    # fallback：冇支撑/阻力 → 用最近价估
    if support is None or resistance is None:
        buy_low = round(last * 0.97, 2)
        buy_high = round(last * 0.99, 2)

        stop = round(buy_low * (1 - risk_pct), 2)  # 用 buy_low 算止损
        pressure = round(last * 1.03, 2)

        if stop >= buy_low:
            stop = round(buy_low * 0.98, 2)

        return buy_low, buy_high, stop, pressure

    buy_low = round(support * 0.995, 2)
    buy_high = round(support * 1.01, 2)

    stop = round(buy_low * (1 - risk_pct), 2)
    pressure = round(resistance, 2)

    if stop >= buy_low:
        stop = round(buy_low * 0.98, 2)

    return buy_low, buy_high, stop, pressure


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


def calc_boll(close: pd.Series, days: int = 20):
    mid = close.rolling(days).mean()
    std = close.rolling(days).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return upper, mid, lower


def calc_atr(df: pd.DataFrame, period: int = 14):
    if df is None or df.empty:
        return None
    if not all(c in df.columns for c in ["High", "Low", "Close"]):
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


def tag_trend(ma20: float | None, ma60: float | None, macd_hist: float | None) -> str | None:
    if ma20 is None or ma60 is None:
        return None
    if ma20 > ma60 and (macd_hist is None or macd_hist >= 0):
        return "📈 多头结构"
    if ma20 < ma60 and (macd_hist is None or macd_hist <= 0):
        return "📉 空头结构"
    return "↔️ 区间结构"


def build_today_status(last: float | None, ma20: float | None, ma60: float | None,
                       rsi14: float | None, macd_hist: float | None,
                       wk_sup: float | None, wk_res: float | None):
    if last is None:
        return None, None, None, None, None

    # 位置
    loc_tag = None
    if wk_sup is not None and abs((last - wk_sup) / wk_sup) <= 0.02:
        loc_tag = "🟢 接近支撑"
    elif wk_res is not None and abs((last - wk_res) / wk_res) <= 0.02:
        loc_tag = "🔴 接近阻力"
    elif wk_sup is not None and wk_res is not None and wk_sup < last < wk_res:
        loc_tag = "⚪ 区间中部"

    # 风险
    if rsi14 is None:
        risk_tag = "🟡 风险中性"
    elif rsi14 < 40 and loc_tag == "🟢 接近支撑":
        risk_tag = "🟢 风险偏低"
    elif 40 <= rsi14 <= 60:
        risk_tag = "🟡 风险中性"
    else:
        risk_tag = "🔴 风险偏高"

    trend_tag = tag_trend(ma20, ma60, macd_hist)

    action_tag = "观望"
    if trend_tag and "多头" in trend_tag and loc_tag == "🟢 接近支撑" and "偏低" in risk_tag:
        action_tag = "可小仓尝试"
    elif trend_tag and "多头" in trend_tag and loc_tag == "🔴 接近阻力":
        action_tag = "勿追"
    elif trend_tag and "空头" in trend_tag and loc_tag == "🟢 接近支撑":
        action_tag = "等确认"
    elif "偏高" in risk_tag:
        action_tag = "观望"

    status_text = f"{trend_tag or '—'} ｜ {loc_tag or '—'} ｜ {risk_tag or '—'}（{action_tag}）"
    status_class = "status-mid"
    if "偏低" in risk_tag:
        status_class = "status-good"
    elif "偏高" in risk_tag:
        status_class = "status-bad"

    return loc_tag, risk_tag, action_tag, status_text, status_class


def calc_week_levels(df: pd.DataFrame):
    """最近 5 个交易日（约1周）High/Low 估支撑/阻力"""
    try:
        if df is None or df.empty:
            return None, None, None
        if not {"High", "Low"}.issubset(set(df.columns)):
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


# =========================
# 基本面：yfinance（估值用）
# =========================
def get_fundamentals_yf(ticker: str):
    t = yf.Ticker(ticker)

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

    try:
        info = t.get_info() or {}
    except Exception:
        info = {}

    shares = (fi.get("shares") if isinstance(fi, dict) else None) or info.get("sharesOutstanding")

    eps_ttm = info.get("trailingEps") or info.get("epsTrailingTwelveMonths") or info.get("trailingEPS")

    pe_ttm = info.get("trailingPE")
    if (pe_ttm is None) and price and eps_ttm and eps_ttm > 0:
        pe_ttm = float(price) / float(eps_ttm)

    rps_ttm = info.get("revenuePerShare")

    forward_eps = info.get("forwardEps")
    forward_pe = info.get("forwardPE")

    if (rps_ttm is None) and shares:
        total_rev = info.get("totalRevenue")
        if total_rev:
            try:
                rps_ttm = float(total_rev) / float(shares)
            except Exception:
                rps_ttm = None

    ps_ttm = info.get("priceToSalesTrailing12Months")
    revenue_ttm = info.get("totalRevenue")

    # extra fundamentals
    market_cap = info.get("marketCap")
    enterprise_value = info.get("enterpriseValue")
    free_cashflow = info.get("freeCashflow")
    total_revenue = info.get("totalRevenue")

    price_to_book = info.get("priceToBook")
    roe = info.get("returnOnEquity")

    earnings_growth = info.get("earningsGrowth")
    revenue_growth = info.get("revenueGrowth")
    peg_ratio = info.get("pegRatio")

    funda = {
        "source": "yfinance",
        "price": to_float(price),
        "shares": to_float(shares),
        "eps_ttm": to_float(eps_ttm),
        "pe_ttm": to_float(pe_ttm),
        "forward_eps": to_float(forward_eps),
        "forward_pe": to_float(forward_pe),
        "revenue_per_share_ttm": to_float(rps_ttm),
        "ps_ttm": to_float(ps_ttm),
        "revenue_ttm": to_float(revenue_ttm),
        "market_cap": to_float(market_cap),
        "enterprise_value": to_float(enterprise_value),
        "free_cashflow": to_float(free_cashflow),
        "total_revenue": to_float(total_revenue),
        "peg_ratio": to_float(peg_ratio),
        "earnings_growth": to_float(earnings_growth),
        "revenue_growth": to_float(revenue_growth),
        "price_to_book": to_float(price_to_book),
        "roe": to_float(roe),
    }

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

    # 1) TTM PE range
    pe_low, pe_high = 18, 26
    if pe_ttm is not None:
        pe_ttm_c = _clamp(pe_ttm, 8, 60)
        if pe_ttm_c:
            pe_low = max(12, min(pe_low, pe_ttm_c * 0.85))
            pe_high = max(pe_high, pe_ttm_c * 1.15)

    ttm_range = _range_from_eps_pe(eps_ttm, pe_low, pe_high)

    # 2) Forward range
    f_pe_low, f_pe_high = 18, 30
    if fwd_pe is not None:
        fwd_pe_c = _clamp(fwd_pe, 8, 80)
        if fwd_pe_c:
            f_pe_low = max(12, fwd_pe_c * 0.85)
            f_pe_high = min(80, fwd_pe_c * 1.15)
            if (f_pe_high - f_pe_low) < 6:
                f_pe_low = max(10, f_pe_low - 3)
                f_pe_high = min(80, f_pe_high + 3)

    forward_range = _range_from_eps_pe(fwd_eps, f_pe_low, f_pe_high)

    # 3) PS range
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

    main = forward_range or ttm_range or ps_range
    fair_low = fair_high = None
    if main:
        fair_low, fair_high = main

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

    peg_calc = None
    if peg:
        peg_calc = peg
    elif pe and eg and eg > 0:
        peg_calc = pe / (eg * 100)  # eg=0.25 => 25%

    fcf_yield = None
    if fcf and mc and mc > 0:
        fcf_yield = (fcf / mc) * 100

    ev_sales = None
    if ev and rev and rev > 0:
        ev_sales = ev / rev

    roe_pct = (roe * 100) if roe is not None else None

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

# =========================
# Market Overview（SPY/QQQ/DIA）— 修正 df/ddf + 去重 keys
# =========================
def _et_now():
    return _dt.now(ET)


def _download_daily(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty or "Close" not in df.columns:
            return None
        return df
    except Exception:
        return None


def get_market_overview(force_refresh: bool = False, auto_refresh_945: bool = True):
    global MARKET_CACHE, MARKET_CACHE_DATE

    now_et = _et_now()
    today_et = now_et.strftime("%Y-%m-%d")

    if (not force_refresh) and MARKET_CACHE_DATE == today_et and MARKET_CACHE:
        return MARKET_CACHE

    # 9:45 前：如果有舊 cache 就先用（避免朝早狂刷新）
    if (not force_refresh) and auto_refresh_945:
        if now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 45):
            if MARKET_CACHE:
                return MARKET_CACHE

    tickers = ["SPY", "QQQ", "DIA"]
    results: list[dict] = []
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

        price = to_float(close.iloc[-1])
        prev = to_float(close.iloc[-2]) if len(close) >= 2 else price
        change_pct = ((price - prev) / prev * 100) if (price is not None and prev) else None
        updated_date = close.index[-1].strftime("%Y-%m-%d")
        dates.append(updated_date)

        rsi_series = calc_rsi(close, 14)
        rsi14 = to_float(rsi_series.dropna().iloc[-1]) if rsi_series is not None and not rsi_series.dropna().empty else None

        ma20 = to_float(close.rolling(20).mean().dropna().iloc[-1]) if len(close) >= 20 else None
        ma60 = to_float(close.rolling(60).mean().dropna().iloc[-1]) if len(close) >= 60 else None

        high_52w = to_float(close.max())
        low_52w = to_float(close.min())

        from_high = ((price - high_52w) / high_52w * 100) if (price is not None and high_52w) else None

        # 1M / 3M
        ret_1m = (to_float(close.iloc[-1]) / to_float(close.iloc[-22]) - 1) * 100 if len(close) >= 22 else None
        ret_3m = (to_float(close.iloc[-1]) / to_float(close.iloc[-64]) - 1) * 100 if len(close) >= 64 else None

        # =========================
        # 趋势 (90日)
        # =========================
        TREND_DAYS = 90
        trend_pct = None
        trend_dir = "flat"
        trend_arrow = "→"

        if close is not None and len(close) >= TREND_DAYS + 1:
            trend_pct = (to_float(close.iloc[-1]) / to_float(close.iloc[-(TREND_DAYS + 1)]) - 1) * 100

            if trend_pct > 0.3:
                trend_dir = "up"
                trend_arrow = "↑"
            elif trend_pct < -0.3:
                trend_dir = "down"
                trend_arrow = "↓"

        # ==============================
        # 支撑 / 阻力（90日）
        # ==============================
        SR_LOOKBACK = 90
        support90 = resist90 = None

        try:
            if {"High", "Low"}.issubset(ddf.columns):
                lowN = ddf["Low"].dropna().tail(SR_LOOKBACK)
                highN = ddf["High"].dropna().tail(SR_LOOKBACK)

                if len(lowN) >= 30 and len(highN) >= 30:
                    support90 = to_float(lowN.min())
                    resist90 = to_float(highN.max())

            # fallback：冇 High/Low 就用 Close
            if (support90 is None or resist90 is None) and "Close" in ddf.columns:
                closeN = ddf["Close"].dropna().tail(SR_LOOKBACK)
                if len(closeN) >= 30:
                    support90 = support90 or to_float(closeN.min())
                    resist90  = resist90  or to_float(closeN.max())

        except Exception:
            support90 = resist90 = None

        # ATR%
        atr_pct = None
        atr_series = calc_atr(ddf, 14)
        if atr_series is not None and not atr_series.dropna().empty and price:
            atr14 = to_float(atr_series.dropna().iloc[-1])
            if atr14:
                atr_pct = (atr14 / price) * 100

        # 距离均线
        dist_ma20 = pct(price, ma20)
        dist_ma60 = pct(price, ma60)

        results.append({
            "ticker": t,
            "last_price": round(price, 2) if price is not None else None,
            "change_pct": round(change_pct, 2) if change_pct is not None else None,
            "updated_date": updated_date,

            "rsi14": round(rsi14, 2) if rsi14 is not None else None,
            "ma20": round(ma20, 2) if ma20 is not None else None,
            "ma60": round(ma60, 2) if ma60 is not None else None,
            "dist_ma20": round(dist_ma20, 2) if dist_ma20 is not None else None,
            "dist_ma60": round(dist_ma60, 2) if dist_ma60 is not None else None,

            "high_52w": round(high_52w, 2) if high_52w is not None else None,
            "low_52w": round(low_52w, 2) if low_52w is not None else None,
            "from_high_pct": round(from_high, 2) if from_high is not None else None,

            "ret_1m": round(ret_1m, 2) if ret_1m is not None else None,
            "ret_3m": round(ret_3m, 2) if ret_3m is not None else None,

            "support90": round(support90, 2) if support90 is not None else None,
            "resist90": round(resist90, 2) if resist90 is not None else None,
            "atr_pct": round(atr_pct, 2) if atr_pct is not None else None,
            "trend_days": TREND_DAYS,
            "trend_arrow": trend_arrow,
            "trend_pct": trend_pct,
            "trend_pct_str": (f"{trend_pct:.2f}%" if trend_pct is not None else "--"),
            "trend_dir": trend_dir
        })

    market_latest_date = max(dates) if dates else None
    for r in results:
        r["is_stale"] = (market_latest_date is not None and r.get("updated_date") != market_latest_date)
        r["market_latest_date"] = market_latest_date

    MARKET_CACHE = results
    MARKET_CACHE_DATE = today_et
    return results


# =========================
# 趨勢分數（你原本）
# =========================
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

    if ma20 and ma60:
        score += 10 if ma20 > ma60 else -10

    if macd_hist is not None:
        score += 5 if macd_hist >= 0 else -5

    if boll_up and boll_low and last_price:
        width = (boll_up - boll_low) / last_price * 100
        if width > 25:
            score -= 5

    return max(0, min(100, int(round(score))))


def _now_ts() -> float:
    return _dt.now().timestamp()


def compute_stock_bundle(ticker: str):
    """
    回傳：indicators, valuation, funda, error
    - 會用 IND_CACHE
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return None, None, None, "請輸入股票代號，例如 NVDA、AAPL、QQQ。"

    now = _now_ts()
    if ticker in IND_CACHE and (now - IND_CACHE_TS.get(ticker, 0)) < IND_CACHE_TTL:
        cached = IND_CACHE[ticker]
        return cached["indicators"], cached["valuation"], cached["funda"], None

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
            last_vol = to_float(vol_series.iloc[-1]) or 0.0
            avg20_vol = to_float(vol_series.tail(20).mean()) or 0.0
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

        last_close = to_float(close.iloc[-1])
        last_date = close.index[-1].strftime("%Y-%m-%d")
        prev_close = to_float(close.iloc[-2])
        change_pct = ((last_close - prev_close) / prev_close * 100) if (last_close is not None and prev_close) else None

        ma20_val = to_float(ma20.dropna().iloc[-1]) if not ma20.dropna().empty else None
        ma60_val = to_float(ma60.dropna().iloc[-1]) if not ma60.dropna().empty else None
        rsi14_val = to_float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else None
        macd_hist_val = to_float(macd_hist.dropna().iloc[-1]) if not macd_hist.dropna().empty else None
        macd_signal_val = to_float(dea.dropna().iloc[-1]) if not dea.dropna().empty else None

        # ===== 估值 =====
        funda = get_fundamentals_yf(ticker)
        valuation = calc_fair_value(funda, ma20=ma20_val, ma60=ma60_val) if funda else None
        extra_val = calc_extra_valuation(funda) if funda else {}
        if valuation is None:
            valuation = {}
        valuation["extra"] = extra_val

        # ===== 周支撑阻力 =====
        wk_sup, wk_res, wk_mid = calc_week_levels(df)

        # trend text（近 3 個月）
        trend_text = ""
        try:
            recent = close.tail(60)
            first_90 = to_float(recent.iloc[0])
            last_90 = to_float(recent.iloc[-1])
            if first_90 and last_90:
                if last_90 > first_90 * 1.05:
                    trend_text = "最近三個月整體屬於向上走勢。"
                elif last_90 < first_90 * 0.95:
                    trend_text = "最近三個月整體偏向下跌或調整。"
                else:
                    trend_text = "最近三個月大致屬於橫行或區間震盪。"
        except Exception:
            trend_text = ""

        # 52w（你本來用 6mo，照舊）
        high_52w = low_52w = None
        try:
            tkr = yf.Ticker(ticker)
            hist_6mo = tkr.history(period="6mo", interval="1d", auto_adjust=True)
            if hist_6mo is not None and not hist_6mo.empty and "Close" in hist_6mo.columns:
                high_52w = to_float(hist_6mo["Close"].max())
                low_52w = to_float(hist_6mo["Close"].min())
        except Exception:
            pass

        from_high_pct = ((last_close - high_52w) / high_52w * 100) if (last_close is not None and high_52w) else None
        rsi_val = round(rsi14_val, 2) if rsi14_val is not None else None

        # RSI label/class
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

        # ✅ FMP 新闻
        news = fmp_stock_news(ticker, limit=6)

        indicators = {
            "ticker": ticker,
            "last_price": round(last_close, 2) if last_close is not None else None,
            "last_date": last_date,
            "change_pct": round(change_pct, 2) if change_pct is not None else None,
            "trend_text": trend_text,

            "rsi": rsi_val,
            "rsi14": rsi_val,
            "rsi_label": rsi_label,
            "rsi_class": rsi_class,

            "ma5": round(to_float(ma5.dropna().iloc[-1]), 2) if not ma5.dropna().empty else None,
            "ma20": round(to_float(ma20.dropna().iloc[-1]), 2) if not ma20.dropna().empty else None,
            "ma60": round(to_float(ma60.dropna().iloc[-1]), 2) if not ma60.dropna().empty else None,

            "macd": round(to_float(macd_hist.dropna().iloc[-1]), 4) if not macd_hist.dropna().empty else None,
            "macd_hist": macd_hist_val,
            "macd_signal": macd_signal_val,

            "boll_upper": round(to_float(boll_upper.dropna().iloc[-1]), 2) if not boll_upper.dropna().empty else None,
            "boll_mid": round(to_float(boll_mid.dropna().iloc[-1]), 2) if not boll_mid.dropna().empty else None,
            "boll_lower": round(to_float(boll_lower.dropna().iloc[-1]), 2) if not boll_lower.dropna().empty else None,

            "volume_str": volume_str,
            "avg20_volume_str": avg20_volume_str,

            "high_52w": round(high_52w, 2) if high_52w is not None else None,
            "low_52w": round(low_52w, 2) if low_52w is not None else None,
            "from_high_pct": round(from_high_pct, 2) if from_high_pct is not None else None,

            "week_support": wk_sup,
            "week_resistance": wk_res,
            "week_mid": wk_mid,

            "fundamentals": funda,
            "valuation": valuation,
            "news": news,
        }

        # ✅ 今日關鍵狀態
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

        # ✅ 短線買入/止蝕/壓力
        buy_low, buy_high, stop, pressure = short_term_levels(last_close, wk_sup, wk_res, risk_pct=0.02)
        indicators.update({
            "st_buy_low": buy_low,
            "st_buy_high": buy_high,
            "st_stop": stop,
            "st_resistance": pressure,
        })

        indicators["trend_score"] = compute_trend_score(indicators)

        # 寫入 cache
        IND_CACHE[ticker] = {"indicators": indicators, "valuation": valuation, "funda": funda}
        IND_CACHE_TS[ticker] = now

        return indicators, valuation, funda, None

    except Exception as e:
        return None, None, None, f"後端錯誤：{e}"


# =========================
# Flask Routes（移走 AI）
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    ticker = ""
    indicators = {
        "ticker": None,
        "last_price": None,
        "last_date": None,
        "change_pct": None,
        "rsi14": None,
        "ma20": None,
        "ma60": None,
        "macd_hist": None,
        "macd_signal": None,
        "week_support": None,
        "week_resistance": None,
        "week_mid": None,
        "trend_tag": None,
        "loc_tag": None,
        "risk_tag": None,
        "action_tag": None,
        "status_text": None,
        "status_class": None,
        "st_buy_low": None,
        "st_buy_high": None,
        "st_stop": None,
        "st_resistance": None,
        "news": [],
    }

    valuation = None
    funda = None
    error = None
    moat = None
    load_13f_flag = False

    market_overview = get_market_overview()
    tape = get_market_snapshot()

    # ✅ POST 才做個股分析
    if request.method == "POST":
        ticker = request.form.get("ticker", "").strip().upper()
        load_13f = request.form.get("load_13f", "").strip()
        load_13f_flag = (load_13f == "1")

        print("DEBUG form:", dict(request.form))
        print("DEBUG load_13f_flag:", load_13f_flag, "ticker:", ticker)

        ind, val, fu, err = compute_stock_bundle(ticker)
        if err:
            error = err
        else:
            indicators = ind
            valuation = val
            funda = fu
        
            if load_13f_flag:
                moat = get_institutional_moat_sec13f(ticker)
                print("DEBUG moat:", moat)

    return render_template(
        "index.html",
        valuation=valuation,
        ticker=ticker,
        indicators=indicators,
        error=error,
        market_overview=market_overview,
        tape=tape,
        moat=moat,
        load_13f_flag=load_13f_flag,
    )
    

@app.post("/refresh_market")
def refresh_market():
    get_market_overview(force_refresh=True, auto_refresh_945=False)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)