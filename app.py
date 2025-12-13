import os
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# 載入 .env（如有）
load_dotenv()

client = OpenAI()  # 用環境變數 OPENAI_API_KEY

app = Flask(__name__)

# =========================================================
# 讀取股票池：universe_tickers.txt  （供「每日精選」使用）
# =========================================================


def load_universe():
    """
    從 universe_tickers.txt 讀入股票池
    每行一個 ticker，例如：
    AAPL
    MSFT
    AMZN
    """
    path = os.path.join(os.path.dirname(__file__), "universe_tickers.txt")
    tickers: list[str] = []

    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                t = line.strip().upper()
                if t:
                    tickers.append(t)

    # 去重 + 排序，方便之後維護
    return list(sorted(set(tickers)))


UNIVERSE = load_universe()
print("UNIVERSE size =", len(UNIVERSE))
print("Sample =", UNIVERSE[:10])

# ====== 指数概览缓存（首页固定 3 个：SPY/QQQ/DIA）======
MARKET_CACHE: list[dict] = []
MARKET_CACHE_DATE: str | None = None

# ====== 精選股票結果緩存（每日只掃一次，避免太慢） ====== #
DAILY_PICKS_CACHE: list[dict] = []
DAILY_PICKS_DATE: str | None = None

# =========================================================
# 技術指標函數（共用：單股分析 + 每日精選）
# =========================================================


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """計算 RSI"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss.replace(0, pd.NA))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(close: pd.Series):
    """計算 MACD (DIF, DEA, HIST)"""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist


def calc_boll(close: pd.Series, days: int = 20):
    """計算布林帶：上軌 / 中軌 / 下軌"""
    mid = close.rolling(days).mean()
    std = close.rolling(days).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return upper, mid, lower


def format_volume(v: float) -> str:
    """成交量格式化：1.2M / 850K"""
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
    """
    趨勢打分（0–100）
    綜合 RSI + 價格 vs MA60 + 均線多頭 / 空頭 + MACD + 布林帶波動
    """
    score = 50

    rsi = ind.get("rsi")
    macd = ind.get("macd")
    ma5 = ind.get("ma5")
    ma20 = ind.get("ma20")
    ma60 = ind.get("ma60")
    last_price = ind.get("last_price")
    boll_up = ind.get("boll_upper")
    boll_low = ind.get("boll_lower")

    # RSI：45–60 最舒服，再偏離會扣分
    if rsi is not None:
        if 45 <= rsi <= 60:
            score += 10
        elif 35 <= rsi < 45 or 60 < rsi <= 70:
            score += 5
        elif rsi < 30 or rsi > 75:
            score -= 10

    # 價格 vs MA60
    if last_price and ma60:
        dist = (last_price - ma60) / ma60 * 100
        if -5 <= dist <= 5:
            score += 10
        elif -10 <= dist < -5:
            score += 5
        elif dist < -15 or dist > 15:
            score -= 10

    # 均線排列：多頭 / 空頭
    if ma5 and ma20 and ma60:
        if ma5 > ma20 > ma60:
            score += 10
        elif ma5 < ma20 < ma60:
            score -= 10

    # MACD
    if macd is not None:
        if macd > 0:
            score += 5
        else:
            score -= 5

    # 布林帶波動太闊 → 風險略高
    if boll_up and boll_low and last_price:
        width = (boll_up - boll_low) / last_price * 100
        if width > 25:
            score -= 5

    score = max(0, min(100, int(round(score))))
    return score


# =========================================================
# AI 相關（單股分析用）
# =========================================================


def build_ai_prompt(ticker: str, ind: dict) -> str:
    """
    用價格 + 簡單走勢 + 技術指標，交畀 AI 做分析
    """
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

系統計算嘅趨勢分數（0–100）：{ind.get("trend_score")}

請你：
1. 用 2–3 句描述最近三個月走勢（偏強、偏弱、定係橫行），可以簡單引用 RSI / 均線 / MACD。
2. 估計一個「可能買入區間」同「大約止蝕位」（例如 100–105；如果風險好大可以建議先觀望）。
3. 提醒 1–2 點需要留意嘅風險（例如跌穿某啲支持位、消息風險、整體大市情況）。
4. 語氣保持中性唔好太肯定，最後加一句：以上只係參考，唔係任何投資建議。

請用分段方式輸出，用換行排版清晰。
"""


def get_ai_advice(prompt: str) -> str:
    try:
        resp = client.responses.create(
            model="gpt-5.1",
            input=prompt,
        )
        return resp.output_text
    except Exception as e:
        return f"AI 分析出錯：{e}"


def build_ai_short_prompt(ticker: str, indicators: dict) -> str:
    """
    一句話技術總評（快速提示）
    要求 AI 清楚講：偏強 / 偏弱 / 橫行 + 建議：觀望 / 分段吸納 / 減持
    """
    return f"""
你係一個專業股票短線交易顧問。

根據以下最新技術指標，請用 **一句話** 總結呢隻股票嘅短線情況：
- 必須包含三樣元素：
  1）簡單判斷：偏強 / 偏弱 / 橫行
  2）建議傾向：觀望 / 分段吸納 / 減持
  3）一句簡短風險提示
- 例如：「目前走勢偏強，可以考慮分段吸納，不過要留意大市回調風險。」

要求：
1. 只輸出一句話，不要分點、不用前綴。
2. 語氣中性，不要太肯定，不要寫「一定」「必然」。

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
趨勢分數（0–100）：{indicators.get("trend_score")}
"""


def get_ai_short_summary(prompt: str) -> str:
    """
    一句話技術總評（AI）
    """
    try:
        resp = client.responses.create(
            model="gpt-5.1",
            input=prompt,
        )
        return resp.output_text
    except Exception as e:
        return f"短評生成失敗：{e}"


# =========================================================
# 每日精選：技術計分 + AI 排名 + 安全買入區 + 合理價
# =========================================================


def _fetch_universe_stock_stats(ticker: str) -> dict | None:
    """
    幫每日精選用：取 6 個月日線，計 RSI14 + MA60 + 今日漲跌
    回傳 dict，失敗就回傳 None
    """
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

        close = df["Close"]
        if len(close) < 60:
            return None

        # 技術指標
        rsi_series = calc_rsi(close)
        ma60_series = close.rolling(60).mean()

        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        change_pct = (last_close - prev_close) / prev_close * 100.0

        rsi = float(rsi_series.iloc[-1])
        ma60_val = float(ma60_series.iloc[-1])

        # 價格相對 MA60 的距離（%）
        dist_pct = (last_close - ma60_val) / ma60_val * 100.0

        # 成交量（簡化版，取最後一日）
        if "Volume" in df.columns:
            last_vol = float(df["Volume"].iloc[-1])
            vol_str = format_volume(last_vol)
        else:
            last_vol = 0.0
            vol_str = "-"

        return {
            "ticker": ticker,
            "last_price": round(last_close, 2),
            "change_pct": round(change_pct, 2),
            "rsi": round(rsi, 1),
            "ma60": round(ma60_val, 1),
            "dist_pct": round(dist_pct, 2),
            "volume_str": vol_str,
        }
    except Exception:
        return None


def _score_candidate(stats: dict) -> float:
    """
    算一個簡單分數，用來排序：
    - 越接近 RSI=35 越高分（偏超賣但未太極端）
    - 越接近 MA60 或略低於 MA60，越高分
    - 跌得太急或 RSI 太高會被扣分
    """
    rsi = stats["rsi"]
    dist = stats["dist_pct"]

    # 以 RSI=35 為中心
    rsi_score = 40 - abs(rsi - 35)  # 最大約 40 分

    # 價格距離 MA60 越近越好，落在 -15% ~ +5% 內較好
    dist_score = 0
    if -20 <= dist <= 10:
        dist_score = 20 - abs(dist)  # 最高約 20 分
    elif dist < -25 or dist > 20:
        dist_score = -10  # 太遠，扣分

    # RSI 太高或太低，額外調整
    if rsi >= 65:
        rsi_score -= 15
    if rsi <= 20:
        rsi_score -= 10  # 太超賣，波動風險亦大

    return rsi_score + dist_score


def build_pick_analysis_prompt(stats: dict, rank: int, total: int) -> str:
    """
    每日精選卡片內那段長一點的 AI 文字
    """
    return f"""
你而家係一個專業股票技術分析顧問，幫我用簡單中文（廣東話）分析一隻已經由程式選出來嘅觀察股。

股票代號：{stats["ticker"]}
目前股價：{stats["last_price"]}
當日升跌：{stats["change_pct"]}%
RSI(14)：{stats["rsi"]}
MA60：{stats["ma60"]}
股價距離 MA60 百分比：{stats["dist_pct"]}%
成交量（最近一日）：{stats.get("volume_str", "-")}
系統排名：第 {rank} / {total} 名（數據根據 RSI + MA60 自動排序）

請用 3–5 句：
1. 簡單描述最近技術走勢（偏弱 / 偏強 / 橫行）。
2. 解釋點解會排喺今日清單嘅第 {rank} 位（例如：RSI 較低、接近 MA60 支持等等）。
3. 簡單講下短線可以留意嘅風險位同需要注意嘅情況（例如跌穿某啲支持位、消息風險）。
4. 最尾加一句：以上只係技術角度嘅觀察，唔構成任何買賣建議。

輸出時請用段落分行，好似正常人寫嘅短分析一樣。
"""


def build_pick_meta_prompt(stats: dict, rank: int, total: int) -> str:
    """
    生成「AI 排名一句話 + 安全買入區 + 合理價」嘅 prompt
    """
    return f"""
你係一位專業技術分析顧問，請根據以下數據，用**極簡短廣東話**比出呢隻股票嘅排名原因同合理價建議。

股票：{stats["ticker"]}
目前股價：{stats["last_price"]}
當日升跌：{stats["change_pct"]}%
RSI(14)：{stats["rsi"]}
MA60：{stats["ma60"]}
股價距離 MA60 百分比：{stats["dist_pct"]}%
系統初步排序：第 {rank} / {total} 名
成交量：{stats.get("volume_str", "-")}

請只輸出以下三行內容（唔好加其他文字）：

1）AI 排名一句話
2）安全買入區一句
3）合理價 / 目標價一句

輸出格式務必如下（三行，每行開頭加 '- '）：
- AI 排名：...
- 安全買入區：...
- 合理價：...
"""


def get_pick_meta_info(prompt: str):
    """
    呼叫 AI，回傳 (rank_text, safe_text, fair_text)
    """
    try:
        resp = client.responses.create(
            model="gpt-5.1-mini",
            input=prompt,
        )
        text = resp.output_text.strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        # 預期三行，但如果少於三行就用空字串頂住
        rank_text = lines[0] if len(lines) >= 1 else ""
        safe_text = lines[1] if len(lines) >= 2 else ""
        fair_text = lines[2] if len(lines) >= 3 else ""

        return rank_text, safe_text, fair_text
    except Exception as e:
        err = f"AI meta 出錯：{e}"
        return "", "", err

from zoneinfo import ZoneInfo
from datetime import datetime as _dt  # 用 _dt 避免同你其他 datetime 撞名

ET = ZoneInfo("America/New_York")

MARKET_CACHE = None
MARKET_CACHE_DATE = None  # 用 ET 日期字串，例如 "2025-12-13"

def _et_now():
    return _dt.now(ET)

def get_market_overview(force_refresh: bool = False, auto_refresh_945: bool = True):
    """
    今日市場概覽：SPY / QQQ / DIA
    - cache 一日一次（以美東日期計）
    - force_refresh=True：按鍵手動刷新
    - auto_refresh_945=True：每日 ET 09:45 之後，第一次有人打開頁面會自動刷新一次
    """
    global MARKET_CACHE, MARKET_CACHE_DATE

    now_et = _et_now()
    today_et = now_et.strftime("%Y-%m-%d")

    # ✅ 有 cache 而且係今日：通常直接用
    if (not force_refresh) and MARKET_CACHE_DATE == today_et and MARKET_CACHE:
        return MARKET_CACHE

    # ✅ 如果你想「9:45 前唔自動刷新」，就用呢段守門
    if (not force_refresh) and auto_refresh_945:
        # 9:45 前：如果有舊 cache 就照用（避免朝早未開市就成日刷新）
        if now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 45):
            if MARKET_CACHE:
                return MARKET_CACHE

    tickers = ["SPY", "QQQ", "DIA"]
    results = []

    for t in tickers:
        try:
            df = yf.download(
                t,
                period="6mo",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if df is None or df.empty or "Close" not in df.columns:
                results.append({"ticker": t, "error": "no data"})
                continue

            close = df["Close"].dropna()
            if close.empty:
                results.append({"ticker": t, "error": "no close"})
                continue

            last_close = float(close.iloc[-1])
            prev_close = float(close.iloc[-2]) if len(close) >= 2 else last_close
            change_pct = ((last_close - prev_close) / prev_close * 100) if prev_close else 0.0

            # ✅ 用最後一日數據日期（同個股一致），唔用 “今日”
            last_dt = close.index[-1]
            updated_date = last_dt.strftime("%Y-%m-%d")

            rsi_series = calc_rsi(close, 14)
            rsi14 = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else None

            ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
            ma60 = float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else None

            results.append({
                "ticker": t,
                "last_price": last_close,
                "change_pct": change_pct,
                "updated_date": updated_date,
                "rsi14": round(rsi14, 2) if rsi14 is not None else None,
                "ma20": round(ma20, 2) if ma20 is not None else None,
                "ma60": round(ma60, 2) if ma60 is not None else None,
            })

        except Exception as e:
            results.append({"ticker": t, "error": str(e)})

    MARKET_CACHE = results
    MARKET_CACHE_DATE = today_et
    return results

def get_daily_picks(num_picks: int = 3):
    """
    每日精選股票：
    - 從 UNIVERSE 入面掃描
    - 根據 RSI + MA60 做簡單排序
    - 再交畀 AI 做文字分析 + 排名理由 + 合理價建議
    - 結果 cache 一日
    """
    global DAILY_PICKS_CACHE, DAILY_PICKS_DATE

    today = datetime.now().strftime("%Y-%m-%d")
    if DAILY_PICKS_DATE == today and DAILY_PICKS_CACHE:
        return DAILY_PICKS_CACHE

    DAILY_PICKS_CACHE = []
    DAILY_PICKS_DATE = today

    if not UNIVERSE:
        return []

    candidates: list[dict] = []
    for ticker in UNIVERSE:
        stats = _fetch_universe_stock_stats(ticker)
        if not stats:
            continue

        score = _score_candidate(stats)
        stats["score"] = score
        candidates.append(stats)

    if not candidates:
        return []

    # 用分數由高到低排，揀前 num_picks 隻
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[:num_picks]
    total = len(top)

    picks: list[dict] = []
    for idx, stats in enumerate(top, start=1):
        # 1) 長一點的技術分析
        analysis_prompt = build_pick_analysis_prompt(stats, idx, total)
        reason = get_ai_advice(analysis_prompt)

        # 2) 排名 + 安全買入區 + 合理價
        meta_prompt = build_pick_meta_prompt(stats, idx, total)
        rank_text, safe_text, fair_text = get_pick_meta_info(meta_prompt)

        pick = {
            **stats,
            "rank": idx,
            "reason": reason,
            "rank_text": rank_text,
            "safe_text": safe_text,
            "fair_text": fair_text,
        }
        picks.append(pick)

    DAILY_PICKS_CACHE = picks
    return picks


# =========================================================
# Flask 主頁
# =========================================================


@app.route("/", methods=["GET", "POST"])
def index():
    ticker = ""
    indicators = None
    ai_advice = None
    ai_summary = None
    chart_labels = []
    chart_prices = []
    error = None

    # 每日精選（內部已經 cache，一日只計一次）
    daily_picks = DAILY_PICKS_CACHE if DAILY_PICKS_CACHE else []
    market_overview = get_market_overview()

    if request.method == "POST":
        ticker = request.form.get("ticker", "").strip().upper()

        if not ticker:
            error = "請輸入股票代號，例如 NVDA、AAPL、QQQ。"
        else:
            try:
                # 1) 下載 6 個月日線數據（用來畫圖 + 計技術指標）
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

                close = df["Close"]

                if len(close) < 2:
                    raise ValueError("數據太少，冇辦法計算漲跌。")

                # 成交量（可能有股票冇 Volume，先安全處理）
                if "Volume" in df.columns:
                    vol_series = df["Volume"]
                    last_vol = float(vol_series.iloc[-1])
                    avg20_vol = float(vol_series.tail(20).mean())
                else:
                    last_vol = 0.0
                    avg20_vol = 0.0
                    vol_series = None

                volume_str = format_volume(last_vol)
                avg20_volume_str = format_volume(avg20_vol)

                # 2) 技術指標
                rsi_series = calc_rsi(close)
                dif, dea, macd_hist = calc_macd(close)
                ma5 = close.rolling(5).mean()
                ma20 = close.rolling(20).mean()
                ma60 = close.rolling(60).mean()
                boll_upper, boll_mid, boll_lower = calc_boll(close, 20)

                # 3) 價格 / 漲跌
                last_close = float(close.iloc[-1])
                last_date = close.index[-1].strftime("%Y-%m-%d")

                prev_close = float(close.iloc[-2])
                change = last_close - prev_close
                change_pct = change / prev_close * 100

                # 4) 最近 90 日用來畫圖
                chart_df = close.tail(90)
                chart_labels = [idx.strftime("%Y-%m-%d") for idx in chart_df.index]
                chart_prices = [float(v) for v in chart_df.values]

                # 5) 簡單判斷 90 日走勢
                first_90 = float(chart_df.iloc[0])
                last_90 = float(chart_df.iloc[-1])

                if last_90 > first_90 * 1.05:
                    trend_text = "最近三個月整體屬於向上走勢。"
                elif last_90 < first_90 * 0.95:
                    trend_text = "最近三個月整體偏向下跌或調整。"
                else:
                    trend_text = "最近三個月大致屬於橫行或窄幅震盪。"

                # 6) 52 週高低價（用 history 1y 估算）
                high_52w = None
                low_52w = None
                try:
                    ticker_obj = yf.Ticker(ticker)
                    hist_1y = ticker_obj.history(period="1y", interval="1d", auto_adjust=True)
                    if hist_1y is not None and not hist_1y.empty:
                        high_52w = float(hist_1y["Close"].max())
                        low_52w = float(hist_1y["Close"].min())
                except Exception:
                    pass

                if high_52w:
                    from_high_pct = (last_close - high_52w) / high_52w * 100
                else:
                    from_high_pct = None

                # 7) 整合成一個 indicators dict，前端用 dot 語法存取
                indicators = {
                    "ticker": ticker,
                    "last_price": round(last_close, 2),
                    "last_date": last_date,
                    "change_pct": round(change_pct, 2),
                    "trend_text": trend_text,
                    "rsi": round(float(rsi_series.iloc[-1]), 2),
                    "ma5": round(float(ma5.iloc[-1]), 2),
                    "ma20": round(float(ma20.iloc[-1]), 2),
                    "ma60": round(float(ma60.iloc[-1]), 2),
                    "macd": round(float(macd_hist.iloc[-1]), 4),
                    "boll_upper": round(float(boll_upper.iloc[-1]), 2),
                    "boll_mid": round(float(boll_mid.iloc[-1]), 2),
                    "boll_lower": round(float(boll_lower.iloc[-1]), 2),
                    "volume": int(last_vol) if last_vol else 0,
                    "avg20_volume": int(avg20_vol) if avg20_vol else 0,
                    "volume_str": volume_str,
                    "avg20_volume_str": avg20_volume_str,
                    "high_52w": round(high_52w, 2) if high_52w else None,
                    "low_52w": round(low_52w, 2) if low_52w else None,
                    "from_high_pct": round(from_high_pct, 2) if from_high_pct is not None else None,
                }

                # 8) 趨勢打分（0–100）
                trend_score = compute_trend_score(indicators)
                indicators["trend_score"] = trend_score

                # 9) 簡單技術信號（給前端顯示標籤）
                tech_signals: list[dict] = []

                rsi = indicators["rsi"]
                if rsi >= 70:
                    tech_signals.append(
                        {"level": "risk", "label": "RSI 超買", "text": "RSI 高於 70，短線有回調風險。"}
                    )
                elif rsi <= 30:
                    tech_signals.append(
                        {"level": "opportunity", "label": "RSI 超賣", "text": "RSI 低於 30，屬於超賣區，或有技術反彈機會。"}
                    )

                if indicators["ma5"] > indicators["ma20"] > indicators["ma60"]:
                    tech_signals.append(
                        {"level": "bull", "label": "多頭排列", "text": "短中長期均線呈多頭排列，整體趨勢偏強。"}
                    )
                elif indicators["ma5"] < indicators["ma20"] < indicators["ma60"]:
                    tech_signals.append(
                        {"level": "bear", "label": "空頭排列", "text": "短中長期均線呈空頭排列，整體趨勢偏弱。"}
                    )

                if indicators["macd"] > 0:
                    tech_signals.append(
                        {"level": "bull", "label": "MACD 正柱", "text": "MACD 柱狀圖在零軸以上，動能偏多。"}
                    )
                else:
                    tech_signals.append(
                        {"level": "risk", "label": "MACD 負柱", "text": "MACD 在零軸以下，需留意下跌動能。"}
                    )

                indicators["tech_signals"] = tech_signals

                # 10) 生成 AI 詳細建議
                prompt = build_ai_prompt(ticker, indicators)
                ai_advice = get_ai_advice(prompt)

                # 11) 一句話技術總評
                short_prompt = build_ai_short_prompt(ticker, indicators)
                ai_summary = get_ai_short_summary(short_prompt)

            except Exception as e:
                error = f"後端錯誤：{e}"

    return render_template(
        "index.html",
        ticker=ticker,
        indicators=indicators,
        ai_advice=ai_advice,
        ai_summary=ai_summary,
        chart_labels=chart_labels,
        chart_prices=chart_prices,
        error=error,
        daily_picks=daily_picks,
        market_overview=market_overview,
    )

# ================================
# 手動刷新每日精選
# ================================
from flask import Flask, redirect, url_for

@app.route("/refresh_picks", methods=["POST"])
def refresh_picks():
    get_daily_picks(num_picks=3)
    return redirect(url_for("index"))

@app.post("/refresh_market")
def refresh_market():
    get_market_overview(force_refresh=True, auto_refresh_945=False)  # 手動就直接刷新
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)