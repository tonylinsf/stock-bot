import os
import json
import time
import random
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# ========= 基本設定 =========
load_dotenv()
warnings.simplefilter("ignore", category=FutureWarning)

client = OpenAI()  # 用環境變數 OPENAI_API_KEY
app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
IS_RENDER = bool(os.getenv("RENDER"))  # Render 會自帶 RENDER=1
DAILY_PICKS_JSON = os.path.join(BASE_DIR, "daily_picks.json")

# Render 上建議少啲（你而家已改 48）
DEFAULT_PICKS = int(os.getenv("DAILY_PICKS_NUM", "3"))
MAX_UNIVERSE_SCAN = int(os.getenv("MAX_UNIVERSE_SCAN", "200"))  # 防止你之後唔小心放返 500+


# =========================================================
# 讀取股票池：universe_tickers.txt  （供「每日精選」使用）
# =========================================================
def load_universe():
    path = os.path.join(BASE_DIR, "universe_tickers.txt")
    tickers: list[str] = []
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                t = line.strip().upper()
                if t:
                    tickers.append(t)
    return list(sorted(set(tickers)))


UNIVERSE = load_universe()
print("UNIVERSE size =", len(UNIVERSE))
print("Sample =", UNIVERSE[:10])


# ====== 記憶體 cache（本地用；Render 主要用 JSON cache） ======
DAILY_PICKS_CACHE: list[dict] = []
DAILY_PICKS_DATE: str | None = None


def _current_pick_key() -> str:
    """
    每日精選日期鍵：
    - 以紐約時間為準
    - 09:45 前仍算前一交易日（避免開市前數據未更新）
    """
    now_east = datetime.now(ZoneInfo("America/New_York"))
    threshold = now_east.replace(hour=9, minute=45, second=0, microsecond=0)

    if now_east < threshold:
        day = (now_east - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        day = now_east.strftime("%Y-%m-%d")
    return day


# =========================================================
# 工具：安全處理 yfinance 回傳（避免 MultiIndex / DataFrame 問題）
# =========================================================
def _as_series(x) -> pd.Series:
    """
    yfinance 有時 df["Close"] 會變 DataFrame（例如 MultiIndex），
    呢度強制變成 Series，避免 float(series) warning / error。
    """
    if x is None:
        return None
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        # 取第一欄
        if x.shape[1] >= 1:
            return x.iloc[:, 0]
        return None
    return None


def _yf_download_one(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    """
    封裝 yfinance download：retry + backoff
    """
    # Render 上較易撞 rate limit，所以加 retry
    max_retry = 3 if IS_RENDER else 2
    for attempt in range(1, max_retry + 1):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,  # Render/雲端較穩
            )
            if df is None or df.empty:
                return None
            return df
        except Exception as e:
            msg = str(e).lower()
            # rate limit / too many requests：等耐啲
            if "rate" in msg or "too many" in msg:
                sleep_s = 2.0 * attempt + random.random()
            else:
                sleep_s = 0.6 * attempt + random.random() * 0.5

            print(f"[yfinance] {ticker} download failed (attempt {attempt}/{max_retry}): {e}")
            if attempt < max_retry:
                time.sleep(sleep_s)
            else:
                return None


# =========================================================
# 技術指標函數（共用：單股分析 + 每日精選）
# =========================================================
def format_volume(vol: float) -> str:
    try:
        v = float(vol)
    except Exception:
        return "-"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    elif v >= 1_000:
        return f"{v / 1_000:.1f}K"
    else:
        return str(int(v))


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
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


# =========================================================
# AI 相關（單股分析用）
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

請你：
1. 用 2–3 句描述最近三個月走勢（偏強、偏弱、定係橫行）。
2. 估計一個「可能買入區間」同「大約止蝕位」。
3. 提醒 1–2 點需要留意嘅風險。
4. 最後加一句：以上只係參考，唔係任何投資建議。

請用分段方式輸出，用換行排版清晰。
"""


def get_ai_advice(prompt: str) -> str:
    try:
        resp = client.responses.create(model="gpt-5.1", input=prompt)
        return resp.output_text
    except Exception as e:
        return f"AI 分析出錯：{e}"


def build_ai_short_prompt(ticker: str, indicators: dict) -> str:
    return f"""
你係一個專業股票短線交易顧問。
根據以下最新技術指標，請用 **一句話** 總結：偏強/偏弱/橫行 + 建議（觀望/吸納/減持）+ 風險提示。
要求：只輸出一句話，不要分點，不要太肯定。

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
"""


def get_ai_short_summary(prompt: str) -> str:
    try:
        resp = client.responses.create(model="gpt-5.1", input=prompt)
        return resp.output_text
    except Exception as e:
        return f"短評生成失敗：{e}"


# =========================================================
# 每日精選：技術計分 + AI 排名 + 安全買入區 + 合理價
# =========================================================
def _fetch_universe_stock_stats(ticker: str) -> dict | None:
    """
    取 6 個月日線，計 RSI14 + MA60 + 今日漲跌
    """
    df = _yf_download_one(ticker, period="6mo", interval="1d")
    if df is None or df.empty:
        return None

    close = _as_series(df.get("Close"))
    if close is None or close.empty or len(close) < 60:
        return None

    vol = _as_series(df.get("Volume"))
    if vol is not None and not vol.empty:
        last_vol = float(vol.iloc[-1])
        avg20_vol = float(vol.tail(20).mean())
        volume_str = format_volume(last_vol)
    else:
        last_vol = 0.0
        avg20_vol = 0.0
        volume_str = "-"

    rsi_series = calc_rsi(close)
    ma60_series = close.rolling(60).mean()

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    change_pct = (last_close - prev_close) / prev_close * 100.0

    rsi = float(rsi_series.iloc[-1])
    ma60_val = float(ma60_series.iloc[-1])
    dist_pct = (last_close - ma60_val) / ma60_val * 100.0

    return {
        "ticker": ticker,
        "last_price": round(last_close, 2),
        "change_pct": round(change_pct, 2),
        "rsi": round(rsi, 1),
        "ma60": round(ma60_val, 1),
        "dist_pct": round(dist_pct, 2),
        "volume": int(last_vol),
        "avg20_volume": int(avg20_vol),
        "volume_str": volume_str,
    }


def _score_candidate(stats: dict) -> float:
    rsi = stats["rsi"]
    dist = stats["dist_pct"]

    rsi_score = 40 - abs(rsi - 35)

    dist_score = 0
    if -20 <= dist <= 10:
        dist_score = 20 - abs(dist)
    elif dist < -25 or dist > 20:
        dist_score = -10

    if rsi >= 65:
        rsi_score -= 15
    if rsi <= 20:
        rsi_score -= 10

    return rsi_score + dist_score


def build_pick_analysis_prompt(stats: dict, rank: int, total: int) -> str:
    return f"""
你係專業股票技術分析顧問，用簡單廣東話寫 3–5 句分析：

股票：{stats["ticker"]}
股價：{stats["last_price"]}
升跌：{stats["change_pct"]}%
RSI(14)：{stats["rsi"]}
MA60：{stats["ma60"]}
距離 MA60：{stats["dist_pct"]}%
排名：第 {rank}/{total}

要求：
1）講趨勢（偏弱/偏強/橫行）
2）講點解排第 {rank}
3）講風險位
4）最後一句：只係技術觀察，唔構成買賣建議
"""


def build_pick_meta_prompt(stats: dict, rank: int, total: int) -> str:
    return f"""
你係專業技術分析顧問，請只輸出三行（每行開頭 '- '）：

股票：{stats["ticker"]}
股價：{stats["last_price"]}
升跌：{stats["change_pct"]}%
RSI：{stats["rsi"]}
MA60：{stats["ma60"]}
距離MA60：{stats["dist_pct"]}%
排名：第 {rank}/{total}

格式必須：
- AI 排名：...
- 安全買入區：...
- 合理價：...
"""


def get_pick_meta_info(prompt: str):
    try:
        resp = client.responses.create(model="gpt-5.1", input=prompt)
        text = resp.output_text.strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        rank_text = lines[0] if len(lines) >= 1 else ""
        safe_text = lines[1] if len(lines) >= 2 else ""
        fair_text = lines[2] if len(lines) >= 3 else ""
        return rank_text, safe_text, fair_text
    except Exception as e:
        err = f"AI meta 出錯：{e}"
        return "", "", err


def _read_daily_picks_file(today_key: str) -> list[dict] | None:
    if not os.path.exists(DAILY_PICKS_JSON):
        return None
    try:
        with open(DAILY_PICKS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("_date") == today_key and isinstance(data.get("picks"), list):
            return data["picks"]
    except Exception as e:
        print("[daily_picks] read cache error:", e)
    return None


def _write_daily_picks_file(today_key: str, picks: list[dict]) -> None:
    try:
        with open(DAILY_PICKS_JSON, "w", encoding="utf-8") as f:
            json.dump({"_date": today_key, "picks": picks}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[daily_picks] write cache error:", e)


def get_daily_picks(num_picks: int = DEFAULT_PICKS):
    """
    Render：優先用 daily_picks.json（每日只算一次，快好多，避免 timeout）
    本地：用 memory cache（同日只計一次）
    """
    global DAILY_PICKS_CACHE, DAILY_PICKS_DATE

    today_key = _current_pick_key()

    # 1) Render：檔案 cache 優先（最快）
    if IS_RENDER:
        cached = _read_daily_picks_file(today_key)
        if cached:
            return cached

    # 2) 記憶體 cache（本地或 Render 次要）
    if DAILY_PICKS_DATE == today_key and DAILY_PICKS_CACHE:
        return DAILY_PICKS_CACHE

    DAILY_PICKS_CACHE = []
    DAILY_PICKS_DATE = today_key

    if not UNIVERSE:
        return []

    # 防止你之後唔小心加太多 ticker
    scan_list = UNIVERSE[:MAX_UNIVERSE_SCAN]

    candidates: list[dict] = []
    failed = 0

    for ticker in scan_list:
        stats = _fetch_universe_stock_stats(ticker)
        if not stats:
            failed += 1
            continue
        stats["score"] = _score_candidate(stats)
        candidates.append(stats)

        # Render：輕微降速避免 yfinance hit rate limit
        if IS_RENDER:
            time.sleep(0.05)

    if not candidates:
        print(f"[daily_picks] no candidates. failed={failed}/{len(scan_list)}")
        return []

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[:num_picks]
    total = len(top)

    picks: list[dict] = []
    for idx, stats in enumerate(top, start=1):
        analysis_prompt = build_pick_analysis_prompt(stats, idx, total)
        reason = get_ai_advice(analysis_prompt)

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

        # Render：OpenAI call 之間稍微隔開，穩定啲
        if IS_RENDER:
            time.sleep(0.2)

    DAILY_PICKS_CACHE = picks

    # Render：寫入檔案 cache
    if IS_RENDER:
        _write_daily_picks_file(today_key, picks)

    return picks


# =========================================================
# Flask routes
# =========================================================
@app.route("/refresh_picks", methods=["POST"])
def refresh_picks():
    """
    手動刷新：
    - 清空 memory cache
    - Render 同時刪 file cache，迫佢重算
    """
    global DAILY_PICKS_CACHE, DAILY_PICKS_DATE
    DAILY_PICKS_CACHE = []
    DAILY_PICKS_DATE = None

    if IS_RENDER:
        try:
            if os.path.exists(DAILY_PICKS_JSON):
                os.remove(DAILY_PICKS_JSON)
        except Exception as e:
            print("[daily_picks] remove cache error:", e)

    # 重算一次（避免首頁返去仍空）
    get_daily_picks()
    return redirect(url_for("index"))


@app.route("/", methods=["GET", "POST"])
def index():
    ticker = ""
    indicators = None
    ai_advice = None
    ai_summary = None
    chart_labels = []
    chart_prices = []
    error = None

    # 每日精選（Render 用檔 cache）
    daily_picks = get_daily_picks()

    if request.method == "POST":
        ticker = request.form.get("ticker", "").strip().upper()

        if not ticker:
            error = "請輸入股票代號，例如 NVDA、AAPL、QQQ。"
        else:
            try:
                df = _yf_download_one(ticker, period="6mo", interval="1d")
                if df is None or df.empty:
                    raise ValueError(f"搵唔到代號 {ticker} 嘅數據，可能打錯或者冇上市。")

                close = _as_series(df.get("Close"))
                if close is None or close.empty or len(close) < 2:
                    raise ValueError("數據太少，冇辦法計算漲跌。")

                vol_series = _as_series(df.get("Volume"))

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

                if vol_series is not None and not vol_series.empty:
                    last_vol = float(vol_series.iloc[-1])
                    avg20_vol = float(vol_series.tail(20).mean())
                    volume_str = format_volume(last_vol)
                    avg20_volume_str = format_volume(avg20_vol)
                else:
                    last_vol = avg20_vol = 0.0
                    volume_str = avg20_volume_str = "-"

                chart_df = close.tail(90)
                chart_labels = [idx.strftime("%Y-%m-%d") for idx in chart_df.index]
                chart_prices = [float(v) for v in chart_df.values]

                first_90 = float(chart_df.iloc[0])
                last_90 = float(chart_df.iloc[-1])

                if last_90 > first_90 * 1.05:
                    trend_text = "最近三個月整體屬於向上走勢。"
                elif last_90 < first_90 * 0.95:
                    trend_text = "最近三個月整體偏向下跌或調整。"
                else:
                    trend_text = "最近三個月大致屬於橫行或窄幅震盪。"

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
                    "volume": int(last_vol),
                    "avg20_volume": int(avg20_vol),
                    "volume_str": volume_str,
                    "avg20_volume_str": avg20_volume_str,
                }

                prompt = build_ai_prompt(ticker, indicators)
                ai_advice = get_ai_advice(prompt)

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
    )


if __name__ == "__main__":
    app.run(debug=True)