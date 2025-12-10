import os
from datetime import datetime

from flask import Flask, render_template, request
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

    rs = avg_gain / avg_loss
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

請你：
1. 用 2–3 句描述最近三個月走勢（偏強、偏弱、定係橫行）。
2. 估計一個「可能買入區間」同「大約止蝕位」（例如 100–105；如果風險好大可以建議先觀望）。
3. 提醒 1–2 點需要留意嘅風險（例如高位震盪、消息面、整體大市情況）。
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
    """
    last_price = indicators.get("last_price")
    change_pct = indicators.get("change_pct")
    rsi = indicators.get("rsi")
    macd = indicators.get("macd")
    ma5 = indicators.get("ma5")
    ma20 = indicators.get("ma20")
    ma60 = indicators.get("ma60")
    boll_up = indicators.get("boll_upper")
    boll_mid = indicators.get("boll_mid")
    boll_low = indicators.get("boll_lower")

    return f"""
你係一個專業股票短線交易顧問。

根據以下最新技術指標，請用 **一句話** 總結：
- 屬於偏強 / 偏弱 / 橫行
- 大約係觀望、分段吸納定係減持為主
- 句尾加一句簡短風險提示

要求：
1. 只輸出一句話，不要分點、不用前綴。
2. 語氣中性，不要太肯定，不要寫「一定」「必然」。

股票：{ticker}
最新收市價：{last_price}
當日漲跌：{change_pct}%
RSI：{rsi}
MACD：{macd}
MA5：{ma5}
MA20：{ma20}
MA60：{ma60}
BOLL 上軌：{boll_up}
BOLL 中軌：{boll_mid}
BOLL 下軌：{boll_low}
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

        return {
            "ticker": ticker,
            "last_price": round(last_close, 2),
            "change_pct": round(change_pct, 2),
            "rsi": round(rsi, 1),
            "ma60": round(ma60_val, 1),
            "dist_pct": round(dist_pct, 2),
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
            model="gpt-5.1",
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
    daily_picks = get_daily_picks()

    if request.method == "POST":
        ticker = request.form.get("ticker", "").strip().upper()

        if not ticker:
            error = "請輸入股票代號，例如 NVDA、AAPL、QQQ。"
        else:
            try:
                # 1) 下載 6 個月日線數據
                df = yf.download(
                    ticker,
                    period="6mo",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                )

                if df is None or df.empty:
                    raise ValueError(f"找不到代號 {ticker} 的數據，可能打錯或者冇上市。")

                close = df["Close"]

                if len(close) < 2:
                    raise ValueError("數據太少，冇辦法計算漲跌。")

                # 2) 技術指標（全部用 Series 計，最後一個值用 iloc[-1]）
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

                # 6) 整合成一個 indicators dict，前端用 dot 語法存取
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
                }

                # 7) 生成 AI 詳細建議
                prompt = build_ai_prompt(ticker, indicators)
                ai_advice = get_ai_advice(prompt)

                # 8) 一句話技術總評
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