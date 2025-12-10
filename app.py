import os
from datetime import datetime, date

from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------------------------------
#  載入 .env & OpenAI Client
# -------------------------------------------------
load_dotenv()
client = OpenAI()  # 用環境變數 OPENAI_API_KEY

app = Flask(__name__)

# -------------------------------------------------
#  讀取股票池：universe_tickers.txt
# -------------------------------------------------
def load_universe():
    """
    從 universe_tickers.txt 讀取股票代號（一行一個）
    檔案放在 app.py 同一層目錄
    """
    path = os.path.join(os.path.dirname(__file__), "universe_tickers.txt")
    tickers = []

    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                t = line.strip().upper()
                if t:
                    tickers.append(t)

    # 去重＋排序，方便你之後維護
    return list(sorted(set(tickers)))


UNIVERSE = load_universe()
print("UNIVERSE size =", len(UNIVERSE))
print("Sample =", UNIVERSE[:10])        

# ====== 精選股結果緩存（每日只掃一次，避免太慢） ====== #
DAILY_PICKS_CACHE: list[dict] = []
DAILY_PICKS_DATE: str | None = None


# -------------------------------------------------
#  技術指標函數
# -------------------------------------------------
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


# -------------------------------------------------
#  AI 提示詞：個股詳細分析
# -------------------------------------------------
def build_ai_prompt(ticker: str, ind: dict) -> str:
    """
    用價格 + 走勢 + 技術指標，交畀 AI 做較長分析
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
2. 估計一個「可能買入區間」同「大約止蝕位」（例如 100–105），語氣唔好太肯定，可以用「大約」「可以留意」呢啲字眼。
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


# -------------------------------------------------
#  AI 提示詞：一句話技術總評
# -------------------------------------------------
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
    try:
        resp = client.responses.create(
            model="gpt-5.1",
            input=prompt,
        )
        return resp.output_text
    except Exception as e:
        return f"短評生成失敗：{e}"


# -------------------------------------------------
#  AI 提示詞：每日精選股票（每一隻嘅解釋）
# -------------------------------------------------
def build_ai_pick_reason(pick: dict, rank: int, total: int) -> str:
    """
    為每日精選嘅其中一隻股，生成一段簡短解說＋安全區間
    """
    t = pick["ticker"]
    p = pick["last_price"]
    cp = pick["change_pct"]
    rsi = pick["rsi"]
    ma60 = pick["ma60"]
    dist = pick["dist_to_ma60"]

    return f"""
你係專業技術分析員，請用廣東話幫我簡短分析以下股票，作為「今日 Smart AI 精選」其中一隻。

這是今日第 {rank} 名推介（共 {total} 隻）。

股票：{t}
最新收市價：{p}
當日漲跌：{cp}%
RSI14：{rsi}
MA60：{ma60}
股價相對 MA60 百分比：{dist:.2f}%

請用 1 段文字輸出：
1. 先用一兩句講解目前技術形態（偏弱、反彈中、仍在調整等）。
2. 按現價估計一個較保守嘅「安全吸納區」或者「可以留意買入區」，例如「大約 170–185 左右可以分段吸納」，語氣唔好太肯定。
3. 提醒 1–2 點風險（例如跌穿某啲支持位、整體大市回調、行業消息等）。
4. 最後加一句：以上只供參考，唔係任何買賣建議。

只需輸出一段文字，唔好用項目符號。
"""


def get_ai_pick_reason(prompt: str) -> str:
    try:
        resp = client.responses.create(
            model="gpt-5.1",
            input=prompt,
        )
        return resp.output_text
    except Exception as e:
        return f"AI 精選解說出錯：{e}"


# -------------------------------------------------
#  計算單一股票的技術指標（給個股頁用）
# -------------------------------------------------
def compute_indicators_for_ticker(df: pd.DataFrame, ticker: str) -> tuple[dict, list, list]:
    """
    傳入 yfinance 回來的 df，計算主要技術指標，回傳：
    indicators dict, chart_labels, chart_prices
    """
    close = df["Close"]

    if len(close) < 2:
        raise ValueError("數據太少，冇辦法計算漲跌。")

    # 技術指標
    rsi_series = calc_rsi(close)
    dif, dea, macd_hist = calc_macd(close)
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    boll_upper, boll_mid, boll_lower = calc_boll(close, 20)

    # 價格 / 漲跌
    last_close = float(close.iloc[-1])
    last_date = close.index[-1].strftime("%Y-%m-%d")
    prev_close = float(close.iloc[-2])
    change = last_close - prev_close
    change_pct = change / prev_close * 100

    # 圖表用最近 90 日
    chart_df = close.tail(90)
    chart_labels = [idx.strftime("%Y-%m-%d") for idx in chart_df.index]
    chart_prices = [float(v) for v in chart_df.values]

    # 走勢文字
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
    }

    return indicators, chart_labels, chart_prices


# -------------------------------------------------
#  每日 Smart AI 精選 3 隻（從 UNIVERSE 中揀）
# -------------------------------------------------
def scan_daily_picks(max_picks: int = 3) -> list[dict]:
    """
    用較簡單的量化規則：
    - RSI 不高於 60（太高當作短線偏熱，唔揀）
    - RSI 越低分數越高（但低於 20 當作風險較大，會扣少少分）
    - 價格低於 MA60 少少會加分（視為偏低估／調整中）
    """
    candidates: list[dict] = []

    for symbol in UNIVERSE:
        try:
            df = yf.download(
                symbol,
                period="6mo",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if df is None or df.empty:
                continue

            close = df["Close"]
            if len(close) < 60:
                continue

            rsi_series = calc_rsi(close)
            ma60_series = close.rolling(60).mean()

            last = float(close.iloc[-1])
            prev = float(close.iloc[-2])
            change_pct = (last - prev) / prev * 100

            rsi = float(rsi_series.iloc[-1])
            ma60 = float(ma60_series.iloc[-1])

            if pd.isna(rsi) or pd.isna(ma60):
                continue

            # 基本篩選：太熱（RSI > 60）唔揀；RSI 太極端 < 10 亦先唔要
            if rsi > 60 or rsi < 10:
                continue

            # 距離 MA60 百分比（負數代表在 MA60 之下）
            dist_to_ma60 = (last - ma60) / ma60 * 100

            # 打分：RSI 越低越高分 + 在 MA60 附近或略低加分
            rsi_score = max(0, 60 - rsi)  # 例：RSI 40 → 20 分
            ma_score = 0
            if -20 <= dist_to_ma60 <= 0:
                # 喺 MA60 之下 0 ~ -20% 區間內，加分（偏低估／回調區）
                ma_score = -dist_to_ma60  # 越低分越高，例如 -8% → 8 分

            total_score = rsi_score + ma_score

            if total_score <= 0:
                continue

            candidates.append(
                {
                    "ticker": symbol,
                    "last_price": round(last, 2),
                    "change_pct": round(change_pct, 2),
                    "rsi": round(rsi, 1),
                    "ma60": round(ma60, 1),
                    "dist_to_ma60": round(dist_to_ma60, 2),
                    "score": round(total_score, 2),
                }
            )

        except Exception:
            # 單隻股票出錯就 skip，不影響其他
            continue

    # 按 score 由高到低排序，揀前 max_picks 隻
    candidates.sort(key=lambda x: x["score"], reverse=True)
    picks = candidates[:max_picks]

    # 用 AI 為每一隻生成解說
    total = len(picks)
    for idx, p in enumerate(picks, start=1):
        prompt = build_ai_pick_reason(p, rank=idx, total=total)
        reason = get_ai_pick_reason(prompt)
        p["reason"] = reason

    return picks


def get_daily_picks() -> list[dict]:
    """
    封裝一層：每日只真正掃一次，之後重用 cache
    """
    global DAILY_PICKS_CACHE, DAILY_PICKS_DATE

    today_str = date.today().isoformat()
    if DAILY_PICKS_DATE == today_str and DAILY_PICKS_CACHE:
        return DAILY_PICKS_CACHE

    picks = scan_daily_picks(max_picks=3)
    DAILY_PICKS_DATE = today_str
    DAILY_PICKS_CACHE = picks
    return picks


# -------------------------------------------------
#  Flask 主頁
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    ticker = ""
    indicators = None
    ai_advice = None
    ai_summary = None
    chart_labels: list[str] = []
    chart_prices: list[float] = []
    error = None

    # 每次進入頁面，都嘗試讀取當日精選（如已 cache 就好快）
    daily_picks = get_daily_picks()

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

                # 計算技術指標 & 圖表數據
                indicators, chart_labels, chart_prices = compute_indicators_for_ticker(df, ticker)

                # 生成 AI 詳細分析
                prompt = build_ai_prompt(ticker, indicators)
                ai_advice = get_ai_advice(prompt)

                # 一句話技術總評
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