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

# ====================== 技術指標函數 ====================== #

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


# ====================== AI 相關 ====================== #

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
    boll_up = indicators.get("boll_up")
    boll_mid = indicators.get("boll_mid")
    boll_low = indicators.get("boll_low")

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


# ====================== Flask 主頁 ====================== #

@app.route("/", methods=["GET", "POST"])
def index():
    ticker = ""
    indicators = None
    ai_advice = None
    chart_labels = []
    chart_prices = []
    error = None

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

                # 7) 生成 AI 建議
                prompt = build_ai_prompt(ticker, indicators)
                ai_advice = get_ai_advice(prompt)

            except Exception as e:
                error = f"後端錯誤：{e}"

    return render_template(
        "index.html",
        ticker=ticker,
        indicators=indicators,
        ai_advice=ai_advice,
        chart_labels=chart_labels,
        chart_prices=chart_prices,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)