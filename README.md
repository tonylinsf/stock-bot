# Smartai Polygon Pro

新版股票分析系统：

- 不用 Moomoo OpenD
- 使用 Polygon / Massive REST API 拉日K
- 大盘过滤：SPY + QQQ
- 技术指标：MA20 / MA60 / MA120 / RSI / MACD / Bollinger / Volume
- 自动选股：Top 5
- AI 建议：需要 OPENAI_API_KEY

## 安装

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 设置 API Key

复制 `.env.example` 成 `.env`：

```bash
cp .env.example .env
```

然后填入：

```txt
POLYGON_API_KEY=你的Polygon Key
OPENAI_API_KEY=你的OpenAI Key
```

## 运行

```bash
python app.py
```

打开：

```txt
http://127.0.0.1:5001
```