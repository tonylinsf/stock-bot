# sec13f_moat.py
import re
import time
import json
import requests
import xml.etree.ElementTree as ET
from functools import lru_cache

# =========================
# 你自己想追踪的机构（可加可减）
# CIK 要 10 位数（左边补 0）
# =========================
CIK_MAP = {
    "BlackRock": "0001364742",
    "Vanguard":  "0000102909",
    "State Street": "0000093751",
    # 你想加多几间可以再加
}

# 一些常见 ticker -> cusip 的兜底（可慢慢加）
CUSIP_MAP = {
    "AAPL": "037833100",
    "MSFT": "594918104",
    "NVDA": "67066G104",
    "TSLA": "88160R101",
    "AVGO": "11135F101",
    "GOOG": "02079K305",
    "GOOGL": "02079K107",
    "AMZN": "023135106",
    "META": "30303M102",
}

# SEC 要求带 User-Agent（写你自己资料，最少要有 contact）
SEC_HEADERS = {
    "User-Agent": "stock_bot_app (your_email@example.com)",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}

def _sec_get(url: str, timeout=20):
    # 基本 rate limit，避免 429
    time.sleep(0.2)
    r = requests.get(url, headers=SEC_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r

def _normalize_cusip(x: str) -> str:
    if not x:
        return ""
    x = x.strip().upper()
    x = re.sub(r"[^0-9A-Z]", "", x)
    return x

def _parse_13f_xml(url: str) -> dict[str, float]:
    """
    解析 13F 信息表 XML，返回：cusip -> shares
    注意：不同 13F XML 的 namespace 可能不同，所以要兼容找 tag
    """
    holdings: dict[str, float] = {}
    xml_text = _sec_get(url).text

    root = ET.fromstring(xml_text)

    # 兼容 namespace：直接用 endswith 去抓
    def findall_endswith(node, suffix: str):
        out = []
        for el in node.iter():
            if el.tag.endswith(suffix):
                out.append(el)
        return out

    info_tables = [el for el in root.iter() if el.tag.endswith("infoTable")]
    if not info_tables:
        # 有啲文件 infoTable 可能包得更深
        info_tables = findall_endswith(root, "infoTable")

    for t in info_tables:
        cusip_el = None
        shares_el = None

        for el in t.iter():
            if el.tag.endswith("cusip"):
                cusip_el = el
            if el.tag.endswith("sshPrnamt"):  # shares amount
                shares_el = el

        cusip = _normalize_cusip(cusip_el.text if cusip_el is not None else "")
        if not cusip:
            continue

        shares_txt = (shares_el.text if shares_el is not None else "") or ""
        try:
            shares = float(shares_txt.replace(",", "").strip())
        except Exception:
            shares = 0.0

        if shares > 0:
            holdings[cusip] = holdings.get(cusip, 0.0) + shares

    return holdings

def _pick_two_recent_13f_xml_urls(cik10: str) -> list[str]:
    """
    用 SEC submissions API 找该机构最近两份 13F-HR 的信息表 XML
    """
    sub_url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    sub = _sec_get(sub_url).json()

    filings = sub.get("filings", {}).get("recent", {})
    forms = filings.get("form", []) or []
    acc_nums = filings.get("accessionNumber", []) or []
    prim_docs = filings.get("primaryDocument", []) or []

    picked = []
    for form, acc, prim in zip(forms, acc_nums, prim_docs):
        if form != "13F-HR":
            continue
        # accession 形如 "000xxxx-yy-zzzzzz"
        acc_nodash = acc.replace("-", "")
        base = f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{acc_nodash}/"

        # 13F 信息表通常叫 "infotable.xml" / "informationtable.xml" / "*infotable*.xml"
        # 最稳：直接猜两个常见名，失败再 fallback 用 index.json 找
        candidates = [
            base + "infotable.xml",
            base + "informationtable.xml",
            base + "infoTable.xml",
        ]

        ok_url = None
        for u in candidates:
            try:
                _sec_get(u, timeout=12)
                ok_url = u
                break
            except Exception:
                pass

        if not ok_url:
            # fallback：拉 index.json，找 xml
            try:
                idx = _sec_get(base + "index.json", timeout=12).json()
                items = idx.get("directory", {}).get("item", []) or []
                for it in items:
                    name = (it.get("name") or "").lower()
                    if name.endswith(".xml") and ("info" in name and "table" in name):
                        ok_url = base + it["name"]
                        break
            except Exception:
                ok_url = None

        if ok_url:
            picked.append(ok_url)

        if len(picked) >= 2:
            break

    return picked

@lru_cache(maxsize=256)
def get_institutional_moat_sec13f(ticker: str, top_n: int = 6) -> dict:
    ticker = (ticker or "").upper().strip()

    result = {
        "title": "🏛 机构持仓动向 (SEC 13F)",
        "summary": "正在匹配 SEC 13F 持仓（按 CUSIP）…",
        "rows": [],
        "note": "数据来自 SEC 13F 报告，存在延迟（季度披露）",
    }

    # 1) 先拿 cusip（yfinance 有时会无/慢，所以加兜底表）
    cusip = None
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        cusip = info.get("cusip")
    except Exception:
        pass

    cusip = _normalize_cusip(cusip or "")
    if not cusip:
        cusip = _normalize_cusip(CUSIP_MAP.get(ticker, ""))

    if not cusip:
        result["summary"] = f"找不到 {ticker} 的 CUSIP，暂时无法匹配 13F 数据"
        return result

    # 2) 对每个机构：取最近两份 13F 信息表，算 delta / pct
    rows = []
    hits = 0

    for name, cik10 in CIK_MAP.items():
        try:
            urls = _pick_two_recent_13f_xml_urls(cik10)
            if len(urls) < 2:
                continue

            now_holdings = _parse_13f_xml(urls[0])
            prev_holdings = _parse_13f_xml(urls[1])

            now_sh = float(now_holdings.get(cusip, 0.0))
            prev_sh = float(prev_holdings.get(cusip, 0.0))

            if now_sh == 0 and prev_sh == 0:
                continue

            delta = now_sh - prev_sh
            pct = (delta / prev_sh * 100.0) if prev_sh > 0 else None

            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
            if pct is None:
                value = f"{arrow} 本季 {now_sh:,.0f} 股（上季 0）"
            else:
                value = f"{arrow} 本季 {now_sh:,.0f} 股，变化 {delta:,.0f}（{pct:+.2f}%）"

            rows.append({"label": name, "value": value, "_now": now_sh, "_abs_delta": abs(delta)})
            hits += 1
        except Exception:
            continue

    if not rows:
        result["summary"] = "未在追踪的机构名单中发现该股持仓记录（可能是匹配不到 CUSIP / 或该机构未持有）。"
        return result

    # ✅ 排序后取 top_n
    rows.sort(key=lambda r: r.get("_now", 0), reverse=True)  # 或改用 _abs_delta
    rows = rows[:top_n]
    for r in rows:
        r.pop("_now", None)
        r.pop("_abs_delta", None)

    result["rows"] = rows
    result["summary"] = f"已找到 {hits} 家机构的季度持仓变化（以 CUSIP 匹配）。"
    return result