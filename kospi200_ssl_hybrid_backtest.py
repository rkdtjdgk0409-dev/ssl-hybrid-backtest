#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KOSPI200 SSL Hybrid fixed-current-universe backtest
---------------------------------------------------
Survivorship bias: INTENTIONALLY IGNORED.
The CURRENT KOSPI200 constituent list is frozen across the full backtest period.

Default strategy mapping from the supplied Pine Script (SSL Hybrid):

  ENTRY signal at today's CLOSE:
    SSL2 BUY Continuation = buy_atr and not buy_atr[1]

    where:
      Baseline (BBMC) = HMA(Close, 60)
      SSL2 type        = JMA, length 5, phase 3, power 1
      ATR              = WMA(True Range, 14)
      atr_crit         = 0.9
      buy_inatr        = Close - ATR * atr_crit < sslDown2
      buy_cont         = Close > BBMC and Close > sslDown2
      buy_atr          = buy_inatr and buy_cont

    Fill: NEXT trading day's OPEN.

  EXIT signal at today's CLOSE:
    EXIT LONG = Close crosses BELOW sslExit

    where:
      sslExit uses HMA(High/Low, 15) with the same stateful Hlv logic
      as the supplied Pine Script.

    Fill: NEXT trading day's OPEN.

No short selling. One full-size long position per stock.

Backtest framework intentionally mirrors the prior CCI/DMI/PSAR KOSPI200 file:
  - current KOSPI200 universe fetched with pykrx (or supplied CSV)
  - FinanceDataReader online mode OR local marcap parquet mode
  - close signal -> next-open execution
  - optional proportional fee/slippage per side
  - forced liquidation at final close
  - by-stock, trades, equal-weight portfolio daily, summary CSV outputs

Input modes:
  A) Online (recommended for GitHub Actions):
       python kospi200_ssl_hybrid_backtest.py --online

     Requires:
       pip install pandas numpy finance-datareader pykrx

  B) Local FinanceData/marcap yearly parquet files:
       python kospi200_ssl_hybrid_backtest.py --marcap-dir ./data

     Files such as:
       marcap-2022.parquet ... marcap-2026.parquet

     Optional frozen universe CSV:
       --universe-csv kospi200_codes.csv
     CSV columns:
       Code (required, 6-digit ticker), Name (optional)

Outputs (default ./output):
  summary.csv
  by_stock.csv
  trades.csv
  portfolio_daily.csv

Example:
  python kospi200_ssl_hybrid_backtest.py \\
      --online \\
      --start 2023-08-10 \\
      --end 2026-08-10 \\
      --fee-side 0.001 \\
      --out output

IMPORTANT:
  This is a faithful Python port of the supplied Pine signal logic for the
  DEFAULT HMA/JMA/HMA configuration, not TradingView's strategy engine.
  Minor numerical differences can occur from initialization / data-vendor
  OHLC differences.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Indicator helpers: Pine-compatible approximations for the supplied defaults
# -----------------------------------------------------------------------------

def wma(s: pd.Series, length: int) -> pd.Series:
    length = int(length)
    if length <= 0:
        raise ValueError("WMA length must be > 0")
    weights = np.arange(1, length + 1, dtype=float)
    denom = weights.sum()
    return s.rolling(length, min_periods=length).apply(
        lambda x: float(np.dot(x, weights) / denom), raw=True
    )


def hma(s: pd.Series, length: int) -> pd.Series:
    """Hull MA matching Pine expression: WMA(2*WMA(src,len/2)-WMA(src,len), round(sqrt(len)))."""
    length = int(length)
    half = max(int(length / 2), 1)
    root = max(int(round(math.sqrt(length))), 1)
    return wma(2.0 * wma(s, half) - wma(s, length), root)


def jma(s: pd.Series, length: int = 5, phase: int = 3, power: int = 1) -> pd.Series:
    """
    Direct iterative port of the JMA block in the supplied Pine Script.

    Pine uses nz(x[1]) -> 0 when unavailable for e0/e1/e2/jma state.
    This implementation follows that initialization behavior.
    """
    src = s.to_numpy(dtype=float)
    out = np.full(len(src), np.nan, dtype=float)

    if length <= 0:
        raise ValueError("JMA length must be > 0")

    phase_ratio = 0.5 if phase < -100 else 2.5 if phase > 100 else phase / 100.0 + 1.5
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2.0)
    alpha = beta ** power

    prev_e0 = 0.0
    prev_e1 = 0.0
    prev_e2 = 0.0
    prev_jma = 0.0

    for i, v in enumerate(src):
        if not np.isfinite(v):
            out[i] = np.nan
            continue

        e0 = (1.0 - alpha) * v + alpha * prev_e0
        e1 = (v - e0) * (1.0 - beta) + beta * prev_e1
        e2 = (
            (e0 + phase_ratio * e1 - prev_jma) * ((1.0 - alpha) ** 2)
            + (alpha ** 2) * prev_e2
        )
        cur_jma = e2 + prev_jma
        out[i] = cur_jma

        prev_e0 = e0
        prev_e1 = e1
        prev_e2 = e2
        prev_jma = cur_jma

    return pd.Series(out, index=s.index, name="JMA")


def true_range(df: pd.DataFrame) -> pd.Series:
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat(
        [
            h - l,
            (h - prev_c).abs(),
            (l - prev_c).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def stateful_ssl_line(close: pd.Series, high_ma: pd.Series, low_ma: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Pine logic:
      Hlv := close > high_ma ? 1 : close < low_ma ? -1 : Hlv[1]
      ssl = Hlv < 0 ? high_ma : low_ma

    Hlv is NA until a direction can first be established.
    """
    c = close.to_numpy(dtype=float)
    hi = high_ma.to_numpy(dtype=float)
    lo = low_ma.to_numpy(dtype=float)
    hlv = np.full(len(c), np.nan, dtype=float)
    ssl = np.full(len(c), np.nan, dtype=float)

    prev = np.nan
    for i in range(len(c)):
        if np.isfinite(c[i]) and np.isfinite(hi[i]) and c[i] > hi[i]:
            cur = 1.0
        elif np.isfinite(c[i]) and np.isfinite(lo[i]) and c[i] < lo[i]:
            cur = -1.0
        else:
            cur = prev

        hlv[i] = cur
        if np.isfinite(cur) and np.isfinite(hi[i]) and np.isfinite(lo[i]):
            ssl[i] = hi[i] if cur < 0 else lo[i]
        prev = cur

    return (
        pd.Series(hlv, index=close.index),
        pd.Series(ssl, index=close.index),
    )


def calc_indicators(
    df: pd.DataFrame,
    baseline_len: int = 60,
    ssl2_len: int = 5,
    exit_len: int = 15,
    atr_len: int = 14,
    atr_crit: float = 0.9,
    jurik_phase: int = 3,
    jurik_power: int = 1,
) -> pd.DataFrame:
    """Port the supplied SSL Hybrid default calculations used by entry/exit."""
    x = df.copy()

    # ATR: Pine default smoothing = WMA.
    x["TR"] = true_range(x)
    x["ATR"] = wma(x["TR"], atr_len)

    # Baseline default: HMA(60) on close.
    x["BBMC"] = hma(x["Close"], baseline_len)

    # SSL2 default: JMA(5) on high/low + stateful Hlv2.
    x["SSL2_MA_HIGH"] = jma(x["High"], ssl2_len, jurik_phase, jurik_power)
    x["SSL2_MA_LOW"] = jma(x["Low"], ssl2_len, jurik_phase, jurik_power)
    x["HLV2"], x["SSL_DOWN2"] = stateful_ssl_line(
        x["Close"], x["SSL2_MA_HIGH"], x["SSL2_MA_LOW"]
    )

    # Exit default: HMA(15) on high/low + stateful Hlv3.
    x["EXIT_HIGH"] = hma(x["High"], exit_len)
    x["EXIT_LOW"] = hma(x["Low"], exit_len)
    x["HLV3"], x["SSL_EXIT"] = stateful_ssl_line(
        x["Close"], x["EXIT_HIGH"], x["EXIT_LOW"]
    )

    # SSL2 continuation logic from Pine.
    x["LOWER_HALF"] = x["Close"] - x["ATR"] * atr_crit
    x["BUY_INATR"] = x["LOWER_HALF"] < x["SSL_DOWN2"]
    x["BUY_CONT"] = (x["Close"] > x["BBMC"]) & (x["Close"] > x["SSL_DOWN2"])
    x["BUY_ATR"] = x["BUY_INATR"] & x["BUY_CONT"]

    # Pine: ssl2_buy_signal = buy_atr and not buy_atr[1]
    prev_buy_atr = x["BUY_ATR"].shift(1).eq(True)
    x["ENTRY_SIGNAL"] = x["BUY_ATR"] & (~prev_buy_atr)

    # Pine: exit_long = ta.crossunder(close, sslExit)
    prev_close = x["Close"].shift(1)
    prev_exit = x["SSL_EXIT"].shift(1)
    x["EXIT_SIGNAL"] = (
        (x["Close"] < x["SSL_EXIT"])
        & (prev_close >= prev_exit)
        & x["SSL_EXIT"].notna()
        & prev_exit.notna()
    )

    return x


# -----------------------------------------------------------------------------
# Backtest engine (same close-signal -> next-open framework as the CCI file)
# -----------------------------------------------------------------------------

def backtest_one(
    df: pd.DataFrame,
    code: str,
    name: str = "",
    start: str = "2023-08-10",
    end: str = "2026-08-10",
    fee_side: float = 0.0,
    warmup: int = 120,
    baseline_len: int = 60,
    ssl2_len: int = 5,
    exit_len: int = 15,
    atr_len: int = 14,
    atr_crit: float = 0.9,
    jurik_phase: int = 3,
    jurik_power: int = 1,
):
    x = df.sort_index().copy()
    x.index = pd.to_datetime(x.index)
    x = x[~x.index.duplicated(keep="last")]

    required = ["Open", "High", "Low", "Close"]
    x = x.dropna(subset=required)
    x = x[(x[required] > 0).all(axis=1)]

    # Calculate on all loaded pre-history; evaluate only requested date range.
    x = calc_indicators(
        x,
        baseline_len=baseline_len,
        ssl2_len=ssl2_len,
        exit_len=exit_len,
        atr_len=atr_len,
        atr_crit=atr_crit,
        jurik_phase=jurik_phase,
        jurik_power=jurik_power,
    )

    eval_mask = (x.index >= pd.Timestamp(start)) & (x.index <= pd.Timestamp(end))
    idx_eval = np.flatnonzero(np.asarray(eval_mask))
    if len(idx_eval) < 2:
        return None, pd.DataFrame(), pd.DataFrame()

    first_eval = int(idx_eval[0])
    first_signal_i = max(first_eval, warmup)

    equity = 1.0
    position = False
    entry_price = None
    entry_date = None
    entry_signal_date = None
    pending = None  # BUY / SELL
    pending_signal_date = None

    trades: list[dict] = []
    daily_rows: list[dict] = []

    prev_equity = equity
    shares = 0.0
    cash = 1.0

    for i in range(first_eval, len(x)):
        dt = x.index[i]
        if dt > pd.Timestamp(end):
            break
        row = x.iloc[i]

        # Execute yesterday's CLOSE signal at today's OPEN.
        if pending == "BUY" and not position:
            px = float(row["Open"]) * (1.0 + fee_side)
            if px > 0:
                shares = cash / px
                cash = 0.0
                position = True
                entry_price = px
                entry_date = dt
                entry_signal_date = pending_signal_date

        elif pending == "SELL" and position:
            raw_open = float(row["Open"])
            px = raw_open * (1.0 - fee_side)
            cash = shares * px
            net_ret = px / entry_price - 1.0 if entry_price else np.nan
            gross_entry = entry_price / (1.0 + fee_side) if entry_price else np.nan
            gross_ret = raw_open / gross_entry - 1.0 if gross_entry and gross_entry > 0 else np.nan

            trades.append(
                {
                    "Code": code,
                    "Name": name,
                    "EntrySignalDate": entry_signal_date,
                    "EntryDate": entry_date,
                    "ExitSignalDate": pending_signal_date,
                    "ExitDate": dt,
                    "EntryPriceNet": entry_price,
                    "ExitPriceNet": px,
                    "Return": net_ret,
                    "GrossApprox": gross_ret,
                    "HoldingDays": (dt - entry_date).days if entry_date is not None else np.nan,
                }
            )

            shares = 0.0
            position = False
            entry_price = None
            entry_date = None
            entry_signal_date = None

        pending = None
        pending_signal_date = None

        # Mark position to today's close.
        equity = cash if not position else shares * float(row["Close"])
        daily_ret = equity / prev_equity - 1.0 if prev_equity > 0 else 0.0
        daily_rows.append(
            {
                "Date": dt,
                "Code": code,
                "Equity": equity,
                "Return": daily_ret,
                "Position": int(position),
            }
        )
        prev_equity = equity

        # Generate today's CLOSE signal for next trading day's OPEN.
        if i >= first_signal_i and i + 1 < len(x) and x.index[i + 1] <= pd.Timestamp(end):
            if not position and bool(row["ENTRY_SIGNAL"]):
                pending = "BUY"
                pending_signal_date = dt
            elif position and bool(row["EXIT_SIGNAL"]):
                pending = "SELL"
                pending_signal_date = dt

    # Forced liquidation at requested period's last close.
    if position and daily_rows:
        last_dt = daily_rows[-1]["Date"]
        last_close = float(x.loc[last_dt, "Close"])
        px = last_close * (1.0 - fee_side)
        cash = shares * px
        net_ret = px / entry_price - 1.0 if entry_price else np.nan

        trades.append(
            {
                "Code": code,
                "Name": name,
                "EntrySignalDate": entry_signal_date,
                "EntryDate": entry_date,
                "ExitSignalDate": pd.NaT,
                "ExitDate": last_dt,
                "EntryPriceNet": entry_price,
                "ExitPriceNet": px,
                "Return": net_ret,
                "GrossApprox": np.nan,
                "HoldingDays": (last_dt - entry_date).days if entry_date is not None else np.nan,
            }
        )

        daily_rows[-1]["Equity"] = cash
        if len(daily_rows) >= 2:
            daily_rows[-1]["Return"] = cash / daily_rows[-2]["Equity"] - 1.0
        else:
            daily_rows[-1]["Return"] = cash - 1.0

    daily = pd.DataFrame(daily_rows).set_index("Date")
    tdf = pd.DataFrame(trades)

    if daily.empty:
        return None, tdf, daily

    total_ret = float(daily["Equity"].iloc[-1] - 1.0)
    days = max((daily.index[-1] - daily.index[0]).days, 1)
    years = days / 365.25
    cagr = (
        daily["Equity"].iloc[-1] ** (1.0 / years) - 1.0
        if years > 0 and daily["Equity"].iloc[-1] > 0
        else np.nan
    )

    peak = daily["Equity"].cummax()
    dd = daily["Equity"] / peak - 1.0
    mdd = float(dd.min())

    r = daily["Return"].replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = (
        float(r.mean() / r.std(ddof=1) * np.sqrt(252))
        if len(r) > 1 and r.std(ddof=1) > 0
        else np.nan
    )

    if len(tdf):
        win_rate = float((tdf["Return"] > 0).mean())
        avg_trade = float(tdf["Return"].mean())
        median_trade = float(tdf["Return"].median())
    else:
        win_rate = avg_trade = median_trade = np.nan

    eval_x = x.loc[(x.index >= pd.Timestamp(start)) & (x.index <= pd.Timestamp(end))]
    bh_entry = float(eval_x["Open"].iloc[0]) * (1.0 + fee_side)
    bh_exit = float(eval_x["Close"].iloc[-1]) * (1.0 - fee_side)
    bh_ret = bh_exit / bh_entry - 1.0

    stats = {
        "Code": code,
        "Name": name,
        "Start": daily.index[0],
        "End": daily.index[-1],
        "Bars": len(daily),
        "StrategyReturn": total_ret,
        "CAGR": cagr,
        "MDD": mdd,
        "Sharpe": sharpe,
        "WinRate": win_rate,
        "AvgTradeReturn": avg_trade,
        "MedianTradeReturn": median_trade,
        "Trades": len(tdf),
        "BuyHoldReturn": bh_ret,
        "ExcessVsBuyHold": total_ret - bh_ret,
    }
    return stats, tdf, daily


# -----------------------------------------------------------------------------
# Universe / data loading
# -----------------------------------------------------------------------------

def load_universe_csv(path: str) -> pd.DataFrame:
    u = pd.read_csv(path, dtype={"Code": str})
    if "Code" not in u.columns:
        raise ValueError("Universe CSV must contain a 'Code' column.")
    u["Code"] = u["Code"].astype(str).str.zfill(6)
    if "Name" not in u.columns:
        u["Name"] = ""
    return u[["Code", "Name"]].drop_duplicates("Code").reset_index(drop=True)


def _fetch_kospi200_pykrx() -> pd.DataFrame:
    """Try KRX/pykrx first. GitHub-hosted runners are sometimes blocked by KRX."""
    from pykrx import stock

    idxs = stock.get_index_ticker_list(market="KOSPI")
    idx_code = next(c for c in idxs if stock.get_index_ticker_name(c) == "코스피 200")

    today = pd.Timestamp.today().normalize()
    tickers = []
    used_date = None
    for n in range(0, 20):
        dt = today - pd.Timedelta(days=n)
        try:
            tickers = stock.get_index_portfolio_deposit_file(idx_code, dt.strftime("%Y%m%d"))
            if tickers:
                used_date = dt
                break
        except Exception:
            continue

    if not tickers:
        raise RuntimeError("pykrx returned no KOSPI200 constituents")

    out = pd.DataFrame({
        "Code": [str(t).zfill(6) for t in tickers],
        "Name": [stock.get_market_ticker_name(t) for t in tickers],
    }).drop_duplicates("Code")
    if len(out) < 190:
        raise RuntimeError(f"pykrx returned only {len(out)} constituents")
    print(f"Universe source: pykrx / KRX ({used_date.date() if used_date is not None else 'unknown'})")
    return out.reset_index(drop=True)


def _fetch_kospi200_hankyung() -> pd.DataFrame:
    """
    Fallback for GitHub Actions when KRX blocks datacenter IPs.

    Korea Economic Daily exposes KOSPI200 component pages with the six-digit
    ticker embedded in the stock-name column. We try several pages and collect
    unique codes. This avoids requiring KRX_ID/KRX_PW secrets.
    """
    import re
    from io import StringIO
    import requests

    base = "https://markets.hankyung.com/index-info/kospi200"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.7,en;q=0.6",
    }
    rows = []
    seen = set()

    # The site may render all constituents on one page or paginate them.
    urls = [base] + [f"{base}?page={i}" for i in range(1, 21)]
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            html = r.text

            # First parse HTML tables (most reliable when server-rendered).
            try:
                tables = pd.read_html(StringIO(html))
            except Exception:
                tables = []

            for tab in tables:
                for col in tab.columns:
                    vals = tab[col].astype(str)
                    for v in vals:
                        m = re.search(r"(.+?)\s+(\d{6})(?:\s|$)", v.strip())
                        if not m:
                            continue
                        name = re.sub(r"\s+", " ", m.group(1)).strip()
                        code = m.group(2)
                        if code not in seen:
                            seen.add(code)
                            rows.append((code, name))

            # Also scan stock-detail links; useful if the visible table is JS-enhanced.
            # Typical links contain a six-digit Korean ticker.
            for m in re.finditer(r'href=["\'][^"\']*(?:code|item|stock)[^"\']*[=/](\d{6})[^"\']*["\'][^>]*>(.*?)</a>', html, re.I | re.S):
                code = m.group(1)
                name = re.sub(r"<[^>]+>", " ", m.group(2))
                name = re.sub(r"\s+", " ", name).strip()
                if code not in seen and name:
                    seen.add(code)
                    rows.append((code, name))

            if len(rows) >= 200:
                break
        except Exception as e:
            print(f"Hankyung universe fallback warning ({url}): {e}")

    out = pd.DataFrame(rows, columns=["Code", "Name"]).drop_duplicates("Code")
    if len(out) < 190:
        raise RuntimeError(f"Hankyung fallback found only {len(out)} constituent codes")
    if len(out) > 200:
        out = out.iloc[:200].copy()
    print(f"Universe source: Hankyung fallback ({len(out)} stocks)")
    return out.reset_index(drop=True)



def _fetch_kospi200_naver() -> pd.DataFrame:
    """
    GitHub Actions-friendly KOSPI200 constituent fetcher using Naver Finance.

    Naver exposes the KOSPI200 constituents in 20 server-rendered pages,
    10 names per page. The stock code is embedded in links like
    /item/main.naver?code=005930. This does not require a KRX login.
    """
    import re
    import time
    import requests
    from bs4 import BeautifulSoup

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.7,en;q=0.6",
        "Referer": "https://finance.naver.com/",
    })

    rows = []
    seen = set()
    errors = []

    for page in range(1, 21):
        url = f"https://finance.naver.com/sise/entryJongmok.naver?&page={page}"
        ok = False
        for attempt in range(3):
            try:
                r = session.get(url, timeout=(8, 20))
                r.raise_for_status()
                # Naver Finance pages are traditionally EUC-KR/CP949.
                # apparent_encoding is safer if headers are incomplete.
                if not r.encoding or r.encoding.lower() in {"iso-8859-1", "ascii"}:
                    r.encoding = r.apparent_encoding or "euc-kr"
                soup = BeautifulSoup(r.text, "lxml")

                found_this_page = 0
                for td in soup.find_all("td", class_="ctg"):
                    a = td.find("a", href=True)
                    if not a:
                        continue
                    m = re.search(r"[?&]code=(\d{6})", a.get("href", ""))
                    if not m:
                        continue
                    code = m.group(1)
                    name = a.get_text(" ", strip=True) or td.get_text(" ", strip=True)
                    if code not in seen:
                        seen.add(code)
                        rows.append((code, name))
                        found_this_page += 1

                # Generic link parser in case the td class changes.
                if found_this_page == 0:
                    for a in soup.find_all("a", href=True):
                        m = re.search(r"[?&]code=(\d{6})", a.get("href", ""))
                        if not m:
                            continue
                        code = m.group(1)
                        name = a.get_text(" ", strip=True)
                        if code not in seen and name:
                            seen.add(code)
                            rows.append((code, name))
                            found_this_page += 1

                if found_this_page > 0:
                    ok = True
                    break
                errors.append(f"page {page}: parsed 0 constituents")
            except Exception as e:
                errors.append(f"page {page} attempt {attempt+1}: {type(e).__name__}: {e}")
                time.sleep(1.0 + attempt)

        if not ok:
            print(f"Naver universe warning: page {page} could not be parsed")
        if len(rows) >= 200:
            break
        time.sleep(0.15)

    out = pd.DataFrame(rows, columns=["Code", "Name"]).drop_duplicates("Code")
    # The endpoint is specifically the KOSPI200 constituent list; insist on exactly 200
    # to prevent silently backtesting an incomplete universe.
    if len(out) != 200:
        tail = "\n".join(errors[-8:])
        raise RuntimeError(
            f"Naver KOSPI200 fetch returned {len(out)} unique constituents, expected 200."
            + (f"\nRecent errors:\n{tail}" if tail else "")
        )

    out["Code"] = out["Code"].astype(str).str.zfill(6)
    print(f"Universe source: Naver Finance ({len(out)} stocks)")
    return out.reset_index(drop=True)

def fetch_current_kospi200_online() -> pd.DataFrame:
    """Fetch current KOSPI200 membership with multiple independent fallbacks."""
    errors = []

    # Naver first on GitHub Actions: it does not depend on the KRX OTP/login flow.
    try:
        return _fetch_kospi200_naver()
    except Exception as e:
        errors.append(f"Naver: {type(e).__name__}: {e}")
        print("Naver universe fetch failed; trying pykrx...")
        print(errors[-1])

    try:
        return _fetch_kospi200_pykrx()
    except Exception as e:
        errors.append(f"pykrx: {type(e).__name__}: {e}")
        print("pykrx universe fetch failed; trying Hankyung fallback...")
        print(errors[-1])

    try:
        return _fetch_kospi200_hankyung()
    except Exception as e:
        errors.append(f"Hankyung: {type(e).__name__}: {e}")

    raise RuntimeError(
        "Could not obtain the current KOSPI200 universe from Naver, pykrx, or Hankyung. "
        "You can also pass --universe-csv with a CSV containing Code,Name.\n"
        + "\n".join(errors)
    )


def prehistory_start(start: str, calendar_days: int) -> str:
    return (pd.Timestamp(start) - pd.Timedelta(days=calendar_days)).strftime("%Y-%m-%d")


def load_marcap(marcap_dir: str, data_start: str, end: str) -> pd.DataFrame:
    d = Path(marcap_dir)
    y0, y1 = pd.Timestamp(data_start).year, pd.Timestamp(end).year
    parts = []

    for y in range(y0, y1 + 1):
        p = d / f"marcap-{y}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        z = pd.read_parquet(p)
        if "Date" in z.columns:
            z["Date"] = pd.to_datetime(z["Date"])
        else:
            z = z.reset_index()
            z["Date"] = pd.to_datetime(z["Date"])
        z["Code"] = z["Code"].astype(str).str.zfill(6)
        parts.append(z)

    return pd.concat(parts, ignore_index=True)


def online_price(code: str, data_start: str, end: str) -> pd.DataFrame:
    import FinanceDataReader as fdr

    z = fdr.DataReader(code, data_start, end).copy()
    z.index = pd.to_datetime(z.index)
    cols = {str(c).lower(): c for c in z.columns}
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in z.columns:
            alt = cols.get(c.lower())
            if alt is not None:
                z[c] = z[alt]
    return z


# -----------------------------------------------------------------------------
# Portfolio aggregation / reporting
# -----------------------------------------------------------------------------

def aggregate_portfolio(dailies: dict[str, pd.DataFrame], start: str, end: str) -> pd.DataFrame:
    """
    Equal initial weight across stocks with usable data.
    Missing dates are treated as zero return, matching the prior CCI file.
    """
    all_dates = pd.date_range(start, end, freq="D")
    rets = []

    for code, d in dailies.items():
        s = d["Return"].reindex(all_dates).fillna(0.0)
        s.name = code
        rets.append(s)

    if not rets:
        return pd.DataFrame()

    mat = pd.concat(rets, axis=1)
    port_ret = mat.mean(axis=1)

    active = pd.Series(False, index=all_dates)
    for d in dailies.values():
        active.loc[active.index.intersection(d.index)] = True

    out = pd.DataFrame({"Return": port_ret, "Active": active})
    out = out[out["Active"]].drop(columns="Active")
    out["Equity"] = (1.0 + out["Return"]).cumprod()
    return out


def metrics_from_equity(d: pd.DataFrame) -> dict:
    if d.empty:
        return {}

    total = float(d["Equity"].iloc[-1] - 1.0)
    days = max((d.index[-1] - d.index[0]).days, 1)
    years = days / 365.25
    cagr = float(d["Equity"].iloc[-1] ** (1.0 / years) - 1.0)
    dd = d["Equity"] / d["Equity"].cummax() - 1.0
    r = d["Return"].dropna()
    sharpe = (
        float(r.mean() / r.std(ddof=1) * np.sqrt(252))
        if len(r) > 1 and r.std(ddof=1) > 0
        else np.nan
    )
    return {"TotalReturn": total, "CAGR": cagr, "MDD": float(dd.min()), "Sharpe": sharpe}


def main() -> None:
    ap = argparse.ArgumentParser(description="KOSPI200 SSL Hybrid backtest")
    ap.add_argument("--start", default="2023-08-10")
    ap.add_argument("--end", default="2026-08-10")
    ap.add_argument("--universe-csv", default=None)
    ap.add_argument("--marcap-dir", default=None)
    ap.add_argument("--online", action="store_true")
    ap.add_argument(
        "--fee-side",
        type=float,
        default=0.0,
        help="Per-side proportional trading cost. Example: 0.001 = 0.10%% each side.",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=120,
        help="Minimum loaded bars before signals are allowed. Default 120.",
    )
    ap.add_argument(
        "--prehistory-days",
        type=int,
        default=400,
        help="Calendar days loaded before --start for indicator warm-up. Default 400.",
    )
    ap.add_argument("--out", default="output")

    # Pine defaults exposed for easy GitHub experiments.
    ap.add_argument("--baseline-len", type=int, default=60)
    ap.add_argument("--ssl2-len", type=int, default=5)
    ap.add_argument("--exit-len", type=int, default=15)
    ap.add_argument("--atr-len", type=int, default=14)
    ap.add_argument("--atr-crit", type=float, default=0.9)
    ap.add_argument("--jurik-phase", type=int, default=3)
    ap.add_argument("--jurik-power", type=int, default=1)
    args = ap.parse_args()

    if not args.online and not args.marcap_dir:
        raise SystemExit("Choose --online or --marcap-dir.")

    if pd.Timestamp(args.end) < pd.Timestamp(args.start):
        raise SystemExit("--end must be >= --start")

    if args.universe_csv:
        universe = load_universe_csv(args.universe_csv)
    else:
        universe = fetch_current_kospi200_online()

    if len(universe) != 200:
        print(f"WARNING: universe count is {len(universe)}, expected 200.")

    # Load enough prior bars to stabilize HMA60/JMA state before evaluation.
    data_start = prehistory_start(args.start, args.prehistory_days)

    marcap = None
    if args.marcap_dir:
        marcap = load_marcap(args.marcap_dir, data_start, args.end)

    all_stats: list[dict] = []
    all_trades: list[pd.DataFrame] = []
    dailies: dict[str, pd.DataFrame] = {}

    for j, row in universe.iterrows():
        code = str(row["Code"]).zfill(6)
        name = str(row["Name"])

        try:
            if marcap is not None:
                z = marcap[marcap["Code"] == code].copy()
                z = z.set_index("Date").sort_index()
            else:
                z = online_price(code, data_start, args.end)

            stats, trades, daily = backtest_one(
                z,
                code,
                name,
                start=args.start,
                end=args.end,
                fee_side=args.fee_side,
                warmup=args.warmup,
                baseline_len=args.baseline_len,
                ssl2_len=args.ssl2_len,
                exit_len=args.exit_len,
                atr_len=args.atr_len,
                atr_crit=args.atr_crit,
                jurik_phase=args.jurik_phase,
                jurik_power=args.jurik_power,
            )

            if stats is not None:
                all_stats.append(stats)
                dailies[code] = daily
                if not trades.empty:
                    all_trades.append(trades)

            print(f"[{j + 1:3d}/{len(universe)}] {code} {name}: OK")

        except Exception as e:
            print(f"[{j + 1:3d}/{len(universe)}] {code} {name}: ERROR {e}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    by_stock = pd.DataFrame(all_stats)
    by_stock.to_csv(outdir / "by_stock.csv", index=False, encoding="utf-8-sig")

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    trades.to_csv(outdir / "trades.csv", index=False, encoding="utf-8-sig")

    portfolio = aggregate_portfolio(dailies, args.start, args.end)
    portfolio.to_csv(outdir / "portfolio_daily.csv", encoding="utf-8-sig")

    pm = metrics_from_equity(portfolio)
    summary = {
        "Strategy": "SSL Hybrid: SSL2 BUY Continuation -> EXIT LONG",
        "Universe": "Current KOSPI200 frozen across full period",
        "SurvivorshipBiasHandled": False,
        "UniverseRequested": 200,
        "StocksWithUsableData": len(by_stock),
        "PeriodStart": args.start,
        "PeriodEnd": args.end,
        "FeePerSide": args.fee_side,
        "BaselineType": "HMA",
        "BaselineLength": args.baseline_len,
        "SSL2Type": "JMA",
        "SSL2Length": args.ssl2_len,
        "ExitType": "HMA",
        "ExitLength": args.exit_len,
        "ATRLength": args.atr_len,
        "ATRSmoothing": "WMA",
        "ATRContinuationCriteria": args.atr_crit,
        "JurikPhase": args.jurik_phase,
        "JurikPower": args.jurik_power,
        **{f"Portfolio_{k}": v for k, v in pm.items()},
        "CrossSection_MeanStrategyReturn": by_stock["StrategyReturn"].mean() if len(by_stock) else np.nan,
        "CrossSection_MedianStrategyReturn": by_stock["StrategyReturn"].median() if len(by_stock) else np.nan,
        "CrossSection_MeanCAGR": by_stock["CAGR"].mean() if len(by_stock) else np.nan,
        "CrossSection_MeanMDD": by_stock["MDD"].mean() if len(by_stock) else np.nan,
        "CrossSection_MeanSharpe": by_stock["Sharpe"].mean() if len(by_stock) else np.nan,
        "TradeWinRate_AllTrades": (trades["Return"] > 0).mean() if len(trades) else np.nan,
        "AvgTradeReturn_AllTrades": trades["Return"].mean() if len(trades) else np.nan,
        "TotalTrades": len(trades),
        "CrossSection_MeanBuyHoldReturn": by_stock["BuyHoldReturn"].mean() if len(by_stock) else np.nan,
        "StrategyBeatsBuyHoldPct": (by_stock["ExcessVsBuyHold"] > 0).mean() if len(by_stock) else np.nan,
    }

    pd.DataFrame([summary]).to_csv(outdir / "summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nSaved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
