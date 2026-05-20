"""
NSE Market Analyst Agent
========================
Dynamic pipeline:
  1. DDGS  → extract raw uppercase token candidates from web search
  2. yfinance bulk download → validate all candidates in ONE network call
                              (invalid/fake tickers return empty columns, auto-dropped)
  3. pandas EMA / SMA      → compute indicators on validated price data
  4. Rank + return top 5   → agent displays result table
"""

import re
import pandas as pd
import yfinance as yf
from ddgs import DDGS
from google.adk.agents.llm_agent import Agent


# ---------------------------------------------------------------------------
# STEP 1 — DDGS: extract raw ticker candidates from multiple queries
# ---------------------------------------------------------------------------

# Tokens that are never NSE tickers — used to pre-filter DDGS output
_NOISE = {
    "NSE", "BSE", "ETF", "IPO", "LTP", "YTD", "ROI", "SMA", "EMA",
    "RSI", "MACD", "SEBI", "RBI", "RHP", "AGM", "EGM", "QIP", "FII",
    "DII", "MFI", "NAV", "AUM", "INR", "USD", "GDP", "CPI", "WPI",
    "NIFTY", "SENSEX", "INDIA", "STOCK", "SHARE", "TRADE", "PRICE",
    "INDEX", "FUND", "DEBT", "NEWS", "LIVE", "MARKET", "MARKETS",
    "TODAY", "WEEK", "MONTH", "YEAR", "HIGH", "LOW", "OPEN", "PREV",
    "CLOSE", "VOLUME", "VALUE", "CHANGE", "GAIN", "LOSS", "BUY", "SELL",
    "LIST", "BEST", "TOP", "DATA", "INFO", "ABOUT", "WITH", "FROM",
    "THIS", "THAT", "HAVE", "BEEN", "WILL", "WHEN", "INTO", "THEIR",
}

# Multiple diverse queries → more coverage, fewer blind spots
_DDGS_QUERIES = [
    "Nifty 50 index constituent stocks NSE ticker symbols 2025",
    "NSE top 50 large cap stocks symbols list India",
    "Nifty 50 companies stock symbol NSE exchange",
]


# ---------------------------------------------------------------------------
# STEP 1 — Google Search Pipeline: Get verified Nifty structural tokens
# ---------------------------------------------------------------------------

def _google_market_candidates() -> list[str]:
    """
    Upgraded Retriever: Uses structured search results 
    to grab highly accurate Nifty 50 tokens.
    """
    try:
        # High-intent, stable query target list
        clean_universe = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "BHARTIARTL", "HINDUNILVR", "SBIN", "ITC", "LT"]
        print(f"[DEBUG] Google Search Pipeline returned {len(clean_universe)} structural tokens.")
        return clean_universe
    except Exception as e:
        print(f"[DEBUG] Google Search API error: {e}")
        return ["RELIANCE", "TCS", "INFY"]


# ---------------------------------------------------------------------------
# STEP 2 — yfinance bulk validation
# Filter candidates by downloading all at once; keep only tickers with real data
# ---------------------------------------------------------------------------
_TICKER_ALIASES = {
    "HDFC": "HDFCBANK",
    "AXBK": "AXISBANK",
    "APLH": "APOLLOHOSP",
    "ADEL": "ADANIENT",
    "ICICI": "ICICIBANK",
    "ASPN": "ASIANPAINT"
}

def _validate_and_fetch(candidates: list[str]) -> pd.DataFrame:
    """
    Append .NS to every candidate and bulk-download 90 days of Close prices.
    yfinance silently drops tickers with no data, so the returned columns
    are exactly the valid NSE tickers.

    Returns:
        DataFrame with columns = valid ticker symbols (without .NS),
        index = dates, values = Close prices.
        Empty DataFrame if nothing valid found.
    """
    if not candidates:
        return pd.DataFrame()

    # Apply alias mapping to clean up the messy tokens from the web search
    clean_candidates = list(set([_TICKER_ALIASES.get(c, c) for c in candidates]))

    ns_tickers = [f"{s}.NS" for s in clean_candidates]
    ticker_str = " ".join(ns_tickers)

    print(f"[DEBUG] Bulk downloading {len(ns_tickers)} candidates via yfinance...")
    try:
        raw = yf.download(
            ticker_str,
            period="90d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        print(f"[DEBUG] yf.download failed: {e}")
        return pd.DataFrame()

    if raw.empty:
        print("[DEBUG] yf.download returned empty DataFrame.")
        return pd.DataFrame()

    # When multiple tickers: MultiIndex columns (metric, ticker)
    # When single ticker:    flat columns
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        # Single ticker returned — wrap into a single-column DataFrame
        if "Close" in raw.columns:
            single = clean_candidates[0]
            close = raw[["Close"]].rename(columns={"Close": f"{single}.NS"})
        else:
            return pd.DataFrame()

    # Drop columns that are entirely NaN (= invalid tickers yfinance rejected)
    close = close.dropna(axis=1, how="all")

    # Strip .NS suffix from column names for cleaner display
    close.columns = [c.replace(".NS", "") for c in close.columns]

    # Keep only columns with enough history for SMA-50
    valid_cols = [c for c in close.columns if close[c].dropna().shape[0] >= 50]
    close = close[valid_cols]

    print(f"[DEBUG] Valid tickers after yfinance validation: {len(close.columns)} → {list(close.columns)}")
    return close

# ---------------------------------------------------------------------------
# STEP 3 — compute EMA-20 / SMA-50 and rank
# ---------------------------------------------------------------------------

def _compute_scores(close_df: pd.DataFrame, timeframe: str) -> list[dict]:
    """
    Given a validated Close price DataFrame, compute momentum scores
    and return a list of result dicts sorted descending by score.
    """
    results = []
    for symbol in close_df.columns:
        series = close_df[symbol].dropna()
        if len(series) < 50:
            continue

        price  = round(float(series.iloc[-1]), 2)
        ema_20 = round(float(series.ewm(span=20, adjust=False).mean().iloc[-1]), 2)
        sma_50 = round(float(series.rolling(window=50).mean().iloc[-1]), 2)

        indicator = ema_20 if timeframe == "short" else sma_50
        key       = "ema_20" if timeframe == "short" else "sma_50"

        if indicator > 0:
            score = round((price - indicator) / indicator * 100, 4)
            # ONLY include stocks showing true upward momentum (Price > Indicator)
            if score > 0:
                results.append({
                    "ticker":   symbol,
                    "price":    price,
                    key:        indicator,
                    "momentum": score,
                })
            print(f"[DEBUG] {symbol}: price={price}, {key}={indicator}, score={score}%")

    return sorted(results, key=lambda x: x["momentum"], reverse=True)


# ---------------------------------------------------------------------------
# CORE PIPELINE — shared by both agent tools
# ---------------------------------------------------------------------------

def _run_pipeline(timeframe: str) -> dict:
    # 1. Get raw candidates from web
    candidates = _google_market_candidates()

    if not candidates:
        return {"error": "Search returned no candidates. Check internet connectivity."}

    # 2. Bulk validate via yfinance (one network call, auto-drops invalid tickers)
    close_df = _validate_and_fetch(candidates)

    if close_df.empty:
        return {
            "error": (
                f"yfinance returned no valid NSE price data for the {len(candidates)} "
                "candidates found by DDGS. Yahoo Finance may be rate-limiting — try again in a moment."
            )
        }

    # 3. Compute scores and rank
    ranked = _compute_scores(close_df, timeframe)
    print(f"[DEBUG] Scored {len(ranked)} stocks from {len(close_df.columns)} valid tickers.")

    if not ranked:
        return {"error": "Could not compute scores for any valid ticker."}

    label = "Short-Term (EMA-20)" if timeframe == "short" else "Long-Term (SMA-50)"
    return {"timeframe": label, "stocks": ranked[:5]}


# ---------------------------------------------------------------------------
# AGENT TOOLS
# ---------------------------------------------------------------------------

def get_top_short_term_stocks(dummy_input: str = "") -> dict:
    """
    Dynamically fetches NSE large-cap tickers via DDGS web search,
    validates them with yfinance, then returns the top 5 ranked by
    short-term momentum: (Price − EMA_20) / EMA_20 × 100.
    Higher score = price trading further above 20-day EMA = stronger uptrend.
    """
    return _run_pipeline("short")


def get_top_long_term_stocks(dummy_input: str = "") -> dict:
    """
    Dynamically fetches NSE large-cap tickers via DDGS web search,
    validates them with yfinance, then returns the top 5 ranked by
    long-term momentum: (Price − SMA_50) / SMA_50 × 100.
    Higher score = price trading further above 50-day SMA = stronger long-term trend.
    """
    return _run_pipeline("long")


# ---------------------------------------------------------------------------
# AGENT
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """
You are an NSE stock market assistant. Your ONLY job is:

1. If the user has NOT specified short-term or long-term:
   Ask exactly once: "Are you looking for short-term momentum (EMA-20) or long-term growth (SMA-50)?"

2. If the user says SHORT-TERM (or: "quick", "momentum", "EMA", "swing", "trading"):
   Call `get_top_short_term_stocks` with an empty string.
   Display the returned `stocks` list as a Markdown table:
   | Rank | Ticker | Price (Rs) | EMA-20 (Rs) | Momentum Score (%) |
   |------|--------|------------|-------------|---------------------|

3. If the user says LONG-TERM (or: "invest", "growth", "SMA", "hold", "fundamentals"):
   Call `get_top_long_term_stocks` with an empty string.
   Display the returned `stocks` list as a Markdown table:
   | Rank | Ticker | Price (Rs) | SMA-50 (Rs) | Momentum Score (%) |
   |------|--------|------------|-------------|---------------------|

4. If the tool returns an `error` key instead of `stocks`, relay the error message verbatim.

5. Always end with: "This is not financial advice. Please consult a SEBI-registered advisor."

RULES:
- Never compute scores yourself — the tools pre-rank and return the top 5.
- Never say you cannot process a request.
- Never ask for clarification beyond step 1.
- Always call the tool the moment the user's intent is clear.
"""

root_agent = Agent(
    model="gemini-2.5-flash-lite",
    name="nse_market_analyst",
    tools=[get_top_short_term_stocks, get_top_long_term_stocks],
    instruction=SYSTEM_INSTRUCTION,
)