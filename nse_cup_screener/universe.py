"""Build the screening universe: every NSE-listed equity, filtered by market cap.

Symbol master comes from NSE itself (EQUITY_L.csv, the official list of listed
securities).  Market caps come from Yahoo's bulk quote endpoint, which accepts
~100 symbols per request, so ~2,100 stocks cost ~21 requests instead of 2,100.
"""

from __future__ import annotations

import csv
import io
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

NSE_EQUITY_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
YAHOO_QUOTE_URLS = (
    "https://query2.finance.yahoo.com/v7/finance/quote",
    "https://query1.finance.yahoo.com/v7/finance/quote",
)
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
CRORE = 1e7  # 1 crore = 10 million rupees


@dataclass
class UniverseStats:
    listed: int = 0
    quoted: int = 0
    passed_mcap: int = 0
    missing_mcap: int = 0


def _cache_is_fresh(path: Path, max_age_days: float) -> bool:
    if not path.exists():
        return False
    age_days = (time.time() - path.stat().st_mtime) / 86400.0
    return age_days <= max_age_days


def fetch_nse_equity_list(cache_dir: Path, max_age_days: float = 7, refresh: bool = False) -> pd.DataFrame:
    """Official NSE list of listed securities. Cached to disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "EQUITY_L.csv"

    if refresh or not _cache_is_fresh(path, max_age_days):
        headers = {"User-Agent": BROWSER_UA, "Accept": "text/csv,*/*", "Referer": "https://www.nseindia.com/"}
        resp = requests.get(NSE_EQUITY_LIST_URL, headers=headers, timeout=60)
        resp.raise_for_status()
        text = resp.text
        if "SYMBOL" not in text.split("\n", 1)[0].upper():
            raise RuntimeError("Unexpected response from NSE equity list endpoint")
        path.write_text(text, encoding="utf-8")

    df = pd.read_csv(path)
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df = df.rename(columns={"NAME_OF_COMPANY": "NAME"})
    return df


def _yahoo_symbol(nse_symbol: str) -> str:
    return f"{nse_symbol}.NS"


def _quote_batch(session_data, symbols: list[str]) -> list[dict]:
    """One bulk quote request. Falls back across Yahoo hosts."""
    last_err: Exception | None = None
    for url in YAHOO_QUOTE_URLS:
        try:
            payload = session_data.get_raw_json(url, params={"symbols": ",".join(symbols)})
            return payload.get("quoteResponse", {}).get("result", []) or []
        except Exception as exc:  # noqa: BLE001 - network layer raises many shapes
            last_err = exc
    raise RuntimeError(f"Yahoo quote request failed: {last_err}")


def fetch_market_caps(
    symbols: list[str],
    cache_dir: Path,
    max_age_days: float = 3,
    refresh: bool = False,
    batch_size: int = 100,
    log=print,
) -> pd.DataFrame:
    """Market cap (INR) + last price + avg volume for each symbol, cached to disk."""
    from yfinance.data import YfData  # imported lazily: pulls in yfinance's session/crumb handling

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "market_caps.csv"

    if not refresh and _cache_is_fresh(path, max_age_days):
        cached = pd.read_csv(path)
        if set(symbols).issubset(set(cached["SYMBOL"])):
            return cached

    yf_data = YfData()
    rows: list[dict] = []
    batches = [symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)]
    for i, batch in enumerate(batches, 1):
        yahoo_syms = [_yahoo_symbol(s) for s in batch]
        try:
            results = _quote_batch(yf_data, yahoo_syms)
        except RuntimeError as exc:
            log(f"  quote batch {i}/{len(batches)} failed ({exc}); retrying once")
            time.sleep(2)
            try:
                results = _quote_batch(yf_data, yahoo_syms)
            except RuntimeError:
                log(f"  quote batch {i}/{len(batches)} failed again; skipping")
                results = []
        for r in results:
            sym = str(r.get("symbol", ""))
            if not sym.endswith(".NS"):
                continue
            rows.append(
                {
                    "SYMBOL": sym[:-3],
                    "YF_SYMBOL": sym,
                    "MARKET_CAP": r.get("marketCap"),
                    "LAST_PRICE": r.get("regularMarketPrice"),
                    "AVG_VOL_3M": r.get("averageDailyVolume3Month"),
                    "CURRENCY": r.get("currency"),
                }
            )
        log(f"  market caps: batch {i}/{len(batches)} ({len(rows)} quoted)")
        time.sleep(0.3)

    df = pd.DataFrame(rows).drop_duplicates(subset="SYMBOL")
    if df.empty:
        raise RuntimeError("Could not retrieve any market caps from Yahoo")
    df["FETCHED_AT"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    df.to_csv(path, index=False)
    return df


def build_universe(
    cache_dir: Path,
    min_market_cap_cr: float = 1000.0,
    series: tuple[str, ...] = ("EQ",),
    refresh: bool = False,
    log=print,
) -> tuple[pd.DataFrame, UniverseStats]:
    """NSE equities of the given series with market cap >= the floor (in ₹ crore)."""
    stats = UniverseStats()

    listing = fetch_nse_equity_list(cache_dir, refresh=refresh)
    if series:
        listing = listing[listing["SERIES"].isin(series)]
    listing = listing[["SYMBOL", "NAME", "SERIES", "DATE_OF_LISTING", "ISIN_NUMBER"]].copy()
    stats.listed = len(listing)
    log(f"NSE listed ({'/'.join(series)}): {stats.listed} symbols")

    caps = fetch_market_caps(listing["SYMBOL"].tolist(), cache_dir, refresh=refresh, log=log)
    merged = listing.merge(caps, on="SYMBOL", how="left")
    stats.quoted = int(merged["MARKET_CAP"].notna().sum())
    stats.missing_mcap = int(merged["MARKET_CAP"].isna().sum())

    merged["MCAP_CR"] = merged["MARKET_CAP"] / CRORE
    keep = merged[merged["MCAP_CR"] >= min_market_cap_cr].copy()
    keep = keep.sort_values("MCAP_CR", ascending=False).reset_index(drop=True)
    stats.passed_mcap = len(keep)

    log(
        f"Market cap >= ₹{min_market_cap_cr:,.0f} Cr: {stats.passed_mcap} symbols "
        f"({stats.missing_mcap} had no quote and were dropped)"
    )
    return keep, stats


def write_universe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["SYMBOL", "NAME", "MCAP_CR", "LAST_PRICE", "AVG_VOL_3M", "ISIN_NUMBER"]
    df[cols].to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def read_universe(path: Path) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(path.read_text(encoding="utf-8")))
