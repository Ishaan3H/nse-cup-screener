"""Monthly OHLCV download with an on-disk cache.

Bars are split-adjusted but not dividend-adjusted (``auto_adjust=False``), so the
price levels in the output — pivots, stops, rims — are levels you can actually
put into an order ticket.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / f"{symbol.replace('/', '_')}.csv"


def _is_fresh(path: Path, max_age_hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) / 3600.0 <= max_age_hours


def _clean(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    df = df.copy()
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return None
    df = df[REQUIRED_COLS]
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df["Close"] > 0) & (df["High"] >= df["Low"])]
    df["Volume"] = df["Volume"].fillna(0)
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def load_cached(cache_dir: Path, symbol: str) -> pd.DataFrame | None:
    path = _cache_path(cache_dir, symbol)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:  # noqa: BLE001 - a corrupt cache file should not kill the run
        return None
    return _clean(df)


def download_monthly(
    symbols: list[str],
    cache_dir: Path,
    period: str = "20y",
    batch_size: int = 60,
    max_age_hours: float = 12,
    refresh: bool = False,
    log=print,
) -> dict[str, pd.DataFrame]:
    """Return {symbol: monthly OHLCV}. Cached symbols are not re-fetched."""
    import yfinance as yf

    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    stale: list[str] = []

    for sym in symbols:
        if not refresh and _is_fresh(_cache_path(cache_dir, sym), max_age_hours):
            df = load_cached(cache_dir, sym)
            if df is not None:
                out[sym] = df
                continue
        stale.append(sym)

    if out:
        log(f"  {len(out)} symbols served from cache, {len(stale)} to download")

    batches = [stale[i : i + batch_size] for i in range(0, len(stale), batch_size)]
    for i, batch in enumerate(batches, 1):
        yahoo = [f"{s}.NS" for s in batch]
        try:
            raw = yf.download(
                yahoo,
                period=period,
                interval="1mo",
                auto_adjust=False,
                actions=False,
                group_by="ticker",
                threads=True,
                progress=False,
                timeout=60,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"  batch {i}/{len(batches)} failed: {type(exc).__name__}: {exc}")
            continue

        for sym, ysym in zip(batch, yahoo):
            try:
                sub = raw[ysym] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            cleaned = _clean(sub)
            if cleaned is None:
                continue
            out[sym] = cleaned
            cleaned.to_csv(_cache_path(cache_dir, sym))

        log(f"  prices: batch {i}/{len(batches)} ({len(out)} symbols with data)")
        time.sleep(0.4)

    return out
