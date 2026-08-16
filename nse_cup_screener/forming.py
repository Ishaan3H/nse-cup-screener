"""Cups that are still being built — the watchlist.

This is staging, not forecasting. A cup's left half is fully observable long
before the right half exists: the advance into the rim, the rim itself, the
decline, and the bottom. Once price turns up off that bottom, the stock is
walking a right side toward a known price level. What this module does is find
those in-progress bases and rank them by how far along and how healthy they
are.

What it deliberately does not do is guess that a stock which has not yet
declined will form a cup. There is no signal for that. Every candidate here has
already put in a rim, a bottom, and the start of a recovery.

The ETA columns are extrapolation and should be read as rough. Two independent
estimates are given because they fail differently:

  - symmetry ETA: classic cups spend about as long on the right side as the
    left, so the remainder is (left months − right months so far). Blind to
    price, but grounded in the shape.
  - rate ETA: fits the recent monthly advance and asks how long that pace needs
    to cover the remaining distance. Grounded in price, but a fast recent run
    flatters it.

When the two disagree wildly, the setup is not behaving like a textbook cup.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .patterns import Params, _fit_roundness, _safe_mean, _smooth

STAGES = ["BOTTOMING", "EARLY_RIGHT", "MID_RIGHT", "APPROACHING_RIM"]


@dataclass
class FormingParams:
    min_months_so_far: int = 24   # rim must already be this far back
    min_right_len: int = 3        # months of recovery needed to call the low in
    min_recovery: float = 0.15    # fraction of the cup's depth won back
    max_recovery: float = 0.88    # past this it belongs to the main screener
    min_months_since_low: int = 3
    min_r2: float = 0.45          # looser than a finished cup: the shape is partial
    min_higher_low: float = 1.05  # recent lows must sit this far above the bottom
    max_eta: int = 36             # ignore candidates more than this many months away
    rate_window: int = 12         # months of advance used for the rate ETA


@dataclass
class Forming:
    symbol: str = ""
    name: str = ""
    stage: str = "BOTTOMING"
    score: float = 0.0

    left_idx: int = 0
    bottom_idx: int = 0
    left_date: str = ""
    bottom_date: str = ""

    left_rim: float = 0.0        # the price level that completes the cup
    cup_low: float = 0.0
    last_close: float = 0.0
    depth_pct: float = 0.0

    months_so_far: int = 0
    left_len_m: int = 0
    right_len_m: int = 0
    months_since_low: int = 0

    recovery_pct: float = 0.0    # how much of the depth is won back
    to_rim_pct: float = 0.0      # gain still needed to reach the rim
    eta_symmetry_m: int = 0
    eta_rate_m: int = 0
    eta_m: int = 0               # the more conservative of the two
    projected_len_m: int = 0
    projected_complete: str = ""

    roundness_r2: float = 0.0
    prior_gain_pct: float = 0.0
    higher_lows: bool = False
    vol_dryup: float = float("nan")
    advance_rate_pct: float = 0.0  # % per month over the rate window

    mcap_cr: float = float("nan")
    notes: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["notes"] = "; ".join(self.notes)
        return row


def _stage(recovery: float) -> str:
    if recovery < 0.30:
        return "BOTTOMING"
    if recovery < 0.55:
        return "EARLY_RIGHT"
    if recovery < 0.75:
        return "MID_RIGHT"
    return "APPROACHING_RIM"


def _score(c: Forming, fp: FormingParams) -> float:
    """How much this deserves a place on a watchlist, 0-100."""
    parts = [
        # Progress: further along the right side is more actionable.
        (25, min(1.0, c.recovery_pct / 0.85)),
        # Shape so far.
        (20, max(0.0, min(1.0, (c.roundness_r2 - fp.min_r2) / (0.95 - fp.min_r2)))),
        # Nearness in time — something 6 months out beats something 30 months out.
        (18, max(0.0, 1.0 - c.eta_m / fp.max_eta)),
        # Depth in a sane band; very deep bases are harder to repair.
        (12, 1.0 if 0.20 <= c.depth_pct / 100 <= 0.60 else 0.45),
        # Structure confirming the turn.
        (10, 1.0 if c.higher_lows else 0.0),
        # The two ETAs agreeing means it is tracking a normal cup shape.
        (
            10,
            1.0 - min(1.0, abs(c.eta_symmetry_m - c.eta_rate_m) / max(6, fp.max_eta / 2))
            if c.eta_rate_m and c.eta_symmetry_m
            else 0.4,
        ),
        # Volume drying up into the low.
        (5, 1.0 - min(1.0, c.vol_dryup) if math.isfinite(c.vol_dryup) else 0.5),
    ]
    return float(sum(w * s for w, s in parts))


def detect_forming(
    df: pd.DataFrame,
    symbol: str = "",
    name: str = "",
    p: Params | None = None,
    fp: FormingParams | None = None,
) -> list[Forming]:
    """In-progress cups for one symbol, best first."""
    p = p or Params()
    fp = fp or FormingParams()
    n = len(df)
    if n < fp.min_months_so_far + 6:
        return []

    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    close = df["Close"].to_numpy(float)
    volume = df["Volume"].to_numpy(float)
    dates = df.index
    last = n - 1
    last_close = float(close[last])

    out: list[Forming] = []

    for L in range(1, n - fp.min_months_so_far):
        if high[L] < high[L - 1] or (L + 1 < n and high[L] < high[L + 1]):
            continue
        left_rim = float(high[L])
        months_so_far = last - L
        if months_so_far < fp.min_months_so_far or months_so_far > p.max_cup_len:
            continue

        # The rim must still be overhead. If price has already reclaimed it the
        # cup is finished, and the main screener owns that case.
        if float(np.max(high[L + 1 :])) > left_rim * (1 + p.interior_tol):
            continue

        B = int(np.argmin(low[L + 1 :])) + L + 1
        bottom = float(low[B])
        if bottom <= 0:
            continue
        depth = (left_rim - bottom) / left_rim
        if not (p.min_depth <= depth <= p.max_depth):
            continue

        left_len, right_len = B - L, last - B
        if left_len < p.min_side_len or right_len < fp.min_right_len:
            continue
        if right_len < fp.min_months_since_low:
            continue

        span = left_rim - bottom
        recovery = (last_close - bottom) / span if span > 0 else 0.0
        if not (fp.min_recovery <= recovery <= fp.max_recovery):
            continue

        # A cup is a pause inside an advance.
        lb = max(0, L - p.prior_lookback)
        prior_low = float(np.min(low[lb : L + 1]))
        prior_gain = left_rim / prior_low - 1.0 if prior_low > 0 else 0.0
        if prior_gain < p.min_prior_gain and L >= p.min_prior_bars:
            continue

        # Shape so far must already curve like a cup, not a straight slide.
        typical = (high[L : last + 1] + low[L : last + 1] + close[L : last + 1]) / 3.0
        y = (typical - bottom) / span
        a, _vertex, r2 = _fit_roundness(_smooth(y, 3 if months_so_far < 48 else 5))
        if a <= 0 or r2 < fp.min_r2:
            continue

        # The turn has to be holding: recent lows above the bottom.
        recent = low[max(B + 1, last - 5) : last + 1]
        higher_lows = bool(recent.size and float(np.min(recent)) >= bottom * fp.min_higher_low)

        # --- the two ETAs ---
        eta_sym = max(0, left_len - right_len)

        window = min(fp.rate_window, right_len)
        if window < 3:
            continue
        seg = close[last - window + 1 : last + 1]
        slope = float(np.polyfit(np.arange(window, dtype=float), seg, 1)[0])
        rate_pct = 100 * slope / max(1e-9, last_close)
        if slope <= 0:
            continue  # right side is not actually advancing; nothing is being built
        eta_rate = int(math.ceil((left_rim - last_close) / slope))

        # The conservative of the two. eta_sym can legitimately be 0 when the
        # right side already outruns the left in time, but the rate estimate is
        # always positive here, so the result never collapses to "arriving now".
        eta = max(eta_sym, eta_rate)
        if eta > fp.max_eta or eta <= 0:
            continue

        projected_len = months_so_far + eta
        if projected_len < p.min_cup_len:
            continue  # would not end up a 3-year-plus cup

        third = max(1, months_so_far // 3)
        vol_dryup = float("nan")
        vol_all = _safe_mean(volume[L : last + 1])
        if vol_all and vol_all > 0:
            vol_dryup = _safe_mean(volume[max(L, B - third // 2) : min(last, B + third // 2) + 1]) / vol_all

        complete = dates[last] + pd.DateOffset(months=int(eta))

        c = Forming(
            symbol=symbol,
            name=name,
            stage=_stage(recovery),
            left_idx=L,
            bottom_idx=B,
            left_date=str(dates[L].date()),
            bottom_date=str(dates[B].date()),
            left_rim=round(left_rim, 2),
            cup_low=round(bottom, 2),
            last_close=round(last_close, 2),
            depth_pct=round(100 * depth, 1),
            months_so_far=months_so_far,
            left_len_m=left_len,
            right_len_m=right_len,
            months_since_low=right_len,
            recovery_pct=round(100 * recovery, 1),
            to_rim_pct=round(100 * (left_rim / last_close - 1), 1),
            eta_symmetry_m=eta_sym,
            eta_rate_m=eta_rate,
            eta_m=eta,
            projected_len_m=projected_len,
            projected_complete=complete.strftime("%b %Y"),
            roundness_r2=round(r2, 3),
            prior_gain_pct=round(100 * prior_gain, 1),
            higher_lows=higher_lows,
            vol_dryup=round(vol_dryup, 2) if math.isfinite(vol_dryup) else float("nan"),
            advance_rate_pct=round(rate_pct, 2),
        )
        c.score = round(_score(c, fp), 1)
        _annotate(c)
        out.append(c)

    if not out:
        return []
    out.sort(key=lambda x: x.score, reverse=True)
    return [out[0]]


def _annotate(c: Forming) -> None:
    if not c.higher_lows:
        c.notes.append("lows not yet rising — the turn is unconfirmed")
    if c.eta_rate_m and c.eta_symmetry_m and abs(c.eta_rate_m - c.eta_symmetry_m) > 12:
        c.notes.append("the two ETAs disagree by over a year — not tracking a textbook shape")
    if c.depth_pct > 60:
        c.notes.append(f"{c.depth_pct:.0f}% deep — a lot of overhead supply to work through")
    if c.recovery_pct >= 75:
        c.notes.append("right side nearly complete — watch for a handle forming")
    if math.isfinite(c.vol_dryup) and c.vol_dryup <= 0.7:
        c.notes.append("volume dried up at the low")
    if c.advance_rate_pct > 4:
        c.notes.append(f"advancing {c.advance_rate_pct:.1f}%/month — rate ETA may be optimistic")


def screen_forming(
    frames: dict[str, pd.DataFrame],
    meta: dict[str, dict] | None = None,
    p: Params | None = None,
    fp: FormingParams | None = None,
    log=None,
) -> list[Forming]:
    p, fp = p or Params(), fp or FormingParams()
    meta = meta or {}
    results: list[Forming] = []
    for i, (sym, df) in enumerate(frames.items(), 1):
        info = meta.get(sym, {})
        try:
            hits = detect_forming(df, symbol=sym, name=info.get("name", ""), p=p, fp=fp)
        except Exception as exc:  # noqa: BLE001
            if log:
                log(f"  {sym}: {type(exc).__name__}: {exc}")
            continue
        for h in hits:
            h.mcap_cr = info.get("mcap_cr", float("nan"))
        results.extend(hits)
        if log and i % 250 == 0:
            log(f"  scanned {i}/{len(frames)}, {len(results)} candidates so far")
    results.sort(key=lambda x: x.score, reverse=True)
    return results
