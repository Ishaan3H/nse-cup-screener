"""Cup and cup-with-handle detection on monthly bars.

The geometry, in the order the detector checks it:

    left rim (L)                                    right rim (R)
        *                                               *
         \                                             /  \\  handle
          \                                           /    \\_/
           \.                                       ./
             \..                                 ../
                \.......   cup bottom (B)  ....../

  1. L is a swing high, and no bar between L and R closes the gap above it —
     the cup's interior must stay under the left rim.
  2. R recovers to within a tolerance band of the left rim.
  3. B, the low between them, sits 12–50% below the rim.
  4. The curve is round, not a V: the fit to a parabola, the number of bars
     resting near the low, and left/right symmetry all have to hold up.
  5. Price rose meaningfully into L — a cup is a rest inside an uptrend, not a
     recovery off a multi-year floor.
  6. The handle, if any, is the pullback after R: shallow, in the upper half of
     the cup, drifting down on lighter volume.

Everything that passes gets scored 0–100 rather than merely accepted, so the
cutoff is yours to move.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #


@dataclass
class Params:
    # --- cup geometry (all lengths in months, since we work on monthly bars) ---
    min_cup_len: int = 36         # left rim -> right rim: multi-year bases only
    max_cup_len: int = 300        # up to 25 years — generational bases count
    min_depth: float = 0.12       # 12% below the rim
    max_depth: float = 0.95       # a 20-year base can round-trip almost the whole move
    rim_tol_low: float = 0.15     # right rim may sit this far below the left rim
    rim_tol_high: float = 0.08    # ...or this far above it
    interior_tol: float = 0.02    # bars inside the cup may poke this far above the rim
    min_side_len: int = 2         # bars on each side of the bottom
    min_symmetry: float = 0.25    # min(left, right) / max(left, right)
    min_bottom_bars: int = 2      # bars resting in the lowest quarter of the cup
    min_curvature_r2: float = 0.55
    vertex_band: tuple[float, float] = (0.20, 0.80)  # where the parabola's low may sit

    # --- context ---
    min_prior_gain: float = 0.30  # advance into the left rim
    prior_lookback: int = 24
    min_prior_bars: int = 6       # bars needed before the rim to judge that advance at all
    strict_prior: bool = False    # True = reject when it cannot be measured, instead of allowing it

    # --- handle ---
    max_handle_len: int = 9
    min_handle_depth: float = 0.03
    max_handle_depth: float = 0.30
    handle_max_retrace: float = 0.50  # handle low must stay in the upper half of the cup

    # --- actionability ---
    max_age: int = 12             # pattern must end within this many bars of today
    near_pivot_pct: float = 0.10  # "near the pivot" band below it
    extended_pct: float = 0.20    # past this much above the pivot, label it extended
    max_extension: float = 1.00   # past this much above the pivot, stop reporting it at all
    require_intact: bool = True   # kill the base if price digs back below the cup's midpoint

    def ideal(self) -> dict[str, tuple[float, float]]:
        """Bands that earn full marks when scoring (vs. the hard limits above).

        The length bands track the configured limits, so a screen tuned for
        multi-year bases does not mark every one of them down for being long.
        """
        return {
            "depth": (0.15, 0.35),
            "cup_len": (self.min_cup_len, min(self.max_cup_len, max(2 * self.min_cup_len, self.min_cup_len + 12))),
            "prior_gain": (0.30, 1.50),
            "handle_depth": (0.05, 0.15),
            "handle_len": (1, max(3, self.max_handle_len // 2)),
        }

    def bottom_bars_required(self, cup_len: int) -> int:
        """A 5-year cup resting on two bars is a V with a long approach."""
        return max(self.min_bottom_bars, int(round(0.08 * cup_len)))


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #


@dataclass
class Pattern:
    symbol: str = ""
    name: str = ""
    pattern: str = "CUP"          # CUP | CUP_HANDLE
    status: str = "FORMING"       # FORMING | NEAR_PIVOT | IN_HANDLE | BREAKOUT | EXTENDED
    score: float = 0.0
    cup_score: float = 0.0
    handle_score: float = 0.0

    # indices into the monthly frame
    left_idx: int = 0
    bottom_idx: int = 0
    right_idx: int = 0
    handle_end_idx: int = -1
    breakout_idx: int = -1
    months_since_breakout: int = -1  # -1 = has not cleared the pivot yet

    left_date: str = ""
    bottom_date: str = ""
    right_date: str = ""
    handle_end_date: str = ""
    breakout_date: str = ""

    left_rim: float = 0.0
    right_rim: float = 0.0
    cup_low: float = 0.0
    depth_pct: float = 0.0
    cup_len_m: int = 0
    left_len_m: int = 0
    right_len_m: int = 0
    symmetry: float = 0.0
    roundness_r2: float = 0.0
    bottom_bars: int = 0
    prior_gain_pct: float = 0.0

    handle_len_m: int = 0
    handle_high: float = 0.0
    handle_low: float = 0.0
    handle_depth_pct: float = 0.0
    handle_retrace: float = 0.0   # how far into the cup the handle dug (0 = at the rim)
    handle_drift: float = 0.0     # slope of handle closes, % per month

    pivot: float = 0.0
    off_ath_pct: float = 0.0      # how far the pivot sits below the highest high on record
    last_close: float = 0.0
    to_pivot_pct: float = 0.0      # +ve = pivot is above price
    above_pivot_pct: float = 0.0   # how far price has run past the pivot; -ve = still below
    gain_since_breakout_pct: float = 0.0
    stop_suggest: float = 0.0
    risk_pct: float = 0.0
    target_pct: float = 0.0       # measured move = cup depth projected off the pivot

    vol_dryup_cup: float = float("nan")
    vol_dryup_handle: float = float("nan")
    vol_breakout: float = float("nan")

    mcap_cr: float = float("nan")
    notes: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["notes"] = "; ".join(self.notes)
        return row


# --------------------------------------------------------------------------- #
# Small scoring helpers
# --------------------------------------------------------------------------- #


def _band_score(value: float, lo: float, hi: float, hard_lo: float, hard_hi: float) -> float:
    """1.0 inside [lo, hi], tapering to 0 at the hard limits."""
    if not math.isfinite(value):
        return 0.0
    if lo <= value <= hi:
        return 1.0
    if value < lo:
        return max(0.0, (value - hard_lo) / (lo - hard_lo)) if lo > hard_lo else 0.0
    return max(0.0, (hard_hi - value) / (hard_hi - hi)) if hard_hi > hi else 0.0


def _ramp(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value) or hi <= lo:
        return 0.0
    return float(min(1.0, max(0.0, (value - lo) / (hi - lo))))


def _safe_mean(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def _smooth(y: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average with edge padding.

    Over a base lasting years, month-to-month noise would drag the parabola fit
    down even for an obviously round shape. Smoothing tests the *shape*.
    """
    if w < 2 or y.size < w:
        return y
    pad = w // 2
    padded = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    kernel = np.ones(w) / w
    return np.convolve(padded, kernel, mode="valid")[: y.size]


def _fit_roundness(y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = ax^2 + bx + c over x in [0, 1]. Returns (a, vertex_x, r2)."""
    n = y.size
    if n < 4:
        return 0.0, 0.5, 0.0
    x = np.linspace(0.0, 1.0, n)
    try:
        a, b, c = np.polyfit(x, y, 2)
    except Exception:  # noqa: BLE001 - degenerate segments
        return 0.0, 0.5, 0.0
    if not np.isfinite([a, b, c]).all() or a == 0:
        return 0.0, 0.5, 0.0
    pred = a * x**2 + b * x + c
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(a), float(-b / (2 * a)), float(r2)


def _evaluate_cup(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    L: int,
    B: int,
    R: int,
    p: Params,
) -> dict | None:
    """Geometry + quality checks for one candidate cup. None = rejected."""
    left_rim = float(high[L])
    right_rim = float(high[R])
    bottom = float(low[B])
    if left_rim <= 0 or bottom <= 0:
        return None

    depth = (left_rim - bottom) / left_rim
    if not (p.min_depth <= depth <= p.max_depth):
        return None

    left_len, right_len = B - L, R - B
    if left_len < p.min_side_len or right_len < p.min_side_len:
        return None
    symmetry = min(left_len, right_len) / max(left_len, right_len)
    if symmetry < p.min_symmetry:
        return None

    # Rounded, not a V: bars must linger near the low.
    span = left_rim - bottom
    band = bottom + 0.25 * span
    bottom_bars = int(np.sum(low[L : R + 1] <= band))
    bottom_bars_req = p.bottom_bars_required(R - L)
    if bottom_bars < bottom_bars_req:
        return None

    # Parabola fit on the typical price of each bar, normalised to the cup's box.
    typical = (high[L : R + 1] + low[L : R + 1] + close[L : R + 1]) / 3.0
    y = (typical - bottom) / span
    a, vertex, r2 = _fit_roundness(_smooth(y, 3 if (R - L) < 48 else 5))
    if a <= 0 or r2 < p.min_curvature_r2:
        return None
    if not (p.vertex_band[0] <= vertex <= p.vertex_band[1]):
        return None

    # A cup is a pause inside an advance. When the rim sits at the very start of
    # the available history — a stock that listed just before it — there is no
    # room to measure that advance. Unmeasurable is not the same as failed, so
    # it passes with the fact recorded rather than being silently dropped.
    lb = max(0, L - p.prior_lookback)
    prior_low = float(np.min(low[lb : L + 1]))
    prior_gain = left_rim / prior_low - 1.0 if prior_low > 0 else 0.0
    prior_measurable = L >= p.min_prior_bars
    if prior_gain < p.min_prior_gain and (prior_measurable or p.strict_prior):
        return None

    # Volume should dry up into the low.
    third = max(1, (R - L) // 3)
    vol_bottom = _safe_mean(volume[max(L, B - third // 2) : min(R, B + third // 2) + 1])
    vol_cup = _safe_mean(volume[L : R + 1])
    vol_dryup = vol_bottom / vol_cup if vol_cup and math.isfinite(vol_cup) and vol_cup > 0 else float("nan")

    return {
        "left_rim": left_rim,
        "right_rim": right_rim,
        "bottom": bottom,
        "depth": depth,
        "left_len": left_len,
        "right_len": right_len,
        "symmetry": symmetry,
        "bottom_bars": bottom_bars,
        "bottom_bars_req": bottom_bars_req,
        "r2": r2,
        "vertex": vertex,
        "prior_gain": prior_gain,
        "prior_measurable": prior_measurable,
        "vol_dryup_cup": vol_dryup,
        "vol_cup": vol_cup,
    }


def _score_cup(m: dict, p: Params) -> float:
    ideal = p.ideal()
    parts = [
        # Fixed reference scale, not p.max_depth: otherwise loosening the cap
        # silently lifts every score and runs stop being comparable.
        (18, _band_score(m["depth"], *ideal["depth"], 0.05, 0.85)),
        (12, _band_score(m["left_len"] + m["right_len"], *ideal["cup_len"], p.min_cup_len, p.max_cup_len)),
        (18, _ramp(m["r2"], p.min_curvature_r2, 0.95)),
        (15, _ramp(m["bottom_bars"], m["bottom_bars_req"], max(m["bottom_bars_req"] + 1, int(0.3 * (m["left_len"] + m["right_len"]))))),
        (12, _ramp(m["symmetry"], p.min_symmetry, 0.85)),
        # Scored against a fixed reference, not the configured tolerance, so
        # scores stay comparable between a strict run and a loose one.
        (10, 1.0 - min(1.0, abs(m["right_rim"] / m["left_rim"] - 1.0) / 0.15)),
        # Unmeasurable prior advance earns half marks, not full — it is missing
        # evidence, not evidence of a good setup.
        (
            10,
            _band_score(m["prior_gain"], *ideal["prior_gain"], p.min_prior_gain, 6.0)
            if m["prior_measurable"]
            else 0.5,
        ),
        (5, 1.0 - min(1.0, m["vol_dryup_cup"]) if math.isfinite(m["vol_dryup_cup"]) else 0.5),
    ]
    return float(sum(w * s for w, s in parts))


def _find_handle(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    R: int,
    n: int,
    p: Params,
) -> int:
    """Last bar of the handle. Returns R when no handle has formed yet.

    The handle runs from the bar after the right rim until price closes above
    the running high of the pattern (that close *is* the breakout, not handle)
    or until it has gone on too long to still be a handle.
    """
    end = R
    running_high = float(high[R])
    for i in range(R + 1, min(R + p.max_handle_len, n - 1) + 1):
        if close[i] > running_high:
            break
        end = i
        running_high = max(running_high, float(high[i]))
    return end


def _evaluate_handle(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    L: int,
    B: int,
    R: int,
    He: int,
    cup: dict,
    p: Params,
) -> dict | None:
    if He <= R:
        return None
    seg = slice(R + 1, He + 1)
    handle_high = float(np.max(high[R : He + 1]))
    handle_low = float(np.min(low[seg]))
    if handle_high <= 0:
        return None

    depth = (handle_high - handle_low) / handle_high
    if not (p.min_handle_depth <= depth <= p.max_handle_depth):
        return None

    # How far down into the cup the handle dug: 0 at the rim, 1 at the cup low.
    span = cup["left_rim"] - cup["bottom"]
    retrace = (cup["left_rim"] - handle_low) / span if span > 0 else 1.0
    if retrace > p.handle_max_retrace:
        return None

    closes = close[seg]
    length = He - R
    if length >= 2:
        x = np.arange(length, dtype=float)
        slope = float(np.polyfit(x, closes, 1)[0]) / max(1e-9, float(closes[0]))
    else:
        slope = float(closes[0] / close[R] - 1.0)

    vol_handle = _safe_mean(volume[seg])
    dryup = vol_handle / cup["vol_cup"] if cup["vol_cup"] and cup["vol_cup"] > 0 else float("nan")

    return {
        "handle_high": handle_high,
        "handle_low": handle_low,
        "handle_depth": depth,
        "handle_retrace": retrace,
        "handle_len": length,
        "drift": slope,
        "vol_dryup_handle": dryup,
    }


def _score_handle(h: dict, p: Params) -> float:
    ideal = p.ideal()
    parts = [
        (30, _band_score(h["handle_depth"], *ideal["handle_depth"], p.min_handle_depth, p.max_handle_depth)),
        (25, 1.0 - min(1.0, h["handle_retrace"] / p.handle_max_retrace)),
        (15, _band_score(h["handle_len"], *ideal["handle_len"], 1, p.max_handle_len)),
        (15, 1.0 if h["drift"] <= 0.01 else max(0.0, 1.0 - (h["drift"] - 0.01) / 0.10)),
        (15, 1.0 - min(1.0, h["vol_dryup_handle"]) if math.isfinite(h["vol_dryup_handle"]) else 0.5),
    ]
    return float(sum(w * s for w, s in parts))


def detect(
    df: pd.DataFrame,
    symbol: str = "",
    name: str = "",
    p: Params | None = None,
    all_patterns: bool = False,
) -> list[Pattern]:
    """Every current cup / cup-with-handle in one symbol's monthly history."""
    p = p or Params()
    n = len(df)
    if n < p.min_cup_len + 6:
        return []

    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)
    dates = df.index

    last = n - 1
    last_close = float(close[last])
    found: list[Pattern] = []

    # The pattern has to still be live, which bounds how far back the right rim can be.
    earliest_right = max(0, last - (p.max_age + p.max_handle_len))

    for L in range(1, n - p.min_cup_len):
        # left rim must be a swing high
        if high[L] < high[L - 1] or (L + 1 < n and high[L] < high[L + 1]):
            continue
        if L + p.min_cup_len > last:
            continue
        left_rim = float(high[L])
        rim_ceiling = left_rim * (1.0 + p.interior_tol)

        interior_max = -np.inf
        bottom = np.inf
        B = -1

        for R in range(L + 2, min(L + p.max_cup_len, last) + 1):
            j = R - 1  # the bar that just moved into the cup's interior
            if high[j] > interior_max:
                interior_max = float(high[j])
            if low[j] < bottom:
                bottom = float(low[j])
                B = j
            if interior_max > rim_ceiling:
                break  # price already took out the rim; no cup can end later than this
            if R - L < p.min_cup_len or R < earliest_right or B < 0:
                continue

            right_rim = float(high[R])
            if not (left_rim * (1 - p.rim_tol_low) <= right_rim <= left_rim * (1 + p.rim_tol_high)):
                continue

            cup = _evaluate_cup(high, low, close, volume, L, B, R, p)
            if cup is None:
                continue

            # Once price digs back below the middle of the cup after the right
            # rim, the base is broken — it is no longer a setup, it is a failure.
            if p.require_intact and R < n - 1:
                floor = cup["bottom"] + (1.0 - p.handle_max_retrace) * (cup["left_rim"] - cup["bottom"])
                if float(np.min(low[R + 1 :])) < floor:
                    continue

            He = _find_handle(high, low, close, R, n, p)
            handle = _evaluate_handle(high, low, close, volume, L, B, R, He, cup, p)

            if handle is not None:
                pattern_type, pivot, end_idx = "CUP_HANDLE", handle["handle_high"], He
            else:
                pattern_type, pivot, end_idx = "CUP", max(left_rim, right_rim), R
                He = R

            # Breakout = first monthly close above the pivot after the pattern ends.
            breakout_idx = -1
            for i in range(end_idx + 1, n):
                if close[i] > pivot:
                    breakout_idx = i
                    break

            # Age runs from whichever came last, the base completing or the
            # breakout out of it — an old rim that broke out last month is live.
            if last - max(end_idx, breakout_idx) > p.max_age:
                continue

            above = last_close / pivot - 1.0
            if breakout_idx >= 0 and above > p.extended_pct:
                status = "EXTENDED"
            elif breakout_idx >= 0 and last_close > pivot:
                status = "BREAKOUT"
            elif pattern_type == "CUP_HANDLE" and He >= last:
                status = "IN_HANDLE"
            elif above >= -p.near_pivot_pct:
                status = "NEAR_PIVOT"
            else:
                status = "FORMING"

            cup_score = _score_cup(cup, p)
            handle_score = _score_handle(handle, p) if handle else 0.0
            score = 0.65 * cup_score + 0.35 * handle_score if handle else cup_score

            vol_breakout = float("nan")
            if breakout_idx >= 0:
                base = _safe_mean(volume[max(0, breakout_idx - 12) : breakout_idx])
                if base and base > 0:
                    vol_breakout = float(volume[breakout_idx]) / base

            stop = handle["handle_low"] if handle else float(low[R])
            if stop >= pivot:
                stop = pivot * 0.92

            pat = Pattern(
                symbol=symbol,
                name=name,
                pattern=pattern_type,
                status=status,
                score=round(score, 1),
                cup_score=round(cup_score, 1),
                handle_score=round(handle_score, 1),
                left_idx=L,
                bottom_idx=B,
                right_idx=R,
                handle_end_idx=He if handle else -1,
                breakout_idx=breakout_idx,
                months_since_breakout=(last - breakout_idx) if breakout_idx >= 0 else -1,
                left_date=str(dates[L].date()),
                bottom_date=str(dates[B].date()),
                right_date=str(dates[R].date()),
                handle_end_date=str(dates[He].date()) if handle else "",
                breakout_date=str(dates[breakout_idx].date()) if breakout_idx >= 0 else "",
                left_rim=round(cup["left_rim"], 2),
                right_rim=round(cup["right_rim"], 2),
                cup_low=round(cup["bottom"], 2),
                depth_pct=round(100 * cup["depth"], 1),
                cup_len_m=R - L,
                left_len_m=cup["left_len"],
                right_len_m=cup["right_len"],
                symmetry=round(cup["symmetry"], 2),
                roundness_r2=round(cup["r2"], 3),
                bottom_bars=cup["bottom_bars"],
                prior_gain_pct=round(100 * cup["prior_gain"], 1),
                handle_len_m=handle["handle_len"] if handle else 0,
                handle_high=round(handle["handle_high"], 2) if handle else 0.0,
                handle_low=round(handle["handle_low"], 2) if handle else 0.0,
                handle_depth_pct=round(100 * handle["handle_depth"], 1) if handle else 0.0,
                handle_retrace=round(handle["handle_retrace"], 2) if handle else 0.0,
                handle_drift=round(100 * handle["drift"], 1) if handle else 0.0,
                pivot=round(pivot, 2),
                off_ath_pct=round(max(0.0, 100 * (1 - pivot / float(np.max(high)))), 1),
                last_close=round(last_close, 2),
                to_pivot_pct=round(100 * (pivot / last_close - 1.0), 1),
                above_pivot_pct=round(100 * above, 1),
                gain_since_breakout_pct=(
                    round(100 * (last_close / float(close[breakout_idx]) - 1.0), 1) if breakout_idx >= 0 else 0.0
                ),
                stop_suggest=round(stop, 2),
                risk_pct=round(100 * (1 - stop / pivot), 1),
                target_pct=round(100 * cup["depth"], 1),
                vol_dryup_cup=round(cup["vol_dryup_cup"], 2) if math.isfinite(cup["vol_dryup_cup"]) else float("nan"),
                vol_dryup_handle=(
                    round(handle["vol_dryup_handle"], 2)
                    if handle and math.isfinite(handle["vol_dryup_handle"])
                    else float("nan")
                ),
                vol_breakout=round(vol_breakout, 2) if math.isfinite(vol_breakout) else float("nan"),
            )
            if not cup["prior_measurable"]:
                pat.notes.append("listed just before the rim — advance into it could not be verified")
            _annotate(pat, p)
            found.append(pat)

    if not found:
        return []

    found.sort(key=lambda x: x.score, reverse=True)
    if all_patterns:
        return _dedupe(found)
    return [found[0]]


def is_actionable(pat: Pattern, p: Params, breakout_age: int = 12) -> bool:
    """Setups still in play: at the pivot, or through it recently enough to matter.

    Two ways in. Price sitting at or under the pivot but within the near band —
    the base is loaded and hasn't fired. Or price already through the pivot,
    provided the breakout is inside `breakout_age` months and hasn't run so far
    that the base is ancient history.

    What this drops: bases still deep in formation, breakouts that failed and
    fell back into the cup, and moves that are long gone.
    """
    if pat.to_pivot_pct > 100 * p.near_pivot_pct and pat.above_pivot_pct < 0:
        return False  # still well under the pivot, nothing imminent
    if pat.breakout_idx >= 0:
        return 0 <= pat.months_since_breakout <= breakout_age and pat.above_pivot_pct <= 100 * p.max_extension
    return True


def _dedupe(patterns: list[Pattern], overlap: float = 0.6) -> list[Pattern]:
    """Drop candidates whose cup spans mostly the same bars as a better one."""
    kept: list[Pattern] = []
    for pat in patterns:
        span = set(range(pat.left_idx, pat.right_idx + 1))
        if any(len(span & set(range(k.left_idx, k.right_idx + 1))) / len(span) > overlap for k in kept):
            continue
        kept.append(pat)
    return kept


def _annotate(pat: Pattern, p: Params) -> None:
    if pat.depth_pct > 40:
        pat.notes.append("deep cup (>40%) — more repair work than a shallow base")
    if pat.roundness_r2 >= 0.85 and pat.symmetry >= 0.6:
        pat.notes.append("well-rounded, symmetric")
    if pat.pattern == "CUP_HANDLE" and pat.handle_retrace <= 0.25:
        pat.notes.append("tight handle in the top quarter of the cup")
    if pat.pattern == "CUP_HANDLE" and pat.handle_drift > 2:
        pat.notes.append("handle drifting up, not down — weaker shakeout")
    if math.isfinite(pat.vol_breakout) and pat.vol_breakout >= 1.5:
        pat.notes.append(f"breakout on {pat.vol_breakout:.1f}x average volume")
    if math.isfinite(pat.vol_dryup_handle) and pat.vol_dryup_handle <= 0.7:
        pat.notes.append("volume dried up through the handle")
    if pat.status == "EXTENDED":
        pat.notes.append("already extended past the pivot")
    if pat.right_rim < pat.left_rim * 0.95:
        pat.notes.append("right rim below left — recovery incomplete")
    if pat.off_ath_pct <= 2:
        pat.notes.append("pivot is at all-time highs")
    elif pat.off_ath_pct >= 25:
        pat.notes.append(f"base sits {pat.off_ath_pct:.0f}% under an older high — overhead supply")


def explain(df: pd.DataFrame, symbol: str, p: Params | None = None, max_rims: int = 3, max_rows: int = 6) -> list[str]:
    """Why a given stock did or did not make the list.

    Walks the same candidate search the detector runs, but reports the verdict
    on every check instead of silently discarding failures.
    """
    p = p or Params()
    out = [f"{symbol}: {len(df)} monthly bars, {df.index[0].date()} → {df.index[-1].date()}"]
    n = len(df)
    if n < p.min_cup_len + 6:
        out.append(f"  too little history — needs {p.min_cup_len + 6} bars for a {p.min_cup_len}-month cup")
        return out

    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    close = df["Close"].to_numpy(float)
    last = n - 1

    # Biggest single-month drop, which usually means a split or demerger the
    # price series was never adjusted for.
    ratio = low[1:] / np.maximum(high[:-1], 1e-9)
    worst = int(np.argmin(ratio))
    if ratio[worst] < 0.55:
        out.append(
            f"  ⚠ {df.index[worst + 1].date()}: price fell {100 * (1 - ratio[worst]):.0f}% from the prior bar's high "
            f"— check for an unadjusted split/demerger, which invalidates any pattern spanning it"
        )

    rims = [i for i in range(1, n - p.min_cup_len) if high[i] >= high[i - 1] and (i + 1 >= n or high[i] >= high[i + 1])]
    rims.sort(key=lambda i: -high[i])
    if not rims:
        out.append("  no swing high qualifies as a left rim")
        return out

    for L in rims[:max_rims]:
        left_rim = float(high[L])
        out.append(f"  left rim {df.index[L].date()} @ {left_rim:,.1f}")
        interior_max, bottom, B, shown = -np.inf, np.inf, -1, 0
        for R in range(L + 2, min(L + p.max_cup_len, last) + 1):
            j = R - 1
            interior_max = max(interior_max, float(high[j]))
            if low[j] < bottom:
                bottom, B = float(low[j]), j
            if interior_max > left_rim * (1 + p.interior_tol):
                out.append(f"      cup ends {df.index[j].date()}: price took out the rim ({high[j]:,.1f})")
                break
            if R - L < p.min_cup_len or B < 0:
                continue
            rr = float(high[R])
            if not (left_rim * (1 - p.rim_tol_low) <= rr <= left_rim * (1 + p.rim_tol_high)):
                continue
            cup = _evaluate_cup(high, low, close, df["Volume"].to_numpy(float), L, B, R, p)
            if cup is None:
                reasons = _cup_failures(high, low, close, L, B, R, p)
                verdict = ", ".join(reasons) if reasons else "rejected"
            else:
                He = _find_handle(high, low, close, R, n, p)
                intact = True
                if p.require_intact and R < n - 1:
                    floor = cup["bottom"] + (1 - p.handle_max_retrace) * (cup["left_rim"] - cup["bottom"])
                    intact = float(np.min(low[R + 1 :])) >= floor
                verdict = "CUP OK" if intact else "base broken — price fell back below the cup midpoint"
            depth = (left_rim - bottom) / left_rim
            out.append(
                f"      right rim {df.index[R].date()} @ {rr:,.1f} · {R - L}m · {depth:.0%} deep · "
                f"{last - R}m ago → {verdict}"
            )
            shown += 1
            if shown >= max_rows:
                break
    return out


def _cup_failures(high, low, close, L: int, B: int, R: int, p: Params) -> list[str]:
    """Every quality check this candidate fails, in plain words."""
    left_rim, bottom = float(high[L]), float(low[B])
    span = left_rim - bottom
    fails: list[str] = []
    depth = (left_rim - bottom) / left_rim
    if not (p.min_depth <= depth <= p.max_depth):
        fails.append(f"depth {depth:.1%} outside {p.min_depth:.1%}–{p.max_depth:.1%}")
    ll, rl = B - L, R - B
    if ll < p.min_side_len or rl < p.min_side_len:
        fails.append("one side of the cup is too short")
    elif min(ll, rl) / max(ll, rl) < p.min_symmetry:
        fails.append(f"lopsided ({min(ll, rl) / max(ll, rl):.2f} < {p.min_symmetry})")
    bb = int(np.sum(low[L : R + 1] <= bottom + 0.25 * span))
    req = p.bottom_bars_required(R - L)
    if bb < req:
        fails.append(f"only {bb} bars near the low, needs {req} (V-shaped)")
    typical = (high[L : R + 1] + low[L : R + 1] + close[L : R + 1]) / 3.0
    a, vx, r2 = _fit_roundness(_smooth((typical - bottom) / span, 3 if (R - L) < 48 else 5))
    if a <= 0:
        fails.append("curve bends the wrong way")
    elif r2 < p.min_curvature_r2:
        fails.append(f"not round enough (R² {r2:.2f} < {p.min_curvature_r2})")
    elif not (p.vertex_band[0] <= vx <= p.vertex_band[1]):
        fails.append(f"low sits off-centre ({vx:.2f})")
    lb = max(0, L - p.prior_lookback)
    plow = float(np.min(low[lb : L + 1]))
    pg = left_rim / plow - 1.0 if plow > 0 else 0.0
    if pg < p.min_prior_gain and L >= p.min_prior_bars:
        fails.append(f"advance into the rim only {pg:+.0%}, needs {p.min_prior_gain:+.0%}")
    return fails


def screen(
    frames: dict[str, pd.DataFrame],
    meta: dict[str, dict] | None = None,
    p: Params | None = None,
    all_patterns: bool = False,
    log=None,
) -> list[Pattern]:
    """Run the detector across every symbol."""
    p = p or Params()
    meta = meta or {}
    results: list[Pattern] = []
    for i, (sym, df) in enumerate(frames.items(), 1):
        info = meta.get(sym, {})
        try:
            hits = detect(df, symbol=sym, name=info.get("name", ""), p=p, all_patterns=all_patterns)
        except Exception as exc:  # noqa: BLE001 - one bad series must not stop the screen
            if log:
                log(f"  {sym}: detector error {type(exc).__name__}: {exc}")
            continue
        for h in hits:
            h.mcap_cr = info.get("mcap_cr", float("nan"))
        results.extend(hits)
        if log and i % 250 == 0:
            log(f"  scanned {i}/{len(frames)} symbols, {len(results)} patterns so far")
    results.sort(key=lambda x: x.score, reverse=True)
    return results
