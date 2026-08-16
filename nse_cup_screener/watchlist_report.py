"""HTML watchlist for cups still under construction.

The chart makes the distinction the data supports: the part of the cup that has
actually happened is drawn solid, and the part that has not is drawn as a faint
projected arc up to the rim. Nothing about the dotted section is a measurement.
"""

from __future__ import annotations

import html
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

from .forming import Forming, FormingParams

STAGE_COLORS = {
    "APPROACHING_RIM": "#22c55e",
    "MID_RIGHT": "#38bdf8",
    "EARLY_RIGHT": "#a78bfa",
    "BOTTOMING": "#94a3b8",
}

STAGE_BLURB = {
    "APPROACHING_RIM": "right side nearly at the old rim",
    "MID_RIGHT": "over halfway back up the right side",
    "EARLY_RIGHT": "recovery under way, still low in the cup",
    "BOTTOMING": "just turning up off the low",
}


def _chart_svg(df: pd.DataFrame, c: Forming, width: int = 880, height: int = 340) -> str:
    pad_l, pad_r, pad_t, pad_b = 8, 62, 14, 22
    vol_h = 50
    price_h = height - pad_t - pad_b - vol_h - 8

    n = len(df)
    start = max(0, c.left_idx - 6)
    view = df.iloc[start:]
    m = len(view)
    if m < 2:
        return ""

    highs = view["High"].to_numpy(float)
    lows = view["Low"].to_numpy(float)
    opens = view["Open"].to_numpy(float)
    closes = view["Close"].to_numpy(float)
    vols = view["Volume"].to_numpy(float)

    # Leave room on the right for the projected remainder of the cup.
    future = max(1, min(c.eta_m, 36))
    total_slots = m + future

    hi = max(float(highs.max()), c.left_rim)
    lo = min(float(lows.min()), c.cup_low)
    span = max(hi - lo, 1e-6)
    hi += span * 0.06
    lo -= span * 0.06
    span = hi - lo

    plot_w = width - pad_l - pad_r
    step = plot_w / total_slots
    body_w = max(1.6, step * 0.6)

    def x(i: float) -> float:
        return pad_l + (i - start + 0.5) * step

    def y(price: float) -> float:
        return pad_t + (hi - price) / span * price_h

    vol_top = pad_t + price_h + 8
    vmax = max(float(vols.max()), 1.0)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" preserveAspectRatio="xMidYMid meet" '
        f'xmlns="http://www.w3.org/2000/svg" class="chart">'
    ]

    # Shade the ground still to be covered.
    parts.append(
        f'<rect x="{pad_l}" y="{y(c.left_rim):.1f}" width="{plot_w:.1f}" '
        f'height="{max(y(float(closes[-1])) - y(c.left_rim), 1):.1f}" fill="#38bdf8" opacity="0.06"/>'
    )

    for frac in (0.0, 0.5, 1.0):
        price = lo + span * frac
        gy = y(price)
        parts.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r:.1f}" y2="{gy:.1f}" stroke="#1e293b"/>'
            f'<text x="{width - pad_r + 6:.1f}" y="{gy + 3.5:.1f}" class="ax">{price:,.0f}</text>'
        )

    # --- the built half of the cup, solid ---
    xl, xb, xn = x(c.left_idx), x(c.bottom_idx), x(n - 1)
    yl, yb, yn = y(c.left_rim), y(c.cup_low), y(float(closes[-1]))
    parts.append(
        f'<path d="M {xl:.1f} {yl:.1f} Q {xl + 0.55 * (xb - xl):.1f} {yb:.1f} {xb:.1f} {yb:.1f} '
        f'Q {xb + 0.5 * (xn - xb):.1f} {yb:.1f} {xn:.1f} {yn:.1f}" fill="none" stroke="#f8fafc" '
        f'stroke-width="1.6" opacity="0.75"/>'
    )
    # --- the part that has not happened, dotted ---
    xf = x(n - 1 + future)
    parts.append(
        f'<path d="M {xn:.1f} {yn:.1f} Q {xn + 0.55 * (xf - xn):.1f} {yl:.1f} {xf:.1f} {yl:.1f}" '
        f'fill="none" stroke="#38bdf8" stroke-width="1.6" stroke-dasharray="2 5" opacity="0.8"/>'
        f'<circle cx="{xf:.1f}" cy="{yl:.1f}" r="3.2" fill="none" stroke="#38bdf8" stroke-width="1.5"/>'
    )

    for k in range(m):
        i = start + k
        cx = x(i)
        col = "#26a69a" if closes[k] >= opens[k] else "#ef5350"
        op = "1" if i >= c.left_idx else "0.4"
        y_o, y_c = y(opens[k]), y(closes[k])
        parts.append(
            f'<line x1="{cx:.1f}" y1="{y(highs[k]):.1f}" x2="{cx:.1f}" y2="{y(lows[k]):.1f}" '
            f'stroke="{col}" opacity="{op}"/>'
            f'<rect x="{cx - body_w / 2:.1f}" y="{min(y_o, y_c):.1f}" width="{body_w:.1f}" '
            f'height="{max(abs(y_c - y_o), 1.2):.1f}" fill="{col}" opacity="{op}"/>'
            f'<rect x="{cx - body_w / 2:.1f}" y="{vol_top + vol_h - (vols[k] / vmax) * vol_h:.1f}" '
            f'width="{body_w:.1f}" height="{max((vols[k] / vmax) * vol_h, 0.5):.1f}" fill="{col}" opacity="0.45"/>'
        )

    # rim (the level that completes the cup) and the current price
    parts.append(
        f'<line x1="{pad_l}" y1="{yl:.1f}" x2="{width - pad_r:.1f}" y2="{yl:.1f}" stroke="#facc15" '
        f'stroke-width="1.4" stroke-dasharray="7 4"/>'
        f'<text x="{width - pad_r + 6:.1f}" y="{yl + 3.5:.1f}" class="ax lbl-rim">{c.left_rim:,.0f}</text>'
        f'<line x1="{pad_l}" y1="{yn:.1f}" x2="{xn:.1f}" y2="{yn:.1f}" stroke="#94a3b8" '
        f'stroke-width="1" stroke-dasharray="2 4" opacity="0.6"/>'
        f'<circle cx="{xl:.1f}" cy="{yl:.1f}" r="3.2" fill="#f8fafc"/>'
        f'<circle cx="{xb:.1f}" cy="{yb:.1f}" r="3.2" fill="#38bdf8"/>'
    )

    last_year = None
    for k in range(m):
        yr = view.index[k].year
        if last_year is None:
            last_year = yr
        elif yr != last_year:
            parts.append(f'<text x="{x(start + k):.1f}" y="{height - 6}" class="ax mid">{yr}</text>')
            last_year = yr

    parts.append("</svg>")
    return "".join(parts)


CSS = """
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#0b1220;color:#e2e8f0;
  font:14px/1.5 ui-sans-serif,-apple-system,"SF Pro Text",Segoe UI,Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:23px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:16px;margin:34px 0 12px;color:#cbd5e1}
.sub{color:#94a3b8;font-size:13px;margin-bottom:18px}
.note{background:#111c33;border-left:3px solid #38bdf8;border-radius:0 8px 8px 0;
  padding:11px 15px;margin:16px 0 24px;color:#cbd5e1;font-size:13px}
.cards{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0 24px}
.card{background:#111c33;border:1px solid #1e293b;border-radius:10px;padding:11px 15px;min-width:112px}
.card .n{font-size:21px;font-weight:650}
.card .k{color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:.06em;margin-top:2px}
.tablewrap{overflow-x:auto;border:1px solid #1e293b;border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13px;min-width:900px}
th,td{padding:7px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid #172033}
th{background:#111c33;position:sticky;top:0;cursor:pointer;user-select:none;font-weight:600;
  color:#cbd5e1;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
th.l,td.l{text-align:left}
tbody tr:hover{background:#111c33}
td a{color:#7dd3fc;text-decoration:none}
td a:hover{text-decoration:underline}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;font-weight:600}
.bar{position:relative;height:6px;background:#1e293b;border-radius:3px;min-width:60px;overflow:hidden}
.bar i{position:absolute;left:0;top:0;bottom:0;background:#38bdf8;border-radius:3px}
.match{background:#0e1729;border:1px solid #1e293b;border-radius:12px;padding:16px 18px;margin:14px 0;
  content-visibility:auto;contain-intrinsic-size:0 520px}
.match header{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px}
.match h3{margin:0;font-size:17px}
.match .co{color:#94a3b8;font-size:13px;flex:1;min-width:150px}
.chart{display:block;margin:6px 0 4px;background:#0b1220;border-radius:8px}
.ax{fill:#64748b;font-size:10px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.ax.mid{text-anchor:middle}
.lbl-rim{fill:#facc15}
.stats{display:flex;flex-wrap:wrap;gap:0 24px;font-size:12.5px;color:#cbd5e1;margin-top:8px}
.stats b{color:#f1f5f9}
.notes{color:#94a3b8;font-size:12.5px;margin-top:7px;font-style:italic}
.legend{color:#64748b;font-size:12px;margin-top:8px}
.legend span{margin-right:16px}
footer{color:#64748b;font-size:12px;margin-top:40px;border-top:1px solid #1e293b;padding-top:16px}
"""

SORT_JS = """
document.querySelectorAll('th[data-i]').forEach(function(th){
  th.addEventListener('click', function(){
    var tb=th.closest('table').tBodies[0], i=+th.dataset.i, num=th.dataset.t==='n';
    var dir=th.dataset.d==='asc'?-1:1; th.dataset.d=dir===1?'asc':'desc';
    Array.from(tb.rows).sort(function(a,b){
      var x=a.cells[i].dataset.v??a.cells[i].innerText, y=b.cells[i].dataset.v??b.cells[i].innerText;
      if(num){x=parseFloat(x);y=parseFloat(y);if(isNaN(x))x=-Infinity;if(isNaN(y))y=-Infinity;return (x-y)*dir;}
      return x.localeCompare(y)*dir;
    }).forEach(function(r){tb.appendChild(r);});
  });
});
"""


def _pill(stage: str) -> str:
    col = STAGE_COLORS.get(stage, "#94a3b8")
    return f'<span class="pill" style="background:{col}22;color:{col}">{stage.replace("_", " ").title()}</span>'


def _fmt(v: float, dp: int = 1) -> str:
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:,.{dp}f}"


def _table(rows: list[Forming]) -> str:
    cols = [
        ("Symbol", "l", "s"), ("Company", "l", "s"), ("Stage", "l", "s"), ("Score", "", "n"),
        ("Mcap ₹Cr", "", "n"), ("Close", "", "n"), ("Rim target", "", "n"), ("To rim %", "", "n"),
        ("Right side done", "l", "n"), ("Cup so far (m)", "", "n"), ("Depth %", "", "n"),
        ("ETA sym (m)", "", "n"), ("ETA rate (m)", "", "n"), ("Projected cup (m)", "", "n"),
        ("Earliest", "l", "s"), ("Rim date", "l", "s"), ("Low date", "l", "s"),
    ]
    head = "".join(f'<th class="{c}" data-i="{i}" data-t="{t}">{html.escape(l)}</th>' for i, (l, c, t) in enumerate(cols))
    body = []
    for r in rows:
        tv = f"https://www.tradingview.com/chart/?symbol=NSE%3A{html.escape(r.symbol)}"
        body.append(
            "<tr>"
            f'<td class="l"><a href="{tv}" target="_blank" rel="noopener">{html.escape(r.symbol)}</a></td>'
            f'<td class="l">{html.escape(r.name[:30])}</td>'
            f'<td class="l" data-v="{r.stage}">{_pill(r.stage)}</td>'
            f'<td data-v="{r.score}"><b>{r.score:.0f}</b></td>'
            f'<td data-v="{r.mcap_cr}">{_fmt(r.mcap_cr, 0)}</td>'
            f'<td data-v="{r.last_close}">{_fmt(r.last_close, 2)}</td>'
            f'<td data-v="{r.left_rim}">{_fmt(r.left_rim, 2)}</td>'
            f'<td data-v="{r.to_rim_pct}">+{r.to_rim_pct:.0f}</td>'
            f'<td class="l" data-v="{r.recovery_pct}"><div class="bar" title="{r.recovery_pct:.0f}%">'
            f'<i style="width:{min(100, max(2, r.recovery_pct)):.0f}%"></i></div></td>'
            f'<td data-v="{r.months_so_far}">{r.months_so_far}</td>'
            f'<td data-v="{r.depth_pct}">{r.depth_pct:.0f}</td>'
            f'<td data-v="{r.eta_symmetry_m}">{r.eta_symmetry_m or "—"}</td>'
            f'<td data-v="{r.eta_rate_m}">{r.eta_rate_m or "—"}</td>'
            f'<td data-v="{r.projected_len_m}">{r.projected_len_m}</td>'
            f'<td class="l">{r.projected_complete}</td>'
            f'<td class="l">{r.left_date}</td>'
            f'<td class="l">{r.bottom_date}</td>'
            "</tr>"
        )
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _block(r: Forming, df: pd.DataFrame) -> str:
    tv = f"https://www.tradingview.com/chart/?symbol=NSE%3A{html.escape(r.symbol)}"
    stats = [
        ("Rim to clear", f"₹{r.left_rim:,.2f} (+{r.to_rim_pct:.0f}% from ₹{r.last_close:,.2f})"),
        ("Right side", f"{r.recovery_pct:.0f}% of the depth won back"),
        ("Cup so far", f"{r.months_so_far} months ({r.months_so_far / 12:.1f} yrs), {r.depth_pct:.0f}% deep"),
        ("Low", f"₹{r.cup_low:,.2f} ({r.bottom_date}), {r.months_since_low} months ago"),
        ("ETA", f"{r.eta_symmetry_m}m by symmetry / {r.eta_rate_m or '—'}m at recent pace → ~{r.projected_complete}"),
        ("If it completes", f"a {r.projected_len_m}-month ({r.projected_len_m / 12:.1f} yr) cup"),
        ("Advance", f"{r.advance_rate_pct:+.1f}%/month recently"),
        ("Shape R²", f"{r.roundness_r2:.2f}"),
        ("Higher lows", "yes" if r.higher_lows else "not yet"),
        ("Market cap", f"₹{r.mcap_cr:,.0f} Cr" if math.isfinite(r.mcap_cr) else "—"),
    ]
    stat_html = "".join(f"<div>{html.escape(k)} <b>{html.escape(v)}</b></div>" for k, v in stats)
    notes = f'<div class="notes">{html.escape(" · ".join(r.notes))}</div>' if r.notes else ""
    return (
        '<div class="match"><header>'
        f'<h3><a href="{tv}" target="_blank" rel="noopener">{html.escape(r.symbol)}</a></h3>'
        f'<span class="co">{html.escape(r.name)}</span>'
        f'<span>score <b>{r.score:.0f}</b></span>{_pill(r.stage)}</header>'
        f'<div class="sub" style="margin:2px 0 6px">{STAGE_BLURB.get(r.stage, "")}</div>'
        f"{_chart_svg(df, r)}"
        '<div class="legend"><span>solid white = the cup so far</span>'
        '<span style="color:#38bdf8">dotted = projected remainder, not a measurement</span>'
        '<span style="color:#facc15">— rim to clear</span></div>'
        f'<div class="stats">{stat_html}</div>{notes}</div>'
    )


def write_watchlist(
    path: Path,
    rows: list[Forming],
    frames: dict[str, pd.DataFrame],
    fp: FormingParams,
    scanned: int,
    min_market_cap_cr: float,
    min_cup_len: int,
    max_charts: int = 60,
    charts: bool = True,
) -> None:
    by_stage: dict[str, int] = {}
    for r in rows:
        by_stage[r.stage] = by_stage.get(r.stage, 0) + 1
    cards = [("Watching", len(rows)), ("Scanned", scanned)] + [
        (k.replace("_", " ").title(), v) for k, v in sorted(by_stage.items(), key=lambda kv: -kv[1])
    ]
    card_html = "".join(f'<div class="card"><div class="n">{v:,}</div><div class="k">{k}</div></div>' for k, v in cards)

    charts_html = (
        "".join(_block(r, frames[r.symbol]) for r in rows[:max_charts] if r.symbol in frames) if charts else ""
    )
    now = datetime.now().strftime("%d %b %Y, %H:%M")

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NSE cup watchlist — {now}</title><style>{CSS}</style></head><body><div class="wrap">
<h1>NSE cup watchlist — bases still being built</h1>
<div class="sub">{now} · {scanned:,} stocks above ₹{min_market_cap_cr:,.0f} Cr market cap ·
rim at least {fp.min_months_so_far} months old · {fp.min_recovery:.0%}–{fp.max_recovery:.0%} of the
depth recovered · would finish as a {min_cup_len}-month-plus cup · completion within {fp.max_eta} months</div>

<div class="note"><b>What this is.</b> Every stock here has already put in a rim, a bottom, and the
start of a recovery — the left half of a cup exists and the right half is under way. It is not a
prediction that a cup <i>will</i> appear; it is a list of cups already half-built, ranked by how far
along and how sound they look. A candidate fails the moment price undercuts the low or stalls.
<b>The two ETAs are extrapolation</b>: one assumes the right side takes as long as the left, the
other projects the recent rate of advance. When they disagree by more than a year, the stock is not
tracking a textbook shape.</div>

<div class="cards">{card_html}</div>
<h2>Watchlist</h2>
{_table(rows) if rows else '<p class="sub">Nothing qualifies right now.</p>'}
<h2>Charts</h2>
{charts_html}
<footer>Monthly bars from Yahoo Finance (split-adjusted), universe from NSE&rsquo;s EQUITY_L list.
Rim target = the left-rim high that would complete the cup; a handle, if one forms, would set the
actual buy pivot slightly differently. Screening output, not investment advice.</footer>
</div><script>{SORT_JS}</script></body></html>"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
