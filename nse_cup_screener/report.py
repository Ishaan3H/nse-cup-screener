"""Self-contained HTML report: a sortable results table plus an annotated
monthly candlestick chart for every match. No external assets, no CDN — the
file opens offline and can be emailed as-is.
"""

from __future__ import annotations

import html
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

from .patterns import Params, Pattern

STATUS_COLORS = {
    "BREAKOUT": "#22c55e",
    "NEAR_PIVOT": "#38bdf8",
    "IN_HANDLE": "#a78bfa",
    "FORMING": "#94a3b8",
    "EXTENDED": "#f59e0b",
}

STATUS_BLURB = {
    "BREAKOUT": "closed above the pivot",
    "NEAR_PIVOT": "within striking distance of the pivot",
    "IN_HANDLE": "handle still forming",
    "FORMING": "right side built, below the pivot",
    "EXTENDED": "ran well past the pivot already",
}


# --------------------------------------------------------------------------- #
# SVG chart
# --------------------------------------------------------------------------- #


def _chart_svg(df: pd.DataFrame, pat: Pattern, width: int = 880, height: int = 360) -> str:
    pad_l, pad_r, pad_t, pad_b = 8, 62, 14, 22
    vol_h = 58
    price_h = height - pad_t - pad_b - vol_h - 8

    n = len(df)
    start = max(0, pat.left_idx - 8)
    end = n - 1
    view = df.iloc[start : end + 1]
    m = len(view)
    if m < 2:
        return ""

    highs = view["High"].to_numpy(float)
    lows = view["Low"].to_numpy(float)
    opens = view["Open"].to_numpy(float)
    closes = view["Close"].to_numpy(float)
    vols = view["Volume"].to_numpy(float)

    hi = max(highs.max(), pat.pivot)
    lo = min(lows.min(), pat.stop_suggest)
    span = max(hi - lo, 1e-6)
    hi += span * 0.06
    lo -= span * 0.06
    span = hi - lo

    plot_w = width - pad_l - pad_r
    step = plot_w / m
    body_w = max(2.0, step * 0.6)

    def x(i: int) -> float:  # i is an absolute index into df
        return pad_l + (i - start + 0.5) * step

    def y(price: float) -> float:
        return pad_t + (hi - price) / span * price_h

    vol_top = pad_t + price_h + 8
    vmax = max(vols.max(), 1.0)

    def vy(v: float) -> float:
        return vol_top + vol_h - (v / vmax) * vol_h

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" preserveAspectRatio="xMidYMid meet" '
        f'xmlns="http://www.w3.org/2000/svg" class="chart">'
    ]

    # --- handle shading, drawn under the candles ---
    if pat.pattern == "CUP_HANDLE" and pat.handle_end_idx > pat.right_idx:
        hx0 = x(pat.right_idx) - body_w / 2
        hx1 = x(pat.handle_end_idx) + body_w / 2
        parts.append(
            f'<rect x="{hx0:.1f}" y="{y(pat.handle_high):.1f}" width="{max(hx1 - hx0, 2):.1f}" '
            f'height="{max(y(pat.handle_low) - y(pat.handle_high), 2):.1f}" fill="#a78bfa" opacity="0.13"/>'
        )

    # --- the cup arc: rim -> bottom -> rim, as two quadratics ---
    xl, xb, xr = x(pat.left_idx), x(pat.bottom_idx), x(pat.right_idx)
    yl, yb, yr = y(pat.left_rim), y(pat.cup_low), y(pat.right_rim)
    c1 = xl + 0.55 * (xb - xl)
    c2 = xr - 0.55 * (xr - xb)
    parts.append(
        f'<path d="M {xl:.1f} {yl:.1f} Q {c1:.1f} {yb:.1f} {xb:.1f} {yb:.1f} '
        f'Q {c2:.1f} {yb:.1f} {xr:.1f} {yr:.1f}" fill="none" stroke="#f8fafc" '
        f'stroke-width="1.6" stroke-dasharray="5 4" opacity="0.65"/>'
    )

    # --- gridlines + price axis ---
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        price = lo + span * frac
        gy = y(price)
        parts.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r:.1f}" y2="{gy:.1f}" '
            f'stroke="#1e293b" stroke-width="1"/>'
            f'<text x="{width - pad_r + 6:.1f}" y="{gy + 3.5:.1f}" class="ax">{price:,.0f}</text>'
        )

    # --- candles + volume ---
    for k in range(m):
        i = start + k
        cx = x(i)
        up = closes[k] >= opens[k]
        col = "#26a69a" if up else "#ef5350"
        in_cup = pat.left_idx <= i <= max(pat.right_idx, pat.handle_end_idx)
        op = "1" if in_cup else "0.45"
        y_hi, y_lo = y(highs[k]), y(lows[k])
        y_o, y_c = y(opens[k]), y(closes[k])
        top, bh = min(y_o, y_c), max(abs(y_c - y_o), 1.2)
        parts.append(
            f'<line x1="{cx:.1f}" y1="{y_hi:.1f}" x2="{cx:.1f}" y2="{y_lo:.1f}" stroke="{col}" '
            f'stroke-width="1" opacity="{op}"/>'
            f'<rect x="{cx - body_w / 2:.1f}" y="{top:.1f}" width="{body_w:.1f}" height="{bh:.1f}" '
            f'fill="{col}" opacity="{op}"/>'
            f'<rect x="{cx - body_w / 2:.1f}" y="{vy(vols[k]):.1f}" width="{body_w:.1f}" '
            f'height="{max(vol_top + vol_h - vy(vols[k]), 0.5):.1f}" fill="{col}" opacity="{0.6 if in_cup else 0.25}"/>'
        )

    # --- pivot and stop ---
    py = y(pat.pivot)
    parts.append(
        f'<line x1="{pad_l}" y1="{py:.1f}" x2="{width - pad_r:.1f}" y2="{py:.1f}" stroke="#facc15" '
        f'stroke-width="1.4" stroke-dasharray="7 4"/>'
        f'<text x="{width - pad_r + 6:.1f}" y="{py + 3.5:.1f}" class="ax lbl-pivot">{pat.pivot:,.0f}</text>'
    )
    sy = y(pat.stop_suggest)
    parts.append(
        f'<line x1="{xl:.1f}" y1="{sy:.1f}" x2="{width - pad_r:.1f}" y2="{sy:.1f}" stroke="#f87171" '
        f'stroke-width="1" stroke-dasharray="3 5" opacity="0.7"/>'
    )

    # --- markers ---
    for idx, price, fill in (
        (pat.left_idx, pat.left_rim, "#f8fafc"),
        (pat.right_idx, pat.right_rim, "#f8fafc"),
        (pat.bottom_idx, pat.cup_low, "#38bdf8"),
    ):
        parts.append(f'<circle cx="{x(idx):.1f}" cy="{y(price):.1f}" r="3.2" fill="{fill}"/>')
    if pat.breakout_idx >= 0:
        bx, by = x(pat.breakout_idx), y(pat.pivot) - 9
        parts.append(f'<path d="M {bx:.1f} {by - 8:.1f} l 5 9 l -10 0 z" fill="#22c55e"/>')

    # --- year ticks ---
    last_year = None
    for k in range(m):
        yr = view.index[k].year
        if yr != last_year and k > 0:
            parts.append(
                f'<text x="{x(start + k):.1f}" y="{height - 6}" class="ax mid">{yr}</text>'
            )
            last_year = yr
        elif last_year is None:
            last_year = yr

    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

CSS = """
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#0b1220;color:#e2e8f0;
  font:14px/1.5 ui-sans-serif,-apple-system,"SF Pro Text",Segoe UI,Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:23px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:16px;margin:38px 0 12px;color:#cbd5e1;font-weight:600}
.sub{color:#94a3b8;font-size:13px;margin-bottom:22px}
.cards{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0 26px}
.card{background:#111c33;border:1px solid #1e293b;border-radius:10px;padding:11px 15px;min-width:118px}
.card .n{font-size:21px;font-weight:650}
.card .k{color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:.06em;margin-top:2px}
.tablewrap{overflow-x:auto;border:1px solid #1e293b;border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13px;min-width:940px}
th,td{padding:7px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid #172033}
th{background:#111c33;position:sticky;top:0;cursor:pointer;user-select:none;
  font-weight:600;color:#cbd5e1;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
th:hover{color:#fff}
th.l,td.l{text-align:left}
tbody tr:hover{background:#111c33}
td a{color:#7dd3fc;text-decoration:none}
td a:hover{text-decoration:underline}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;font-weight:600}
.score{font-weight:650}
.match{background:#0e1729;border:1px solid #1e293b;border-radius:12px;padding:16px 18px;margin:14px 0}
.match header{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px;margin-bottom:2px}
.match h3{margin:0;font-size:17px}
.match .co{color:#94a3b8;font-size:13px;flex:1;min-width:160px}
.chart{display:block;margin:6px 0 4px;background:#0b1220;border-radius:8px}
.ax{fill:#64748b;font-size:10px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.ax.mid{text-anchor:middle}
.lbl-pivot{fill:#facc15}
.stats{display:flex;flex-wrap:wrap;gap:0 26px;font-size:12.5px;color:#cbd5e1;margin-top:8px}
.stats div{padding:2px 0}
.stats b{color:#f1f5f9;font-weight:600}
.notes{color:#94a3b8;font-size:12.5px;margin-top:7px;font-style:italic}
.legend{color:#64748b;font-size:12px;margin-top:8px}
.legend span{margin-right:16px}
footer{color:#64748b;font-size:12px;margin-top:44px;border-top:1px solid #1e293b;padding-top:16px}
code{background:#111c33;padding:1px 5px;border-radius:4px;font-size:12px}
"""

SORT_JS = """
document.querySelectorAll('th[data-i]').forEach(function(th){
  th.addEventListener('click', function(){
    var tb = th.closest('table').tBodies[0];
    var i = +th.dataset.i, num = th.dataset.t === 'n';
    var dir = th.dataset.d === 'asc' ? -1 : 1;
    th.dataset.d = dir === 1 ? 'asc' : 'desc';
    var rows = Array.from(tb.rows);
    rows.sort(function(a,b){
      var x = a.cells[i].dataset.v ?? a.cells[i].innerText;
      var y = b.cells[i].dataset.v ?? b.cells[i].innerText;
      if(num){ x = parseFloat(x); y = parseFloat(y);
        if(isNaN(x)) x = -Infinity; if(isNaN(y)) y = -Infinity; return (x-y)*dir; }
      return x.localeCompare(y)*dir;
    });
    rows.forEach(function(r){ tb.appendChild(r); });
  });
});
"""


def _fmt(v: float, dp: int = 1, suffix: str = "") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:,.{dp}f}{suffix}"


def _pill(status: str) -> str:
    color = STATUS_COLORS.get(status, "#94a3b8")
    return f'<span class="pill" style="background:{color}22;color:{color}">{status.replace("_", " ").title()}</span>'


def _table(patterns: list[Pattern]) -> str:
    cols = [
        ("Symbol", "l", "s"), ("Company", "l", "s"), ("Pattern", "l", "s"), ("Status", "l", "s"),
        ("Score", "", "n"), ("Mcap ₹Cr", "", "n"), ("Close", "", "n"), ("Pivot", "", "n"),
        ("To pivot %", "", "n"), ("B/O age (m)", "", "n"), ("Since b/o %", "", "n"),
        ("Off ATH %", "", "n"), ("Depth %", "", "n"), ("Cup (m)", "", "n"),
        ("Handle (m)", "", "n"), ("Handle %", "", "n"), ("Stop", "", "n"), ("Risk %", "", "n"),
        ("Cup start", "l", "s"), ("Cup low", "l", "s"),
    ]
    head = "".join(
        f'<th class="{c}" data-i="{i}" data-t="{t}">{html.escape(label)}</th>'
        for i, (label, c, t) in enumerate(cols)
    )
    rows = []
    for p in patterns:
        tv = f"https://www.tradingview.com/chart/?symbol=NSE%3A{html.escape(p.symbol)}"
        rows.append(
            "<tr>"
            f'<td class="l"><a href="{tv}" target="_blank" rel="noopener">{html.escape(p.symbol)}</a></td>'
            f'<td class="l">{html.escape(p.name[:34])}</td>'
            f'<td class="l">{"Cup + Handle" if p.pattern == "CUP_HANDLE" else "Cup"}</td>'
            f'<td class="l" data-v="{p.status}">{_pill(p.status)}</td>'
            f'<td class="score" data-v="{p.score}">{p.score:.0f}</td>'
            f'<td data-v="{p.mcap_cr}">{_fmt(p.mcap_cr, 0)}</td>'
            f'<td data-v="{p.last_close}">{_fmt(p.last_close, 2)}</td>'
            f'<td data-v="{p.pivot}">{_fmt(p.pivot, 2)}</td>'
            f'<td data-v="{p.to_pivot_pct}">{p.to_pivot_pct:+.1f}</td>'
            f'<td data-v="{p.months_since_breakout}">{p.months_since_breakout if p.breakout_idx >= 0 else "—"}</td>'
            f'<td data-v="{p.gain_since_breakout_pct}">'
            f'{f"{p.gain_since_breakout_pct:+.0f}" if p.breakout_idx >= 0 else "—"}</td>'
            f'<td data-v="{p.off_ath_pct}">{p.off_ath_pct:.0f}</td>'
            f'<td data-v="{p.depth_pct}">{p.depth_pct:.0f}</td>'
            f'<td data-v="{p.cup_len_m}">{p.cup_len_m}</td>'
            f'<td data-v="{p.handle_len_m}">{p.handle_len_m or "—"}</td>'
            f'<td data-v="{p.handle_depth_pct}">{_fmt(p.handle_depth_pct, 0) if p.handle_len_m else "—"}</td>'
            f'<td data-v="{p.stop_suggest}">{_fmt(p.stop_suggest, 2)}</td>'
            f'<td data-v="{p.risk_pct}">{_fmt(p.risk_pct, 1)}</td>'
            f'<td class="l">{p.left_date}</td>'
            f'<td class="l">{p.bottom_date}</td>'
            "</tr>"
        )
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _match_block(pat: Pattern, df: pd.DataFrame) -> str:
    tv = f"https://www.tradingview.com/chart/?symbol=NSE%3A{html.escape(pat.symbol)}"
    kind = "Cup with handle" if pat.pattern == "CUP_HANDLE" else "Cup"
    stats = [
        ("Pivot", f"₹{pat.pivot:,.2f}"),
        ("Last", f"₹{pat.last_close:,.2f} ({pat.to_pivot_pct:+.1f}% to pivot)"),
        ("Cup", f"{pat.cup_len_m} months ({pat.cup_len_m / 12:.1f} yrs), {pat.depth_pct:.0f}% deep"),
        ("Started", pat.left_date),
        ("Rims", f"₹{pat.left_rim:,.0f} → ₹{pat.right_rim:,.0f}"),
        ("Low", f"₹{pat.cup_low:,.0f} ({pat.bottom_date})"),
        ("Run-up into cup", f"+{pat.prior_gain_pct:.0f}%"),
        ("Pivot vs record high", "at highs" if pat.off_ath_pct <= 2 else f"{pat.off_ath_pct:.0f}% below"),
        ("Roundness R²", f"{pat.roundness_r2:.2f}"),
        ("Symmetry", f"{pat.symmetry:.2f}"),
    ]
    if pat.pattern == "CUP_HANDLE":
        stats += [
            ("Handle", f"{pat.handle_len_m} months, {pat.handle_depth_pct:.0f}% deep"),
            ("Handle sits", f"{pat.handle_retrace * 100:.0f}% down into the cup"),
        ]
    stats += [
        (
            "Breakout",
            f"{pat.breakout_date}, {pat.months_since_breakout} month(s) ago "
            f"({pat.gain_since_breakout_pct:+.0f}% since, {pat.above_pivot_pct:+.0f}% vs pivot)"
            if pat.breakout_idx >= 0
            else "not yet — pivot still overhead",
        ),
        ("Stop", f"₹{pat.stop_suggest:,.2f} ({pat.risk_pct:.1f}% risk)"),
        ("Measured move", f"+{pat.target_pct:.0f}% from pivot"),
        ("Market cap", f"₹{pat.mcap_cr:,.0f} Cr" if math.isfinite(pat.mcap_cr) else "—"),
    ]
    stat_html = "".join(f"<div>{html.escape(k)} <b>{html.escape(v)}</b></div>" for k, v in stats)
    notes = f'<div class="notes">{html.escape("· ".join(pat.notes))}</div>' if pat.notes else ""
    return (
        '<div class="match"><header>'
        f'<h3><a href="{tv}" target="_blank" rel="noopener">{html.escape(pat.symbol)}</a></h3>'
        f'<span class="co">{html.escape(pat.name)}</span>'
        f'<span>{kind} · score <b class="score">{pat.score:.0f}</b></span>{_pill(pat.status)}'
        "</header>"
        f'<div class="sub" style="margin:2px 0 6px">{STATUS_BLURB.get(pat.status, "")}</div>'
        f"{_chart_svg(df, pat)}"
        '<div class="legend"><span>⟋ dashed white = cup arc</span><span style="color:#facc15">— pivot</span>'
        '<span style="color:#f87171">— stop</span><span style="color:#a78bfa">▩ handle</span>'
        "<span>bars below = monthly volume</span></div>"
        f'<div class="stats">{stat_html}</div>{notes}</div>'
    )


def write_report(
    path: Path,
    patterns: list[Pattern],
    frames: dict[str, pd.DataFrame],
    params: Params,
    universe_size: int,
    scanned: int,
    min_market_cap_cr: float,
    max_charts: int = 120,
    stage_note: str = "",
) -> None:
    by_status: dict[str, int] = {}
    for p in patterns:
        by_status[p.status] = by_status.get(p.status, 0) + 1
    n_handle = sum(1 for p in patterns if p.pattern == "CUP_HANDLE")

    cards = [
        ("Matches", len(patterns)),
        ("Cup + handle", n_handle),
        ("Cup only", len(patterns) - n_handle),
        ("Scanned", scanned),
    ] + [(k.replace("_", " ").title(), v) for k, v in sorted(by_status.items(), key=lambda kv: -kv[1])]
    card_html = "".join(f'<div class="card"><div class="n">{v:,}</div><div class="k">{k}</div></div>' for k, v in cards)

    charts = "".join(_match_block(p, frames[p.symbol]) for p in patterns[:max_charts] if p.symbol in frames)
    more = (
        f'<p class="sub">Charts shown for the top {max_charts} by score; the full list is in the table above '
        f"and in the CSV.</p>"
        if len(patterns) > max_charts
        else ""
    )
    now = datetime.now().strftime("%d %b %Y, %H:%M")

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NSE monthly cup &amp; handle screen — {now}</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>NSE cup &amp; cup-with-handle screen — monthly charts</h1>
<div class="sub">{now} · {universe_size:,} NSE stocks above ₹{min_market_cap_cr:,.0f} Cr market cap ·
{scanned:,} with enough monthly history · cup {params.min_cup_len}–{params.max_cup_len} months
({params.min_cup_len / 12:.0f}–{params.max_cup_len / 12:.0f} years), depth
{params.min_depth:.0%}–{params.max_depth:.0%}, handle ≤ {params.max_handle_len} months in the
upper half of the cup, prior advance ≥ {params.min_prior_gain:.0%}{stage_note}</div>
<div class="cards">{card_html}</div>
<h2>All matches</h2>
{_table(patterns) if patterns else '<p class="sub">Nothing passed the filters this run.</p>'}
<h2>Charts</h2>{more}
{charts}
<footer>Monthly bars from Yahoo Finance (split-adjusted), universe from NSE&rsquo;s official
EQUITY_L list. Pivot = handle high, or the cup rim when there is no handle. Stop = handle low
(cup right-side low without a handle). Screening output, not investment advice — every setup here
still needs your own eyes on the chart.</footer>
</div><script>{SORT_JS}</script></body></html>"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
