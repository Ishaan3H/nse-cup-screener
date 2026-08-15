# NSE monthly cup & cup-with-handle screener

Scans every NSE-listed equity above a market-cap floor for **cup** and
**cup-with-handle** bases on **monthly** charts, scores each one 0–100, and
writes a CSV plus a self-contained HTML report with an annotated candlestick
chart for every match.

Defaults are tuned for **multi-year bases that are live right now** — cups
between 3 and 25 years long, sitting at the pivot or already through it within
the last 12 months.

## Install

Needs **Python 3.9 or newer** and git. Pick your platform.

<details open>
<summary><b>macOS</b></summary>

macOS ships with Python 3, which is enough. If `python3 --version` says 3.8 or
older, install a newer one with [Homebrew](https://brew.sh): `brew install python`.

```bash
git clone https://github.com/Ishaan3H/nse-cup-screener.git
cd nse-cup-screener
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
chmod +x screen
./screen
```
</details>

<details>
<summary><b>Linux</b></summary>

Install Python and the venv module first if they are missing:

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git   # Debian/Ubuntu
# sudo dnf install -y python3 python3-pip git                                 # Fedora/RHEL
# sudo pacman -S python git                                                   # Arch
```

Then:

```bash
git clone https://github.com/Ishaan3H/nse-cup-screener.git
cd nse-cup-screener
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
chmod +x screen
./screen
```
</details>

<details>
<summary><b>Windows</b></summary>

Install Python from [python.org/downloads](https://www.python.org/downloads/)
and **tick "Add python.exe to PATH"** on the first screen of the installer.
Then open PowerShell or Command Prompt:

```bat
git clone https://github.com/Ishaan3H/nse-cup-screener.git
cd nse-cup-screener
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
screen.bat
```

Use `screen.bat` wherever this README says `./screen` — for example
`screen.bat --explain BHEL`. If `py` is not recognised, the PATH box was
missed during install; re-run the installer and choose *Modify*.

No git? Download the ZIP from the repo's green **Code** button, extract it, and
`cd` into the folder instead of cloning.
</details>

## Run it

```bash
./screen
```

That is the whole thing. First run takes ~3 minutes (it downloads ~1,400
symbols); later runs finish in ~20 seconds off the cache. Output lands in
`out/`:

- `out/cup_screen.html` — the report: sortable table + annotated charts
- `out/cup_screen.csv` — every match with all 40-odd measured fields
- `out/universe.csv` — the stocks that passed the market-cap filter

## What counts as a cup

Checked in this order, on monthly bars:

1. **Left rim** is a swing high, and nothing inside the cup closes back above
   it — the interior has to stay under the rim.
2. **Right rim** climbs back to within 15% below / 8% above the left rim.
3. **Depth** from rim to low is 12–95%. A 15-year base can give back nearly the
   whole prior move and still round-trip it — BHEL fell 95% from its 2010 high
   before recovering — so the cap is deliberately loose and depth is **scored**
   rather than merely gated. A 55% cup scores well below a 25% one, and
   anything past 85% earns zero of the 18 depth points while still being
   allowed through. That scoring scale is fixed, so changing `--max-depth`
   does not silently inflate every score.
4. **Round, not a V** — a parabola is fit to the smoothed typical price
   (R² ≥ 0.55, curving upward, low near the middle), at least 8% of the bars
   have to rest in the lowest quarter of the cup, and the two sides have to be
   within 4:1 of each other in length.
5. **Prior advance** of ≥ 30% into the left rim. A cup is a rest inside an
   uptrend, not a recovery off a multi-year floor. When the rim sits in the
   first 6 bars of the available history — a stock that listed just before it,
   like ABCAPITAL — that advance cannot be measured at all. Unmeasurable is not
   the same as failed, so the cup passes, scores half marks on this component,
   and is annotated. `--strict-runup` rejects those instead.
6. **Handle** (optional): the pullback after the right rim, up to 9 months,
   3–30% deep, bottoming in the **upper half** of the cup, drifting sideways
   or down, ideally on lighter volume.
7. **Base still intact** — if price digs back below the cup's midpoint after
   the right rim, the base has failed and it is dropped.

**Pivot** = handle high, or the cup rim when there is no handle.
**Stop** = handle low (right-side low without a handle).

### Scoring

Cup score (100) weighs depth 18, roundness 18, bars resting at the low 15,
duration 12, symmetry 12, rim alignment 10, prior advance 10, volume dry-up 5.
Handle score (100) weighs depth 30, how high in the cup it sits 25, length 15,
downward drift 15, volume dry-up 15. A cup-with-handle's final score is
`0.65 × cup + 0.35 × handle`; a plain cup scores on the cup alone.

### Statuses

| Status | Meaning |
|---|---|
| `NEAR_PIVOT` | within 10% below the pivot — about to go |
| `IN_HANDLE` | handle still forming |
| `BREAKOUT` | closed above the pivot, up to 20% past it |
| `EXTENDED` | more than 20% past the pivot |
| `FORMING` | right side built but still well under the pivot |

`--stage actionable` (the default) keeps a match on either of two grounds:

- price is **at or under the pivot but inside the 10% near band** — loaded and
  hasn't fired; or
- price is **through the pivot**, the breakout is within `--breakout-age`
  months (default 12), and it hasn't run more than `--max-extension` past the
  pivot (default +100%).

So year-old breakouts are included, including ones labelled `EXTENDED` — the
label tells you the entry is behind you, it no longer hides the row. What gets
dropped: bases still deep in formation, breakouts that failed and fell back
into the cup, and moves that have long since run away. `--stage all` keeps
everything.

## Why isn't stock X in the list?

```bash
./screen --explain VEDL,ABCAPITAL
```

Walks the same candidate search the screener runs, but prints the verdict on
every check instead of discarding failures silently — each left rim it tried,
each right rim, the depth, the age, and which test rejected it. It also flags
any single month where price fell more than 45% from the previous bar's high,
which usually means a split or demerger the price series was never adjusted
for; any pattern spanning such a bar is measuring a corporate action rather
than a chart.

## Useful variations

```bash
# widen the net: right sides that have not fully recovered to the old rim
./screen --rim-tol-low 0.25 --min-score 50
```

```bash
# only setups still at the pivot — no breakouts that already went
./screen --breakout-age 1 --max-extension 0.10
```

```bash
# breakouts from the last two years, however far they have run
./screen --breakout-age 24 --max-age 24 --max-extension 99
```

```bash
# include bases that resolved years ago and have run far past the pivot
# (~145 matches vs ~21 — mostly EXTENDED, the entry long since gone)
./screen --include-resolved
```

```bash
# every stage of every base, not just the ones near the pivot
./screen --stage all --min-score 60
```

```bash
# tighter, more classic proportions — fewer and shallower bases
./screen --max-cup-len 120 --max-depth 0.60
```

```bash
# classic O'Neil proportions on short bases instead of multi-year ones
./screen --min-cup-len 5 --max-cup-len 30 --max-depth 0.50 --max-handle-len 6
```

```bash
# cup-with-handle only, biggest 300 companies, fresh data
./screen --pattern handle --limit 300 --refresh-prices
```

```bash
# check specific names
./screen --symbols BIOCON,NYKAA,LALPATHLAB --stage all --min-score 0
```

`./screen --help` lists every knob: `--min-depth`, `--max-depth`, `--min-r2`,
`--min-symmetry`, `--rim-tol-low`, `--min-prior-gain`, `--max-handle-depth`,
`--max-age`, `--near-pivot`, `--extended-pct`, `--max-extension`,
`--breakout-age`, `--start-before`, `--top`, and more.

## Data

- **Universe** — NSE's own `EQUITY_L.csv` (the official list of listed
  securities), series `EQ`, cached for 7 days.
- **Market cap** — Yahoo's bulk quote endpoint, ~100 symbols per request, so
  the whole list costs ~22 requests. Cached for 3 days. Default floor is
  ₹1,000 crore (`--min-mcap`).
- **Prices** — monthly OHLCV via `yfinance`, split-adjusted but **not**
  dividend-adjusted, so pivots and stops are levels you can put in an order.
  Full available history (`--period max`, about 30 years back to 1996 for older
  listings) — a 25-year cup needs it. Cached per symbol in `data/monthly/`,
  refreshed after 12 hours.

A handful of symbols (recent listings, demergers, renamed tickers) have no
Yahoo history and are skipped; the run logs how many. Stocks with fewer than
42 months of history can't hold a 36-month cup and are skipped too. A cup whose
left rim sits at the very start of the available data is also rejected, because
there is no room left to verify the 30% advance into it.

## Layout

```
nse_cup_screener/
  universe.py   NSE symbol list + bulk market caps
  prices.py     monthly OHLCV download and cache
  patterns.py   detection, scoring, status  ← the core
  report.py     HTML report and inline SVG charts
  cli.py        argument parsing and pipeline
screen          wrapper that runs it with the project venv (macOS/Linux)
screen.bat      the same for Windows
```

Rebuild the venv at any time with:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Dependencies are just `yfinance`, `pandas`, `numpy` and `requests` — the charts
and the HTML report are generated from scratch, with no plotting or templating
library involved.

## Caveat

This finds and ranks chart geometry. It does not know about earnings,
promoter pledges, results dates, or index inclusion, and a high score is a
statement about shape, not about a company. Every candidate still needs your
own eyes on the chart.
