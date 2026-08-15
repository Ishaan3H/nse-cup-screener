"""Command line entry point: build universe -> fetch monthly bars -> detect -> report."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

from . import __version__
from .patterns import Params, detect, explain, is_actionable, screen
from .prices import download_monthly
from .report import write_report
from .universe import build_universe, write_universe

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
STATUSES = ["BREAKOUT", "NEAR_PIVOT", "IN_HANDLE", "FORMING", "EXTENDED"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="nse-cup-screener",
        description="Screen NSE-listed stocks for cup and cup-with-handle patterns on monthly charts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--min-mcap", type=float, default=1000.0, help="minimum market cap in ₹ crore")
    ap.add_argument("--series", default="EQ", help="NSE series to include, comma separated")
    ap.add_argument("--symbols", default="", help="screen only these symbols (comma separated), skips the universe build")
    ap.add_argument(
        "--explain",
        default="",
        help="print why these symbols do or do not qualify, check by check, then exit",
    )
    ap.add_argument("--limit", type=int, default=0, help="cap the universe to the N largest by market cap (0 = all)")
    ap.add_argument("--period", default="max", help="how much monthly history to pull (max ≈ 30 years)")

    ap.add_argument("--min-score", type=float, default=55.0, help="drop matches scoring below this")
    ap.add_argument(
        "--stage",
        choices=["actionable", "all"],
        default="actionable",
        help="'actionable' keeps setups at the pivot or through it within --breakout-age; 'all' keeps every stage",
    )
    ap.add_argument("--breakout-age", type=int, default=12, help="how many months back a breakout still counts")
    ap.add_argument(
        "--include-resolved",
        action="store_true",
        help="also report bases that already broke out years ago and have run far past the pivot "
        "(widens --max-age/--breakout-age to 42 and --max-extension to 3.0 unless you set them yourself)",
    )
    ap.add_argument(
        "--max-extension",
        type=float,
        default=Params.max_extension,
        help="drop breakouts that have run more than this far past the pivot, e.g. 1.0 = +100%%",
    )
    ap.add_argument("--status", default="", help=f"keep only these statuses, comma separated ({', '.join(STATUSES)}); overrides --stage")
    ap.add_argument("--pattern", choices=["all", "handle", "cup"], default="all", help="restrict pattern type")
    ap.add_argument("--all-patterns", action="store_true", help="report every distinct pattern per stock, not just the best")
    ap.add_argument("--top", type=int, default=0, help="keep only the N highest-scoring matches (0 = all)")
    ap.add_argument(
        "--start-before",
        default="",
        help="keep only cups whose left rim formed before this date, e.g. 2024-01-01",
    )
    ap.add_argument("--max-charts", type=int, default=120, help="how many charts to draw in the report")

    g = ap.add_argument_group("pattern shape")
    g.add_argument("--min-cup-len", type=int, default=Params.min_cup_len, help="minimum cup length in months")
    g.add_argument("--max-cup-len", type=int, default=Params.max_cup_len, help="maximum cup length in months")
    g.add_argument("--min-depth", type=float, default=Params.min_depth, help="minimum cup depth, e.g. 0.12")
    g.add_argument("--max-depth", type=float, default=Params.max_depth, help="maximum cup depth")
    g.add_argument("--min-r2", type=float, default=Params.min_curvature_r2, help="minimum parabola fit (U vs V)")
    g.add_argument(
        "--rim-tol-low",
        type=float,
        default=Params.rim_tol_low,
        help="how far below the left rim the right side may stop (0.25 catches bases still climbing)",
    )
    g.add_argument("--rim-tol-high", type=float, default=Params.rim_tol_high, help="how far above the left rim the right rim may run")
    g.add_argument("--min-symmetry", type=float, default=Params.min_symmetry, help="min(left, right) / max(left, right)")
    g.add_argument("--min-prior-gain", type=float, default=Params.min_prior_gain, help="required advance into the left rim")
    g.add_argument(
        "--strict-runup",
        action="store_true",
        help="also reject cups whose rim is too near the start of the data to measure that advance",
    )
    g.add_argument("--max-handle-len", type=int, default=Params.max_handle_len, help="maximum handle length in months")
    g.add_argument("--max-handle-depth", type=float, default=Params.max_handle_depth, help="maximum handle depth")
    g.add_argument("--max-age", type=int, default=Params.max_age, help="pattern must end within N months of today")
    g.add_argument("--near-pivot", type=float, default=Params.near_pivot_pct, help="band below the pivot counted as 'near'")
    g.add_argument("--extended-pct", type=float, default=Params.extended_pct, help="past this far above the pivot, flag as extended")
    g.add_argument(
        "--allow-broken",
        action="store_true",
        help="keep bases that have since fallen back below the cup's midpoint",
    )

    io = ap.add_argument_group("io")
    io.add_argument("--out", default=str(ROOT / "out"), help="output directory")
    io.add_argument("--cache", default=str(ROOT / "data"), help="cache directory")
    io.add_argument("--refresh", action="store_true", help="ignore caches and refetch everything")
    io.add_argument("--refresh-prices", action="store_true", help="refetch price data only")
    io.add_argument("--price-max-age", type=float, default=12, help="hours before cached prices are considered stale")
    io.add_argument("--quiet", action="store_true", help="less logging")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return ap


def params_from_args(a: argparse.Namespace) -> Params:
    return Params(
        min_cup_len=a.min_cup_len,
        max_cup_len=a.max_cup_len,
        min_depth=a.min_depth,
        max_depth=a.max_depth,
        min_curvature_r2=a.min_r2,
        rim_tol_low=a.rim_tol_low,
        rim_tol_high=a.rim_tol_high,
        min_symmetry=a.min_symmetry,
        min_prior_gain=a.min_prior_gain,
        strict_prior=a.strict_runup,
        max_handle_len=a.max_handle_len,
        max_handle_depth=a.max_handle_depth,
        max_age=a.max_age,
        near_pivot_pct=a.near_pivot,
        extended_pct=a.extended_pct,
        max_extension=a.max_extension,
        require_intact=not a.allow_broken,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    log = (lambda *a, **k: None) if args.quiet else print

    if args.include_resolved:
        # Only fill in the knobs the caller left alone, so explicit flags win.
        for name, value in (("max_age", 42), ("breakout_age", 42), ("max_extension", 3.0)):
            if getattr(args, name) == parser.get_default(name):
                setattr(args, name, value)

    out_dir = Path(args.out)
    cache_dir = Path(args.cache)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    log(f"NSE cup screener v{__version__}")

    # 0. explain mode ---------------------------------------------------------
    if args.explain:
        wanted = [s.strip().upper() for s in args.explain.split(",") if s.strip()]
        frames = download_monthly(
            wanted,
            cache_dir / "monthly",
            period=args.period,
            max_age_hours=args.price_max_age,
            refresh=args.refresh or args.refresh_prices,
            log=lambda *a, **k: None,
        )
        p = params_from_args(args)
        for sym in wanted:
            if sym not in frames:
                print(f"\n{sym}: no monthly data from Yahoo")
                continue
            print()
            for line in explain(frames[sym], sym, p):
                print(line)
            hits = detect(frames[sym], symbol=sym, p=p)
            if hits and is_actionable(hits[0], p, args.breakout_age):
                print(f"  → QUALIFIES: score {hits[0].score:.0f}, {hits[0].status}")
            elif hits:
                print(f"  → pattern found but filtered out: {hits[0].status}, "
                      f"{hits[0].months_since_breakout}m since breakout, "
                      f"{hits[0].above_pivot_pct:+.0f}% vs pivot")
            else:
                print("  → no qualifying pattern")
        return 0

    # 1. universe -------------------------------------------------------------
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        meta = {s: {"name": "", "mcap_cr": float("nan")} for s in symbols}
        log(f"Screening {len(symbols)} explicitly requested symbols")
    else:
        log("\n[1/4] Building universe")
        universe, _ = build_universe(
            cache_dir,
            min_market_cap_cr=args.min_mcap,
            series=tuple(s.strip().upper() for s in args.series.split(",") if s.strip()),
            refresh=args.refresh,
            log=log,
        )
        if args.limit:
            universe = universe.head(args.limit)
            log(f"  limited to the {len(universe)} largest by market cap")
        write_universe(universe, out_dir / "universe.csv")
        symbols = universe["SYMBOL"].tolist()
        meta = {
            r.SYMBOL: {"name": str(r.NAME), "mcap_cr": float(r.MCAP_CR)}
            for r in universe.itertuples()
        }

    if not symbols:
        log("Universe is empty — nothing to screen.")
        return 1

    # 2. prices ---------------------------------------------------------------
    log(f"\n[2/4] Fetching monthly bars for {len(symbols)} symbols")
    frames = download_monthly(
        symbols,
        cache_dir / "monthly",
        period=args.period,
        max_age_hours=args.price_max_age,
        refresh=args.refresh or args.refresh_prices,
        log=log,
    )
    p = params_from_args(args)
    usable = {s: df for s, df in frames.items() if len(df) >= p.min_cup_len + 6}
    log(f"  {len(usable)} symbols with enough monthly history ({len(symbols) - len(usable)} short or missing)")

    # 3. detect ---------------------------------------------------------------
    log("\n[3/4] Scanning for cups")
    patterns = screen(usable, meta=meta, p=p, all_patterns=args.all_patterns, log=log)
    log(f"  {len(patterns)} raw patterns")

    if args.pattern == "handle":
        patterns = [x for x in patterns if x.pattern == "CUP_HANDLE"]
    elif args.pattern == "cup":
        patterns = [x for x in patterns if x.pattern == "CUP"]
    if args.status:
        wanted = {s.strip().upper() for s in args.status.split(",") if s.strip()}
        patterns = [x for x in patterns if x.status in wanted]
    elif args.stage == "actionable":
        before = len(patterns)
        patterns = [x for x in patterns if is_actionable(x, p, args.breakout_age)]
        log(
            f"  {len(patterns)} at the pivot or through it within {args.breakout_age} months "
            f"({before - len(patterns)} still basing, failed, or long gone)"
        )
    if args.start_before:
        cutoff = str(pd.Timestamp(args.start_before).date())
        patterns = [x for x in patterns if x.left_date < cutoff]
    patterns = [x for x in patterns if x.score >= args.min_score]
    if args.top:
        patterns = patterns[: args.top]
    log(f"  {len(patterns)} after filters (score >= {args.min_score:g})")

    # 4. output ---------------------------------------------------------------
    log("\n[4/4] Writing output")
    csv_path = out_dir / "cup_screen.csv"
    if patterns:
        pd.DataFrame([x.to_row() for x in patterns]).to_csv(csv_path, index=False)
    else:
        pd.DataFrame(columns=["symbol"]).to_csv(csv_path, index=False)

    html_path = out_dir / "cup_screen.html"
    write_report(
        html_path,
        patterns,
        usable,
        p,
        universe_size=len(symbols),
        scanned=len(usable),
        min_market_cap_cr=args.min_mcap,
        max_charts=args.max_charts,
        stage_note=(
            f" · showing setups at the pivot, plus breakouts from the last "
            f"{args.breakout_age} months that have not run more than "
            f"{args.max_extension:.0%} past it"
            if args.stage == "actionable" and not args.status
            else ""
        ),
    )

    log(f"  {csv_path}")
    log(f"  {html_path}")

    if patterns and not args.quiet:
        print(f"\nAll {len(patterns)} matches")
        print(
            f"{'SYMBOL':<14}{'TYPE':<14}{'STATUS':<12}{'SCORE':>6}{'CUP START':>12}{'MONTHS':>8}"
            f"{'CLOSE':>11}{'PIVOT':>11}{'TO PIVOT':>10}{'B/O AGO':>9}{'SINCE B/O':>11}{'DEPTH':>8}"
        )
        for x in patterns:
            kind = "cup + handle" if x.pattern == "CUP_HANDLE" else "cup"
            bo_age = f"{x.months_since_breakout}m" if x.breakout_idx >= 0 else "—"
            bo_gain = f"{x.gain_since_breakout_pct:+.0f}%" if x.breakout_idx >= 0 else "—"
            print(
                f"{x.symbol:<14}{kind:<14}{x.status.lower():<12}{x.score:>6.0f}{x.left_date[:7]:>12}"
                f"{x.cup_len_m:>8}{x.last_close:>11,.2f}{x.pivot:>11,.2f}"
                f"{x.to_pivot_pct:>9.1f}%{bo_age:>9}{bo_gain:>11}{x.depth_pct:>7.0f}%"
            )

    log(f"\nDone in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
