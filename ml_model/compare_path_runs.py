"""Compare path benchmark runs trip by trip: planning 0, 30 and 60 minutes ahead,
or the same departures under different edge models (--title, --drift-heading).

    python3 -m ml_model.compare_path_runs \\
        --runs now=docs/benchmarks/paths_2026-sample30_lead0.md \\
               30min=docs/benchmarks/paths_2026-sample30_lead30.md \\
               60min=docs/benchmarks/paths_2026-sample30_lead60.md \\
        --out docs/benchmarks/paths_2026-sample30_leads.md

Runs are joined on (departure time, origin, destination) -- benchmark_paths
draws station pairs from the departure time, so runs over the same test dates
and seed share trips -- and only trips every listed router replayed in every
run are kept. The same riders, leaving at the same moments, so differences
come from what differs between the runs.

Confidence intervals resample departure times, since trips leaving together
share conditions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml_model.benchmark_gbm import format_table

KEY = ["t0", "origin", "destination"]
ESTIMATES = {"scheduled": "Scheduled", "predicted": "LightGBM", "predicted_waits": "LightGBM + waits"}
ROUTER_NAMES = {"sched_live": "schedule", "model_live": "LightGBM", "model_waits_live": "LightGBM + waits"}
# --edge-costs NAME=CSV sets from benchmark_paths: routers NAME_live / NAME_waits_live,
# estimates pred_NAME / pred_NAME_waits.
for _name in ("rgnn", "gat", "gwnet"):
    ESTIMATES |= {f"pred_{_name}": _name.upper(), f"pred_{_name}_waits": f"{_name.upper()} + waits"}
    ROUTER_NAMES |= {f"{_name}_live": _name.upper(), f"{_name}_waits_live": f"{_name.upper()} + waits"}


def bootstrap_mean(values: pd.Series, groups: pd.Series, draws: int = 2000,
                   seed: int = 0) -> tuple[float, float]:
    by = values.groupby(groups).agg(["sum", "count"])
    idx = np.random.default_rng(seed).integers(0, len(by), (draws, len(by)))
    means = by["sum"].to_numpy()[idx].sum(1) / by["count"].to_numpy()[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, metavar="NAME=REPORT_MD")
    parser.add_argument("--routers", nargs="+", default=["sched_live", "model_live", "model_waits_live"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Path benchmark: planning ahead")
    parser.add_argument("--run-label", default="planned",
                        help="column naming what each run varies")
    parser.add_argument("--drift-heading", default="Cost of planning ahead")
    args = parser.parse_args()

    runs = {}
    for spec in args.runs:
        name, _, report = spec.partition("=")
        report = Path(report)
        runs[name] = pd.read_csv(report.with_name(report.stem + "_trips.csv")).set_index(KEY)

    common = None
    for trips in runs.values():
        ok = trips[np.logical_and.reduce([trips[f"{r}_status"] == "ok" for r in args.routers])].index
        common = ok if common is None else common.intersection(ok)
    runs = {name: trips.loc[common].reset_index() for name, trips in runs.items()}
    first = next(iter(runs.values()))
    t0 = first["t0"]

    def m(seconds):
        return f"{seconds / 60:.1f}"

    def signed_m(seconds):
        return f"{seconds / 60:+.1f}"

    def pct(share):
        return f"{share:.0%}"

    quote_rows = []
    for name, trips in runs.items():
        for key, label in ESTIMATES.items():
            column = f"sched_live_{key}"
            if column not in trips:
                continue
            err = trips["sched_live_true"] - trips[column]
            quote_rows.append({args.run_label: name, "estimate": label,
                               "mean error": signed_m(err.mean()),
                               "median abs error": m(err.abs().median()),
                               "actual took longer": pct((err > 0).mean()),
                               "within ±5 min": pct((err.abs() <= 300).mean())})

    route_rows = []
    for name, trips in runs.items():
        base = trips["sched_live_true"]
        for r in args.routers:
            change = trips[f"{r}_true"] - base
            lo, hi = bootstrap_mean(change, t0) if r != "sched_live" else (0.0, 0.0)
            route_rows.append({args.run_label: name, "router": ROUTER_NAMES.get(r, r),
                               "actual mean": m(trips[f"{r}_true"].mean()),
                               "vs schedule (s)": f"{change.mean():+.0f}",
                               "95% CI (s)": "" if r == "sched_live" else f"{lo:+.0f} to {hi:+.0f}",
                               "transfers": f"{trips[f'{r}_transfers'].mean():.2f}"})

    names = list(runs)
    drift_rows = []
    for name in names[1:]:
        for r in args.routers:
            change = runs[name][f"{r}_true"] - runs[names[0]][f"{r}_true"]
            lo, hi = bootstrap_mean(change, t0)
            drift_rows.append({"router": ROUTER_NAMES.get(r, r),
                               "comparison": f"{name} vs {names[0]}",
                               "actual time change (s)": f"{change.mean():+.0f}",
                               "95% CI (s)": f"{lo:+.0f} to {hi:+.0f}",
                               "same route": pct((runs[name][f"{r}_path"]
                                                  == runs[names[0]][f"{r}_path"]).mean())})

    sections = [
        f"# {args.title}",
        f"Runs: {', '.join(f'`{s}`' for s in args.runs)}. Joined on departure time, origin and "
        f"destination; {len(common):,} trips that every router ({', '.join(args.routers)}) "
        "replayed in every run. Times in minutes unless marked; CIs resample departure times.",
        "## Quoted vs actual\n\nOn the schedule router's path, from arrival at the origin "
        "station. Positive error means the trip took longer than quoted.\n\n"
        + format_table(quote_rows, [args.run_label, "estimate", "mean error", "median abs error",
                                    "actual took longer", "within ±5 min"]),
        "## Route actually taken\n\nEach router's own route, replayed.\n\n"
        + format_table(route_rows, [args.run_label, "router", "actual mean", "vs schedule (s)",
                                    "95% CI (s)", "transfers"]),
    ]
    if drift_rows:
        sections.append(
            f"## {args.drift_heading}\n\nThe same router in each run against "
            f"`{names[0]}`, on the same trips.\n\n"
            + format_table(drift_rows, ["router", "comparison", "actual time change (s)",
                                        "95% CI (s)", "same route"]))
    args.out.write_text("\n\n".join(sections) + "\n")
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
