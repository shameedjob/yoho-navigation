"""Benchmark the quantile GBM on a held-out week against simple baselines.

    python3 -m ml_model.benchmark_gbm --data data/training/training_data_2025-01.csv

Splits by service date into three consecutive spans, so no day's disruptions
reach two of them:

  train   everything before validation
  valid   --valid-days, used only for early stopping
  test    the last --test-days, touched once, for the numbers reported

A 7-day test span covers every day of the week once.

Every predictor is scored on the same test rows, and every one of them has a
value on every row (each baseline falls back down a chain rather than going
null), so no method looks better by skipping the hard rows.

  schedule          graph_edge_sec -- what Dijkstra routes on today. A mean-ish
                    cost, so its coverage shows how often today's plan is late
  schedule_x_q      schedule times one factor fit on train so it covers q
  edge_hour_q       per-edge q-quantile of train edge_sec by (weekend, hour)
  last_observed     obs_last_edge_sec, the most recent earlier train's time
  gbm               QuantileGBM on MODEL_FEATURES

Edge metrics are per traversal. Journey metrics sum consecutive edges of one
train's run (reconstructed by chaining each traversal to the one that left
from its to_node at the second it arrived), because a route's cost is a sum:
a per-edge q90 does not make a q90 journey.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from graph.subway_loader import build_subway_graph
from ml_model import model_data
from ml_model.gbm import OBJECTIVES, QuantileGBM
from ml_model.metrics import evaluate
from ml_model.replay import journey_ids, split_by_date

EDGE = ["from_node", "to_node"]
RUSH_HOURS = {7, 8, 9, 16, 17, 18}
MIN_GROUP_ROWS = 20
JOURNEY_EDGES = (5, 10, 20)


def quantile_lookup(train: pd.DataFrame, keys: list[str], q: float) -> pd.DataFrame:
    """q-quantile of edge_sec per group, keeping groups with enough rows."""
    grouped = train.groupby(keys, observed=True)["edge_sec"]
    table = grouped.quantile(q).rename("value").to_frame()
    table["rows"] = grouped.size()
    return table[table["rows"] >= MIN_GROUP_ROWS][["value"]].reset_index()


def baselines(train: pd.DataFrame, test: pd.DataFrame, q: float) -> dict[str, np.ndarray]:
    global_q = float(train["edge_sec"].quantile(q))

    edge_q = quantile_lookup(train, EDGE, q)
    edge_hour_q = quantile_lookup(train, EDGE + ["is_weekend", "hour"], q)
    fine = test[EDGE + ["is_weekend", "hour"]].merge(
        edge_hour_q, how="left", on=EDGE + ["is_weekend", "hour"])["value"]
    coarse = test[EDGE].merge(edge_q, how="left", on=EDGE)["value"]
    edge_hour = fine.fillna(coarse).fillna(global_q).to_numpy()

    edge_median = quantile_lookup(train, EDGE, 0.5)
    median = test[EDGE].merge(edge_median, how="left", on=EDGE)["value"].to_numpy()
    fallback_median = float(train["edge_sec"].median())

    def schedule_of(rows: pd.DataFrame, per_edge_median: np.ndarray) -> np.ndarray:
        cost = rows["graph_edge_sec"].astype("float64").to_numpy()
        return np.where(np.isnan(cost),
                        np.where(np.isnan(per_edge_median), fallback_median, per_edge_median),
                        cost)

    schedule = schedule_of(test, median)
    train_median = train[EDGE].merge(edge_median, how="left", on=EDGE)["value"].to_numpy()
    ratio = train["edge_sec"].to_numpy(dtype=float) / schedule_of(train, train_median)
    factor = float(np.quantile(ratio, q))

    last = test["obs_last_edge_sec"].astype("float64").to_numpy()
    return {
        "schedule": schedule,
        "schedule_x_q": schedule * factor,
        "edge_hour_q": edge_hour,
        "last_observed": np.where(np.isnan(last), edge_hour, last),
    }, {"schedule_x_q_factor": factor, "global_q": global_q}


def journey_scores(test: pd.DataFrame, preds: dict[str, np.ndarray], run: np.ndarray,
                   position: np.ndarray, q: float) -> dict[int, dict[str, dict]]:
    """Every window of n consecutive edges in a run: summed prediction vs
    summed observed time."""
    order = np.lexsort((position, run))
    run_sorted = run[order]
    target = test["edge_sec"].to_numpy(dtype=float)[order]
    out: dict[int, dict[str, dict]] = {}
    for n in JOURNEY_EDGES:
        # window [k, k+n) is valid when its first and last rows share a run
        valid = np.flatnonzero(run_sorted[:len(order) - n + 1] == run_sorted[n - 1:])
        if len(valid) == 0:
            continue
        def window_sum(values):
            c = np.r_[0.0, np.cumsum(values)]
            return c[valid + n] - c[valid]
        actual = window_sum(target)
        out[n] = {name: evaluate(window_sum(p[order]), actual, q) for name, p in preds.items()}
    return out


def slices(test: pd.DataFrame) -> dict[str, np.ndarray]:
    schedule = test["graph_edge_sec"].astype("float64")
    return {
        "all": np.ones(len(test), dtype=bool),
        "has_schedule=0 (censored)": test["has_schedule"].to_numpy() == 0,
        "not in graph (graph_edge_sec null)": schedule.isna().to_numpy(),
        "no recent observation": test["obs_last_edge_sec"].isna().to_numpy(),
        "station alert live": test["station_alert_count"].to_numpy() > 0,
        "weekend": test["is_weekend"].to_numpy() == 1,
        "rush hour (7-10, 16-19)": test["hour"].isin(RUSH_HOURS).to_numpy(),
        "overnight (0-5)": test["hour"].between(0, 5).to_numpy(),
        "delayed edge (>60s over schedule)":
            (test["edge_sec"].astype("float64") > schedule + 60).fillna(False).to_numpy(),
    }


def format_table(rows: list[dict], columns: list[str]) -> str:
    def cell(value):
        if isinstance(value, float):
            return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
        return str(value)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(r[c]) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument("--objective", choices=OBJECTIVES, default="quantile",
                        help="l1 / huber train a point model; baselines and pinball then use q=0.5")
    parser.add_argument("--huber-delta", type=float, default=30.0,
                        help="seconds where huber switches from squared to absolute loss")
    parser.add_argument("--valid-days", type=int, default=4)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--num-boost-round", type=int, default=5000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--params", type=json.loads, default=None)
    parser.add_argument("--model-out", type=Path, default=Path("ml_model/checkpoints/gbm_benchmark"))
    parser.add_argument("--report", type=Path, default=None,
                        help="markdown report path; default docs/benchmarks/gbm_<test span>.md")
    args = parser.parse_args()
    q = args.quantile if args.objective == "quantile" else 0.5

    frame = pd.concat([pd.read_csv(p, low_memory=False) for p in args.data], ignore_index=True)
    frame = model_data.prepare(frame, build_subway_graph(args.data_dir / "gtfs_subway"))
    frame = frame[frame["edge_sec"].notna()].reset_index(drop=True)
    frame["edge_sec"] = frame["edge_sec"].astype("float64")
    train, valid, test, spans = split_by_date(frame, args.valid_days, args.test_days)
    train, valid, test = (part.reset_index(drop=True) for part in (train, valid, test))
    for name, dates in spans.items():
        size = {"train": train, "valid": valid, "test": test}[name]
        print(f"{name}: {dates[0]} .. {dates[-1]} ({len(dates)} days, {len(size):,} rows)")

    model = QuantileGBM(model_data.SPEC, q, args.params, args.objective, args.huber_delta)
    model.fit(train, valid, num_boost_round=args.num_boost_round,
              early_stopping_rounds=args.early_stopping_rounds)
    model.save(args.model_out)

    preds, fitted = baselines(train, test, q)
    preds["gbm"] = model.predict(test)

    target = test["edge_sec"].to_numpy()
    slice_masks = slices(test)
    edge_results = {
        slice_name: {name: evaluate(p[mask], target[mask], q) for name, p in preds.items()}
        for slice_name, mask in slice_masks.items() if mask.any()
    }
    per_day = {
        day: evaluate(preds["gbm"][mask], target[mask], q)
        for day, mask in ((d, (test["service_date"] == d).to_numpy())
                          for d in spans["test"])
    }
    run, position = journey_ids(test)
    journeys = journey_scores(test, preds, run, position, q)
    gain = pd.Series(model.booster.feature_importance("gain"),
                     index=model.booster.feature_name())
    importance = (gain / gain.sum()).sort_values(ascending=False)

    results = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": [str(p) for p in args.data],
        "quantile": q,
        "objective": args.objective,
        "huber_delta": args.huber_delta if args.objective == "huber" else None,
        "spans": {k: [v[0], v[-1], len(v)] for k, v in spans.items()},
        "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
        "best_iteration": model.booster.best_iteration,
        "features": model_data.SPEC.to_dict(),
        "fitted": fitted,
        "edge": edge_results,
        "journey": {str(n): v for n, v in journeys.items()},
        "gbm_per_day": per_day,
        "gbm_gain_share": importance.round(5).to_dict(),
        "linked_runs": {"rows_in_runs_of_2_plus": float((np.bincount(run)[run] > 1).mean())},
    }

    report = args.report or Path(
        f"docs/benchmarks/gbm_{spans['test'][0]}_{spans['test'][-1]}.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.with_suffix(".json").write_text(json.dumps(results, indent=2))

    names = list(preds)
    sections = [
        f"# GBM benchmark, test {spans['test'][0]} to {spans['test'][-1]}",
        f"Generated by `python3 -m ml_model.benchmark_gbm` on {results['created']}. "
        f"Objective {args.objective}"
        + (f" (delta {args.huber_delta:g}s)" if args.objective == "huber" else "")
        + f", scored at quantile {q}. Raw numbers in `{report.with_suffix('.json').name}`.",
        "## Split\n\n" + format_table(
            [{"span": k, "dates": f"{v[0]} .. {v[-1]}", "days": len(v),
              "rows": results["rows"][k]} for k, v in spans.items()],
            ["span", "dates", "days", "rows"]),
        f"GBM stopped at iteration {model.booster.best_iteration}. "
        f"`schedule_x_q` factor fit on train: {fitted['schedule_x_q_factor']:.3f}.",
    ]
    overall = [{"method": n, **edge_results["all"][n]} for n in names]
    sections.append("## Per edge, all test rows\n\n" + format_table(
        overall, ["method", "rows", "pinball", "coverage", "mae"]))
    slice_rows = []
    for slice_name, by_method in edge_results.items():
        row = {"slice": slice_name, "rows": by_method["gbm"]["rows"]}
        for n in names:
            row[n] = f"{by_method[n]['pinball']:.2f} / {by_method[n]['coverage']:.2f}"
        slice_rows.append(row)
    sections.append("## Per edge by slice\n\nEach cell is pinball / coverage.\n\n"
                    + format_table(slice_rows, ["slice", "rows", *names]))
    journey_rows = []
    for n, by_method in journeys.items():
        row = {"edges": n, "windows": by_method["gbm"]["rows"]}
        for m in names:
            row[m] = f"{by_method[m]['pinball']:.1f} / {by_method[m]['coverage']:.2f}"
        journey_rows.append(row)
    sections.append("## Summed over consecutive edges of one run\n\n"
                    "Each cell is pinball / coverage of the summed cost.\n\n"
                    + format_table(journey_rows, ["edges", "windows", *names]))
    sections.append("## GBM by test day\n\n" + format_table(
        [{"day": d, **s} for d, s in per_day.items()],
        ["day", "rows", "pinball", "coverage", "mae"]))
    sections.append("## GBM feature gain share\n\n" + format_table(
        [{"feature": f, "share": float(v)} for f, v in importance.items()],
        ["feature", "share"]))
    report.write_text("\n\n".join(sections) + "\n")
    print(f"wrote {report} and {report.with_suffix('.json')}")
    print("\n\n".join(sections[3:6]))


if __name__ == "__main__":
    main()
