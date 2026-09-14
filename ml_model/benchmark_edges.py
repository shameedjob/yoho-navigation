"""Score every edge model on the same test traversals from the same snapshots.

    python3 -m ml_model.benchmark_edges --data data/training/training_data_2026-sample30.csv \\
        --model ml_model/checkpoints/gbm_2026-sample30 \\
        --edge-costs rgnn=ml_model/checkpoints/rgnn_2026-sample30/rgnn_test_predictions.csv \\
                     gat=... gwnet=... \\
        --out docs/benchmarks/edges_2026-sample30.md

benchmark_gbm scores LightGBM per traversal with features at that exact
moment, and the graph trainers report validation windows; neither is
comparable to the other. Here every model predicts from a snapshot at the
same times t0 -- the path benchmark's departure times, where the graph models
wrote their predictions -- and is scored on the same labels: every test
traversal departing in [t0, t0 + --horizon-min), the target the graph models
were trained on. LightGBM's features come from replay.snapshot_features at t0,
exactly as in the path benchmark.

Only rows every model predicts are scored. Differences against the reference
model (--reference, LightGBM by default) are paired per traversal, with 95%
CIs from resampling snapshot times.

With --wait-models, transfers are scored too: every transfer taken off a
sampled share of test arrivals within the horizon of a snapshot, labelled walk
+ wait for the next train (waits.transfer_cost_labels). Scored against it: the
walk plus the transfer-wait model's wait with features at the snapshot (as the
path benchmark prices transfers), the graph's schedule cost (walk + half the
headway), and the transfer rows of any --edge-costs CSV that has them
(train_gat --with-transfers).

No torch here: the graph models' predictions are read from their CSVs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from graph.subway_loader import build_subway_graph
from ml_model import model_data, waits
from ml_model.benchmark_gbm import RUSH_HOURS, format_table
from ml_model.gbm import QuantileGBM
from ml_model.metrics import evaluate
from ml_model.replay import (TZ, Network, Observed, departure_times, state_of_time,
                             snapshot_features, snapshot_labels, test_dates)

EDGE = ["from_node", "to_node"]


def pinball(pred: np.ndarray, target: np.ndarray, q: float) -> np.ndarray:
    err = target - pred
    return np.maximum(q * err, (q - 1) * err)


def bootstrap_mean(values: np.ndarray, groups: np.ndarray, draws: int = 2000,
                   seed: int = 0) -> tuple[float, float]:
    by = pd.Series(values).groupby(groups).agg(["sum", "count"])
    idx = np.random.default_rng(seed).integers(0, len(by), (draws, len(by)))
    means = by["sum"].to_numpy()[idx].sum(1) / by["count"].to_numpy()[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, required=True, help="LightGBM q90 edge model")
    parser.add_argument("--gbm", nargs="*", default=[], metavar="NAME=DIR",
                        help="more LightGBM checkpoints, predicted from the same snapshots")
    parser.add_argument("--edge-costs", nargs="*", default=[], metavar="NAME=CSV")
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--step-min", type=int, default=20,
                        help="snapshot spacing; must match the prediction CSVs")
    parser.add_argument("--horizon-min", type=int, default=10,
                        help="label traversals departing this soon after each snapshot")
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument("--reference", default="lightgbm")
    parser.add_argument("--wait-models", type=Path, default=None,
                        help="train_waits output; adds the transfer section")
    parser.add_argument("--transfer-fraction", type=float, default=0.25,
                        help="share of test arrivals whose transfers are labelled")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    q = args.quantile

    gtfs = args.data_dir / "gtfs_subway"
    graph = build_subway_graph(gtfs)
    network = Network(graph, gtfs)
    rows = pd.concat([pd.read_csv(p, low_memory=False) for p in args.data], ignore_index=True)
    rows = rows[rows["edge_sec"].notna()]
    dates = test_dates(rows, args.test_days)
    before = {(pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y-%m-%d") for d in dates}
    rows = rows[rows["service_date"].isin(set(dates) | before)].reset_index(drop=True)
    t0s = departure_times(rows, dates, args.step_min)

    features = snapshot_features(rows, Observed(rows, network), network, t0s)
    prepared = model_data.prepare(features, graph)
    table = features[["t0", *EDGE]].copy()
    table["lightgbm"] = QuantileGBM.load(args.model).predict(prepared)
    table["schedule"] = prepared["graph_edge_sec"].to_numpy()
    table["recent_obs"] = features["obs_last_edge_sec"].notna().to_numpy()
    names = ["schedule", "lightgbm"]
    for spec in args.gbm:
        name, _, path = spec.partition("=")
        table[name] = QuantileGBM.load(Path(path)).predict(prepared)
        names.append(name)
    for spec in args.edge_costs:
        name, _, path = spec.partition("=")
        preds = pd.read_csv(path).rename(columns={"pred": name})
        if "is_transfer" in preds:
            preds = preds[~preds["is_transfer"].astype(bool)]
        missing = set(t0s.tolist()) - set(preds["t0"].unique().tolist())
        if missing:
            raise SystemExit(f"{path} lacks {len(missing)} snapshot times")
        table = table.merge(preds[["t0", *EDGE, name]], how="left", on=["t0", *EDGE])
        names.append(name)
    print(f"{len(table):,} snapshot edge predictions from {len(t0s)} snapshots", flush=True)

    # Labels: test-date traversals departing within the horizon of a snapshot.
    labels = snapshot_labels(rows[rows["service_date"].isin(dates)], t0s, args.horizon_min * 60)
    scored = labels.merge(table, how="inner", on=["t0", *EDGE])
    complete = scored[names].notna().all(axis=1)
    print(f"{len(scored):,} labelled traversals; {int(complete.sum()):,} with every "
          f"model's prediction", flush=True)
    scored = scored[complete].reset_index(drop=True)

    target = scored["edge_sec"].to_numpy(dtype=float)
    local = pd.to_datetime(scored["t0"], unit="s", utc=True).dt.tz_convert(TZ)
    slices = {
        "all": np.ones(len(scored), bool),
        "weekday": (local.dt.dayofweek < 5).to_numpy(),
        "weekend": (local.dt.dayofweek >= 5).to_numpy(),
        "rush hour (7-10, 16-19)": local.dt.hour.isin(RUSH_HOURS).to_numpy(),
        "overnight (0-5)": local.dt.hour.between(0, 5).to_numpy(),
        "edge seen in last 30 min": scored["recent_obs"].to_numpy(bool),
        "edge not seen in last 30 min": ~scored["recent_obs"].to_numpy(bool),
    }
    results = {s: {n: evaluate(scored[n].to_numpy()[m], target[m], q) for n in names}
               for s, m in slices.items() if m.any()}

    ref = args.reference
    ref_loss = pinball(scored[ref].to_numpy(), target, q)
    groups = scored["t0"].to_numpy()
    paired = []
    for n in names:
        if n == ref:
            continue
        diff = pinball(scored[n].to_numpy(), target, q) - ref_loss
        lo, hi = bootstrap_mean(diff, groups)
        by_snapshot = pd.Series(diff).groupby(groups).mean()
        paired.append({"model": n, "pinball vs " + ref: f"{diff.mean():+.3f}",
                       "95% CI": f"{lo:+.3f} to {hi:+.3f}",
                       "relative": f"{diff.mean() / ref_loss.mean():+.1%}",
                       "snapshots better": f"{(by_snapshot < 0).mean():.0%}",
                       "_diff": float(diff.mean()), "_ci": [lo, hi]})
    day = local.dt.strftime("%Y-%m-%d").to_numpy()
    per_day = [{"day": d, "rows": int((day == d).sum()),
                **{n: f"{evaluate(scored[n].to_numpy()[day == d], target[day == d], q)['pinball']:.2f}"
                   for n in names}} for d in dates]

    summary = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": [str(p) for p in args.data], "model": str(args.model),
        "edge_costs": args.edge_costs, "test_dates": dates, "snapshots": len(t0s),
        "step_min": args.step_min, "horizon_min": args.horizon_min, "quantile": q,
        "rows_scored": len(scored), "results": results,
        "paired": [{k: v for k, v in p.items()} for p in paired],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str))

    sections = [
        "# Edge benchmark: every model from the same snapshots",
        f"Generated by `python3 -m ml_model.benchmark_edges` on {summary['created']}. "
        f"{len(t0s)} snapshots every {args.step_min} min on test dates {', '.join(dates)}. "
        f"Labels: each test traversal departing within {args.horizon_min} min after a snapshot, "
        f"{len(scored):,} of them, all scored by every model. lightgbm `{args.model}`; "
        + ", ".join(f"{s.partition('=')[0]} `{s.partition('=')[2]}`"
                    for s in [*args.gbm, *args.edge_costs])
        + f". Quantile {q}; pinball and MAE in seconds per edge. `schedule` is the graph's "
        "cost, a mean-ish number shown for scale.",
        "## All test traversals\n\n" + format_table(
            [{"model": n, **results["all"][n]} for n in names],
            ["model", "rows", "pinball", "coverage", "mae"]),
        f"## Paired against {ref}\n\nPer-traversal pinball difference (negative: better than "
        f"{ref}). CIs resample snapshot times. `snapshots better` is the share of snapshots "
        f"where the model's mean pinball beat {ref}'s.\n\n" + format_table(
            paired, ["model", f"pinball vs {ref}", "95% CI", "relative", "snapshots better"]),
    ]
    slice_rows = []
    for s, by in results.items():
        slice_rows.append({"slice": s, "rows": by[ref]["rows"],
                           **{n: f"{by[n]['pinball']:.2f} / {by[n]['coverage']:.2f}" for n in names}})
    sections.append("## By slice\n\nEach cell is pinball / coverage.\n\n"
                    + format_table(slice_rows, ["slice", "rows", *names]))
    sections.append("## Pinball by test day\n\n" + format_table(per_day, ["day", "rows", *names]))
    if args.wait_models:
        transfer_summary, transfer_sections = score_transfers(args, graph, network, rows, dates,
                                                             t0s, q)
        summary["transfers"] = transfer_summary
        args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str))
        sections.extend(transfer_sections)
    args.out.write_text("\n\n".join(sections) + "\n")
    print("\n\n".join(sections))


def score_transfers(args, graph, network, rows, dates, t0s, q) -> tuple[dict, list[str]]:
    rng = np.random.default_rng(args.seed)
    transfer_idx = np.flatnonzero(network.is_transfer)
    transfers = pd.DataFrame({"from_node": [network.edge_keys[e][0] for e in transfer_idx],
                              "to_node": [network.edge_keys[e][1] for e in transfer_idx],
                              "walk_sec": network.walk[transfer_idx]})
    edge_cost = {(f, t): b for (f, t, x), b in zip(network.edge_keys, network.base) if not x}
    deps = waits.departures(rows, edge_cost)
    events = waits.transfer_cost_labels(rows[rows["service_date"].isin(dates)], deps, transfers,
                                        args.transfer_fraction, rng)
    events = pd.merge_asof(events.sort_values("arrived"), pd.DataFrame({"t0": np.sort(t0s)}),
                           left_on="arrived", right_on="t0", direction="backward",
                           tolerance=args.horizon_min * 60 - 1)
    events = events.dropna(subset=["t0"]).astype({"t0": np.int64}).reset_index(drop=True)
    events = events.merge(transfers, on=["from_node", "to_node"])

    # the transfer-wait model and schedule, once per (snapshot, transfer)
    pairs = events[["t0", "from_node", "to_node", "walk_sec"]].drop_duplicates(
        ["t0", "from_node", "to_node"]).reset_index(drop=True)
    typical = pd.read_csv(args.wait_models / "typical_headway.csv")
    frame = waits.build_transfer(typical, pd.DataFrame({"node": pairs["to_node"], "t": pairs["t0"]}),
                                 pairs["from_node"], pairs["walk_sec"])
    pairs["wait_model"] = pairs["walk_sec"] + QuantileGBM.load(
        args.wait_models / "transfer").predict(frame)
    periods = {t0: state_of_time(datetime.fromtimestamp(int(t0), TZ)) for t0 in pairs["t0"].unique()}
    base = {(f, t): b for (f, t, x), b in zip(network.edge_keys, network.base) if x}
    pairs["schedule"] = [graph.edge_time(f, t, True, base[(f, t)], periods[t0])
                         for f, t, t0 in zip(pairs["from_node"], pairs["to_node"], pairs["t0"])]
    names = ["schedule", "wait_model"]
    for spec in args.edge_costs:
        name, _, path = spec.partition("=")
        preds = pd.read_csv(path)
        if "is_transfer" not in preds or not preds["is_transfer"].astype(bool).any():
            continue
        preds = preds[preds["is_transfer"].astype(bool)].rename(columns={"pred": name})
        pairs = pairs.merge(preds[["t0", "from_node", "to_node", name]], how="left",
                            on=["t0", "from_node", "to_node"])
        names.append(name)
    scored = events.merge(pairs.drop(columns="walk_sec"), on=["t0", "from_node", "to_node"])
    scored = scored[np.isfinite(scored[names]).all(axis=1)].reset_index(drop=True)
    target = scored["edge_sec"].to_numpy(dtype=float)
    results = {n: evaluate(scored[n].to_numpy(), target, q) for n in names}
    ref_loss = pinball(scored["wait_model"].to_numpy(), target, q)
    paired = []
    for n in names:
        if n == "wait_model":
            continue
        diff = pinball(scored[n].to_numpy(), target, q) - ref_loss
        lo, hi = bootstrap_mean(diff, scored["t0"].to_numpy())
        paired.append({"model": n, "pinball vs wait_model": f"{diff.mean():+.2f}",
                       "95% CI": f"{lo:+.2f} to {hi:+.2f}",
                       "relative": f"{diff.mean() / ref_loss.mean():+.1%}"})
    local = pd.to_datetime(scored["t0"], unit="s", utc=True).dt.tz_convert(TZ)
    slices = {"weekday": (local.dt.dayofweek < 5).to_numpy(),
              "weekend": (local.dt.dayofweek >= 5).to_numpy(),
              "rush hour (7-10, 16-19)": local.dt.hour.isin(RUSH_HOURS).to_numpy(),
              "overnight (0-5)": local.dt.hour.between(0, 5).to_numpy(),
              "walk 0 (same platform area)": (scored["walk_sec"] == 0).to_numpy()}
    slice_rows = [{"slice": s_name, "rows": int(m.sum()),
                   **{n: "{pinball:.1f} / {coverage:.2f}".format(
                       **evaluate(scored[n].to_numpy()[m], target[m], q)) for n in names}}
                  for s_name, m in slices.items() if m.any()]
    summary = {"events": len(scored), "results": results, "paired": paired}
    sections = [
        "## Transfers: walk + wait\n\n"
        f"{len(scored):,} transfers taken off {args.transfer_fraction:.0%} of test arrivals "
        f"within {args.horizon_min} min of a snapshot, each labelled walk + wait for the next "
        "train on the far platform. `wait_model` is the walk plus the transfer-wait model "
        f"(`{args.wait_models}`) with features at the snapshot; `schedule` is the graph's "
        "walk + half-headway cost. Seconds per transfer.\n\n"
        + format_table([{"model": n, **results[n]} for n in names],
                       ["model", "rows", "pinball", "coverage", "mae"]),
        "Paired against `wait_model`, CIs resampling snapshot times:\n\n"
        + format_table(paired, ["model", "pinball vs wait_model", "95% CI", "relative"]),
        "By slice, pinball / coverage:\n\n" + format_table(slice_rows, ["slice", "rows", *names]),
    ]
    return summary, sections


if __name__ == "__main__":
    main()
