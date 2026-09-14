"""Train the LightGBM edge model on snapshot rows, the way the graph models learn.

    python3 -m ml_model.train_gbm_snapshots --data data/training/training_data_2026-sample30.csv \\
        --out ml_model/checkpoints/gbm_snapshots_2026-sample30

benchmark_gbm trains on one row per traversal with features taken at that
traversal's own moment. Live, the model sees a snapshot taken before the
train leaves, so its inputs are up to a step older than in training. The graph
models (train_rgnn, train_gat) train on exactly the live shape instead: a
snapshot every --step-min, labelled with the traversals departing in the
following --horizon-min.

This builds the same thing for LightGBM: for each train/valid day, the
MODEL_FEATURES row of every ride edge at every snapshot time
(replay.Day.features, the grid the graph models use) joined to the traversals
departing within the horizon (replay.snapshot_features / snapshot_labels). A
snapshot with two trains on an edge gives two rows with the same features.
Same date split, quantile and LightGBM parameters as benchmark_gbm, so the
only difference from gbm_2026-sample30 is the row shape.

Score it with ml_model.benchmark_edges.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from graph.subway_loader import build_subway_graph
from ml_model import model_data
from ml_model.gbm import QuantileGBM
from ml_model.replay import EDGE, Day, Network, snapshot_labels, split_by_date


def snapshot_rows(rows, network, dates, step_sec, horizon_sec, log) -> pd.DataFrame:
    parts = []
    for date in dates:
        began = time.time()
        day = Day(rows, network, date, step_sec)
        features = model_data.prepare(day.features(network))
        labels = snapshot_labels(day.rows[day.rows["service_date"] == date], day.times,
                                 horizon_sec)
        joined = labels[["t0", *EDGE, "edge_sec"]].merge(features, how="inner",
                                                         on=["t0", *EDGE])
        parts.append(joined.assign(service_date=date))
        log(f"  {date}: {len(features):,} snapshot rows, {len(joined):,} labelled "
            f"[{time.time() - began:.0f}s]")
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--valid-days", type=int, default=4)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--step-min", type=int, default=10)
    parser.add_argument("--horizon-min", type=int, default=10)
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument("--num-boost-round", type=int, default=5000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--params", type=json.loads, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    def log(message: str) -> None:
        print(message, flush=True)

    gtfs = args.data_dir / "gtfs_subway"
    graph = build_subway_graph(gtfs)
    network = Network(graph, gtfs)
    rows = pd.concat([pd.read_csv(p, low_memory=False) for p in args.data], ignore_index=True)
    rows = rows[rows["edge_sec"].notna()].reset_index(drop=True)
    _, _, _, spans = split_by_date(rows, args.valid_days, args.test_days)
    step_sec, horizon_sec = args.step_min * 60, args.horizon_min * 60

    log("train days:")
    train = snapshot_rows(rows, network, spans["train"], step_sec, horizon_sec, log)
    log("valid days:")
    valid = snapshot_rows(rows, network, spans["valid"], step_sec, horizon_sec, log)
    for frame in (train, valid):
        frame["edge_sec"] = frame["edge_sec"].astype("float64")

    began = time.time()
    model = QuantileGBM(model_data.SPEC, args.quantile, args.params)
    model.fit(train, valid, num_boost_round=args.num_boost_round,
              early_stopping_rounds=args.early_stopping_rounds)
    model.save(args.out)
    log(f"stopped at iteration {model.booster.best_iteration} [{time.time() - began:.0f}s]")
    (args.out / "training.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": [str(p) for p in args.data],
        "spans": {k: [v[0], v[-1], len(v)] for k, v in spans.items()},
        "rows": {"train": len(train), "valid": len(valid)},
        "step_min": args.step_min, "horizon_min": args.horizon_min,
        "best_iteration": model.booster.best_iteration,
        "fit_seconds": round(time.time() - began),
    }, indent=2))


if __name__ == "__main__":
    main()
