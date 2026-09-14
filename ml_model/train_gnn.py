"""Train the line-graph DCRNN to forecast a quantile of per-edge cost.

    python3 -m ml_model.train_gnn \\
        --features snapshots.csv --traversals training_data.csv \\
        --numeric obs_median_edge_sec obs_last_age_sec --categorical route

Features are whatever columns are named on the command line (see
ml_model/sequence.py for how rows become a time grid). The per-edge feature
history doesn't exist yet, so --synthetic fabricates random frames over the
real subway line graph, with a target that depends on the last step of one
feature -- enough to see the loop learn something end to end. Do not keep a
model trained that way.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from graph.subway_loader import build_subway_graph
from ml_model.features import EDGE_KEY, FeatureSpec
from ml_model.forecaster import EdgeForecaster, ForecastConfig
from ml_model.line_graph import LineGraph, build_line_graph
from ml_model.metrics import evaluate
from ml_model.model import pinball_loss
from ml_model.sequence import (EdgeFeatureEncoder, EdgeSequenceDataset,
                               build_feature_tensor, build_targets,
                               split_origins, time_grid)


def make_synthetic_frames(line_graph: LineGraph, num_steps: int, step_sec: int,
                          num_features: int, seed: int = 0):
    """(features, traversals, numeric columns). Every edge gets a row per
    step with ~20% of values missing; each step each edge has a 30% chance of
    a traversal departing in the next step, costing 90s plus 20s per unit of
    that edge's latest f0 plus noise."""
    rng = np.random.default_rng(seed)
    E = len(line_graph)
    columns = [f"f{i}" for i in range(num_features)]
    start = 1_735_707_600  # 2025-01-01 00:00 New York, a multiple of 300

    t = np.repeat(np.arange(num_steps), E)
    e = np.tile(np.arange(E), num_steps)
    keys = np.array(line_graph.edge_keys, dtype=object)
    values = rng.standard_normal((len(t), num_features))
    features = pd.DataFrame(values, columns=columns)
    features.insert(0, "snapshot_ts", start + t * step_sec)
    for i, name in enumerate(EDGE_KEY):
        features.insert(1 + i, name, keys[e, i])
    signal = values[:, 0].copy()
    features[columns] = features[columns].mask(rng.random(values.shape) < 0.2)

    ride = rng.random(len(t)) < 0.3
    depart = start + t[ride] * step_sec + rng.integers(0, step_sec, ride.sum())
    edge_sec = np.clip(90 + 20 * signal[ride] + rng.gamma(2, 10, ride.sum()), 20, 1200).round()
    traversals = pd.DataFrame({name: keys[e[ride], i] for i, name in enumerate(EDGE_KEY)})
    traversals["edge_sec"] = edge_sec
    traversals["ts"] = depart + edge_sec.astype(np.int64)
    return features, traversals, columns


def run_epoch(forecaster: EdgeForecaster, loader, quantile: float,
              optimizer: torch.optim.Optimizer | None):
    """Mean training loss over the epoch, plus the (pred, target) pairs in
    seconds when evaluating."""
    model, enc = forecaster.model, forecaster.encoder
    model.train(optimizer is not None)
    losses, preds, targets = [], [], []
    with torch.set_grad_enabled(optimizer is not None):
        for X, sample, edge, horizon, value in loader:
            if len(value) == 0:
                continue
            out = model(X, forecaster.edge_index, forecaster.edge_weight)
            pred = out[sample, edge, horizon]
            loss = pinball_loss(pred, (value - enc.target_mean) / enc.target_std, quantile)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                preds.append(forecaster.to_cost_units(pred).clamp(min=0))
                targets.append(value)
            losses.append(loss.item())
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    if optimizer is not None or not preds:
        return mean_loss, None
    return mean_loss, evaluate(torch.cat(preds).numpy(), torch.cat(targets).numpy(), quantile)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--features", type=Path, nargs="*", default=[])
    parser.add_argument("--traversals", type=Path, nargs="*", default=[])
    parser.add_argument("--numeric", nargs="*", default=[])
    parser.add_argument("--categorical", nargs="*", default=[])
    parser.add_argument("--target", default="edge_sec")
    parser.add_argument("--time-col", default="snapshot_ts", help="feature row timestamp")
    parser.add_argument("--traversal-time-col", default="ts")
    parser.add_argument("--duration-col", default="edge_sec",
                        help="subtracted from --traversal-time-col to get departure; "
                             "'' if that column is already departure")
    parser.add_argument("--synthetic", type=int, default=0, metavar="STEPS",
                        help="train on this many steps of random frames instead")
    parser.add_argument("--synthetic-features", type=int, default=4)
    parser.add_argument("--step-sec", type=int, default=300)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizons", type=int, default=6)
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--k", type=int, default=2, help="DCRNN diffusion step count")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--checkpoint", type=Path, default=Path("ml_model/checkpoints/dcrnn.pt"))
    args = parser.parse_args()

    graph = build_subway_graph(args.data_dir / "gtfs_subway")
    line_graph = build_line_graph(graph)
    print(f"line graph: {len(line_graph)} edges, {line_graph.edge_index.shape[1]} links")

    if args.synthetic:
        features, traversals, numeric = make_synthetic_frames(
            line_graph, args.synthetic, args.step_sec, args.synthetic_features)
        spec = FeatureSpec(numeric=numeric, target="edge_sec")
        time_col, duration_col = "snapshot_ts", "edge_sec"
    else:
        if not args.features or not args.traversals:
            parser.error("--features and --traversals are required without --synthetic")
        features = pd.concat([pd.read_csv(p) for p in args.features], ignore_index=True)
        traversals = pd.concat([pd.read_csv(p) for p in args.traversals], ignore_index=True)
        spec = FeatureSpec(numeric=args.numeric, categorical=args.categorical,
                           target=args.target)
        time_col, duration_col = args.time_col, args.duration_col or None
        spec.check(features)

    times = time_grid(int(features[time_col].min()), int(features[time_col].max()),
                      args.step_sec)
    train_origins, valid_origins = split_origins(len(times), args.window,
                                                 args.horizons, args.valid_fraction)

    # Scaling statistics come from the training span only.
    train_end = times[train_origins[-1]]
    departs = traversals[args.traversal_time_col]
    if duration_col is not None:
        departs = departs - traversals[duration_col]
    encoder = EdgeFeatureEncoder(spec).fit(
        features[features[time_col] <= train_end],
        traversals.loc[departs < train_end + args.horizons * args.step_sec, spec.target])

    X = torch.from_numpy(build_feature_tensor(features, line_graph, encoder, times, time_col))
    targets = build_targets(traversals, line_graph, times, args.horizons, spec.target,
                            args.traversal_time_col, duration_col)
    print(f"grid: {len(times)} steps x {len(line_graph)} edges x {encoder.num_features} "
          f"channels; {len(targets.value)} labels")
    print(f"windows: {len(train_origins)} train, {len(valid_origins)} valid")

    def loader(origins, shuffle):
        return torch.utils.data.DataLoader(
            EdgeSequenceDataset(X, targets, args.window, origins),
            batch_size=args.batch_size, shuffle=shuffle,
            collate_fn=EdgeSequenceDataset.collate)

    config = ForecastConfig(window=args.window, step_sec=args.step_sec,
                            horizons=args.horizons, quantile=args.quantile,
                            hidden_channels=args.hidden_channels, K=args.k,
                            num_layers=args.num_layers, time_col=time_col)
    forecaster = EdgeForecaster.create(encoder, line_graph, config)
    optimizer = torch.optim.Adam(forecaster.model.parameters(), lr=args.lr)
    train_loader, valid_loader = loader(train_origins, True), loader(valid_origins, False)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(forecaster, train_loader, args.quantile, optimizer)
        valid_loss, scores = run_epoch(forecaster, valid_loader, args.quantile, None)
        detail = "  ".join(f"{k} {v:.4g}" for k, v in (scores or {}).items())
        print(f"epoch {epoch:3d}  train pinball {train_loss:.4f}  "
              f"valid pinball {valid_loss:.4f}  ({detail})")
        if valid_loss < best:
            best = valid_loss
            forecaster.save(args.checkpoint)

    print(f"saved best checkpoint (valid pinball {best:.4f}) to {args.checkpoint}")


if __name__ == "__main__":
    main()
