"""Train the recurrent GNN edge model and write its predictions for the path benchmark.

    python3 -m ml_model.train_rgnn --data data/training/training_data_2026-sample30.csv

The model is the line-graph DCRNN (ml_model/model.py): a GRU whose gates are
diffusion graph convolutions over ride edges, so each edge's cost is predicted
from its own recent history and its neighbours' along and across lines. It is
the recurrent counterpart to the LightGBM edge model, which sees each edge at
one moment on its own.

Inputs match what LightGBM gets. Every --step-min, each subway ride edge gets
the MODEL_FEATURES row a live snapshot would show at that moment, rebuilt from
traversal rows by ml_model.replay.snapshot_features -- the same function the
path benchmark uses to feed LightGBM. The labels are the traversals departing
in the following step, scored with pinball loss at --quantile.

Days are handled independently, so the sampled dates needn't be consecutive:
each day's grid runs from its first step to midnight, and windows reaching
before the first step are padded with the encoder's "no row" vector -- the
same padding prediction uses for early departures.

Same date split as benchmark_gbm and train_waits (train, --valid-days for model
selection, last --test-days held out). After training, it predicts every
ride edge at the path benchmark's departure times on the test dates
(replay.departure_times) and writes them to --predictions-out, for

    python3 -m ml_model.benchmark_paths ... --edge-costs rgnn=<predictions-out>

Run in its own process: it uses torch, and the path benchmark uses LightGBM
(the two can't share a process on macOS, OMP Error #15).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from graph.subway_loader import build_subway_graph
from ml_model.features import EDGE_KEY, MODEL_FEATURES, FeatureSpec
from ml_model.forecaster import EdgeForecaster, ForecastConfig
from ml_model.line_graph import build_line_graph
from ml_model.metrics import evaluate
from ml_model.model import pinball_loss
from ml_model.replay import (TZ, Day, Network, departure_times, split_by_date, test_dates)
from ml_model.sequence import (EdgeFeatureEncoder, EdgeSequenceDataset, SparseTargets,
                               build_feature_tensor, build_targets)

# MODEL_FEATURES minus two categoricals that one-hot into dozens of channels
# per edge per step: route (the graph structure and graph_edge_sec already
# locate an edge) and station_alert_types (the combinations run to hundreds;
# the counts and ages stay).
RGNN_FEATURES = FeatureSpec(
    numeric=list(MODEL_FEATURES.numeric),
    categorical=[c for c in MODEL_FEATURES.categorical if c not in ("route", "station_alert_types")],
    target="edge_sec",
)


def build_days(rows, network, line_graph, encoder, dates, step_sec, window, log,
               target_graph=None, extra_labels=None):
    """Stacked [T, E, F] tensor over `dates` (each padded with window - 1 empty
    steps in front), the sparse labels re-indexed into it, and each day's
    valid origins and grid offset.

    target_graph: index labels into this line graph instead (one whose first
    nodes are line_graph's, e.g. with transfer nodes appended).
    extra_labels(day) -> more label rows (EDGE_KEY, ts, edge_sec) for the day."""
    target_graph = target_graph or line_graph
    blocks, parts, origins, offsets = [], [], [], {}
    pad = np.broadcast_to(encoder.absent(), (window - 1, len(line_graph), encoder.num_features))
    cursor = 0
    for date in dates:
        began = time.time()
        day = Day(rows, network, date, step_sec)
        X = build_feature_tensor(day.features(network), line_graph, encoder, day.times, "t0")
        labelled = day.rows[day.rows["service_date"] == date].assign(is_transfer=False)
        if extra_labels is not None:
            labelled = pd.concat([labelled[[*EDGE_KEY, "ts", "edge_sec"]], extra_labels(day)],
                                 ignore_index=True)
        targets = build_targets(labelled, target_graph, day.times, horizons=1)
        blocks.extend([pad, X])
        first = cursor + window - 1
        offsets[date] = (first, day.times)
        parts.append((targets.origin + first, targets.edge, targets.horizon, targets.value))
        origins.append(np.arange(first, first + len(day.times)))
        cursor = first + len(day.times)
        log(f"  {date}: {len(day.times)} steps, {len(targets.value):,} labels "
            f"[{time.time() - began:.0f}s]")
    X = torch.from_numpy(np.concatenate(blocks).astype(np.float32))
    origin, edge, horizon, value = (np.concatenate(p) for p in zip(*parts))
    order = np.argsort(origin, kind="stable")
    origin = origin[order]
    targets = SparseTargets(origin, edge[order], horizon[order], value[order],
                            np.searchsorted(origin, np.arange(len(X) + 1)))
    return X, targets, np.concatenate(origins), offsets


def fit_encoder(rows, network, dates, step_sec, rng, sample_days: int) -> EdgeFeatureEncoder:
    """Scaling statistics from a few training days' snapshot rows and their labels."""
    chosen = sorted(str(d) for d in rng.choice(dates, size=min(sample_days, len(dates)), replace=False))
    frames, labels = [], []
    for date in chosen:
        day = Day(rows, network, date, step_sec)
        frames.append(day.features(network))
        labels.append(day.rows.loc[day.rows["service_date"] == date, "edge_sec"])
    return EdgeFeatureEncoder(RGNN_FEATURES).fit(pd.concat(frames), pd.concat(labels))


def run_epoch(forecaster, loader, quantile, optimizer=None):
    model, enc = forecaster.model, forecaster.encoder
    model.train(optimizer is not None)
    losses, preds, values = [], [], []
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
                values.append(value)
            losses.append(loss.item())
    scores = (evaluate(torch.cat(preds).numpy(), torch.cat(values).numpy(), quantile)
              if preds else None)
    return float(np.mean(losses)), scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--valid-days", type=int, default=4)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument("--step-min", type=int, default=10, help="feature grid step")
    parser.add_argument("--window", type=int, default=6, help="steps of history per prediction")
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--k", type=int, default=2, help="diffusion hops per layer")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--windows-per-epoch", type=int, default=2000,
                        help="training windows sampled per epoch")
    parser.add_argument("--valid-windows", type=int, default=800)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--encoder-days", type=int, default=4)
    parser.add_argument("--benchmark-step-min", type=int, default=20,
                        help="the path benchmark's --step-min; departure times to predict")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=Path("ml_model/checkpoints/rgnn.pt"))
    parser.add_argument("--predictions-out", type=Path, default=None,
                        help="default: <checkpoint dir>/rgnn_test_predictions.csv")
    args = parser.parse_args()
    if args.benchmark_step_min % args.step_min:
        parser.error("--benchmark-step-min must be a multiple of --step-min")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    step_sec = args.step_min * 60

    def log(message: str) -> None:
        print(message, flush=True)

    gtfs = args.data_dir / "gtfs_subway"
    graph = build_subway_graph(gtfs)
    network = Network(graph, gtfs)
    line_graph = build_line_graph(graph)
    rows = pd.concat([pd.read_csv(p, low_memory=False) for p in args.data], ignore_index=True)
    rows = rows[rows["edge_sec"].notna()].reset_index(drop=True)
    _, _, _, spans = split_by_date(rows, args.valid_days, args.test_days)
    log(f"line graph {len(line_graph)} edges; spans "
        + ", ".join(f"{k} {v[0]}..{v[-1]} ({len(v)})" for k, v in spans.items()))

    encoder = fit_encoder(rows, network, spans["train"], step_sec, rng, args.encoder_days)
    log(f"encoder: {encoder.num_features} channels")
    log("train days:")
    X_train, y_train, origins_train, _ = build_days(rows, network, line_graph, encoder,
                                                    spans["train"], step_sec, args.window, log)
    log("valid days:")
    X_valid, y_valid, origins_valid, _ = build_days(rows, network, line_graph, encoder,
                                                    spans["valid"], step_sec, args.window, log)

    config = ForecastConfig(window=args.window, step_sec=step_sec, horizons=1,
                            quantile=args.quantile, hidden_channels=args.hidden_channels,
                            K=args.k, num_layers=args.num_layers, time_col="t0")
    forecaster = EdgeForecaster.create(encoder, line_graph, config)
    optimizer = torch.optim.Adam(forecaster.model.parameters(), lr=args.lr)
    collate = EdgeSequenceDataset.collate
    valid_pick = rng.choice(origins_valid, size=min(args.valid_windows, len(origins_valid)),
                            replace=False)
    valid_loader = torch.utils.data.DataLoader(
        EdgeSequenceDataset(X_valid, y_valid, args.window, np.sort(valid_pick)),
        batch_size=args.batch_size, collate_fn=collate)

    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        began = time.time()
        pick = rng.choice(origins_train, size=min(args.windows_per_epoch, len(origins_train)),
                          replace=False)
        train_loader = torch.utils.data.DataLoader(
            EdgeSequenceDataset(X_train, y_train, args.window, pick),
            batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        train_loss, _ = run_epoch(forecaster, train_loader, args.quantile, optimizer)
        valid_loss, scores = run_epoch(forecaster, valid_loader, args.quantile)
        history.append({"epoch": epoch, "train_pinball": train_loss, "valid_pinball": valid_loss,
                        **{f"valid_{k}": v for k, v in (scores or {}).items()}})
        log(f"epoch {epoch:2d}  train {train_loss:.4f}  valid {valid_loss:.4f}  "
            + "  ".join(f"{k} {v:.4g}" for k, v in (scores or {}).items())
            + f"  [{time.time() - began:.0f}s]")
        if valid_loss < best:
            best, stale = valid_loss, 0
            forecaster.save(args.checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                log(f"no improvement for {args.patience} epochs; stopping")
                break
    del X_train, y_train
    forecaster = EdgeForecaster.load(args.checkpoint)

    # Predictions at the path benchmark's departure times, from the best checkpoint.
    dates = test_dates(rows, args.test_days)
    log("test days:")
    X_test, _, _, offsets = build_days(rows, network, line_graph, encoder, dates, step_sec,
                                       args.window, log)
    t0s = departure_times(rows, dates, args.benchmark_step_min)
    index = []
    for t0 in t0s:
        date = datetime.fromtimestamp(int(t0), TZ).strftime("%Y-%m-%d")
        first, times = offsets[date]
        index.append(first + int(np.searchsorted(times, t0)))
    keys = forecaster.line_graph.edge_keys
    frames = []
    forecaster.model.eval()
    with torch.no_grad():
        for lo in range(0, len(t0s), args.batch_size):
            batch = index[lo:lo + args.batch_size]
            X = torch.stack([X_test[i - args.window + 1:i + 1] for i in batch])
            out = forecaster.model(X, forecaster.edge_index, forecaster.edge_weight)[:, :, 0]
            pred = forecaster.to_cost_units(out).clamp(min=0).numpy()
            for t0, row in zip(t0s[lo:lo + args.batch_size], pred):
                frames.append(pd.DataFrame({"t0": int(t0), EDGE_KEY[0]: [k[0] for k in keys],
                                            EDGE_KEY[1]: [k[1] for k in keys], "pred": row}))
    out_path = args.predictions_out or args.checkpoint.with_name("rgnn_test_predictions.csv")
    pd.concat(frames).round({"pred": 1}).to_csv(out_path, index=False)
    log(f"wrote {len(t0s)} departure times x {len(keys)} edges to {out_path}")
    args.checkpoint.with_suffix(".json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spans": {k: [v[0], v[-1], len(v)] for k, v in spans.items()},
        "test_dates": dates, "config": vars(args) | {"data": [str(p) for p in args.data],
                                                     "data_dir": str(args.data_dir),
                                                     "checkpoint": str(args.checkpoint),
                                                     "predictions_out": str(out_path)},
        "features": RGNN_FEATURES.to_dict(), "history": history, "best_valid_pinball": best,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
