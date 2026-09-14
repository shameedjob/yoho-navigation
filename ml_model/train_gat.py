"""Train the temporal GAT or Graph WaveNet edge model and write its predictions for the path benchmark.

    python3 -m ml_model.train_gat --data data/training/training_data_2026-sample30.csv \\
        --checkpoint ml_model/checkpoints/gat_2026-sample30/gat.pt

    python3 -m ml_model.train_gat --architecture gwnet --data ... \\
        --checkpoint ml_model/checkpoints/gwnet_2026-sample30/gwnet.pt

--architecture gat (default) is ml_model/gat.py: temporal attention over each
edge's window with a cyclical time-of-day / day-of-week encoding, then GATv2
over the line graph. gwnet is ml_model/graph_wavenet.py: gated dilated
temporal convolutions and diffusion over the line graph plus a learned
adjacency, with the same time encoding as input channels.

Everything else matches ml_model/train_rgnn.py so the two are comparable: the
same snapshot feature grid, feature set (RGNN_FEATURES), labels, date split,
per-day padding, and prediction CSV format for

    python3 -m ml_model.benchmark_paths ... --edge-costs gat=<predictions-out>

Run in its own process (torch; see train_rgnn on OMP Error #15).
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
from ml_model.features import EDGE_KEY
from ml_model.gat import GATConfig, GATForecaster
from ml_model import waits
from ml_model.graph_wavenet import GWNetConfig, GWNetForecaster
from ml_model.line_graph import build_line_graph
from ml_model.metrics import evaluate
from ml_model.model import pinball_loss
from ml_model.replay import TZ, Network, departure_times, split_by_date, test_dates
from ml_model.sequence import EdgeSequenceDataset
from ml_model.train_rgnn import RGNN_FEATURES, build_days, fit_encoder


def step_ends(offsets: dict, num_steps: int, step_sec: int, window: int) -> torch.Tensor:
    """Unix time each row of a build_days tensor ends at, padding included."""
    ends = np.zeros(num_steps, dtype=np.int64)
    for first, times in offsets.values():
        lo = first - (window - 1)
        ends[lo:first + len(times)] = times[0] + step_sec * np.arange(-(window - 1), len(times))
    return torch.from_numpy(ends)


class TimedDataset(EdgeSequenceDataset):
    """EdgeSequenceDataset that also yields each window's origin index."""

    def __getitem__(self, i: int):
        return (*super().__getitem__(i), int(self.origins[i]))

    @staticmethod
    def collate(batch):
        return (*EdgeSequenceDataset.collate([item[:4] for item in batch]),
                torch.tensor([item[4] for item in batch]))


def transfer_labels(network, transfer_keys, fraction, rng):
    """extra_labels for build_days: for a sampled share of the day's train
    arrivals at a node with transfers, each transfer's walk plus the wait for
    the next train on the far platform (waits.platform_state, the transfer-wait
    model's label), as a row departing when the rider stepped off."""
    wanted = set(transfer_keys)
    idx = [e for e, key in enumerate(network.edge_keys) if key in wanted]
    transfers = pd.DataFrame({"from_node": [network.edge_keys[e][0] for e in idx],
                              "to_node": [network.edge_keys[e][1] for e in idx],
                              "walk_sec": network.walk[idx]})
    edge_cost = {(f, t): b for (f, t, x), b in zip(network.edge_keys, network.base) if not x}

    def labels(day) -> pd.DataFrame:
        deps = waits.departures(day.rows, edge_cost)
        today = day.rows[day.rows["service_date"] == day.date]
        out = waits.transfer_cost_labels(today, deps, transfers, fraction, rng)
        return out[["from_node", "to_node", "is_transfer", "ts", "edge_sec"]]

    return labels


def fit_target_scaling(forecaster, targets) -> None:
    """Per-node label mean/std, one pair for rides and one for transfers."""
    model, rides = forecaster.model, forecaster.num_rides
    is_transfer = targets.edge >= rides
    for mask, nodes in ((~is_transfer, slice(0, rides)), (is_transfer, slice(rides, None))):
        model.target_mean[nodes] = float(targets.value[mask].mean())
        model.target_std[nodes] = float(targets.value[mask].std())


def run_epoch(forecaster, loader, ends, quantile, optimizer=None):
    forecaster.model.train(optimizer is not None)
    rides = getattr(forecaster, "num_rides", None)
    losses, preds, values, edges = [], [], [], []
    with torch.set_grad_enabled(optimizer is not None):
        for X, sample, edge, horizon, value, origin in loader:
            if len(value) == 0:
                continue
            pred = forecaster(X, ends[origin])[sample, edge, horizon]
            loss = pinball_loss(pred, forecaster.standardize(value, edge), quantile)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(forecaster.model.parameters(), 1.0)
                optimizer.step()
            else:
                preds.append(forecaster.to_cost_units(pred, edge).clamp(min=0))
                values.append(value)
                edges.append(edge)
            losses.append(loss.item())
    if not preds:
        return float(np.mean(losses)), None
    pred, value, edge = (torch.cat(v).numpy() for v in (preds, values, edges))
    if rides is not None and (edge >= rides).any():
        scores = {f"ride_{k}": v for k, v in evaluate(pred[edge < rides], value[edge < rides],
                                                       quantile).items()}
        scores |= {f"transfer_{k}": v for k, v in evaluate(pred[edge >= rides],
                                                            value[edge >= rides], quantile).items()}
        # model selection keeps weighting the two the way the loss does
        scores["pinball"] = float(np.mean(losses))
    else:
        scores = evaluate(pred, value, quantile)
    return float(np.mean(losses)), scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--architecture", choices=["gat", "gwnet"], default="gat")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--valid-days", type=int, default=4)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument("--step-min", type=int, default=10, help="feature grid step")
    parser.add_argument("--window", type=int, default=6, help="steps of history per prediction")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--gat-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=None,
                        help="default 0.1 for gat, 0.3 for gwnet")
    parser.add_argument("--skip-channels", type=int, default=64, help="gwnet")
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 1, 2],
                        help="gwnet temporal conv dilations; receptive field 1 + sum")
    parser.add_argument("--hops", type=int, default=2, help="gwnet diffusion hops per support")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="gwnet without the learned adjacency")
    parser.add_argument("--with-transfers", action="store_true",
                        help="gwnet also prices transfer edges (walk + wait); see graph_wavenet")
    parser.add_argument("--transfer-fraction", type=float, default=0.25,
                        help="share of train arrivals labelled for transfers")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--windows-per-epoch", type=int, default=2000)
    parser.add_argument("--valid-windows", type=int, default=800)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--encoder-days", type=int, default=4)
    parser.add_argument("--benchmark-step-min", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=Path("ml_model/checkpoints/gat.pt"))
    parser.add_argument("--predictions-out", type=Path, default=None,
                        help="default: <checkpoint dir>/<architecture>_test_predictions.csv")
    args = parser.parse_args()
    if args.benchmark_step_min % args.step_min:
        parser.error("--benchmark-step-min must be a multiple of --step-min")
    if args.with_transfers and args.architecture != "gwnet":
        parser.error("--with-transfers needs --architecture gwnet")
    if args.architecture == "gat" and args.hidden % args.heads:
        parser.error("--hidden must be divisible by --heads")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    step_sec = args.step_min * 60

    def log(message: str) -> None:
        print(message, flush=True)

    gtfs = args.data_dir / "gtfs_subway"
    graph = build_subway_graph(gtfs)
    network = Network(graph, gtfs)
    line_graph = build_line_graph(graph)
    target_graph = build_line_graph(graph, include_transfers=True) if args.with_transfers \
        else line_graph
    extra = (transfer_labels(network, [k for k in target_graph.edge_keys if k[2]],
                             args.transfer_fraction, rng) if args.with_transfers else None)
    rows = pd.concat([pd.read_csv(p, low_memory=False) for p in args.data], ignore_index=True)
    rows = rows[rows["edge_sec"].notna()].reset_index(drop=True)
    _, _, _, spans = split_by_date(rows, args.valid_days, args.test_days)
    log(f"line graph {len(line_graph)} edges; spans "
        + ", ".join(f"{k} {v[0]}..{v[-1]} ({len(v)})" for k, v in spans.items()))

    encoder = fit_encoder(rows, network, spans["train"], step_sec, rng, args.encoder_days)
    log(f"encoder: {encoder.num_features} channels")
    log("train days:")
    X_train, y_train, origins_train, off_train = build_days(
        rows, network, line_graph, encoder, spans["train"], step_sec, args.window, log,
        target_graph, extra)
    ends_train = step_ends(off_train, len(X_train), step_sec, args.window)
    log("valid days:")
    X_valid, y_valid, origins_valid, off_valid = build_days(
        rows, network, line_graph, encoder, spans["valid"], step_sec, args.window, log,
        target_graph, extra)
    ends_valid = step_ends(off_valid, len(X_valid), step_sec, args.window)

    if args.architecture == "gat":
        Forecaster = GATForecaster
        config = GATConfig(window=args.window, step_sec=step_sec, quantile=args.quantile,
                           hidden=args.hidden, heads=args.heads, gat_layers=args.gat_layers,
                           dropout=0.1 if args.dropout is None else args.dropout)
    else:
        Forecaster = GWNetForecaster
        config = GWNetConfig(window=args.window, step_sec=step_sec, quantile=args.quantile,
                             channels=args.hidden, skip_channels=args.skip_channels,
                             dilations=tuple(args.dilations), hops=args.hops,
                             adaptive=not args.no_adaptive, transfers=args.with_transfers,
                             dropout=0.3 if args.dropout is None else args.dropout)
    forecaster = Forecaster.create(encoder, target_graph, config)
    if args.with_transfers:
        fit_target_scaling(forecaster, y_train)
        keys = target_graph.edge_keys[forecaster.num_rides:]
        walk = {k: w for k, w in zip(network.edge_keys, network.walk) if k[2]}
        forecaster.model.transfer_walk[:] = torch.tensor([walk.get(k, 0.0) for k in keys])
        log(f"transfer nodes {len(keys):,}; label scaling "
            f"ride {forecaster.model.target_mean[0]:.0f}±{forecaster.model.target_std[0]:.0f}s, "
            f"transfer {forecaster.model.target_mean[-1]:.0f}±{forecaster.model.target_std[-1]:.0f}s")
    log(f"parameters: {sum(p.numel() for p in forecaster.model.parameters()):,}")
    optimizer = torch.optim.AdamW(forecaster.model.parameters(), lr=args.lr)
    valid_pick = rng.choice(origins_valid, size=min(args.valid_windows, len(origins_valid)),
                            replace=False)
    valid_loader = torch.utils.data.DataLoader(
        TimedDataset(X_valid, y_valid, args.window, np.sort(valid_pick)),
        batch_size=args.batch_size, collate_fn=TimedDataset.collate)

    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        began = time.time()
        pick = rng.choice(origins_train, size=min(args.windows_per_epoch, len(origins_train)),
                          replace=False)
        train_loader = torch.utils.data.DataLoader(
            TimedDataset(X_train, y_train, args.window, pick),
            batch_size=args.batch_size, shuffle=True, collate_fn=TimedDataset.collate)
        train_loss, _ = run_epoch(forecaster, train_loader, ends_train, args.quantile, optimizer)
        valid_loss, scores = run_epoch(forecaster, valid_loader, ends_valid, args.quantile)
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
    forecaster = Forecaster.load(args.checkpoint)

    dates = test_dates(rows, args.test_days)
    log("test days:")
    X_test, _, _, offsets = build_days(rows, network, line_graph, encoder, dates, step_sec,
                                       args.window, log, target_graph, extra)
    ends_test = step_ends(offsets, len(X_test), step_sec, args.window)
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
            out = forecaster(X, ends_test[batch])[:, :, 0]
            pred = forecaster.to_cost_units(out).clamp(min=0).numpy()
            for t0, row in zip(t0s[lo:lo + args.batch_size], pred):
                frames.append(pd.DataFrame({"t0": int(t0), EDGE_KEY[0]: [k[0] for k in keys],
                                            EDGE_KEY[1]: [k[1] for k in keys],
                                            EDGE_KEY[2]: [k[2] for k in keys], "pred": row}))
    out_path = args.predictions_out or args.checkpoint.with_name(f"{args.architecture}_test_predictions.csv")
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
