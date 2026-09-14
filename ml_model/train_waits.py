"""Train the origin- and transfer-wait models and benchmark them.

    python3 -m ml_model.train_waits --data data/training/training_data_2026-01.csv

Same date split as benchmark_gbm (train, --valid-days for early stopping, last
--test-days reported), so the wait models never see the week the path
benchmark replays. See ml_model/waits.py for labels and features.

Baselines on the test week:

  origin    zero             what the router charges today
            typical / 2      half the node's typical headway for that hour
            node_hour_q      q-quantile of training waits per (node, weekend, hour)
  transfer  headway / 2      today's transfer cost minus its walk
            node_hour_q      as above, on transfer samples

Writes the models and the typical-headway table they need to --out, and a
report to --report.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from graph.service_states import states_of
from graph.subway_loader import build_subway_graph
from ml_model import waits
from ml_model.benchmark_gbm import format_table
from ml_model.replay import split_by_date
from ml_model.replay import TZ, Network
from ml_model.gbm import QuantileGBM
from ml_model.metrics import evaluate


def span_bounds(dates: list[str]) -> tuple[int, int]:
    start = pd.Timestamp(dates[0], tz=TZ)
    end = pd.Timestamp(dates[-1], tz=TZ) + pd.Timedelta(days=1)
    return int(start.timestamp()), int(end.timestamp())


def node_hour_quantile(train: pd.DataFrame, test: pd.DataFrame, q: float) -> np.ndarray:
    keys = ["node", "is_weekend", "hour"]
    table = train.groupby(keys)["wait_sec"].quantile(q).rename("value").reset_index()
    fine = test[keys].merge(table, how="left", on=keys)["value"]
    coarse = test[["node"]].merge(train.groupby("node")["wait_sec"].quantile(q)
                                  .rename("value").reset_index(), how="left", on="node")["value"]
    return fine.fillna(coarse).fillna(train["wait_sec"].quantile(q)).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument("--valid-days", type=int, default=4)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--origin-step-sec", type=int, default=600)
    parser.add_argument("--transfer-fraction", type=float, default=0.1)
    parser.add_argument("--num-boost-round", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("ml_model/checkpoints/waits"))
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    q, rng = args.quantile, np.random.default_rng(args.seed)

    gtfs = args.data_dir / "gtfs_subway"
    graph = build_subway_graph(gtfs)
    network = Network(graph, gtfs)
    edge_cost = {(f, t): b for (f, t, x), b in zip(network.edge_keys, network.base) if not x}
    transfers = pd.DataFrame({
        "from_node": [k[0] for k, x in zip(network.edge_keys, network.is_transfer) if x],
        "to_node": [k[1] for k, x in zip(network.edge_keys, network.is_transfer) if x],
        "walk_sec": network.walk[network.is_transfer],
    })

    rows = pd.concat([pd.read_csv(p, low_memory=False) for p in args.data], ignore_index=True)
    rows = rows[rows["edge_sec"].notna()].reset_index(drop=True)
    _, _, _, spans = split_by_date(rows, args.valid_days, args.test_days)
    deps = waits.departures(rows, edge_cost)
    in_train = pd.to_datetime(deps["dep"], unit="s", utc=True).dt.tz_convert(TZ) \
        .dt.strftime("%Y-%m-%d").isin(spans["train"])
    typical = waits.typical_headways(deps[in_train])
    print(f"{len(deps):,} departures; spans {({k: (v[0], v[-1]) for k, v in spans.items()})}",
          flush=True)

    samples = {}
    for name, dates in spans.items():
        span_rows = rows[rows["service_date"].isin(dates)]
        # One day at a time: sampled dates needn't be consecutive, and a grid
        # over the gaps would be millions of samples with no trains.
        samples[("origin", name)] = pd.concat([
            waits.origin_samples(deps, typical, *span_bounds([date]), args.origin_step_sec, rng)
            for date in dates], ignore_index=True)
        samples[("transfer", name)] = waits.transfer_samples(
            span_rows, deps, typical, transfers, args.transfer_fraction, rng)
        print(f"{name}: {len(samples[('origin', name)]):,} origin, "
              f"{len(samples[('transfer', name)]):,} transfer samples", flush=True)

    results, gains = {}, {}
    for kind, spec in (("origin", waits.ORIGIN_FEATURES), ("transfer", waits.TRANSFER_FEATURES)):
        train, valid, test = (samples[(kind, s)] for s in ("train", "valid", "test"))
        model = QuantileGBM(spec, q).fit(train, valid, num_boost_round=args.num_boost_round,
                                          early_stopping_rounds=100)
        model.save(args.out / kind)
        target = test["wait_sec"].to_numpy(dtype=float)
        preds = {"model": model.predict(test),
                 "node_hour_q": node_hour_quantile(train, test, q),
                 "typical / 2": test["typical_headway_sec"].to_numpy(dtype=float) / 2}
        if kind == "origin":
            preds["zero (today)"] = np.zeros(len(test))
        else:
            _, period = states_of(test["t"])
            base = {(f, t): b for (f, t, x), b in zip(network.edge_keys, network.base) if x}
            today = [graph.edge_time(f, t, True, base[(f, t)], p)
                     for f, t, p in zip(test["from_node"], test["node"], period)]
            preds["headway / 2 (today)"] = np.array(
                [c if c != math.inf else np.nan for c in today]) - test["walk_sec"].to_numpy()
        results[kind] = {
            "rows": {s: int(len(samples[(kind, s)])) for s in ("train", "valid", "test")},
            "best_iteration": model.booster.best_iteration,
            "mean_actual_wait": float(target.mean()),
            "methods": {n: evaluate(np.nan_to_num(p, nan=np.nanmedian(p)), target, q)
                        for n, p in preds.items()},
        }
        gain = pd.Series(model.booster.feature_importance("gain"), index=model.booster.feature_name())
        gains[kind] = (gain / gain.sum()).sort_values(ascending=False).round(4).to_dict()
    typical.to_csv(args.out / "typical_headway.csv", index=False)
    (args.out / "meta.json").write_text(json.dumps(
        {"quantile": q, "spans": {k: [v[0], v[-1]] for k, v in spans.items()}}, indent=2))

    report = args.report or Path(f"docs/benchmarks/waits_{spans['test'][0]}_{spans['test'][-1]}.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    summary = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "quantile": q, "spans": {k: [v[0], v[-1]] for k, v in spans.items()},
               "results": results, "gain_share": gains}
    report.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    sections = [f"# Wait models, test {spans['test'][0]} to {spans['test'][-1]}",
                f"Generated by `python3 -m ml_model.train_waits` on {summary['created']}. "
                f"Quantile {q}. Train {spans['train'][0]}..{spans['train'][-1]}, early stopping on "
                f"{spans['valid'][0]}..{spans['valid'][-1]}. Wait in seconds; coverage is the share "
                "of real waits at or under the prediction."]
    for kind, label in (("origin", "Origin wait: rider arrives at a platform"),
                        ("transfer", "Transfer wait: rider steps off a train and walks over")):
        r = results[kind]
        sections.append(
            f"## {label}\n\n{r['rows']['test']:,} test samples, mean actual wait "
            f"{r['mean_actual_wait']:.0f}s. Model stopped at iteration {r['best_iteration']}.\n\n"
            + format_table([{"method": n, **m} for n, m in r["methods"].items()],
                           ["method", "rows", "pinball", "coverage", "mae"])
            + "\n\nFeature gain share: "
            + ", ".join(f"{f} {v:.2f}" for f, v in gains[kind].items() if v >= 0.01))
    report.write_text("\n\n".join(sections) + "\n")
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
