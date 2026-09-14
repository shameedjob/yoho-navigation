"""Train the quantile GBM on per-edge rows.

    python3 -m ml_model.train_gbm --data training_data.csv --feature-set model_data

    python3 -m ml_model.train_gbm --data training_data.csv \\
        --numeric sched_edge_sec prior_delay_sec minute_of_day dow \\
        --categorical route_id direction

`--feature-set model_data` uses ml_model.features.MODEL_FEATURES, the set a
live snapshot serves, with the row prep in ml_model/model_data.py.
Otherwise features are whatever columns are named on the command line.

Validation holds out the *last* values of --split-col
(service dates by default) rather than a random sample: rows from one day
share that day's disruptions, so a random split leaks them across the line and
flatters the score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from graph.subway_loader import build_subway_graph
from ml_model import model_data
from ml_model.features import FeatureSpec
from ml_model.gbm import QuantileGBM
from ml_model.metrics import evaluate


def time_split(frame: pd.DataFrame, split_col: str,
               valid_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the earliest values of `split_col`, validate on the rest.
    At least one value goes to each side.
    """
    values = sorted(frame[split_col].dropna().unique())
    if len(values) < 2:
        raise ValueError(f"need at least 2 distinct {split_col} values to split, "
                         f"got {len(values)}")
    n_valid = min(max(1, round(len(values) * valid_fraction)), len(values) - 1)
    valid_values = set(values[-n_valid:])
    is_valid = frame[split_col].isin(valid_values)
    return frame[~is_valid], frame[is_valid]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+", required=True,
                        help="CSV files of per-edge rows, concatenated")
    parser.add_argument("--feature-set", choices=["model_data"],
                        help="a predefined feature set instead of --numeric/--categorical")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="for the subway graph that --feature-set model_data joins")
    parser.add_argument("--numeric", nargs="*", default=[])
    parser.add_argument("--categorical", nargs="*", default=[])
    parser.add_argument("--target", default="edge_sec")
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument("--split-col", default="service_date")
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--num-boost-round", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--params", type=json.loads, default=None,
                        help='LightGBM overrides as JSON, e.g. \'{"num_leaves": 31}\'')
    parser.add_argument("--out", type=Path, default=Path("ml_model/checkpoints/gbm"))
    args = parser.parse_args()

    frame = pd.concat([pd.read_csv(path, low_memory=False) for path in args.data],
                      ignore_index=True)
    if args.feature_set == "model_data":
        if args.numeric or args.categorical:
            parser.error("--feature-set replaces --numeric/--categorical")
        spec = model_data.SPEC
        frame = model_data.prepare(frame, build_subway_graph(args.data_dir / "gtfs_subway"))
    else:
        spec = FeatureSpec(numeric=args.numeric, categorical=args.categorical,
                           target=args.target)
    spec.check(frame, need_target=True)
    train, valid = time_split(frame, args.split_col, args.valid_fraction)
    print(f"rows: {len(train)} train, {len(valid)} valid")

    model = QuantileGBM(spec, args.quantile, args.params)
    model.fit(train, valid, num_boost_round=args.num_boost_round,
              early_stopping_rounds=args.early_stopping_rounds)

    target = pd.to_numeric(valid[spec.target]).astype("float64")
    scores = evaluate(model.predict(valid), target, args.quantile)
    print("valid: " + "  ".join(f"{k} {v:.4g}" for k, v in scores.items()))

    model.save(args.out)
    print(f"saved model to {args.out}")


if __name__ == "__main__":
    main()
