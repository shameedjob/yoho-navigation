"""Row prep for the canonical feature set, ml_model.features.MODEL_FEATURES.

QuantileGBM is feature-agnostic; this is the one place that turns a training
row (scripts/training_data.py) or a live snapshot row (snapshot/build.py) into
exactly the columns MODEL_FEATURES names. How each feature is derived on the
two sides is docs/FEATURE_PARITY.md's job.

What prep does beyond selecting columns:

  graph columns   training rows don't store route, service_period or
                  graph_edge_sec; they are joined from the graph with the same
                  code a snapshot uses (snapshot.build.add_graph_columns).
  alert types     `station_alert_types` is "" when no alert is live, as both
                  sides write it, but "" reads back null through a CSV round
                  trip. Filled back to "" so "no alert" is one category, the
                  same one a live snapshot sends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from graph import Graph
from ml_model.features import MODEL_FEATURES
from snapshot.build import add_graph_columns

SPEC = MODEL_FEATURES

GRAPH_COLUMNS = ("route", "service_period", "graph_edge_sec")


def prepare(rows: pd.DataFrame, graph: Graph | None = None) -> pd.DataFrame:
    """A copy of `rows` with every SPEC column present.

    Rows lacking the graph columns (training rows) need `graph`; snapshot rows
    already carry them.
    """
    missing = [c for c in GRAPH_COLUMNS if c not in rows.columns]
    if missing:
        if graph is None:
            raise ValueError(f"rows lack {missing}; pass the graph to join them")
        rows = add_graph_columns(rows, graph)
    else:
        rows = rows.copy()
    rows["station_alert_types"] = rows["station_alert_types"].fillna("")
    return rows


def predict_snapshot(model, edges: pd.DataFrame) -> np.ndarray:
    """Predictions in the row order of a GraphSnapshot's edges, for
    GraphSnapshot.weighted_graph.

    Transfer rows come back NaN, keeping their headway-based cost: every
    training row is a ride, so a prediction for a transfer would be the model
    guessing outside anything it has seen.
    """
    pred = model.predict(prepare(edges))
    return np.where(edges["is_transfer"].astype(bool).to_numpy(), np.nan, pred)
