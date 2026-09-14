"""Wait labels and features, built from observed traversals.

Ride edges are priced by the edge model; what the router doesn't price is
waiting. Two waits matter (see docs/benchmarks/paths_*.md, where the missing
wait is the largest planned-vs-actual gap):

  origin wait    a rider reaches a platform at t and waits for the next train.
                 Priced at routing time, which *is* t, so the platform's live
                 state (time since the last train, where the next one is) is
                 fair to use.
  transfer wait  a rider steps off one train, walks, and waits on another
                 platform. Priced when the route is planned, often half an
                 hour before the rider gets there, so only what is knowable
                 in advance is used: route pair, time of day, walk, typical
                 headway. Labelled from real arrivals, so timed cross-platform
                 transfers and bunching show up in the labels.

Both come from training rows (scripts/training_data.py), the same traversals
the path benchmark replays: a departure from node n at time d is a row with
from_node == n and ts - edge_sec == d (the train's arrival at from_node, when
a rider boards). Waits are per node, i.e. the next train on that platform and
route; at the few nodes where a route branches, the next train may not go the
rider's way.

The live service computes the same features from LiveState in
snapshot/waits.py and serves them at /waits and /waits/transfers; how the two
sides differ is in docs/FEATURE_PARITY.md ("Waits"). Change both together.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ml_model.features import FeatureSpec

TZ = ZoneInfo("America/New_York")
MAX_WAIT_SEC = 3600  # a platform with no train in an hour has no service, not a wait
FALLBACK_EDGE_SEC = 90.0

ORIGIN_FEATURES = FeatureSpec(
    numeric=["age_since_last_sec", "last_headway_sec", "prev_headway_sec",
             "typical_headway_sec", "next_eta_sched_sec", "next_train_delay_sec",
             "hour", "minute_of_day", "dow", "is_weekend"],
    categorical=["route", "direction"],
    target="wait_sec",
)

TRANSFER_FEATURES = FeatureSpec(
    numeric=["walk_sec", "typical_headway_sec", "hour", "minute_of_day", "dow", "is_weekend"],
    categorical=["route", "direction", "from_route"],
    target="wait_sec",
)


def route_of(node: pd.Series) -> pd.Series:
    return node.str.split("::").str[-1]


def direction_of(node: pd.Series) -> pd.Series:
    return node.str.split("::").str[0].str[-1]


def calendar(t: pd.Series) -> pd.DataFrame:
    local = pd.to_datetime(t, unit="s", utc=True).dt.tz_convert(TZ)
    return pd.DataFrame({
        "hour": local.dt.hour, "minute_of_day": local.dt.hour * 60 + local.dt.minute,
        "dow": local.dt.dayofweek, "is_weekend": (local.dt.dayofweek >= 5).astype(int),
    }, index=t.index)


def departures(rows: pd.DataFrame, edge_cost: dict[tuple[str, str], float]) -> pd.DataFrame:
    """One row per train departure from a node, with its run and how far along
    its run's *schedule* it is (sched_cum: scheduled seconds from the run's
    first observed stop to this node). Rows must be reset-indexed traversals.

    Scheduled time per edge is the graph's cost, else the row's
    sched_edge_sec, else FALLBACK_EDGE_SEC -- all known ahead of time.
    """
    from ml_model.replay import journey_ids

    run, position = journey_ids(rows)
    graph = pd.Series([edge_cost.get(k) for k in zip(rows["from_node"], rows["to_node"])],
                      index=rows.index, dtype="float64")
    step = graph.fillna(pd.to_numeric(rows["sched_edge_sec"], errors="coerce")) \
        .fillna(FALLBACK_EDGE_SEC)
    deps = pd.DataFrame({
        "node": rows["from_node"].to_numpy(),
        "dep": (rows["ts"] - rows["edge_sec"]).to_numpy(dtype=np.int64),
        "run": run, "position": position, "step": step.to_numpy(),
        "delay": pd.to_numeric(rows["prior_delay_sec"], errors="coerce").to_numpy(),
    })
    deps = deps.sort_values(["run", "position"], kind="stable")
    deps["sched_cum"] = deps.groupby("run")["step"].cumsum() - deps["step"]
    deps = deps.sort_values(["node", "dep"], kind="stable").reset_index(drop=True)
    by_node = deps.groupby("node")["dep"]
    deps["headway"] = deps["dep"] - by_node.shift(1)
    deps["prev_headway"] = by_node.shift(1) - by_node.shift(2)
    return deps


def typical_headways(deps: pd.DataFrame) -> pd.DataFrame:
    """Median headway per (node, is_weekend, hour) -- fit on training days only
    and saved with the models, since it stands in for a timetable."""
    cal = calendar(deps["dep"])
    frame = deps[["node"]].assign(is_weekend=cal["is_weekend"], hour=cal["hour"],
                                  headway=deps["headway"])
    frame = frame[frame["headway"].between(30, MAX_WAIT_SEC)]
    return (frame.groupby(["node", "is_weekend", "hour"])["headway"].median()
            .rename("typical_headway_sec").reset_index())


def with_typical_headway(queries: pd.DataFrame, typical: pd.DataFrame) -> pd.DataFrame:
    return queries.merge(typical, how="left", on=["node", "is_weekend", "hour"])


def platform_state(deps: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """ORIGIN_FEATURES state columns and the label for (node, t) queries.

    Everything but next_dep uses departures at or before t. The next train's
    identity is its run, which a live router has from trip updates; its
    position is its latest departure at or before t, and next_eta_sched_sec is
    when it would reach this node if it ran to schedule from there.
    """
    queries = queries.reset_index(drop=True).assign(_q=lambda f: np.arange(len(f)))
    by_t = queries.sort_values("t")
    right = deps.sort_values("dep")

    last = pd.merge_asof(by_t, right[["node", "dep", "headway", "prev_headway"]],
                         left_on="t", right_on="dep", by="node", direction="backward",
                         tolerance=MAX_WAIT_SEC)
    nxt = pd.merge_asof(by_t[["_q", "node", "t"]],
                        right[["node", "dep", "run", "sched_cum"]].rename(
                            columns={"dep": "next_dep", "sched_cum": "next_sched_cum"}),
                        left_on="t", right_on="next_dep", by="node", direction="forward",
                        tolerance=MAX_WAIT_SEC)
    out = queries.copy()
    out.loc[last["_q"], "age_since_last_sec"] = (last["t"] - last["dep"]).to_numpy()
    out.loc[last["_q"], "last_headway_sec"] = last["headway"].to_numpy()
    out.loc[last["_q"], "prev_headway_sec"] = last["prev_headway"].to_numpy()
    out.loc[nxt["_q"], "next_dep"] = nxt["next_dep"].to_numpy()
    out.loc[nxt["_q"], "run"] = nxt["run"].to_numpy()
    out.loc[nxt["_q"], "next_sched_cum"] = nxt["next_sched_cum"].to_numpy()

    has_next = out["run"].notna()
    trains = out.loc[has_next, ["_q", "t", "run"]].astype({"run": np.int64}).sort_values("t")
    progress = deps.sort_values("dep")[["run", "dep", "sched_cum", "delay"]].rename(
        columns={"dep": "at_dep", "sched_cum": "at_sched_cum"})
    trains = pd.merge_asof(trains, progress, left_on="t", right_on="at_dep", by="run",
                           direction="backward")
    out.loc[trains["_q"], "next_eta_sched_sec"] = (
        trains["at_dep"] + (out.loc[trains["_q"], "next_sched_cum"].to_numpy()
                            - trains["at_sched_cum"]) - trains["t"]).to_numpy()
    out.loc[trains["_q"], "next_train_delay_sec"] = trains["delay"].to_numpy()
    out["wait_sec"] = out["next_dep"] - out["t"]
    return out.drop(columns=["_q", "run", "next_sched_cum"])


def origin_samples(deps: pd.DataFrame, typical: pd.DataFrame, start: int, end: int,
                   step_sec: int, rng: np.random.Generator) -> pd.DataFrame:
    """A rider at every node every ~step_sec in [start, end), jittered so no
    sample lines up with a timetable. Samples with no train within the hour
    are dropped: that is no service, which the live service filter handles."""
    nodes = deps["node"].unique()
    grid = np.arange(start, end, step_sec)
    queries = pd.DataFrame({"node": np.repeat(nodes, len(grid)),
                            "t": np.tile(grid, len(nodes))})
    queries["t"] += rng.integers(0, step_sec, len(queries))
    return build_origin(deps, typical, queries).dropna(subset=["wait_sec"])


def build_origin(deps: pd.DataFrame, typical: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    frame = platform_state(deps, queries)
    frame = pd.concat([frame, calendar(frame["t"])], axis=1)
    frame["route"], frame["direction"] = route_of(frame["node"]), direction_of(frame["node"])
    return with_typical_headway(frame, typical)


def transfer_samples(rows: pd.DataFrame, deps: pd.DataFrame, typical: pd.DataFrame,
                     transfers: pd.DataFrame, fraction: float,
                     rng: np.random.Generator) -> pd.DataFrame:
    """Every observed arrival at a node with transfer edges, times each of its
    transfers, subsampled to `fraction`. transfers: from_node, to_node, walk_sec.
    The rider reaches to_node's platform at arrival + walk_sec."""
    arrivals = rows[["to_node", "ts"]].rename(columns={"to_node": "from_node", "ts": "arrived"})
    arrivals = arrivals[rng.random(len(arrivals)) < fraction]
    events = arrivals.merge(transfers, on="from_node")
    queries = pd.DataFrame({"node": events["to_node"].to_numpy(),
                            "t": (events["arrived"] + events["walk_sec"]).to_numpy(dtype=np.int64)})
    frame = build_transfer(typical, queries, events["from_node"], events["walk_sec"])
    labelled = platform_state(deps, queries)
    frame["wait_sec"] = labelled["wait_sec"].to_numpy()
    return frame.dropna(subset=["wait_sec"])


def transfer_cost_labels(arrival_rows: pd.DataFrame, deps: pd.DataFrame,
                         transfers: pd.DataFrame, fraction: float,
                         rng: np.random.Generator) -> pd.DataFrame:
    """Walk + wait per transfer taken, for models that price the whole transfer
    edge (ml_model/graph_wavenet.py with transfers). For a sampled `fraction`
    of the train arrivals in arrival_rows, every transfer out of the arrival
    node: the walk, plus the wait for the next departure on the far platform
    (platform_state, the transfer-wait model's label). One row per event,
    shaped like a traversal: from_node, to_node, is_transfer, arrived, ts
    (arrived + cost) and edge_sec (the cost). No-service transfers are dropped."""
    arrivals = arrival_rows[["to_node", "ts"]].rename(columns={"to_node": "from_node",
                                                               "ts": "arrived"})
    arrivals = arrivals[rng.random(len(arrivals)) < fraction]
    events = arrivals.merge(transfers, on="from_node")
    queries = pd.DataFrame({"node": events["to_node"].to_numpy(),
                            "t": (events["arrived"] + events["walk_sec"]).to_numpy(dtype=np.int64)})
    cost = events["walk_sec"].to_numpy() + platform_state(deps, queries)["wait_sec"].to_numpy()
    out = pd.DataFrame({"from_node": events["from_node"].to_numpy(),
                        "to_node": events["to_node"].to_numpy(), "is_transfer": True,
                        "arrived": events["arrived"].to_numpy(dtype=np.int64),
                        "edge_sec": cost})
    out = out.dropna(subset=["edge_sec"]).reset_index(drop=True)
    out["ts"] = (out["arrived"] + out["edge_sec"]).astype(np.int64)
    return out


def build_transfer(typical: pd.DataFrame, queries: pd.DataFrame, from_nodes: pd.Series,
                   walk_sec: pd.Series) -> pd.DataFrame:
    """TRANSFER_FEATURES for riders reaching `queries.node` at `queries.t`."""
    frame = queries.reset_index(drop=True).copy()
    frame = pd.concat([frame, calendar(frame["t"])], axis=1)
    frame["walk_sec"] = np.asarray(walk_sec, dtype=float)
    frame["route"], frame["direction"] = route_of(frame["node"]), direction_of(frame["node"])
    frame["from_node"] = np.asarray(from_nodes)
    frame["from_route"] = route_of(frame["from_node"])
    return with_typical_headway(frame, typical)
