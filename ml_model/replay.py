"""Replaying observed traversals: the model-free core the path benchmark and
the recurrent GNN share.

  journey_ids        chain traversals into train runs
  Network            the subway graph flattened for fast station-to-station routing
  Observed           test-span traversals indexed for replaying a path
  snapshot_features  MODEL_FEATURES for every ride edge as a live snapshot at t0
                     would see them, rebuilt from traversal rows

Kept free of LightGBM and torch on purpose: the two can't share a process on
macOS (OMP Error #15), and both the LightGBM path benchmark and the torch GNN
scripts need these.
"""

from __future__ import annotations

import heapq
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from graph.service_states import state_at, states_of
from graph.subway_loader import load_parent_stations, load_transfer_walk_times
from ml_model.features import EDGE_KEY


def journey_ids(test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(run id, position in run) per row. A row continues the row whose train
    arrived at this row's from_node at the second this row departed it, on the
    same route. Links that aren't one-to-one are cut rather than guessed."""
    rows = pd.DataFrame({
        "i": np.arange(len(test)),
        "route_id": test["route_id"].astype(str).to_numpy(),
        "node": test["from_node"].to_numpy(),
        "t": (test["ts"] - test["edge_sec"]).to_numpy(dtype=np.int64),
    })
    ends = pd.DataFrame({
        "prev": np.arange(len(test)),
        "route_id": rows["route_id"],
        "node": test["to_node"].to_numpy(),
        "t": test["ts"].to_numpy(dtype=np.int64),
    })
    links = rows.merge(ends, on=["route_id", "node", "t"])
    links = links[~links["i"].duplicated(keep=False) & ~links["prev"].duplicated(keep=False)]

    prev = np.full(len(test), -1)
    prev[links["i"].to_numpy()] = links["prev"].to_numpy()
    # Pointer jumping: each round doubles how far back a row can see, so after
    # log2(longest run) rounds every row points at its run's first row.
    head = np.where(prev < 0, np.arange(len(test)), prev)
    while True:
        jumped = head[head]
        if np.array_equal(jumped, head):
            break
        head = jumped
    # Position by ordering departure times within each run.
    order = np.lexsort((rows["t"].to_numpy(), head))
    position = np.empty(len(test), dtype=np.int64)
    run_sorted = head[order]
    starts = np.r_[0, np.flatnonzero(np.diff(run_sorted)) + 1]
    counts = np.diff(np.r_[starts, len(order)])
    position[order] = np.arange(len(order)) - np.repeat(starts, counts)
    return head, position


def split_by_date(frame: pd.DataFrame, valid_days: int, test_days: int):
    dates = sorted(str(d) for d in frame["service_date"].unique())
    if len(dates) <= valid_days + test_days:
        raise ValueError(f"{len(dates)} dates can't hold {valid_days} valid "
                         f"+ {test_days} test days and still train")
    test_dates = dates[-test_days:]
    valid_dates = dates[-test_days - valid_days:-test_days]
    train_dates = dates[:-test_days - valid_days]
    part = frame["service_date"]
    spans = {"train": train_dates, "valid": valid_dates, "test": test_dates}
    return (frame[part.isin(train_dates)], frame[part.isin(valid_dates)],
            frame[part.isin(test_dates)], spans)


TZ = ZoneInfo("America/New_York")
EDGE = ["from_node", "to_node"]
MAX_WAIT_SEC = 3600
NEXT_TRAIN_WINDOW_SEC = 3600  # how far ahead a snapshot looks for an edge's next train
OBS_WINDOW_SEC = 1800  # same as training's add_last_observed and a live snapshot
JOURNEY_BUDGET_SEC = 3 * 3600  # t0 must leave this much observed data after it
SAME_SEC = 60  # true times this close count as a tie

NEXT_TRAIN_COLUMNS = ["sched_edge_sec", "has_schedule", "hour", "minute_of_day", "dow",
                      "is_weekend", "station_alert_count", "station_alert_age_sec",
                      "station_alert_types"]

DEPARTURE_PERIODS = [("overnight 0-6", 0, 6), ("am rush 6-10", 6, 10),
                     ("midday 10-16", 10, 16), ("pm rush 16-20", 16, 20),
                     ("evening 20-24", 20, 24)]


def state_of_time(local: datetime) -> str:
    """The graph's schedule state for a local time (graph/service_states.py)."""
    return state_at(local)


class Network:
    """The graph flattened to integer node/edge arrays for fast routing."""

    def __init__(self, graph, gtfs_dir: Path) -> None:
        self.graph = graph
        parent_of = load_parent_stations(gtfs_dir)
        walk_times = load_transfer_walk_times(gtfs_dir)
        self.node_ids = list(graph)
        self.index = {n: i for i, n in enumerate(self.node_ids)}
        self.station = [parent_of.get(graph.get_node(n).stop_id, graph.get_node(n).stop_id)
                        for n in self.node_ids]
        self.nodes_at: dict[str, list[int]] = {}
        for i, station in enumerate(self.station):
            self.nodes_at.setdefault(station, []).append(i)

        keys, frm, to, transfer, base, walk = [], [], [], [], [], []
        for from_id in self.node_ids:
            for to_id, time, is_transfer in graph.get_node(from_id).paths:
                if to_id not in self.index:
                    continue
                keys.append((from_id, to_id, is_transfer))
                frm.append(self.index[from_id])
                to.append(self.index[to_id])
                transfer.append(is_transfer)
                base.append(time)
                walk.append(walk_times.get((self.station[self.index[from_id]],
                                            self.station[self.index[to_id]]), 0)
                            if is_transfer else 0)
        self.edge_keys = keys
        self.edge_index = {k: i for i, k in enumerate(keys)}
        self.edge_from, self.edge_to = np.array(frm), np.array(to)
        self.is_transfer = np.array(transfer)
        self.base = base
        self.walk = np.array(walk, dtype=float)
        self.adjacency: list[list[int]] = [[] for _ in self.node_ids]
        for e, f in enumerate(frm):
            self.adjacency[f].append(e)
        self.edges = pd.DataFrame(keys, columns=list(EDGE_KEY))

    def costs(self, graph, period: str) -> np.ndarray:
        return np.array([graph.edge_time(f, t, x, b, period)
                         for (f, t, x), b in zip(self.edge_keys, self.base)], dtype=float)

    def route(self, costs: np.ndarray, origin: str, destination: str,
              start: np.ndarray | None = None) -> list[int] | None:
        """Edge indices of the cheapest path from any platform at `origin` to
        the first platform reached at `destination`. `start` is a per-node cost
        of starting there -- the wait for its first train -- else 0."""
        targets = set(self.nodes_at[destination])
        best = {s: (float(start[s]) if start is not None else 0.0)
                for s in self.nodes_at[origin]}
        previous: dict[int, int] = {}
        queue = [(c, s) for s, c in best.items()]
        heapq.heapify(queue)
        done = set()
        while queue:
            cost, node = heapq.heappop(queue)
            if node in done:
                continue
            done.add(node)
            if node in targets:
                path = []
                while node in previous:
                    edge = previous[node]
                    path.append(edge)
                    node = self.edge_from[edge]
                return path[::-1]
            for edge in self.adjacency[node]:
                step = costs[edge]
                if step == math.inf:
                    continue
                nxt = self.edge_to[edge]
                if cost + step < best.get(nxt, math.inf):
                    best[nxt] = cost + step
                    previous[nxt] = edge
                    heapq.heappush(queue, (cost + step, nxt))
        return None


class Observed:
    """Test-span traversals indexed for replay."""

    def __init__(self, rows: pd.DataFrame, network: Network) -> None:
        self.dep = (rows["ts"] - rows["edge_sec"]).to_numpy(dtype=np.int64)
        self.arr = rows["ts"].to_numpy(dtype=np.int64)
        run, position = journey_ids(rows)
        order = np.lexsort((position, run))
        self.next_in_run = np.full(len(rows), -1)
        same = run[order][1:] == run[order][:-1]
        self.next_in_run[order[:-1][same]] = order[1:][same]
        self.run = run
        self.edge = np.array([network.edge_index.get((f, t, False), -1)
                              for f, t in zip(rows["from_node"], rows["to_node"])])
        self.by_edge: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        known = np.flatnonzero(self.edge >= 0)
        order = known[np.lexsort((self.dep[known], self.edge[known]))]
        edges_sorted = self.edge[order]
        bounds = np.flatnonzero(np.diff(edges_sorted)) + 1
        for chunk in np.split(order, bounds):
            if len(chunk):
                self.by_edge[int(self.edge[chunk[0]])] = (self.dep[chunk], chunk)

    def replay(self, network: Network, path: list[int], t0: int):
        """(cumulative true seconds per edge, first boarding time, status)."""
        clock, row, boarded, cumulative = t0, -1, None, []
        for edge in path:
            if network.is_transfer[edge]:
                clock += network.walk[edge]
                row = -1
            else:
                stay = self.next_in_run[row] if row >= 0 else -1
                if stay >= 0 and self.edge[stay] == edge:
                    row = stay
                else:
                    if edge not in self.by_edge:
                        return cumulative, boarded, "edge never observed"
                    deps, rows = self.by_edge[edge]
                    k = np.searchsorted(deps, clock)
                    if k == len(deps) or deps[k] - clock > MAX_WAIT_SEC:
                        return cumulative, boarded, "no train within an hour"
                    row = rows[k]
                if boarded is None:
                    boarded = int(self.dep[row])
                clock = int(self.arr[row])
            cumulative.append(clock - t0)
        return cumulative, boarded, "ok"


def snapshot_features(rows: pd.DataFrame, observed: Observed, network: Network,
                      t0s: np.ndarray) -> pd.DataFrame:
    """MODEL_FEATURES for every (t0, ride edge), as a snapshot at t0 would
    see them. See the module docstring for each column's source."""
    ride = network.edges[~network.is_transfer].reset_index(drop=True)
    grid = ride.loc[ride.index.repeat(len(t0s))].reset_index(drop=True)
    grid["t0"] = np.tile(t0s, len(ride))
    grid = grid.sort_values("t0", kind="stable").reset_index(drop=True)

    traversals = rows[EDGE + NEXT_TRAIN_COLUMNS + ["edge_sec", "prior_delay_sec"]].copy()
    traversals["dep"] = observed.dep
    traversals["arr"] = observed.arr
    traversals["run"] = observed.run

    upcoming = traversals.sort_values("dep")[EDGE + NEXT_TRAIN_COLUMNS + ["dep", "run"]]
    grid = pd.merge_asof(grid, upcoming, left_on="t0", right_on="dep", by=EDGE,
                         direction="forward", tolerance=NEXT_TRAIN_WINDOW_SEC)

    finished = traversals.sort_values("arr")[EDGE + ["arr", "edge_sec"]] \
        .rename(columns={"arr": "obs_arr", "edge_sec": "obs_last_edge_sec"})
    grid = pd.merge_asof(grid, finished, left_on="t0", right_on="obs_arr", by=EDGE,
                         direction="backward", tolerance=OBS_WINDOW_SEC)
    clock = grid["dep"].fillna(grid["t0"]).clip(lower=grid["t0"])
    grid["obs_last_age_sec"] = clock - grid["obs_arr"]

    has_train = grid["run"].notna()
    lateness = traversals.sort_values("dep")[["run", "dep", "prior_delay_sec"]] \
        .rename(columns={"dep": "delay_at"})
    trains = grid.loc[has_train, ["t0", "run"]].reset_index()
    trains["run"] = trains["run"].astype(np.int64)
    trains = pd.merge_asof(trains.sort_values("t0"), lateness, left_on="t0",
                           right_on="delay_at", by="run", direction="backward")
    grid["prior_delay_sec"] = np.nan
    grid.loc[trains["index"], "prior_delay_sec"] = trains["prior_delay_sec"].to_numpy(dtype=float)

    local = pd.to_datetime(grid["t0"], unit="s", utc=True).dt.tz_convert(TZ)
    fallback = {"hour": local.dt.hour, "minute_of_day": local.dt.hour * 60 + local.dt.minute,
                "dow": local.dt.dayofweek, "is_weekend": (local.dt.dayofweek >= 5).astype(int)}
    for column, values in fallback.items():
        grid[column] = grid[column].fillna(values)
    grid["has_schedule"] = grid["has_schedule"].fillna(0)
    grid["station_alert_count"] = grid["station_alert_count"].fillna(0)

    # Service day and schedule state at t0, as a live snapshot takes them at its
    # own time (snapshot.build.build_snapshot).
    grid["service_period"], grid["service_state"] = states_of(grid["t0"])
    graph = network.graph
    grid["route"] = grid["to_node"].map(lambda n: graph.get_node(n).vehicle)
    grid["direction"] = grid["to_node"].map(lambda n: graph.get_node(n).stop_id[-1])
    base = {(f, t): b for (f, t, x), b in zip(network.edge_keys, network.base) if not x}
    cost = [graph.edge_time(f, t, False, base[(f, t)], p)
            for f, t, p in zip(grid["from_node"], grid["to_node"], grid["service_state"])]
    grid["graph_edge_sec"] = [c if c != math.inf else np.nan for c in cost]
    grid["is_transfer"] = False
    return grid


# Routers compared. "live" drops ride edges with no train due within
# NEXT_TRAIN_WINDOW_SEC of t0 -- something a live router can know from trip
# updates and today's router does not do. Without it both routers send riders
# onto service that isn't running (period costs have no time of day, and the
# 2026 graph has edges 2025 service never ran), so comparing only the trips
# that happen to replay would score the easy ones.


def test_dates(rows: pd.DataFrame, days: int | None = None,
               dates: list[str] | None = None) -> list[str]:
    """Explicit `dates`, else the latest `days` service dates in `rows` --
    the same test span benchmark_gbm.split_by_date holds out."""
    if dates:
        return sorted(str(d) for d in dates)
    available = sorted(str(d) for d in rows["service_date"].unique())
    return available[-days:]


def departure_times(rows: pd.DataFrame, dates: list[str], step_min: int,
                    warmup_min: int = 30, lead_min: int = 0) -> np.ndarray:
    """Departure times t0 for the path benchmark, per test date.

    From `warmup_min` + `lead_min` after local midnight -- so a snapshot taken
    `lead_min` before departure still has recent observations, even when the
    day before isn't in the data, as with randomly sampled days -- to the end
    of the day, stopping JOURNEY_BUDGET_SEC before that date's last observed
    arrival so every journey has data to replay. Departures are aligned to
    multiples of `step_min` whatever the lead, so runs with different leads
    share departures; snapshots sit `lead_min` before them.
    """
    step = step_min * 60
    times = []
    for date in dates:
        day = rows[rows["service_date"] == date]
        start = int(pd.Timestamp(date, tz=TZ).timestamp()) + (warmup_min + lead_min) * 60
        start = -(-start // step) * step
        end = min(int(pd.Timestamp(date, tz=TZ).timestamp()) + 86400,
                  int(day["ts"].max()) - JOURNEY_BUDGET_SEC)
        times.append(np.arange(start, end, step, dtype=np.int64))
    return np.concatenate(times) if times else np.array([], dtype=np.int64)


def day_grid(date: str, step_sec: int) -> np.ndarray:
    """Step ends from the first step after local midnight to the next midnight."""
    start = int(pd.Timestamp(date, tz=TZ).timestamp())
    return np.arange(start + step_sec, start + 86400 + 1, step_sec, dtype=np.int64)


class Day:
    """One service date's rows, replay index and feature grid."""

    def __init__(self, rows: pd.DataFrame, network: Network, date: str, step_sec: int) -> None:
        before = (pd.Timestamp(date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        self.date = date
        self.rows = rows[rows["service_date"].isin([date, before])].reset_index(drop=True)
        self.observed = Observed(self.rows, network)
        self.times = day_grid(date, step_sec)

    def features(self, network: Network, times: np.ndarray | None = None) -> pd.DataFrame:
        return snapshot_features(self.rows, self.observed, network,
                                 self.times if times is None else times)


def snapshot_labels(rows: pd.DataFrame, t0s: np.ndarray, horizon_sec: int) -> pd.DataFrame:
    """Traversals in `rows` departing within horizon_sec after a snapshot in
    `t0s`, each tagged with that snapshot's t0: what a model predicting from
    the snapshot at t0 is scored on. Departure is ts - edge_sec, as in
    ml_model.sequence.build_targets."""
    labels = rows[EDGE + ["ts", "edge_sec"]].copy()
    labels["dep"] = labels["ts"].astype(np.int64) - labels["edge_sec"].astype(np.int64)
    labels = pd.merge_asof(labels.sort_values("dep"),
                           pd.DataFrame({"t0": np.sort(np.asarray(t0s, dtype=np.int64))}),
                           left_on="dep", right_on="t0", direction="backward",
                           tolerance=horizon_sec - 1)
    return labels.dropna(subset=["t0"]).astype({"t0": np.int64}).reset_index(drop=True)
