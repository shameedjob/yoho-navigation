"""Origin waits from live state: when the next train reaches each platform.

One table, rebuilt with each snapshot: origin_waits, one row per graph node,
ORIGIN_FEATURES for a rider reaching that platform now, plus the wait the
router charges for it (`with_origin_wait`): the raw ETA, else 0.9 x the
typical headway, else no service.

ORIGIN_FEATURES still follow ml_model/waits.py's training rows (see
docs/FEATURE_PARITY.md, "Waits"): live reads LiveState.arrivals and the trip
update feeds, and measures the next train's progress from its latest observed
stop plus graph edge costs over the stops left to this platform. No model
reads them live any more; `next_eta_sched_sec` and `typical_headway_sec` feed
the wait, and the rest are served for inspection. Transfer waits are priced
by the graph model in the agent.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from graph import Graph
from ml_model.waits import FALLBACK_EDGE_SEC, MAX_WAIT_SEC, ORIGIN_FEATURES
from snapshot.build import TZ
from snapshot.state import ARRIVAL_TOLERANCE_SEC, LiveState

ORIGIN_COLUMNS = ["snapshot_ts", "node", *ORIGIN_FEATURES.columns, "next_train_trip",
                  "next_train_feed_eta_sec"]


def calendar_at(now: int) -> dict:
    local = datetime.fromtimestamp(now, TZ)
    return {"hour": local.hour, "minute_of_day": local.hour * 60 + local.minute,
            "dow": local.weekday(), "is_weekend": int(local.weekday() >= 5)}


def load_typical(path: Path) -> dict[tuple[str, int, int], float]:
    table = pd.read_csv(path)
    return {(n, int(w), int(h)): float(v) for n, w, h, v in
            zip(table["node"], table["is_weekend"], table["hour"], table["typical_headway_sec"])}


class WaitFeatures:
    """Static pieces (ride edge costs) built once per graph."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.ride_cost: dict[tuple[str, str], float] = {
            (from_id, to_id): time
            for from_id in graph
            for to_id, time, is_transfer in graph.get_node(from_id).paths if not is_transfer}

    def origin_waits(self, state: LiveState, now: int,
                     typical: dict[tuple[str, int, int], float] | None) -> pd.DataFrame:
        """ORIGIN_FEATURES for every node at `now`.

        The next train at a node is the trip with the earliest predicted time
        at that stop among stops it will leave again (a trip's last listed stop
        is not a departure, as in training). Predictions more than
        ARRIVAL_TOLERANCE_SEC in the past are skipped: a stop still listed
        minutes after its predicted time is a stale listing, not a train a
        rider can board, and training's next train is always a real departure
        at or after t. next_eta_sched_sec is that
        train's latest observed arrival plus graph ride costs over the stops
        between there and this node, minus now; null until the train has been
        observed at a stop. next_train_feed_eta_sec is the feed's own
        prediction, served for comparison only -- the model never saw it.
        """
        cal = calendar_at(now)
        nxt: dict[str, tuple[int, tuple, str, list, int]] = {}
        for trip_key, route_id, stops in state.trips_as_of(now):
            for i, (stop_id, eta) in enumerate(stops[:-1]):
                if eta < now - ARRIVAL_TOLERANCE_SEC:
                    continue
                node = state.node_id(stop_id, route_id)
                if node not in nxt or eta < nxt[node][0]:
                    nxt[node] = (eta, trip_key, route_id, stops, i)

        rows = []
        for node_id in self.graph:
            node = self.graph.get_node(node_id)
            times = sorted(t for t in state.arrivals.get(node_id, ()) if t <= now)
            age = last_hw = prev_hw = None
            if times and now - times[-1] <= MAX_WAIT_SEC:
                age = now - times[-1]
                if len(times) >= 2:
                    last_hw = times[-1] - times[-2]
                if len(times) >= 3:
                    prev_hw = times[-2] - times[-3]

            eta_sched = delay = trip = feed_eta = None
            if node_id in nxt:
                feed_eta, trip_key, route_id, stops, index = nxt[node_id]
                trip = f"{trip_key[0]}|{trip_key[1]}"
                feed_eta -= now
                seen = state.last_stop(trip_key)
                if seen is not None and seen[1] <= now:
                    path = [seen[0]] + [state.node_id(s, route_id) for s, _ in stops[:index + 1]]
                    remaining = sum(self.ride_cost.get(pair, FALLBACK_EDGE_SEC)
                                    for pair in zip(path, path[1:]))
                    eta_sched = seen[1] + remaining - now
                observed = state.trip_delay.get(trip_key)
                if observed is not None and observed[0] <= now:
                    delay = observed[1]

            rows.append({
                "snapshot_ts": now, "node": node_id,
                "age_since_last_sec": age, "last_headway_sec": last_hw,
                "prev_headway_sec": prev_hw,
                "typical_headway_sec": (typical or {}).get(
                    (node_id, cal["is_weekend"], cal["hour"])),
                "next_eta_sched_sec": eta_sched, "next_train_delay_sec": delay,
                **cal, "route": node.vehicle, "direction": node.stop_id[-1],
                "next_train_trip": trip, "next_train_feed_eta_sec": feed_eta,
            })
        frame = pd.DataFrame(rows, columns=ORIGIN_COLUMNS)
        for column in ("age_since_last_sec", "last_headway_sec", "prev_headway_sec",
                       "typical_headway_sec", "next_eta_sched_sec", "next_train_delay_sec",
                       "next_train_feed_eta_sec"):
            frame[column] = frame[column].astype("Float64")
        return frame


# The live origin-wait rule (docs/HANDOFF_TIME_BUCKETS.md, benchmarked in
# ml_model.benchmark_paths as the joint_eta routers): the raw ETA; where a
# platform has none, this fraction of its typical headway, the 90th percentile
# of a uniformly random wait; with neither, no service.
TYPICAL_HEADWAY_FRACTION = 0.9


def with_origin_wait(frame: pd.DataFrame) -> pd.DataFrame:
    """`frame` (origin_waits) plus wait_sec and wait_source ("eta", "typical"
    or "none"). A platform with no service costs MAX_WAIT_SEC, as in the
    benchmark, so a route never starts there unless nothing else runs."""
    frame = frame.copy()
    eta = frame["next_eta_sched_sec"].astype(float).clip(lower=0)
    typical = TYPICAL_HEADWAY_FRACTION * frame["typical_headway_sec"].astype(float)
    frame["wait_sec"] = eta.fillna(typical).fillna(MAX_WAIT_SEC).round(1)
    frame["wait_source"] = np.where(eta.notna(), "eta",
                                    np.where(typical.notna(), "typical", "none"))
    return frame
