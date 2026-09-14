"""Path benchmark: route on schedule costs vs model costs, replay both against
what trains actually did.

    python3 -m ml_model.benchmark_paths --data data/training/training_data_2025-01.csv

This is the benchmark that matters most: the edge benchmark (benchmark_gbm.py)
scores costs, this scores the routes a rider would be sent on. For each
(departure time t0, origin station, destination station):

  sched           Dijkstra on the graph's schedule costs for t0's service
                  state (day x six-hour bucket) -- what the router does today
  sched_day       the same on whole-day average costs, the router before
                  time-of-day buckets
  model           Dijkstra on the same graph reweighted by the model
                  (GraphSnapshot.weighted_graph), from a snapshot as of t0
  *_live          either, with ride edges that have no train due within an
                  hour of t0 removed (see ROUTERS)

and at every node along either path three cumulative costs:

  scheduled   sum of schedule edge costs so far
  predicted   sum of model edge costs so far (transfers keep schedule cost)
  true        replayed seconds since t0, from the test span's observed
              traversals

Snapshot as of t0. Rebuilt from training rows with nothing after t0 except
what a live snapshot also has in advance or as a prediction. Per ride edge,
the "next train" is the first traversal arriving at from_node at or after t0:
its schedule columns, and its calendar and alert columns (measured at its
arrival, standing in for the live predicted arrival) are used as-is.
obs_last_* uses traversals finished by t0, aged to the next train's arrival
at from_node. prior_delay_sec is that train's lateness at the latest stop it
had reached by t0, as a live snapshot measures it. graph_edge_sec and
service_state come from the snapshot time, as a live snapshot takes them. Alerts posted
between t0 and the next train's arrival leak in; alert features carry ~0.4%
of the model's gain.

Replay. The rider is at the origin station at t0. For a ride edge they stay on
the train they're on if its run continues along the edge, else board the
first train on that edge arriving at from_node at or after their clock. A
transfer costs its transfers.txt walk only; the real wait shows up as the gap
before the next boarding. The trip fails if an edge has no train within
MAX_WAIT_SEC -- including graph edges the 2025 data never observes, since the
graph is built from the 2026 GTFS. Each comparison uses trips where both of
its routers' paths replay.

With --lead-min N the rider plans ahead: the snapshot, and so every model
input (ride costs, waits, the live service filter), is taken N minutes before
departure, while the replay starts at departure. Station pairs are drawn from
the departure time, so runs with different leads route the same trips;
ml_model/compare_path_runs.py compares them trip by trip.

Schedule and predicted costs exclude the wait for the first train (the graph
has no origin-wait term), so planned costs are compared with true time from
first boarding as well as from t0.

Joint routers. An --edge-costs CSV with is_transfer rows (ml_model/train_gat.py
--with-transfers) prices transfers too, as walk + wait. With --wait-models,
each such cost set NAME also gets
  NAME_joint       its ride and transfer costs; the origin-wait model (the
                   next train's live ETA, corrected) for the first train
  NAME_joint_eta   the same, but the raw ETA (next_eta_sched_sec) for the
                   first train, falling back to the origin model without one
each with a _live variant, and estimates pred_NAME_joint / pred_NAME_joint_eta.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from graph.service_states import DAYS, STATES, day_of
from graph.subway_loader import build_subway_graph
from ml_model import model_data, waits
from ml_model.benchmark_gbm import format_table
from ml_model.gbm import QuantileGBM
from ml_model.replay import (DEPARTURE_PERIODS, SAME_SEC, TZ, Network, Observed,
                            departure_times, state_of_time, snapshot_features, test_dates)
from snapshot.build import GraphSnapshot

LIGHTGBM = "model"  # the cost set from --model; key kept for older reports


def cost_set_name(cost_set: str) -> str:
    return "LightGBM" if cost_set == LIGHTGBM else cost_set.upper()


def router_key(cost_set: str, waits_on: bool, live: bool) -> str:
    return cost_set + ("_waits" if waits_on else "") + ("_live" if live else "")


JOINT_MODES = {"joint": "origin-wait model", "joint_eta": "live ETA, else 0.9 x typical headway"}
# The chosen live origin rule's fallback where a platform has no ETA: this
# fraction of its typical headway, the 90th percentile of a uniformly random wait.
ORIGIN_HEADWAY_FRACTION = 0.9


def estimate_key(cost_set: str, waits_on: bool) -> str:
    if cost_set == LIGHTGBM:
        return "predicted_waits" if waits_on else "predicted"
    return f"pred_{cost_set}" + ("_waits" if waits_on else "")


def describe(cost_sets: list[str], waits_on: bool,
             joint_sets: list[str] = ()) -> tuple[dict, dict, dict, list]:
    """(routers, router display names, estimate display names, comparison pairs)."""
    routers = {"sched": "schedule costs (today's router)",
               "sched_live": "schedule costs, live service filter",
               "sched_day": "whole-day average schedule costs (before time-of-day buckets)",
               "sched_day_live": "whole-day average schedule costs, live service filter"}
    router_names = {"sched": "schedule, no filter", "sched_live": "schedule",
                    "sched_day": "day-average schedule, no filter",
                    "sched_day_live": "day-average schedule"}
    estimate_names = {"scheduled": "Scheduled"}
    pairs = [("schedule (buckets) vs day-average schedule", "sched_day_live", "sched_live")]
    for cs in cost_sets:
        name = cost_set_name(cs)
        for w in ([False, True] if waits_on else [False]):
            label = name + (" + waits" if w else "")
            detail = f"{name} ride costs" + (" + origin and transfer wait models" if w else "")
            routers[router_key(cs, w, False)] = detail
            routers[router_key(cs, w, True)] = detail + ", live service filter"
            router_names[router_key(cs, w, False)] = label + ", no filter"
            router_names[router_key(cs, w, True)] = label
            estimate_names[estimate_key(cs, w)] = label
            pairs.append((f"{label} vs schedule", "sched_live", router_key(cs, w, True)))
    for cs in joint_sets if waits_on else ():
        name = cost_set_name(cs)
        for mode, origin in JOINT_MODES.items():
            label = f"{name} joint, origin {origin}"
            detail = f"{name} ride and transfer (walk + wait) costs; first train from the {origin}"
            routers[f"{cs}_{mode}"] = detail
            routers[f"{cs}_{mode}_live"] = detail + ", live service filter"
            router_names[f"{cs}_{mode}"] = label + ", no filter"
            router_names[f"{cs}_{mode}_live"] = label
            estimate_names[f"pred_{cs}_{mode}"] = label
            pairs.append((f"{label} vs schedule", "sched_live", f"{cs}_{mode}_live"))
            pairs.append((f"{label} vs LightGBM + waits", router_key(LIGHTGBM, True, True),
                          f"{cs}_{mode}_live"))
        pairs.append((f"{name} joint vs {name} + waits", router_key(cs, True, True),
                      f"{cs}_joint_live"))
    extra = [cs for cs in cost_sets if cs != LIGHTGBM]
    for cs in extra:
        for w in ([False, True] if waits_on else [False]):
            pairs.append((f"{cost_set_name(cs)} vs LightGBM" + (", both + waits" if w else ""),
                          router_key(LIGHTGBM, w, True), router_key(cs, w, True)))
    return routers, router_names, estimate_names, pairs


def compare(frame: pd.DataFrame, a: str, b: str) -> dict:
    d = frame[f"{b}_true"] - frame[f"{a}_true"]
    return {
        "trips": int(len(frame)),
        "a_true_mean": float(frame[f"{a}_true"].mean()),
        "b_true_mean": float(frame[f"{b}_true"].mean()),
        "a_true_median": float(frame[f"{a}_true"].median()),
        "b_true_median": float(frame[f"{b}_true"].median()),
        "a_true_p90": float(frame[f"{a}_true"].quantile(0.9)),
        "b_true_p90": float(frame[f"{b}_true"].quantile(0.9)),
        "mean_change": float(d.mean()),
        "share_b_faster": float((d < -SAME_SEC).mean()),
        "share_b_slower": float((d > SAME_SEC).mean()),
        "share_tie": float((d.abs() <= SAME_SEC).mean()),
    }


# The path every estimate is scored on. Chosen by schedule costs, so it isn't
# a path any model picked to suit its own numbers, and filtered to running
# service so most trips replay.
ACCURACY_ROUTER = "sched_live"
# Trips count as alerted when this router's route passes a station with a live
# incident alert at the snapshot: the route a rider gets without any model, so
# the subset doesn't depend on which way a model chose to go.
ALERTED_ROUTER = "sched_live"
STOP_BUCKETS = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 10_000)]


def estimate_errors(actual: pd.Series, estimate: pd.Series) -> dict:
    err = actual - estimate
    return {
        "mean_error": float(err.mean()),
        "median_abs_error": float(err.abs().median()),
        "share_actual_later": float((err > 0).mean()),
        "share_within_2min": float((err.abs() <= 120).mean()),
        "share_within_5min": float((err.abs() <= 300).mean()),
    }


def spread(values: pd.Series) -> dict:
    return {"mean": float(values.mean()), "median": float(values.median()),
            "p90": float(values.quantile(0.9))}


def accuracy(trips: pd.DataFrame, nodes: pd.DataFrame, router: str,
             estimates: list[str]) -> dict:
    """Every estimate and the actual time on the same paths, on two clocks:
    from arrival at the origin station (what a rider experiences) and from
    boarding the first train (ride and transfer time only)."""
    ok = trips[trips[f"{router}_status"] == "ok"]
    estimates = [e for e in estimates if f"{router}_{e}" in ok.columns]
    from_station = ok[f"{router}_true"]
    from_boarding = ok[f"{router}_true_boarded"]

    def boarding_estimate(e: str) -> pd.Series:
        # waits estimates include their own origin-wait prediction; the others never had one
        if e.endswith("_waits") or e.endswith("_joint"):
            return ok[f"{router}_{e}"] - ok[f"{router}_origin_wait_pred"]
        if e.endswith("_joint_eta"):
            return ok[f"{router}_{e}"] - ok[f"{router}_origin_eta_pred"]
        return ok[f"{router}_{e}"]

    along = nodes[(nodes["path"] == router) & nodes["trip_id"].isin(ok["trip_id"])]
    along = along.assign(stops=along["step"] + 1)
    by_stops = []
    for lo, hi in STOP_BUCKETS:
        part = along[(along["stops"] >= lo) & (along["stops"] <= hi)]
        if len(part):
            row = {"stops": f"{lo}-{hi}" if hi < 10_000 else f"{lo}+", "nodes": int(len(part)),
                   "actual": float(part["true"].mean())}
            for e in estimates:
                row[e] = float(part[e].mean())
                row[f"{e}_late"] = float((part["true"] > part[e]).mean())
            by_stops.append(row)

    by_departure = []
    for label, lo, hi in DEPARTURE_PERIODS:
        part = ok[(ok["depart_hour"] >= lo) & (ok["depart_hour"] < hi)]
        if len(part):
            by_departure.append({"departure": label, "trips": int(len(part)),
                                 **{e: estimate_errors(part[f"{router}_true"], part[f"{router}_{e}"])
                                    for e in estimates}})

    result = {
        "router": router, "trips": int(len(ok)), "estimates": estimates,
        "journey": {**{e: spread(ok[f"{router}_{e}"]) for e in estimates},
                    "actual": spread(from_station), "actual_boarded": spread(from_boarding)},
        "first_wait": spread(ok[f"{router}_initial_wait"]),
        "errors_station": {e: estimate_errors(from_station, ok[f"{router}_{e}"]) for e in estimates},
        "errors_boarding": {e: estimate_errors(from_boarding, boarding_estimate(e))
                            for e in estimates},
        "along_journey": by_stops,
        "by_departure": by_departure,
    }
    if f"{router}_origin_wait_pred" in ok.columns:
        result["first_wait_pred"] = spread(ok[f"{router}_origin_wait_pred"])
    return result


def route_choice(trips: pd.DataFrame, routers: list[str]) -> dict | None:
    """Actual time of each live router's own route, on trips where all of them replay."""
    live = [r for r in routers if r.endswith("_live")]
    ok = trips[np.logical_and.reduce([trips[f"{r}_status"] == "ok" for r in live])]
    if not len(ok):
        return None
    base = ok["sched_live_true"]
    return {"trips": int(len(ok)), "routers": {
        r: {**spread(ok[f"{r}_true"]),
            "transfers": float(ok[f"{r}_transfers"].mean()),
            "differs_from_schedule": float((ok[f"{r}_path"] != ok["sched_live_path"]).mean()),
            "faster_than_schedule": float((ok[f"{r}_true"] < base - SAME_SEC).mean()),
            "slower_than_schedule": float((ok[f"{r}_true"] > base + SAME_SEC).mean()),
            "mean_change_vs_schedule": float((ok[f"{r}_true"] - base).mean())}
        for r in live}}


def summarize(trips: pd.DataFrame, nodes: pd.DataFrame, cost_sets: list[str]) -> dict:
    waits_on = any(c.endswith("_origin_wait_pred") for c in trips.columns)
    joint_sets = [cs for cs in cost_sets if f"{cs}_joint_live_status" in trips.columns]
    router_desc, router_names, estimate_names, pair_specs = describe(cost_sets, waits_on,
                                                                     joint_sets)
    routers = [r for r in router_desc if f"{r}_status" in trips.columns]
    pairs = []
    for label, a, b in pair_specs:
        if a not in routers or b not in routers:
            continue
        ok = trips[(trips[f"{a}_status"] == "ok") & (trips[f"{b}_status"] == "ok")]
        differ = ok[ok[f"{a}_path"] != ok[f"{b}_path"]]
        slices = {"all trips": compare(ok, a, b)} if len(ok) else {}
        if len(differ):
            slices["paths differ"] = compare(differ, a, b)
        pairs.append({"label": label, "a": a, "b": b,
                      "paths_differ": float(len(differ) / len(ok)) if len(ok) else None,
                      "slices": slices})
    alerted = None
    if f"{ALERTED_ROUTER}_alerted" in trips.columns:
        flagged = trips[trips[f"{ALERTED_ROUTER}_alerted"].fillna(False).astype(bool)]
        replayed = flagged[flagged[f"{ACCURACY_ROUTER}_status"] == "ok"]
        alerted = {"trips": int(len(flagged)), "share": float(len(flagged) / len(trips)),
                   "replayed": int(len(replayed))}
        if len(replayed):
            alerted["accuracy"] = accuracy(flagged, nodes[nodes["trip_id"].isin(flagged["trip_id"])],
                                           ACCURACY_ROUTER, list(estimate_names))
            alerted["route_choice"] = route_choice(flagged, routers)
    return {
        "trips_sampled": int(len(trips)),
        "alerted": alerted,
        "cost_sets": cost_sets,
        "routers": routers,
        "router_descriptions": {r: router_desc[r] for r in routers},
        "router_names": router_names,
        "estimate_names": estimate_names,
        "accuracy": accuracy(trips, nodes, ACCURACY_ROUTER, list(estimate_names)),
        "route_choice": route_choice(trips, routers),
        "status": {r: trips[f"{r}_status"].value_counts().to_dict() for r in routers},
        "pairs": pairs,
    }


def load_edge_costs(spec: str, t0s: np.ndarray) -> tuple[str, dict[int, dict], dict[int, dict]]:
    """NAME=CSV with columns t0, from_node, to_node, pred and optionally is_transfer ->
    (name, t0 -> {(from, to): ride pred}, t0 -> {(from, to): transfer pred}; empty
    when the CSV has no transfer rows)."""
    name, _, path = spec.partition("=")
    if not name or not path or not name.isidentifier():
        raise SystemExit(f"--edge-costs expects NAME=CSV with a plain name, got {spec!r}")
    table = pd.read_csv(path)
    missing = sorted(set(t0s.tolist()) - set(table["t0"].unique().tolist()))
    if missing:
        raise SystemExit(f"{path} has no predictions for {len(missing)} departure times, "
                         f"e.g. {missing[0]}; build it for the same test dates and --step-min")
    table = table[table["t0"].isin(t0s)]
    is_transfer = table["is_transfer"].astype(bool) if "is_transfer" in table else \
        pd.Series(False, index=table.index)

    def by_t0(part_table):
        return {int(t0): dict(zip(zip(part["from_node"], part["to_node"]), part["pred"]))
                for t0, part in part_table.groupby("t0")}

    return name, by_t0(table[~is_transfer]), by_t0(table[is_transfer])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+")
    parser.add_argument("--rerender", type=Path, metavar="REPORT_MD",
                        help="rebuild an existing report from its saved CSVs and JSON, no replay")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, default=Path("ml_model/checkpoints/gbm_benchmark"),
                        help="LightGBM edge model; must not have trained on the test dates")
    parser.add_argument("--wait-models", type=Path, default=None,
                        help="directory from ml_model.train_waits; adds the + waits routers")
    parser.add_argument("--edge-costs", action="append", default=[], metavar="NAME=CSV",
                        help="another edge model's ride predictions (t0, from_node, to_node, "
                             "pred), e.g. from ml_model.train_rgnn, keyed by snapshot time -- "
                             "with --lead-min, departure minus the lead; adds NAME routers. "
                             "Repeatable")
    parser.add_argument("--test-days", type=int, default=7,
                        help="test on the latest N service dates in --data")
    parser.add_argument("--test-dates", nargs="+", default=None, help="explicit test dates instead")
    parser.add_argument("--step-min", type=int, default=20)
    parser.add_argument("--lead-min", type=int, default=0,
                        help="take every model input from a snapshot this many minutes before "
                             "the rider leaves, as when planning a trip ahead")
    parser.add_argument("--pairs-per-time", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="report path; default docs/benchmarks/paths_<first>_<last test date>.md")
    args = parser.parse_args()

    if args.rerender:
        report = args.rerender
        trips = pd.read_csv(report.with_name(report.stem + "_trips.csv"))
        nodes = pd.read_csv(report.with_name(report.stem + "_nodes.csv"))
        meta = json.loads(report.with_suffix(".json").read_text())
        summary = summarize(trips, nodes, meta.get("cost_sets") or [LIGHTGBM])
        summary.update({k: meta.get(k) for k in (
            "created", "model", "wait_models", "edge_costs", "test_dates", "test_start",
            "last_departure", "step_min", "lead_min", "pairs_per_time", "seed")})
        report.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str))
        report.write_text(render(summary, report))
        print(render(summary, report))
        return
    if not args.data:
        parser.error("--data is required unless --rerender is given")

    gtfs = args.data_dir / "gtfs_subway"
    graph = build_subway_graph(gtfs)
    network = Network(graph, gtfs)
    day_graph = build_subway_graph(gtfs, bucketed=False)

    rows = pd.concat([pd.read_csv(p, low_memory=False) for p in args.data], ignore_index=True)
    rows = rows[rows["edge_sec"].notna()]
    dates = test_dates(rows, args.test_days, args.test_dates)
    # the day before each test date, when the data has it, so early departures
    # see the previous evening's trains
    before = {(pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y-%m-%d") for d in dates}
    rows = rows[rows["service_date"].isin(set(dates) | before)].reset_index(drop=True)
    observed = Observed(rows, network)
    # t0s are snapshot times: every model input is computed at them, and riders
    # leave lead_min later.
    lead_sec = args.lead_min * 60
    t0s = departure_times(rows, dates, args.step_min, lead_min=args.lead_min) - lead_sec
    print(f"test dates {dates}; {len(rows):,} traversals; {len(t0s)} departure times "
          f"x {args.pairs_per_time} pairs, snapshot {args.lead_min} min before departure; "
          f"{len(network.edge_keys)} edges", flush=True)

    features = snapshot_features(rows, observed, network, t0s)
    features["pred"] = QuantileGBM.load(args.model).predict(model_data.prepare(features))
    print(f"predicted {len(features):,} snapshot edge rows", flush=True)
    loaded = [load_edge_costs(spec, t0s) for spec in args.edge_costs]
    external = {name: rides for name, rides, _ in loaded}
    joint = {name: transfers for name, _, transfers in loaded if transfers}
    if joint and not args.wait_models:
        raise SystemExit("transfer predictions need --wait-models for the origin wait")
    cost_sets = [LIGHTGBM, *external]
    routers = list(describe(cost_sets, bool(args.wait_models), list(joint))[0])

    origin_wait, origin_eta, transfer_model = {}, {}, None
    transfer_idx = np.flatnonzero(network.is_transfer)
    if args.wait_models:
        typical = pd.read_csv(args.wait_models / "typical_headway.csv")
        edge_cost = {(f, t): b for (f, t, x), b in zip(network.edge_keys, network.base) if not x}
        deps = waits.departures(rows, edge_cost)
        queries = pd.DataFrame({"node": np.repeat(network.node_ids, len(t0s)),
                                "t": np.tile(t0s, len(network.node_ids))})
        frame = waits.build_origin(deps, typical, queries)
        frame["pred"] = QuantileGBM.load(args.wait_models / "origin").predict(frame)
        index = {n: i for i, n in enumerate(network.node_ids)}
        # raw ETA: when the next train reaches the platform if it runs to schedule
        # from where it is now. Where there's no ETA, ORIGIN_HEADWAY_FRACTION of
        # the typical headway (the q90 of a random wait), as the live service
        # will serve it; with neither, no service.
        frame["eta"] = (frame["next_eta_sched_sec"].clip(lower=0)
                        .fillna(ORIGIN_HEADWAY_FRACTION * frame["typical_headway_sec"])
                        .fillna(waits.MAX_WAIT_SEC))
        for t0, part in frame.groupby("t"):
            nodes_at = [index[n] for n in part["node"]]
            vector = np.zeros(len(network.node_ids))
            vector[nodes_at] = part["pred"].to_numpy()
            origin_wait[int(t0)] = vector
            vector = np.zeros(len(network.node_ids))
            vector[nodes_at] = part["eta"].to_numpy()
            origin_eta[int(t0)] = vector
        transfer_model = QuantileGBM.load(args.wait_models / "transfer")
        transfer_from = pd.Series([network.edge_keys[e][0] for e in transfer_idx])
        transfer_to = [network.edge_keys[e][1] for e in transfer_idx]
        print(f"predicted origin waits for {len(frame):,} (node, t0)", flush=True)

    stations = sorted(s for s, nodes in network.nodes_at.items()
                      if any(not network.is_transfer[e] for n in nodes for e in network.adjacency[n]))
    schedule_costs = {p: network.costs(graph, p) for p in STATES}
    # An edge the day-average graph lacks doesn't run in it.
    day_costs = {d: np.array([day_graph._period_times.get(k, {}).get(d, math.inf)
                              for k in network.edge_keys], dtype=float) for d in DAYS}

    trips, node_rows = [], []
    for snap, snapshot_rows in features.groupby("t0", sort=True):
        snap = int(snap)
        t0 = snap + lead_sec  # when the rider leaves
        local = datetime.fromtimestamp(t0, TZ)
        # The timetable for the departure state; everything else is as of the snapshot.
        period = state_of_time(local)
        keys = list(zip(snapshot_rows["from_node"], snapshot_rows["to_node"]))
        # Ride edges arriving at a station with a live incident alert at the snapshot.
        alerted_edges = {network.edge_index[(f, t, False)] for (f, t), count
                         in zip(keys, snapshot_rows["station_alert_count"]) if count > 0}
        predictions = {LIGHTGBM: dict(zip(keys, snapshot_rows["pred"])),
                       **{name: by_t0[snap] for name, by_t0 in external.items()}}
        running = {network.edge_index[(f, t, False)]
                   for (f, t), due in zip(keys, snapshot_rows["dep"].notna()) if due}
        not_running = np.array([not x and e not in running
                                for e, x in enumerate(network.is_transfer)])

        costs = {"sched": schedule_costs[period]}
        costs["sched_live"] = np.where(not_running, math.inf, costs["sched"])
        costs["sched_day"] = day_costs[day_of(period)]
        costs["sched_day_live"] = np.where(not_running, math.inf, costs["sched_day"])
        start = start_eta = None
        if args.wait_models:
            start, start_eta = origin_wait[snap], origin_eta[snap]
            frame = waits.build_transfer(typical, pd.DataFrame({"node": transfer_to, "t": snap}),
                                         transfer_from, network.walk[transfer_idx])
            runs = np.isfinite(schedule_costs[period][transfer_idx])
            transfer_cost = np.where(runs, network.walk[transfer_idx]
                                     + transfer_model.predict(frame), math.inf)
        for cs, predicted in predictions.items():
            edges = network.edges.copy()
            edges["pred"] = [predicted.get((f, t), np.nan) if not x else np.nan
                             for f, t, x in network.edge_keys]
            reweighted = GraphSnapshot(at=local, service_period=day_of(period),
                                       service_state=period, edges=edges) \
                .weighted_graph(graph, edges["pred"])
            ride = network.costs(reweighted, period)
            costs[router_key(cs, False, False)] = ride
            costs[router_key(cs, False, True)] = np.where(not_running, math.inf, ride)
            if args.wait_models:
                waited = ride.copy()
                waited[transfer_idx] = transfer_cost
                costs[router_key(cs, True, False)] = waited
                costs[router_key(cs, True, True)] = np.where(not_running, math.inf, waited)
            if cs in joint:
                predicted_transfers = joint[cs][snap]
                own = np.array([predicted_transfers.get(network.edge_keys[e][:2], np.nan)
                                for e in transfer_idx])
                joined = ride.copy()
                # a transfer the model doesn't cover keeps the transfer-wait model's cost
                joined[transfer_idx] = np.where(runs, np.where(np.isnan(own), transfer_cost, own),
                                                math.inf)
                for mode in JOINT_MODES:
                    costs[f"{cs}_{mode}"] = joined
                    costs[f"{cs}_{mode}_live"] = np.where(not_running, math.inf, joined)

        # Pairs drawn from the departure time, so runs with different --lead-min
        # route the same trips and can be compared trip by trip.
        rng = np.random.default_rng([args.seed, t0])
        for _ in range(args.pairs_per_time):
            origin, destination = rng.choice(stations, size=2, replace=False)
            trip = {"trip_id": len(trips), "t0": t0, "snapshot_t": snap,
                    "lead_min": args.lead_min, "depart_local": local.isoformat(),
                    "depart_hour": local.hour, "weekend": int(local.weekday() >= 5),
                    "origin": origin, "destination": destination}
            for name in routers:
                router_start = (start_eta if "_joint_eta" in name
                                else start if "_waits" in name or "_joint" in name else None)
                path = network.route(costs[name], origin, destination, router_start)
                if path is None:
                    trip.update({f"{name}_status": "no route", f"{name}_path": ""})
                    continue
                true, boarded, status = observed.replay(network, path, t0)
                cumulative = {"scheduled": np.cumsum(costs["sched"][path])}
                first = float(start[network.edge_from[path[0]]]) if args.wait_models else 0.0
                first_eta = float(start_eta[network.edge_from[path[0]]]) if args.wait_models \
                    else 0.0
                for cs in cost_sets:
                    cumulative[estimate_key(cs, False)] = np.cumsum(costs[cs][path])
                    if args.wait_models:
                        cumulative[estimate_key(cs, True)] = \
                            first + np.cumsum(costs[router_key(cs, True, False)][path])
                    if cs in joint:
                        joined_path = np.cumsum(costs[f"{cs}_joint"][path])
                        cumulative[f"pred_{cs}_joint"] = first + joined_path
                        cumulative[f"pred_{cs}_joint_eta"] = first_eta + joined_path
                trip.update({
                    f"{name}_status": status,
                    f"{name}_alerted": bool(alerted_edges.intersection(int(e) for e in path)),
                    f"{name}_path": ">".join(network.node_ids[network.edge_to[e]] for e in path),
                    f"{name}_edges": len(path),
                    f"{name}_transfers": int(network.is_transfer[path].sum()),
                    **{f"{name}_{e}": float(values[-1]) for e, values in cumulative.items()},
                })
                if args.wait_models:
                    trip[f"{name}_origin_wait_pred"] = first
                    trip[f"{name}_origin_eta_pred"] = first_eta
                if status == "ok":
                    # A walk-only path (a transfer between linked stations) never boards.
                    wait = boarded - t0 if boarded is not None else 0
                    trip.update({f"{name}_true": float(true[-1]),
                                 f"{name}_true_boarded": float(true[-1] - wait),
                                 f"{name}_initial_wait": float(wait)})
                for step, edge in enumerate(path):
                    node_rows.append({
                        "trip_id": trip["trip_id"], "path": name, "step": step,
                        "node": network.node_ids[network.edge_to[edge]],
                        "via_transfer": bool(network.is_transfer[edge]),
                        **{e: float(values[step]) for e, values in cumulative.items()},
                        "true": float(true[step]) if step < len(true) else np.nan,
                    })
            trips.append(trip)

    trips = pd.DataFrame(trips)
    nodes = pd.DataFrame(node_rows)
    summary = summarize(trips, nodes, cost_sets)
    summary.update({
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": str(args.model), "test_dates": dates,
        "wait_models": str(args.wait_models) if args.wait_models else None,
        "edge_costs": args.edge_costs,
        "last_departure": datetime.fromtimestamp(int(t0s[-1]) + lead_sec, TZ).isoformat(),
        "step_min": args.step_min, "lead_min": args.lead_min,
        "pairs_per_time": args.pairs_per_time, "seed": args.seed,
    })

    lead = f"_lead{args.lead_min}" if args.lead_min else ""
    report = args.out or Path(f"docs/benchmarks/paths_{dates[0]}_{dates[-1]}{lead}.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str))
    trips.to_csv(report.with_name(report.stem + "_trips.csv"), index=False)
    nodes.to_csv(report.with_name(report.stem + "_nodes.csv"), index=False)
    report.write_text(render(summary, report))
    print(f"wrote {report}")
    print(render(summary, report))


def render(s: dict, report: Path) -> str:
    def m(seconds):
        return f"{seconds / 60:.1f}"

    def signed_m(seconds):
        return f"{seconds / 60:+.1f}"

    def pct(share):
        return f"{share:.0%}"

    acc = s["accuracy"]
    est = acc["estimates"]
    j = acc["journey"]
    names = {e: s["estimate_names"][e] for e in est}
    router_names = s["router_names"]

    def error_table(errors: dict) -> str:
        return format_table([
            {"estimate": names[e], "mean error": signed_m(errors[e]["mean_error"]),
             "median abs error": m(errors[e]["median_abs_error"]),
             "actual took longer": pct(errors[e]["share_actual_later"]),
             "within ±2 min": pct(errors[e]["share_within_2min"]),
             "within ±5 min": pct(errors[e]["share_within_5min"])}
            for e in est],
            ["estimate", "mean error", "median abs error", "actual took longer",
             "within ±2 min", "within ±5 min"])

    bullets = ["- **Scheduled**: the graph's schedule costs added up (what the router quotes today)"]
    for cs in s["cost_sets"]:
        name = cost_set_name(cs)
        bullets.append(f"- **{name}**: {name}'s ride costs added up; transfers keep the schedule's "
                       "walk + half-headway cost; no wait for the first train")
    if any(e.endswith("_joint") for e in est):
        bullets.append("- **… joint**: one model's ride costs *and* transfer costs (walk + wait "
                       "for the next train); the first train's wait from the origin-wait model "
                       "(its live ETA, corrected) or the raw live ETA")
    if any(e.endswith("_waits") for e in est):
        bullets.append("- **… + waits**: the same ride costs, plus the origin-wait model's wait for "
                       "the first train and the transfer-wait model's wait at each transfer")
    bullets.append("- **Actual**: replayed from what trains did, from arriving at the origin "
                   "station to arriving at the destination")
    first_wait = (f"The actual wait for the first train averaged {m(acc['first_wait']['mean'])} "
                  f"(median {m(acc['first_wait']['median'])})")
    if "first_wait_pred" in acc:
        first_wait += (f"; the origin-wait model predicted {m(acc['first_wait_pred']['mean'])} on "
                       f"average (median {m(acc['first_wait_pred']['median'])})")
    if s.get("test_dates"):
        span = f"on test dates {', '.join(s['test_dates'])} (to {s['last_departure']})"
    else:
        span = f"from {s['test_start']} to {s['last_departure']}"
    models = f"Models: LightGBM `{s['model']}`"
    if s.get("wait_models"):
        models += f", waits `{s['wait_models']}`"
    for spec in s.get("edge_costs") or []:
        models += f", {spec.partition('=')[0].upper()} `{spec.partition('=')[2]}`"

    lead_note = ""
    if s.get("lead_min"):
        lead_note = (f" **Planned {s['lead_min']} min ahead:** every model input -- ride costs, "
                     "waits, the service filter -- comes from a snapshot taken that long before "
                     "the rider leaves; actual times are replayed from the departure.")
    parts = [
        "# Path benchmark: estimated vs actual journey time",
        f"Generated by `python3 -m ml_model.benchmark_paths` on {s['created']}. {models}. "
        f"Departures every {s['step_min']} min {span}, {s['pairs_per_time']} random station "
        f"pairs each (seed {s['seed']}), {s['trips_sampled']} trips. All times in minutes. "
        f"Per-trip rows in `{report.stem}_trips.csv`, per-node costs in `{report.stem}_nodes.csv`."
        + lead_note,

        "## 1. How close is each estimate to the actual time?\n\n"
        f"All numbers are for the **same path per trip**: the route schedule costs pick, "
        f"skipping lines with no train due ({acc['trips']} trips that could be replayed).\n\n"
        + "\n".join(bullets) + "\n\n"
        + format_table(
            [{"": names[e], "mean": m(j[e]["mean"]), "median": m(j[e]["median"]),
              "90th pct": m(j[e]["p90"])} for e in est]
            + [{"": "**Actual**", "mean": m(j["actual"]["mean"]),
                "median": m(j["actual"]["median"]), "90th pct": m(j["actual"]["p90"])}],
            ["", "mean", "median", "90th pct"])
        + "\n\n**Error per trip**, actual minus estimate (positive: the trip took longer than "
        "quoted):\n\n" + error_table(acc["errors_station"])
        + f"\n\n{first_wait}. The models predict the 90th percentile of each piece, so they are "
        "meant to overestimate: \"actual took longer\" should be low, not 50%.\n\n"
        "Same comparison **from boarding the first train**, leaving the first wait out of both "
        "sides:\n\n" + error_table(acc["errors_boarding"]),

        "## 2. Along the journey\n\n"
        "Mean time to reach each node since arriving at the origin station, grouped by how many "
        "stops in it is.\n\n"
        + format_table([
            {"stops in": r["stops"], "nodes": r["nodes"],
             **{names[e]: m(r[e]) for e in est}, "actual": m(r["actual"]),
             **{f"longer than {names[e]}": pct(r[f"{e}_late"]) for e in est}}
            for r in acc["along_journey"]],
            ["stops in", "nodes", *(names[e] for e in est), "actual",
             *(f"longer than {names[e]}" for e in est)]),

        "## 3. By departure time\n\nFrom arrival at the origin station.\n\n"
        + format_table([
            {"departure": r["departure"], "trips": r["trips"],
             **{f"{names[e]} mean error": signed_m(r[e]["mean_error"]) for e in est},
             **{f"longer than {names[e]}": pct(r[e]["share_actual_later"]) for e in est}}
            for r in acc["by_departure"]],
            ["departure", "trips", *(f"{names[e]} mean error" for e in est),
             *(f"longer than {names[e]}" for e in est)]),
    ]

    def choice_table(choice: dict) -> str:
        return format_table([
            {"router": router_names.get(r, r), "actual mean": m(v["mean"]),
             "median": m(v["median"]), "90th pct": m(v["p90"]),
             "vs schedule (s)": f"{v.get('mean_change_vs_schedule', 0):+.0f}",
             "transfers": f"{v['transfers']:.2f}",
             "route differs": pct(v["differs_from_schedule"]),
             "faster": pct(v["faster_than_schedule"]),
             "slower": pct(v["slower_than_schedule"])}
            for r, v in choice["routers"].items()],
            ["router", "actual mean", "median", "90th pct", "vs schedule (s)", "transfers",
             "route differs", "faster", "slower"])

    choice = s.get("route_choice")
    if choice:
        parts.append(
            "## 4. Does routing on a model pick faster routes?\n\n"
            "A different question from the estimates: route once per router (all skipping lines "
            "with no train due) and compare the **actual** time of each router's own route, from "
            f"arrival at the origin station, on the {choice['trips']} trips where all of them "
            "replay. Faster/slower means by more than a minute against the schedule's route.\n\n"
            + choice_table(choice))

    alerted = s.get("alerted")
    if alerted:
        section = (
            "## 5. Trips through live alerts\n\n"
            f"Trips whose schedule route passes a station with a live incident alert at the "
            f"snapshot: {alerted['trips']} of {s['trips_sampled']} ({pct(alerted['share'])}), "
            f"{alerted['replayed']} of them replayable. Where alerts matter, if anywhere. "
            "Small samples: read differences of a minute or two as noise.")
        if "accuracy" in alerted:
            section += ("\n\n**Error per trip** on these trips, same path per trip, actual "
                        "minus estimate:\n\n" + error_table(alerted["accuracy"]["errors_station"]))
        if alerted.get("route_choice"):
            section += (f"\n\n**Route choice** on the {alerted['route_choice']['trips']} alerted "
                        "trips where every router replays:\n\n"
                        + choice_table(alerted["route_choice"]))
        parts.append(section)

    statuses = sorted({k for counts in s["status"].values() for k in counts})
    parts.append(
        "## Appendix: routers and replay\n\n" + format_table(
            [{"router": r, "costs": s["router_descriptions"][r],
              **{st: s["status"][r].get(st, 0) for st in statuses}}
             for r in s["routers"]], ["router", "costs", *statuses])
        + "\n\nThe live service filter removes ride edges with no train due within an hour "
        "of departure, which a live router knows from trip updates. Without it, today's "
        "router picks lines that aren't running on most trips, so those trips can't be replayed."
        "\n\nActual time of each router pair's routes, from arrival at the station, on trips "
        "where both replay:\n\n" + format_table(
            [{"comparison": f"{p['label']}: {p['a']} → {p['b']}",
              "trips": p["slices"]["all trips"]["trips"],
              "first": m(p["slices"]["all trips"]["a_true_mean"]),
              "second": m(p["slices"]["all trips"]["b_true_mean"]),
              "second faster": pct(p["slices"]["all trips"]["share_b_faster"]),
              "second slower": pct(p["slices"]["all trips"]["share_b_slower"])}
             for p in s["pairs"] if p["slices"]],
            ["comparison", "trips", "first", "second", "second faster", "second slower"]))
    return "\n\n".join(parts) + "\n"


if __name__ == "__main__":
    main()
