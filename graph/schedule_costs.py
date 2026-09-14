"""Per-state schedule costs from GTFS stop times, shared by the subway and bus
loaders so both apply the same rules.

The cost logic is unchanged from the per-day graph, just computed on each
state's trips (see graph/service_states.py):

  ride      mean scheduled time between two consecutive stops, over trips
            arriving at the second stop in the state
  headway   mean gap between a route's arrivals at a stop in the state
  transfer  walk + headway/2 of the route being boarded

A state with fewer than MIN_BUCKET_TRIPS trips takes the whole day's figure,
and one with none leaves the edge closed.
"""

from __future__ import annotations

from collections import defaultdict

from graph.service_states import DAYS, mean_gap, per_state, stop_time_states

# The state an edge's period-agnostic `time` describes, and the one routing
# assumes when given none: weekday midday, the most typical service.
DEFAULT_STATE = "Weekday:10-16"


def _nested_lists():
    return defaultdict(list)


class ScheduleCosts:
    """Pass bucketed=False to key costs by service day alone ("Weekday"), as
    the graph did before time-of-day buckets -- kept to benchmark against."""

    def __init__(self, bucketed: bool = True) -> None:
        self.bucketed = bucketed
        # (stop, route) -> state -> timeline arrival seconds; and -> day -> raw
        self.arrivals: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(_nested_lists)
        self.day_arrivals: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(_nested_lists)
        # (route, from_stop, to_stop) -> state or day -> ride seconds
        self.legs: dict[tuple[str, str, str], dict[str, list[int]]] = defaultdict(_nested_lists)
        self.day_legs: dict[tuple[str, str, str], dict[str, list[int]]] = defaultdict(_nested_lists)
        self._sorted = False

    def add_stop_time(self, days: tuple[str, ...], route: str, stop: str, arrival: int,
                      prev_stop: str | None = None, prev_departure: int | None = None) -> None:
        """Record a trip reaching `stop` at `arrival` (GTFS seconds) on each of
        `days`, and the leg from `prev_stop` when the trip came from one."""
        leg = arrival - prev_departure if prev_stop is not None and prev_departure is not None else None
        for day in days:
            self.day_arrivals[(stop, route)][day].append(arrival)
            if leg is not None:
                self.day_legs[(route, prev_stop, stop)][day].append(leg)
            if not self.bucketed:
                continue
            for name, t in stop_time_states(day, arrival):
                self.arrivals[(stop, route)][name].append(t)
                if leg is not None:
                    self.legs[(route, prev_stop, stop)][name].append(leg)
        self._sorted = False

    def _sort(self) -> None:
        if self._sorted:
            return
        for table in (self.arrivals, self.day_arrivals):
            for by_key in table.values():
                for times in by_key.values():
                    times.sort()
        self._sorted = True

    def ride_times(self, key: tuple[str, str, str]) -> dict[str, float]:
        """State -> mean ride seconds for a (route, from_stop, to_stop) leg."""
        if not self.bucketed:
            return {day: sum(t) / len(t) for day, t in self.day_legs.get(key, {}).items() if t}
        by_state = self.legs.get(key, {})
        return per_state(
            {name: sum(t) / len(t) for name, t in by_state.items() if t},
            {name: len(t) for name, t in by_state.items()},
            {day: sum(t) / len(t) for day, t in self.day_legs.get(key, {}).items() if t},
        )

    def headways(self, stop: str, route: str) -> dict[str, float]:
        """State -> mean headway seconds of `route` at `stop`."""
        self._sort()
        by_state = self.arrivals.get((stop, route), {})
        by_day = self.day_arrivals.get((stop, route), {})
        if not self.bucketed:
            return {day: gap for day in DAYS if (gap := mean_gap(by_day.get(day, []))) is not None}
        return per_state(
            {name: gap for name, t in by_state.items() if (gap := mean_gap(t)) is not None},
            {name: len(t) for name, t in by_state.items()},
            {day: gap for day in DAYS if (gap := mean_gap(by_day.get(day, []))) is not None},
        )

    def served(self):
        """Every (stop, route) with at least one arrival."""
        return self.arrivals.keys() | self.day_arrivals.keys()


def base_cost(by_state: dict[str, float]) -> int:
    """The period-agnostic weight: the default state's (or, unbucketed, the
    default day's) cost when the edge runs then, else the mean over the states
    it does run in."""
    for default in (DEFAULT_STATE, DEFAULT_STATE.split(":")[0]):
        if default in by_state:
            return round(by_state[default])
    return round(sum(by_state.values()) / len(by_state))
