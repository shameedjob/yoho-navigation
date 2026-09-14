"""Leaving modes and lines out of a route: graph search, parsing, and the tools."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from agent import routing
from agent.routing import Avoid, parse_avoid
from graph import Graph
from graph.stop_node import StopNode


def small_graph() -> Graph:
    """A -> B on the 6 express (fast) or the 4 (slower); A -> B on a bus (slowest)."""
    g = Graph(default_period=None)
    for node_id, vehicle, mode in [("A6X", "6X", "subway"), ("B6X", "6X", "subway"), ("A4", "4", "subway"),
                                   ("B4", "4", "subway"), ("Abus", "M101", "bus"), ("Bbus", "M101", "bus")]:
        g.add_node(StopNode(id=node_id, stop_id=node_id, vehicle=vehicle, mode=mode, lat=0, lon=0, paths=[]))
    g.add_edge("A6X", "B6X", 60)
    g.add_edge("A4", "B4", 120)
    g.add_edge("Abus", "Bbus", 600)
    return g


def best(g, **ignore):
    """Like agent.routing._route: the start and end ids come from stops still allowed."""
    avoid = Avoid(frozenset(ignore.get("ignore_modes", ())), frozenset(ignore.get("ignore_routes", ())))
    starts = {n: 0.0 for n in ("A6X", "A4", "Abus") if avoid.allows(g.get_node(n))}
    ends = {n: 0.0 for n in ("B6X", "B4", "Bbus") if avoid.allows(g.get_node(n))}
    if not starts or not ends:
        return None
    result = g.shortest_path(next(iter(starts)), next(iter(ends)), start_costs=starts, end_costs=ends,
                             service_period=None, **ignore)
    return result[0] if result else None


def test_ignore_routes_excludes_a_line_and_its_express_variant():
    g = small_graph()
    assert best(g) == ["A6X", "B6X"]
    assert best(g, ignore_routes={"6"}) == ["A4", "B4"]       # "6" also drops the 6X
    assert best(g, ignore_routes={"6", "4"}) == ["Abus", "Bbus"]
    assert best(g, ignore_modes={"bus"}, ignore_routes={"6", "4"}) is None


def test_parse_avoid_reads_loose_names(monkeypatch):
    monkeypatch.setattr(routing, "known_lines", lambda: {"L", "4", "5", "B38", "M15"})
    avoid = parse_avoid(["Buses"], ["the L train", "4", "b38"])
    assert avoid.modes == {"bus"} and avoid.lines == {"L", "4", "B38"}
    assert parse_avoid(["trains"]).modes == {"subway"}
    with pytest.raises(ValueError, match="nothing left"):
        parse_avoid(["subway", "bus"])
    with pytest.raises(ValueError, match="unknown line"):
        parse_avoid(lines=["Q99"])
    with pytest.raises(ValueError, match="subway' or 'bus"):
        parse_avoid(["ferry"])


def test_route_tool_passes_avoid_through_and_reports_it(monkeypatch):
    from agent.agent_interaction import RouteLog, make_route_tools
    monkeypatch.setattr(routing, "known_lines", lambda: {"L", "4"})
    seen = []

    def get_path(start, end, departure_time=None, avoid=None):
        seen.append(avoid)
        if avoid and "L" in avoid.lines and "4" in avoid.lines:
            raise ValueError("no path found while avoiding {'lines': ['4', 'L']}")
        return {"steps": [{"stop_id": "1", "stop_name": "A", "mode": "subway", "route": "G", "lat": 40.7, "lon": -73.9},
                          {"stop_id": "2", "stop_name": "B", "mode": "subway", "route": "G", "lat": 40.71, "lon": -73.91}],
                "total_time_sec": 600, "walk_in_sec": 60, "walk_out_sec": 60, "service_state": "Weekday:10-16"}

    monkeypatch.setitem(sys.modules, "agent.tools", SimpleNamespace(get_path=get_path))
    tool = {t.tool_name: t for t in make_route_tools(RouteLog())}["get_path"]
    ok = tool(start=(40.7, -73.9), end=(40.71, -73.91), avoid_modes=["bus"], avoid_lines=["the L"])
    assert seen[-1] == Avoid(frozenset({"bus"}), frozenset({"L"})) and ok["avoided"] == {"modes": ["bus"], "lines": ["L"]}
    assert tool(start=(40.7, -73.9), end=(40.71, -73.91), avoid_lines=["L", "4"])["error"] == "no_route"
    assert tool(start=(40.7, -73.9), end=(40.71, -73.91), avoid_lines=["Q99"])["error"] == "bad_avoid"
    assert "avoided" not in tool(start=(40.7, -73.9), end=(40.71, -73.91))
