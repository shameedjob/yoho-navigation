"""Station alerts/delays (agent/station_status.py) and multi-stop trips -- faked
live data and routing: no network, no torch."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from accounts.home import Home, save_home
from agent import station_status as SS
from agent.agent_interaction import RouteLog, make_route_tools, make_user_tools
from tests.test_web import UID, WRITE, FakeAgent, env, login  # noqa: F401  (env is a fixture)

NY = ZoneInfo("America/New_York")


def node(stop_id, route, alerts=0, types=""):
    return {"node_id": f"{stop_id}::{route}", "stop_id": stop_id, "route": route,
            "station_alert_count": alerts, "station_alert_types": types}


NODES = [node("626N", "4", 1, "delays"), node("626N", "6X"), node("626S", "4"),   # 86 St (Lexington)
         node("635N", "4"), node("635S", "6"),                                   # 14 St-Union Sq
         node("L03N", "L")]                                                      # Union Sq on the L
WAITS = {"626N::4": {"next_train_delay_sec": 150.0}, "635N::4": {"next_train_delay_sec": 420.0},
         "635S::6": {"next_train_delay_sec": 30.0}, "L03N::L": {"next_train_delay_sec": 0.0}}
live = lambda: (NODES, WAITS)
headers = lambda node_id: ["Uptown 4/6 trains are delayed: door problem at 86 St."] if node_id == "626N::4" else []


def test_platforms_roll_up_to_stations_with_worst_delay_and_alert_text():
    rows = {s["station_id"]: s for s in SS.station_status(["626", "635", "L03"], live=live, alert_headers=headers)}
    assert rows["626"]["status"] == "alert" and rows["626"]["alerts"] == ["Uptown 4/6 trains are delayed: door problem at 86 St."]
    assert rows["626"]["lines"] == ["4", "6X"] and rows["626"]["delay_sec"] == 150
    assert rows["635"]["status"] == "delayed" and rows["635"]["delay_sec"] == 420 and rows["635"]["delayed_lines"] == ["4"]
    assert rows["L03"]["status"] == "ok" and rows["L03"]["name"] == "14 St-Union Sq"


def test_line_filter_matches_express_variant_and_only_problems_drops_ok():
    rows = SS.station_status(line="6", live=live, alert_headers=headers)
    assert [(r["station_id"], r["lines"]) for r in rows] == [("635", ["6"]), ("626", ["6X"])]  # both on time: bigger delay first
    rows = SS.station_status(line="4", only_problems=True, live=live, alert_headers=headers)
    assert [r["status"] for r in rows] == ["alert", "delayed"]  # alerts first, then worst delay


def test_route_problems_only_counts_delays_on_lines_the_route_rides():
    step = lambda stop, route: {"stop_id": stop, "stop_name": stop, "mode": "subway", "route": route, "lat": 0, "lon": 0}
    # the 6 from 14 St-Union Sq to 86 St: the 4 is 7 min late at Union Sq, but the route doesn't ride it
    problems = SS.route_problems([step("635S", "6"), step("626N", "6")], live=live, alert_headers=headers)
    assert problems == [{"name": "86 St", "alerts": ["Uptown 4/6 trains are delayed: door problem at 86 St."],
                         "alert_types": ["delays"], "delays": []}]
    on_the_4 = SS.route_problems([step("635N", "4"), step("626N", "4")], live=live, alert_headers=headers)
    assert [(p["name"], p["delays"]) for p in on_the_4] == [
        ("14 St-Union Sq", [{"mode": "subway", "route": "4", "delay_min": 7}]),
        ("86 St", [{"mode": "subway", "route": "4", "delay_min": 2}])]
    assert SS.route_problems([step("L03N", "L"), step("L03N", "L")], live=live, alert_headers=headers) == []


def test_find_stations_by_loose_name():
    names = {SS.stations()[i]["name"] for i in SS.find_stations("union square")}
    assert names == {"14 St-Union Sq"}
    assert "626" in SS.find_stations("86 Street")
    assert not {SS.stations()[i]["name"] for i in SS.find_stations("86 St")} - {"86 St"}  # not 186/286 St
    assert {SS.stations()[i]["name"] for i in SS.find_stations("Jay Street Metro")} == {"Jay St-MetroTech"}


def test_station_status_tool_checks_the_route_just_planned(monkeypatch):
    log = RouteLog()
    steps = [{"stop_id": "635N", "stop_name": "14 St-Union Sq", "mode": "subway", "route": "4", "lat": 40.73, "lon": -73.99},
             {"stop_id": "626N", "stop_name": "86 St", "mode": "subway", "route": "4", "lat": 40.78, "lon": -73.95}]
    monkeypatch.setitem(sys.modules, "agent.tools", SimpleNamespace())
    tool = {t.tool_name: t for t in make_route_tools(log)}["station_status"]
    assert tool(along_route=True)["error"] == "no_route"
    log.record(steps, 900)
    checked = []
    out = SS.station_status(["635", "626"], live=live, alert_headers=headers)
    monkeypatch.setattr("agent.station_status.station_status",
                        lambda ids, line=None, only_problems=False: checked.append(sorted(ids)) or out)
    result = tool(along_route=True)
    assert checked == [["626", "635"]]  # the route's stations, from its stop ids
    assert {s["name"] for s in result["stations"]} == {"86 St", "14 St-Union Sq"} and result["shown_on_map"]
    assert log.stations == out and "lat" not in result["stations"][0]  # coordinates go to the map, not the model


# --- multi-stop trips -----------------------------------------------------------

def fake_path(calls):
    def get_path(start, end, departure_time=None, avoid=None):
        calls.append((start, end, departure_time))
        # stations near, not at, the ends -- as real routes are (a stop at home would leak home's coordinates)
        return {"steps": [{"stop_id": "A1N", "stop_name": "A", "mode": "subway", "route": "4", "lat": 40.7001, "lon": -73.9001},
                          {"stop_id": "B1N", "stop_name": "B", "mode": "subway", "route": "4", "lat": 40.7501, "lon": -73.9501}],
                "total_time_sec": 1200, "walk_in_sec": 60, "walk_out_sec": 60, "service_state": "Weekday:10-16"}
    return get_path


@pytest.fixture
def trip_tool(env, monkeypatch):  # noqa: F811
    login(env)
    save_home(env.store, env.cipher, UID, Home("home", 40.6782, -73.9442))
    calls = []
    monkeypatch.setitem(sys.modules, "agent.tools", SimpleNamespace(get_path=fake_path(calls)))
    places = {"Katz's Deli": {"name": "Katz's Delicatessen", "address": "205 East Houston Street", "lat": 40.7223, "lon": -73.9874},
              "MoMA": {"name": "MoMA", "address": "11 West 53rd Street", "lat": 40.7616, "lon": -73.9775}}
    monkeypatch.setattr("geocoding.search_places", lambda q, limit=5: [places[q]] if q in places else [])
    log = RouteLog()
    tool = {t.tool_name: t for t in make_user_tools(UID, env.store, env.cipher, log)}["plan_multi_stop_trip"]
    return SimpleNamespace(tool=tool, calls=calls, log=log)


def test_multi_stop_trip_chains_legs_in_time_with_stays(trip_tool):
    result = trip_tool.tool(stops=["home", "Katz's Deli", "MoMA"], depart_at="2026-09-14 09:00", stay_minutes=[45])
    start = int(datetime(2026, 9, 14, 9, 0, tzinfo=NY).timestamp())
    assert [c[2] for c in trip_tool.calls] == [start, start + 1200 + 45 * 60]  # leg 2 leaves after arriving + stay
    assert [(l["from"], l["to"]) for l in result["legs"]] == [
        ("home", "Katz's Delicatessen, 205 East Houston Street"),
        ("Katz's Delicatessen, 205 East Houston Street", "MoMA, 11 West 53rd Street")]
    assert result["total_travel_minutes"] == 40 and result["legs"][1]["departs"] == "Mon Sep 14, 10:05 AM"


def test_multi_stop_trip_draws_one_route_with_waypoints_and_no_home_coordinates(trip_tool):
    trip_tool.tool(stops=["home", "Katz's Deli", "40.7616,-73.9775"])
    route = trip_tool.log.route
    assert route["start"] == {"kind": "home"} and route["end"] == {"kind": "place", "lat": 40.7616, "lon": -73.9775}
    assert route["waypoints"] == [{"kind": "place", "lat": 40.7223, "lon": -73.9874,
                                   "name": "Katz's Delicatessen, 205 East Houston Street", "order": 1}]
    assert route["directions"].startswith("Leg 1, to Katz's Delicatessen") and "Leg 2, to 40.7616,-73.9775" in route["directions"]
    assert "40.6782" not in json.dumps(route)


def test_multi_stop_trip_starts_here_from_the_device_location(trip_tool, env):  # noqa: F811
    here = (40.7359, -73.9911)
    tool = {t.tool_name: t for t in make_user_tools(UID, env.store, env.cipher, trip_tool.log, location=here)}["plan_multi_stop_trip"]
    result = tool(stops=["here", "MoMA"])
    assert trip_tool.calls[0][:2] == (here, (40.7616, -73.9775)) and result["legs"][0]["from"] == "your location"
    assert trip_tool.tool(stops=["my location", "MoMA"])["error"] == "no_location"  # none shared


def test_multi_stop_trip_names_the_stop_it_cannot_find(trip_tool):
    result = trip_tool.tool(stops=["home", "Nowhere Special"])
    assert result["error"] == "stop_not_found" and "Nowhere Special" in result["message"]


def test_chat_sends_station_status_for_the_map(env, monkeypatch):  # noqa: F811
    login(env)
    original = FakeAgent.__call__
    stations = [{"station_id": "626", "name": "86 St", "lat": 40.78, "lon": -73.96, "status": "alert"}]

    def call(self, prompt):
        self.route_log = SimpleNamespace(route=None, places=None, stations=stations)
        return original(self, prompt)

    monkeypatch.setattr(FakeAgent, "__call__", call)
    body = env.client.post("/api/chat", json={"message": "delays at 86 St?"}, headers=WRITE).get_json()
    assert body["stations"] == stations
