"""Route legs and the route payload's trip ends (agent/directions.py, RouteLog)."""

from __future__ import annotations

import json

from agent.agent_interaction import HOME, RouteLog
from agent.directions import route_legs

STEPS = [{"stop_id": "1", "stop_name": "Bedford Av", "mode": "subway", "route": "L", "lat": 40.717, "lon": -73.957},
         {"stop_id": "2", "stop_name": "1 Av", "mode": "subway", "route": "L", "lat": 40.731, "lon": -73.982},
         {"stop_id": "3", "stop_name": "14 St-Union Sq", "mode": "subway", "route": "L", "lat": 40.735, "lon": -73.990}]
START, END = (40.7150, -73.9510), (40.7370, -73.9896)


def test_legs_walk_from_the_start_and_to_the_destination():
    legs = route_legs(STEPS, 240, 120, start=START, end=END)
    first, last = legs[0], legs[-1]
    assert first["type"] == "walk" and (first["from"]["lat"], first["from"]["lon"]) == START
    assert first["to"]["stop_name"] == "Bedford Av" and first["sec"] == 240
    assert last["from"]["stop_name"] == "14 St-Union Sq" and (last["to"]["lat"], last["to"]["lon"]) == END


def test_ends_without_coordinates_stay_placeholders():
    legs = route_legs(STEPS, 240, 120)
    assert legs[0]["from"] == "start" and legs[-1]["to"] == "destination"


def test_walk_only_trip_is_one_walk_between_the_ends():
    [leg] = route_legs(STEPS[:1], 400, 0, start=START, end=END)
    assert leg["type"] == "walk" and leg["from"]["stop_name"] == "Start" and leg["to"]["stop_name"] == "Destination"


def test_route_payload_has_start_and_end_but_never_home_coordinates():
    log = RouteLog()
    summary = log.record(STEPS, 1500, 240, 120, start=START, end=END)
    assert log.route["start"] == {"kind": "place", "lat": START[0], "lon": START[1]}
    assert log.route["end"] == {"kind": "place", "lat": END[0], "lon": END[1]}
    assert "start" not in summary and "end" not in summary  # the model gets the summary only

    home = (40.6782, -73.9442)
    log.record(STEPS, 1500, 240, 120, start=HOME, end=END)
    assert log.route["start"] == {"kind": "home"} and log.route["legs"][0]["from"] == "start"
    assert str(home[0]) not in json.dumps(log.route)
