"""Place search (geocoding/search.py) with fake Photon/Nominatim clients: no network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from geocoding.search import NYC_BBOX, geocode, search_places
from tests.test_web import WRITE, FakeAgent, env, login  # noqa: F401  (env is a fixture)


class FakeSource:
    def __init__(self, results=None, error=None):
        self.results, self.error, self.calls = results or [], error, []

    def search(self, query, limit=5, bbox=None):
        self.calls.append((query, limit, bbox))
        if self.error:
            raise self.error
        return list(self.results)


def place(name, lat, lon, kind="school", address=""):
    return {"name": name, "address": address, "lat": lat, "lon": lon, "kind": kind}


SCHOOL = place("P.S. 26 - The Jesse Owens School", 40.6916, -73.9314, address="1014 Lafayette Avenue, Brooklyn")


def test_place_names_try_photon_first_within_nyc():
    photon, nominatim = FakeSource([SCHOOL]), FakeSource([])
    assert search_places("PS 26 the Jesse Owens School", photon=photon, nominatim=nominatim)[0] == SCHOOL
    assert photon.calls == [("PS 26 the Jesse Owens School", 5, NYC_BBOX)]


def test_street_addresses_try_nominatim_first():
    photon, nominatim = FakeSource([place("A", 1, 1)]), FakeSource([place("11 West 53rd Street", 40.76, -73.98)])
    assert search_places("11 W 53rd St", photon=photon, nominatim=nominatim)[0]["name"] == "11 West 53rd Street"


def test_results_are_merged_deduplicated_and_capped():
    entrances = [place("Atlantic Av-Barclays Ctr", 40.6844, -73.9783, "stop"),
                 place("Atlantic Av-Barclays Ctr", 40.6829, -73.9794, "stop")]  # ~200 m apart: same station
    photon = FakeSource([place("Barclays Center", 40.6825, -73.9753, "stadium"), *entrances])
    nominatim = FakeSource([place("Barclays Center", 40.6826, -73.9752, "stadium"), place("Other", 40.7, -73.9)])
    names = [p["name"] for p in search_places("Barclays Center", photon=photon, nominatim=nominatim)]
    assert names == ["Barclays Center", "Atlantic Av-Barclays Ctr", "Other"]


def test_a_failing_service_is_skipped():
    photon, nominatim = FakeSource(error=RuntimeError("503")), FakeSource([SCHOOL])
    assert search_places("Jesse Owens School", photon=photon, nominatim=nominatim) == [SCHOOL]


def test_geocode_takes_the_best_match_or_raises():
    assert geocode("Jesse Owens", photon=FakeSource([SCHOOL]), nominatim=FakeSource()) == (40.6916, -73.9314)
    with pytest.raises(ValueError, match="search_places"):
        geocode("nowhere at all", photon=FakeSource(), nominatim=FakeSource())


def test_chat_sends_searched_places_to_pin_unless_a_route_was_planned(env, monkeypatch):  # noqa: F811
    login(env)
    original = FakeAgent.__call__
    log = SimpleNamespace(route=None, places=[SCHOOL])

    def call(self, prompt):
        self.route_log = log
        return original(self, prompt)

    monkeypatch.setattr(FakeAgent, "__call__", call)
    body = env.client.post("/api/chat", json={"message": "PS 26"}, headers=WRITE).get_json()
    assert body["places"] == [SCHOOL] and body["route"] is None

    log.route = {"legs": [], "directions": "Take the G."}
    body = env.client.post("/api/chat", json={"message": "route there"}, headers=WRITE).get_json()
    assert body["places"] is None and body["route"] == log.route
