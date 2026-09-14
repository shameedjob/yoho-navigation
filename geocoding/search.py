"""Find places in New York City by address or name, for the agent and the
alert scheduler.

  geocode(query)          one best match (lat, lon), or ValueError
  search_places(query)    several candidates to choose from

Both are restricted to the NYC area. Photon handles loose place names;
Nominatim is the fallback (and the first try for plain street addresses, which
it matches precisely). Neither needs a key; both ask for light use.
"""

from __future__ import annotations

import logging
import math
import re

from .nominatim import NominatimClient
from .photon import PhotonClient

log = logging.getLogger(__name__)

USER_AGENT = "yoho-navigation-agent/0.1"
# west, south, east, north: the five boroughs with a little margin.
NYC_BBOX = (-74.2591, 40.4774, -73.7004, 40.9176)
_SAME_PLACE_M = 300


def _looks_like_address(query: str) -> bool:
    return bool(re.match(r"\s*\d+[a-zA-Z]?\s+\S", query))


def _distance_m(a: dict, b: dict) -> float:
    dlat = math.radians(b["lat"] - a["lat"])
    dlon = math.radians(b["lon"] - a["lon"]) * math.cos(math.radians(a["lat"]))
    return 6_371_000 * math.hypot(dlat, dlon)


def _dedupe(places: list[dict]) -> list[dict]:
    """Drop repeats: same name within _SAME_PLACE_M (e.g. a station's entrances)."""
    kept: list[dict] = []
    for place in places:
        name = place["name"].casefold()
        if not any(k["name"].casefold() == name and _distance_m(k, place) < _SAME_PLACE_M for k in kept):
            kept.append(place)
    return kept


def search_places(query: str, limit: int = 5, *, photon=None, nominatim=None) -> list[dict]:
    """Up to `limit` NYC places for `query`, best first:
    [{"name", "address", "lat", "lon", "kind"}]. Empty when nothing matches.
    A service that errors is skipped, not fatal."""
    photon = photon or PhotonClient(USER_AGENT)
    nominatim = nominatim or NominatimClient(USER_AGENT)
    sources = [nominatim, photon] if _looks_like_address(query) else [photon, nominatim]
    places: list[dict] = []
    for source in sources:
        try:
            found = source.search(query, limit=limit, bbox=NYC_BBOX)
        except Exception as exc:
            log.warning("place search via %s failed: %s", type(source).__name__, exc)
            continue
        places = _dedupe(places + found)
        if len(places) >= limit:
            break
    return places[:limit]


def geocode(query: str, **clients) -> tuple[float, float]:
    """(lat, lon) of the best NYC match for an address or place name."""
    places = search_places(query, limit=1, **clients)
    if not places:
        raise ValueError(f"no place in New York City matches {query!r}; try search_places with fewer words")
    return places[0]["lat"], places[0]["lon"]
