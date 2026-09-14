"""Client for Photon (photon.komoot.io), an OpenStreetMap search engine.

Nominatim wants the words roughly as OSM has them ("PS 26 the Jesse Owens
School" finds nothing; "Jesse Owens School" does). Photon is typo- and
word-order tolerant, so it's the first try for place names. The public
instance is free, needs no key, and asks for fair use: no bulk geocoding.
"""

from __future__ import annotations

import requests

PHOTON_URL = "https://photon.komoot.io/api/"


class PhotonClient:
    def __init__(self, user_agent: str, timeout: float = 10.0):
        self._session = requests.Session()
        self._session.headers["User-Agent"] = user_agent
        self._timeout = timeout

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "PhotonClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def search(self, query: str, limit: int = 5,
               bbox: tuple[float, float, float, float] | None = None) -> list[dict]:
        """Up to `limit` places matching `query`, inside `bbox` (west, south, east,
        north) when given: [{"name", "address", "lat", "lon", "kind"}]."""
        params: dict = {"q": query, "limit": limit}
        if bbox:
            params["bbox"] = ",".join(str(v) for v in bbox)
        response = self._session.get(PHOTON_URL, params=params, timeout=self._timeout)
        response.raise_for_status()
        places = []
        for feature in response.json().get("features", []):
            p = feature.get("properties", {})
            lon, lat = feature["geometry"]["coordinates"][:2]
            street = " ".join(str(p[k]) for k in ("housenumber", "street") if p.get(k))
            area = p.get("district") or p.get("locality") or p.get("city")
            address = ", ".join(x for x in (street, area, p.get("postcode")) if x)
            places.append({"name": p.get("name") or street or query, "address": address,
                           "lat": float(lat), "lon": float(lon), "kind": p.get("osm_value") or p.get("type") or ""})
        return places
