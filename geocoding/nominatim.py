"""Client for OpenStreetMap's Nominatim geocoding API."""

from __future__ import annotations

import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


class NominatimClient:
    """Geocodes addresses to (lat, lon) using OpenStreetMap's Nominatim API.

    Nominatim's usage policy
    (https://operations.osmfoundation.org/policies/nominatim/) requires a
    descriptive User-Agent identifying the calling application, and caps
    the public instance at 1 request/second -- this client doesn't
    rate-limit for you, so space out calls if geocoding more than one
    address.
    """

    def __init__(self, user_agent: str, timeout: float = 10.0):
        self._session = requests.Session()
        self._session.headers["User-Agent"] = user_agent
        self._timeout = timeout

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "NominatimClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def geocode(self, address: str) -> tuple[float, float] | None:
        """Return (lat, lon) for the given address, or None if not found."""
        response = self._session.get(
            NOMINATIM_URL,
            params={"q": address, "format": "json", "limit": 1},
            timeout=self._timeout,
        )
        response.raise_for_status()
        results = response.json()
        if not results:
            return None
        return float(results[0]["lat"]), float(results[0]["lon"])

    def search(self, query: str, limit: int = 5,
               bbox: tuple[float, float, float, float] | None = None) -> list[dict]:
        """Up to `limit` places matching `query`, restricted to `bbox`
        (west, south, east, north) when given: [{"name", "address", "lat", "lon", "kind"}]."""
        params = {"q": query, "format": "jsonv2", "limit": limit}
        if bbox:
            west, south, east, north = bbox
            params.update(viewbox=f"{west},{north},{east},{south}", bounded=1)
        response = self._session.get(NOMINATIM_URL, params=params, timeout=self._timeout)
        response.raise_for_status()
        places = []
        for r in response.json():
            parts = [p.strip() for p in r.get("display_name", "").split(",")]
            name = r.get("name") or (parts[0] if parts else query)
            rest = [p for p in parts if p and p != name]
            places.append({"name": name, "address": ", ".join(rest[:4]), "lat": float(r["lat"]),
                           "lon": float(r["lon"]), "kind": r.get("type") or r.get("category") or ""})
        return places
