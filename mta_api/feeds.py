"""Feed identifiers for the MTA's real-time GTFS feeds.

Reference: https://api.mta.info/#/subwayRealTimeFeeds
"""

BASE_URL = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds"

# Maps a subway line/group to its GTFS-realtime feed path.
SUBWAY_FEEDS = {
    "1234567S": "nyct%2Fgtfs",
    "ACE": "nyct%2Fgtfs-ace",
    "BDFM": "nyct%2Fgtfs-bdfm",
    "G": "nyct%2Fgtfs-g",
    "JZ": "nyct%2Fgtfs-jz",
    "NQRW": "nyct%2Fgtfs-nqrw",
    "L": "nyct%2Fgtfs-l",
    "SIR": "nyct%2Fgtfs-si",
}

ALERTS_FEED = "camsys%2Fall-alerts"


def feed_url(feed_key: str) -> str:
    """Return the full URL for a subway feed key (e.g. "ACE", "G")."""
    try:
        path = SUBWAY_FEEDS[feed_key]
    except KeyError:
        raise ValueError(
            f"Unknown feed key {feed_key!r}; valid keys are {sorted(SUBWAY_FEEDS)}"
        ) from None
    return f"{BASE_URL}/{path}"


def line_to_feed_key(line: str) -> str:
    """Map an individual subway line (e.g. "A", "6", "G") to its feed key."""
    line = line.upper()
    for feed_key in SUBWAY_FEEDS:
        if line in feed_key:
            return feed_key
    raise ValueError(f"Unknown subway line {line!r}")
