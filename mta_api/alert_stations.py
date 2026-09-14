"""Resolve the stations an alert header refers to, against the official station
list: https://data.ny.gov/Transportation/MTA-Subway-Stations/39hk-dx4f

Alert headers name places in prose ("...a signal problem at 5 Av/53 St",
"...between 69 St and 82 St-Jackson Hts"), and the resulting station ids join
straight to the `station_id` column of the delay output, since both are GTFS
stop ids with the N/S suffix stripped.

Why the matching works the way it does, measured over a sample week of 994
alert updates:

  * MTA writes station names verbatim from its own list, so a plain substring
    scan already resolves 95.2% of headers. Fuzzy matching is the fallback for
    the rest, not the primary mechanism -- running it first would be slower and
    strictly less accurate.
  * Of the 4.8% that no substring resolves, the large majority name no station
    at all: bridges and tunnels (24), a borough only (9), or nothing beyond the
    route (7). These are not near-misses and must not be forced to a match.
    Against those headers the best-scoring station name in the whole list
    reaches only 0.37 Dice / 0.61 sequence ratio, so the thresholds below sit
    far above the noise floor rather than being tuned to a guess.
  * Restricting candidates to the alert's own routes *loses* 5.3% recall
    outright, because `daytime_routes` omits shuttle codes and because alerts
    legitimately name stations off the affected line (a reroute's destination,
    a shared-track conflict). Routes are therefore a ranking boost, never a
    filter -- see ROUTE_ALIASES for the shuttle codes involved.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import requests

DOMAIN = "data.ny.gov"
STATIONS_DATASET = "39hk-dx4f"
STATIONS_URL = f"https://{DOMAIN}/resource/{STATIONS_DATASET}.json"

# A local GTFS feed, when present, is preferred over the Socrata dataset: it
# carries the identical 496 stations under the identical ids and names (checked
# field by field, zero disagreements), needs no network round trip, and -- see
# load_stations_gtfs -- describes routes in a vocabulary that matches the alerts
# archive exactly, which the Socrata column does not.
DEFAULT_GTFS_DIR = Path("data/gtfs_subway")

# Only needed for the Socrata source, whose `daytime_routes` column uses public
# route letters rather than the alerts archive's codes: SI alerts name stations
# listed as SIR, and the GS and H shuttles name stations listed as plain "S".
# The GTFS source needs none of this and ignores it.
ROUTE_ALIASES = {"SI": {"SIR"}, "GS": {"S"}, "H": {"A", "S"}, "FS": {"S"}}

# Trigrams, padded, scored by Dice coefficient. Trigrams suit these names
# because the discriminating content is short and numeric ("50 St" vs "59 St"),
# where whole-token overlap would tie and character edit distance alone would
# call them near-identical.
NGRAM_N = 3

# A span must clear BOTH scores to be accepted. Dice catches shared substrings
# regardless of order; the sequence ratio (a Levenshtein-family alignment score)
# penalizes transpositions Dice would forgive. Requiring both is what keeps
# "southbound e" from resolving to "South Ferry", which scored 0.37/0.61.
MIN_DICE = 0.55
MIN_RATIO = 0.70

# Token windows to slide over the header when scanning for a fuzzy name. NYC
# station names run from one token ("Astoria Blvd" normalizes to two) up to
# roughly eight ("42 St-Port Authority Bus Terminal"), so this brackets the
# realistic range without scoring every substring of the sentence.
MIN_WINDOW, MAX_WINDOW = 2, 8

DIRECTION_SUFFIX = re.compile(r"[NS]$")

# A header says which way a train is heading by naming its terminal: "8 Av-bound
# L trains", "Forest Hills-71 Av-bound M". That terminal is a direction label,
# not a place the incident touches, and it is the single biggest trap in reading
# these headers as ranges: of 215 alerts naming two or more stations in a sample
# week, the majority are one incident location plus one of these labels. Treating
# such a pair as an endpoint pair would expand a single-station delay into half a
# line. Stripped before matching, never used as a range endpoint.
BOUND_LABEL = re.compile(r"\S[^,;.]*?-bound\b", re.IGNORECASE)

# The phrasings that genuinely delimit a stretch of line. "between A and B" is
# the explicit form; "from A to B" is used for reroutes and express runs
# ("running express from 36 St to Atlantic Av-Barclays Ctr"). Both are read only
# after BOUND_LABEL has been removed, so "from Jay St-MetroTech to W 4 St" is a
# range while "8 Av-bound ... at Broadway Junction" is not.
# Both endpoints are confined to one sentence. Letting the first group run
# greedily across a full stop produced false ranges: "service to/from
# Jamaica-179 St is not running. Take F trains ... at Kew Gardens-Union Tpke"
# was read as a Jamaica-to-Kew-Gardens segment spanning two unrelated sentences.
RANGE_PATTERNS = [
    re.compile(r"\bbetween\b(?P<a>[^.;\n]+?)\band\b(?P<b>[^.;\n]+?)(?=[,.;\n]|$)",
               re.IGNORECASE),
    re.compile(r"\bfrom\b(?P<a>[^.;\n]+?)\bto\b(?P<b>[^.;\n]+?)(?=[,.;\n]|$)",
               re.IGNORECASE),
]


def digits(text: str) -> list[str]:
    return re.findall(r"\d+", text)


def digits_agree(span: str, name: str) -> bool:
    """Reject a fuzzy pair whose numbers disagree.

    Roughly a third of the system is named by number alone -- 121 St, 125 St,
    135 St, 155 St all exist, several on the same line -- so for those the
    digits *are* the identity and character similarity is actively misleading:
    "12 St" scores 0.67 against "125 St" and 0.80 against "155 St" while the
    intended station was 121 St. Without this guard, fuzzy top-1 accuracy on
    numeric names measured 0% under a character transposition. With it, those
    become no-match instead of a confident wrong answer, which is the right
    trade for a join key.

    Names carrying no digits on either side (Delancey St-Essex St) are
    unaffected, and a pair whose numbers already agree ("34 st hudson yds" vs
    "34 St-Hudson Yards") still passes on its text similarity.
    """
    return digits(span) == digits(name)


def normalize(text: str) -> str:
    """Casefold, drop punctuation, collapse whitespace.

    Punctuation is noise here: the list writes "W 4 St-Wash Sq" and
    "Central Park North (110 St)" while headers vary the separators freely.
    Digits are kept because they carry most of the discriminating signal.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(text).lower())).strip()


def ngrams(text: str, n: int = NGRAM_N) -> Counter:
    padded = f"{' ' * (n - 1)}{text}{' ' * (n - 1)}"
    return Counter(padded[i:i + n] for i in range(len(padded) - n + 1))


def dice(a: Counter, b: Counter) -> float:
    """Dice coefficient over n-gram multisets: 2|A∩B| / (|A|+|B|)."""
    total = sum(a.values()) + sum(b.values())
    if not total:
        return 0.0
    return 2 * sum((a & b).values()) / total


def ratio(a: str, b: str) -> float:
    """Normalized alignment score in [0,1], 1.0 being identical.

    This is the Levenshtein-family half of the check. It is computed from the
    edit distance directly rather than via difflib, whose SequenceMatcher uses
    a longest-matching-block heuristic that is not a true edit distance and
    scores some transpositions too generously.
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,          # deletion
                current[j - 1] + 1,       # insertion
                previous[j - 1] + (ca != cb),  # substitution
            ))
        previous = current
    return 1.0 - previous[-1] / max(len(a), len(b))


@dataclass
class Station:
    gtfs_stop_id: str
    stop_name: str
    routes: set[str] = field(default_factory=set)
    borough: str | None = None

    @property
    def normalized(self) -> str:
        return normalize(self.stop_name)


@dataclass
class StationMatch:
    """One station an alert header refers to."""

    stop_name: str
    gtfs_stop_ids: list[str]
    score: float
    method: str          # "exact" or "fuzzy"
    span: str            # the header text that produced the match
    on_alert_route: bool


def load_stations(app_token: str | None = None, timeout: float = 30.0) -> list[Station]:
    """Fetch the 496-row station list. Small enough to hold in memory and to
    scan linearly, so no local cache is kept."""
    headers = {"X-App-Token": app_token} if app_token else {}
    response = requests.get(
        STATIONS_URL,
        headers=headers,
        timeout=timeout,
        params={"$select": "gtfs_stop_id,stop_name,daytime_routes,borough",
                "$limit": 2000},
    )
    response.raise_for_status()
    return [
        Station(
            gtfs_stop_id=row["gtfs_stop_id"],
            stop_name=row["stop_name"],
            routes=set(str(row.get("daytime_routes", "")).split()),
            borough=row.get("borough"),
        )
        for row in response.json()
        if row.get("gtfs_stop_id") and row.get("stop_name")
    ]


def load_stations_gtfs(gtfs_dir=DEFAULT_GTFS_DIR) -> list[Station]:
    """Build the station list from a local GTFS feed instead of the network.

    Stations are the `location_type == 1` parent rows of stops.txt; the child
    rows are per-direction platforms sharing the parent's name. Routes are
    derived by walking stop_times.txt back through trips.txt, which is better
    than the Socrata `daytime_routes` column in two ways that matter here:

      * The codes come out in exactly the alerts archive's vocabulary. Every
        route token the archive used over a quarter of alerts exists as a GTFS
        route_id, so the ROUTE_ALIASES patching above is simply unnecessary:
        GTFS says SI where Socrata says SIR, FS where Socrata says S.
      * It covers night and weekend patterns, which `daytime_routes` omits by
        definition. 220 of 496 stations gain at least one route, including the
        6X/7X/FX express variants that Socrata folds away entirely.
    """
    gtfs_dir = Path(gtfs_dir)

    def read(name):
        with open(gtfs_dir / name, newline="", encoding="utf-8-sig") as handle:
            yield from csv.DictReader(handle)

    route_of_trip = {row["trip_id"]: row["route_id"] for row in read("trips.txt")}

    routes_by_station: dict[str, set[str]] = {}
    for row in read("stop_times.txt"):
        # Child stop ids carry an N/S direction suffix; the parent station id is
        # the same string without it.
        parent = DIRECTION_SUFFIX.sub("", row["stop_id"])
        route = route_of_trip.get(row["trip_id"])
        if route:
            routes_by_station.setdefault(parent, set()).add(route)

    return [
        Station(
            gtfs_stop_id=row["stop_id"],
            stop_name=row["stop_name"],
            routes=routes_by_station.get(row["stop_id"], set()),
        )
        for row in read("stops.txt")
        if row.get("location_type") == "1"
    ]


def load_route_patterns(gtfs_dir=DEFAULT_GTFS_DIR) -> dict[str, list[tuple[str, ...]]]:
    """Ordered station sequences per route, for expanding a range to the stops
    inside it.

    Every distinct stop sequence a route runs is kept rather than one canonical
    list, because several routes branch: the A splits toward Lefferts Blvd and
    the Rockaways, the 5 toward Dyre Av and Nereid Av. A range's two endpoints
    only both appear on the branch the alert is talking about, so keeping all
    patterns lets the expansion pick that branch instead of guessing.
    """
    gtfs_dir = Path(gtfs_dir)

    def read(name):
        with open(gtfs_dir / name, newline="", encoding="utf-8-sig") as handle:
            yield from csv.DictReader(handle)

    route_of_trip = {row["trip_id"]: row["route_id"] for row in read("trips.txt")}

    sequences: dict[str, list[tuple[int, str]]] = {}
    for row in read("stop_times.txt"):
        sequences.setdefault(row["trip_id"], []).append(
            (int(row["stop_sequence"]), DIRECTION_SUFFIX.sub("", row["stop_id"])))

    patterns: dict[str, set[tuple[str, ...]]] = {}
    for trip_id, stops in sequences.items():
        route = route_of_trip.get(trip_id)
        if not route:
            continue
        ordered = tuple(stop for _, stop in sorted(stops))
        patterns.setdefault(route, set()).add(ordered)

    # Longest first so a range resolves against the fullest pattern that holds
    # both endpoints, rather than a short turn-back run that happens to contain
    # them.
    return {route: sorted(pats, key=len, reverse=True)
            for route, pats in patterns.items()}


class StationMatcher:
    """Resolves header prose to stations. Build once, reuse across alerts."""

    def __init__(self, stations: list[Station] | None = None,
                 gtfs_dir=DEFAULT_GTFS_DIR, patterns=None, **kwargs):
        local = Path(gtfs_dir).joinpath("stops.txt").exists()
        if stations is None:
            # Prefer the local feed, fall back to Socrata when it isn't there.
            stations = load_stations_gtfs(gtfs_dir) if local else load_stations(**kwargs)
        self.stations = stations

        # Range expansion needs stop order, which only the GTFS feed carries;
        # without it the matcher still resolves stations, just not segments.
        if patterns is None:
            patterns = load_route_patterns(gtfs_dir) if local else {}
        self.patterns = patterns

        # 496 platforms collapse to 379 distinct names: "Canal St" alone covers
        # 6 separate stations. A name is therefore resolved to the *set* of stop
        # ids carrying it, and the route hint narrows that set rather than
        # picking one arbitrarily.
        self.by_name: dict[str, list[Station]] = {}
        for station in self.stations:
            self.by_name.setdefault(station.normalized, []).append(station)

        # Longest first, so "50 St-8 Av" is tried before the "50 St" nested
        # inside it and the more specific name wins.
        self.names_by_length = sorted(self.by_name, key=len, reverse=True)
        self.ngram_cache = {name: ngrams(name) for name in self.by_name}

    def expand_routes(self, routes) -> set[str]:
        expanded = set()
        for route in routes or ():
            route = str(route).strip().upper()
            expanded.add(route)
            expanded |= ROUTE_ALIASES.get(route, set())
        return expanded

    def _build(self, name: str, score: float, method: str, span: str,
               routes: set[str]) -> StationMatch:
        candidates = self.by_name[name]
        on_route = [s for s in candidates if s.routes & routes] if routes else []
        chosen = on_route or candidates
        return StationMatch(
            stop_name=chosen[0].stop_name,
            gtfs_stop_ids=sorted(s.gtfs_stop_id for s in chosen),
            score=score,
            method=method,
            span=span,
            on_alert_route=bool(on_route),
        )

    def match(self, header: str, routes=None, *, fuzzy: bool = True) -> list[StationMatch]:
        """Return every station the header refers to, best first.

        A header routinely names more than one -- "between 69 St and 82 St" is
        a segment, not a point -- so this returns a list rather than a single
        best guess. `routes` (the alert's affected lines) only breaks ties
        between same-named stations and sets `on_alert_route`; it never
        excludes a candidate, because alerts do name off-line stations.
        """
        text = normalize(header)
        if not text:
            return []
        routes = self.expand_routes(routes)

        matches, consumed = [], []
        for name in self.names_by_length:
            if not name or name not in text:
                continue
            # Skip a name nested inside one already matched, so "50 St-8 Av"
            # doesn't also report a bare "50 St" at the same position.
            if any(name in longer for longer in consumed):
                continue
            consumed.append(name)
            matches.append(self._build(name, 1.0, "exact", name, routes))

        if matches or not fuzzy:
            return self._rank(matches)

        return self._rank(self._fuzzy(text, routes))

    def _fuzzy(self, text: str, routes: set[str]) -> list[StationMatch]:
        """Slide token windows over the header, scoring each against every
        station name. Only reached when no exact name is present, which is
        about 5% of headers, so the linear scan is affordable."""
        tokens = text.split()
        best: dict[str, tuple[float, str]] = {}
        for width in range(MIN_WINDOW, MAX_WINDOW + 1):
            for start in range(len(tokens) - width + 1):
                span = " ".join(tokens[start:start + width])
                span_grams = ngrams(span)
                for name, name_grams in self.ngram_cache.items():
                    # Length gate first: it is far cheaper than the edit
                    # distance and discards most pairs outright.
                    if not 0.5 <= len(span) / max(len(name), 1) <= 2.0:
                        continue
                    d = dice(span_grams, name_grams)
                    if d < MIN_DICE:
                        continue
                    if not digits_agree(span, name):
                        continue
                    if ratio(span, name) < MIN_RATIO:
                        continue
                    if d > best.get(name, (0.0, ""))[0]:
                        best[name] = (d, span)

        results = []
        for name, (score, span) in best.items():
            if any(name != other and name in other for other in best):
                continue
            results.append(self._build(name, score, "fuzzy", span, routes))
        return results

    def segment(self, route: str, start_ids, end_ids) -> list[str]:
        """Stop ids from `start_ids` to `end_ids` inclusive, along `route`.

        Endpoints arrive as id *sets* because a station name can belong to
        several platforms. The shortest slice spanning any pairing wins, which
        picks the intended stretch when a name like Canal St appears at more
        than one point on the network.
        """
        start_ids, end_ids = set(start_ids), set(end_ids)
        best: list[str] = []
        for pattern in self.patterns.get(str(route).strip().upper(), ()):
            index = {stop: i for i, stop in enumerate(pattern)}
            heads = [index[s] for s in start_ids if s in index]
            tails = [index[e] for e in end_ids if e in index]
            for head in heads:
                for tail in tails:
                    lo, hi = sorted((head, tail))
                    span = list(pattern[lo:hi + 1])
                    if not best or len(span) < len(best):
                        best = span
        return best

    def affected_stations(self, header: str, routes=None) -> dict[str, list[str]]:
        """Every stop id an alert touches, per route, with ranges expanded.

        A header that delimits a stretch of line ("no 6 train service between
        3 Av-138 St and Pelham Bay Park") marks every stop in between, not just
        the two named. A header that merely names a location marks that station
        alone. Returns {route: [stop_id, ...]}, so an alert on two routes
        expands along each separately -- the same two station names can sit any
        number of stops apart depending on which line you travel.

        Falls back to the plain point matches for any route whose range cannot
        be resolved, which happens when the endpoints never share a pattern
        (a reroute naming a station the route does not normally serve).
        """
        routes = [str(r).strip().upper() for r in (routes or []) if str(r).strip()]
        text = BOUND_LABEL.sub(" ", str(header))

        # Match against the stripped text, not the raw header: a terminal named
        # only to say which way the train faces is not a place the alert
        # touches, so it must not survive into the fallback either.
        points = self.match(text, routes)
        fallback = sorted({i for m in points for i in m.gtfs_stop_ids})

        ranges = []
        for pattern in RANGE_PATTERNS:
            for hit in pattern.finditer(text):
                left = self.match(hit.group("a"), routes, fuzzy=False)
                right = self.match(hit.group("b"), routes, fuzzy=False)
                if left and right:
                    ranges.append((left[0], right[0]))

        result = {}
        for route in routes or [""]:
            stops = []
            for left, right in ranges:
                stops += self.segment(route, left.gtfs_stop_ids, right.gtfs_stop_ids)
            result[route] = sorted(set(stops)) if stops else fallback
        return result

    @staticmethod
    def _rank(matches: list[StationMatch]) -> list[StationMatch]:
        # On-route matches first, then score, then the more specific (longer)
        # name, so the caller can take the head of the list as the best guess.
        return sorted(
            matches,
            key=lambda m: (-m.on_alert_route, -m.score, -len(m.stop_name)),
        )
