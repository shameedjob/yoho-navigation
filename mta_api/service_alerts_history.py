"""Query the NY Open Data historical MTA service-alerts archive:
https://data.ny.gov/Transportation/MTA-Service-Alerts-Beginning-April-2020/7kct-peq7

This is a *different* data source from `mta_api.client`: that module reads MTA's
live GTFS-realtime feed (current alerts only), while this dataset is a
Socrata-hosted, queryable history of every alert back to April 2020 — useful for
"how often does line X have delays" style questions the live feed can't answer.

Note on OData: Socrata exposes this dataset at an OData v4 endpoint
(https://data.ny.gov/api/odata/v4/7kct-peq7), but as of this writing its
`$filter` support is broken for *every* comparison operator (`eq`, `ne`, `gt`,
`ge`, `lt`, `le`) on this dataset — each one fails server-side with a type
-mismatch error ("The types 'Edm.Boolean' and 'Edm.String' are not
compatible."), independent of the field's actual type (confirmed against both
string and numeric columns). Only string *functions* like `contains`/
`startswith` work, which isn't enough to express a time-span filter.

Socrata's native query language (SoQL, via `$where` on the `/resource/<id>.json`
endpoint) supports the same filtering OData would and works correctly against
this dataset, so that's what this module uses.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import requests

from .models import ServiceAlertRecord

DOMAIN = "data.ny.gov"
DATASET_ID = "7kct-peq7"
RESOURCE_URL = f"https://{DOMAIN}/resource/{DATASET_ID}.json"

_PAGE_SIZE = 1000
_SELECT = "event_id,update_number,status_label,date,affected,header"

# Chunk size for the `event_id in (...)` follow-up query, kept well under the
# point where the generated URL would outgrow what Socrata accepts.
_ID_CHUNK = 200


def _soql_escape(value: str) -> str:
    """Escape a string for safe interpolation into a SoQL string literal."""
    return value.replace("'", "''")


def _soql_timestamp(dt: datetime) -> str:
    # The dataset's `date` column is a floating (timezone-less) timestamp, so
    # any tzinfo on `dt` is dropped rather than converted, matching how the
    # column itself has no timezone.
    return dt.replace(tzinfo=None).isoformat(timespec="seconds")


def _fetch_rows(
    session: requests.Session,
    where_clause: str,
    *,
    headers: dict[str, str],
    timeout: float,
) -> list[dict]:
    """Run one paginated SoQL query and return every matching row."""
    rows: list[dict] = []
    offset = 0
    while True:
        response = session.get(
            RESOURCE_URL,
            headers=headers,
            timeout=timeout,
            params={
                "$select": _SELECT,
                "$where": where_clause,
                "$order": "date ASC",
                "$limit": _PAGE_SIZE,
                "$offset": offset,
            },
        )
        response.raise_for_status()
        page = response.json()
        rows.extend(page)
        if len(page) < _PAGE_SIZE:
            return rows
        offset += _PAGE_SIZE


def _affected_trains(row: dict) -> list[str]:
    return [token.strip() for token in (row.get("affected") or "").split("|") if token.strip()]


def fetch_subway_service_alerts(
    start: datetime,
    end: datetime,
    *,
    agency: str = "NYCT Subway",
    app_token: str | None = None,
    timeout: float = 10.0,
) -> list[ServiceAlertRecord]:
    """Fetch historical service-alert *events* for `agency` that were announced
    between `start` and `end` (inclusive), ordered oldest to newest.

    The archive stores one row per alert update; this collapses the updates
    sharing an `event_id` into a single record spanning `time` to `end_time`.
    See `ServiceAlertRecord` for what that end time does and doesn't mean.

    The window bounds the event's *first* update, not its updates individually,
    so an event announced just before `end` still reports its true end time
    even when later updates fall outside the window -- those trailing updates
    are pulled in by a follow-up query keyed on `event_id`. Without that, an
    event straddling the boundary would look like it ended at `end`.

    Example:
        from datetime import datetime
        alerts = fetch_subway_service_alerts(
            datetime(2024, 1, 1), datetime(2024, 1, 2)
        )
        for a in alerts:
            print(a.time, a.duration, a.status_label, a.affected_trains)
    """
    agency_clause = f"agency = '{_soql_escape(agency)}'"
    headers = {"X-App-Token": app_token} if app_token else {}

    with requests.Session() as session:
        in_window = _fetch_rows(
            session,
            f"{agency_clause} "
            f"AND date >= '{_soql_timestamp(start)}' "
            f"AND date <= '{_soql_timestamp(end)}'",
            headers=headers,
            timeout=timeout,
        )

        event_ids = {row["event_id"] for row in in_window if row.get("event_id")}

        rows_by_event: dict[str, list[dict]] = defaultdict(list)
        ordered_ids = sorted(event_ids)
        for index in range(0, len(ordered_ids), _ID_CHUNK):
            chunk = ordered_ids[index : index + _ID_CHUNK]
            id_list = ", ".join(f"'{_soql_escape(event_id)}'" for event_id in chunk)
            for row in _fetch_rows(
                session,
                f"{agency_clause} AND event_id in ({id_list})",
                headers=headers,
                timeout=timeout,
            ):
                rows_by_event[row["event_id"]].append(row)

    window_start = start.replace(tzinfo=None)
    window_end = end.replace(tzinfo=None)

    records: list[ServiceAlertRecord] = []
    for event_id, rows in rows_by_event.items():
        # `date` order alone defines the span. `update_number` is *not* trusted
        # for ordering: on a handful of events (4 of 471 over a sample week) it
        # runs backwards relative to `date`, so using it to find the first
        # update would mis-report both the start time and which events fall in
        # the window.
        rows.sort(key=lambda row: row["date"])
        first, last = rows[0], rows[-1]

        # Drop events carried over from before the window. They were matched
        # only by a mid-event update, so reporting them would either truncate
        # their start time or duplicate them against an earlier query.
        started = datetime.fromisoformat(first["date"])
        if not window_start <= started <= window_end:
            continue

        # The affected-line set can grow or shrink as an event develops (59 of
        # 471 events over a sample week), so report the union: a line touched
        # at any point during the event was affected by it.
        affected: list[str] = []
        for row in rows:
            for train in _affected_trains(row):
                if train not in affected:
                    affected.append(train)

        records.append(
            ServiceAlertRecord(
                status_label=first["status_label"],
                time=started,
                end_time=datetime.fromisoformat(last["date"]),
                affected_trains=affected,
                event_id=event_id,
                update_count=len(rows),
                headers=[row.get("header", "") for row in rows],
            )
        )

    records.sort(key=lambda record: record.time)
    return records
