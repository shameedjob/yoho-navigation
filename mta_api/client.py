"""Client for the MTA's real-time GTFS feeds (subway trip updates, vehicle
positions, and service alerts).
"""

from __future__ import annotations

from datetime import datetime

import requests
from google.transit import gtfs_realtime_pb2

from . import feeds
from .models import Alert, StopTimeUpdate, TripUpdate, VehiclePosition, _to_datetime

# Extension field number MTA uses on GTFS-rt Alert for its own metadata.
MERCURY_ALERT_FIELD = 1001


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return result, pos


def _wire_fields(data: bytes):
    """Yield (field_number, wire_type, value) from raw protobuf bytes."""
    pos = 0
    while pos < len(data):
        key, pos = _read_varint(data, pos)
        number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, pos = _read_varint(data, pos)
        elif wire_type == 2:
            length, pos = _read_varint(data, pos)
            value = data[pos:pos + length]
            pos += length
        elif wire_type == 5:
            value, pos = data[pos:pos + 4], pos + 4
        elif wire_type == 1:
            value, pos = data[pos:pos + 8], pos + 8
        else:
            raise ValueError(f"unsupported wire type {wire_type}")
        yield number, wire_type, value


def _mercury_fields(alert: gtfs_realtime_pb2.Alert) -> dict[int, object]:
    """First value of each field in MTA's alert extension, or {} if absent.

    Read from the re-serialized message because the upb protobuf runtime
    keeps the extension as unknown fields and doesn't expose an accessor.
    Observed layout on the all-alerts feed: 1 and 2 are unix timestamps that
    behave as created and last-updated, 3 is the condition string.
    """
    try:
        for number, wire_type, value in _wire_fields(alert.SerializeToString()):
            if number == MERCURY_ALERT_FIELD and wire_type == 2:
                fields: dict[int, object] = {}
                for inner, _, inner_value in _wire_fields(value):
                    fields.setdefault(inner, inner_value)
                return fields
    except (IndexError, ValueError):
        pass
    return {}


def _translation(text: gtfs_realtime_pb2.TranslatedString, language: str = "en") -> str:
    """The plain-text translation. The feed also carries "en-html"; it
    happens to come second today, but picking by position would silently
    start feeding markup to the header parser if that order changed."""
    for translation in text.translation:
        if translation.language == language:
            return translation.text
    return text.translation[0].text if text.translation else ""


class MTAClient:
    """Fetches and parses the MTA's GTFS-realtime protobuf feeds.

    Example:
        client = MTAClient()
        updates = client.get_trip_updates("A")
        positions = client.get_vehicle_positions("A")
        alerts = client.get_alerts()
    """

    def __init__(self, api_key: str | None = None, timeout: float = 10.0):
        self._session = requests.Session()
        if api_key:
            self._session.headers["x-api-key"] = api_key
        self._timeout = timeout

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "MTAClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _fetch_feed(self, url: str) -> gtfs_realtime_pb2.FeedMessage:
        response = self._session.get(url, timeout=self._timeout)
        response.raise_for_status()
        feed_message = gtfs_realtime_pb2.FeedMessage()
        feed_message.ParseFromString(response.content)
        return feed_message

    def get_feed_for_line(self, line: str) -> gtfs_realtime_pb2.FeedMessage:
        """Fetch the raw GTFS-realtime feed covering the given subway line."""
        feed_key = feeds.line_to_feed_key(line)
        return self._fetch_feed(feeds.feed_url(feed_key))

    def get_trip_updates(self, line: str) -> list[TripUpdate]:
        return _parse_trip_updates(self.get_feed_for_line(line))

    def get_feed_trip_updates(self, feed_key: str) -> tuple[datetime | None, list[TripUpdate]]:
        """Trip updates for one feed group (e.g. "ACE"), with the feed's own
        header timestamp. Polling by feed key rather than by line avoids
        fetching the same group once per line it carries."""
        feed_message = self._fetch_feed(feeds.feed_url(feed_key))
        return _to_datetime(feed_message.header.timestamp), _parse_trip_updates(feed_message)

    def get_vehicle_positions(self, line: str) -> list[VehiclePosition]:
        feed_message = self.get_feed_for_line(line)
        positions = []
        for entity in feed_message.entity:
            if not entity.HasField("vehicle"):
                continue
            vehicle = entity.vehicle
            positions.append(
                VehiclePosition(
                    trip_id=vehicle.trip.trip_id,
                    route_id=vehicle.trip.route_id,
                    current_stop_id=vehicle.stop_id or None,
                    status=gtfs_realtime_pb2.VehiclePosition.VehicleStopStatus.Name(
                        vehicle.current_status
                    )
                    if vehicle.HasField("current_status")
                    else None,
                    timestamp=_to_datetime(vehicle.timestamp),
                )
            )
        return positions

    def get_alerts(self) -> list[Alert]:
        url = f"{feeds.BASE_URL}/{feeds.ALERTS_FEED}"
        feed_message = self._fetch_feed(url)
        alerts = []
        for entity in feed_message.entity:
            if not entity.HasField("alert"):
                continue
            alert = entity.alert
            header = _translation(alert.header_text)
            description = _translation(alert.description_text)
            mercury = _mercury_fields(alert)
            alert_type = mercury.get(3)
            created, updated = mercury.get(1), mercury.get(2)
            route_ids = [
                informed.route_id
                for informed in alert.informed_entity
                if informed.route_id
            ]
            alerts.append(
                Alert(
                    alert_id=entity.id,
                    header_text=header,
                    description_text=description,
                    affected_route_ids=route_ids,
                    active_periods=[
                        (_to_datetime(period.start), _to_datetime(period.end))
                        for period in alert.active_period
                    ],
                    agency_ids=sorted({
                        informed.agency_id
                        for informed in alert.informed_entity
                        if informed.agency_id
                    }),
                    alert_type=(alert_type.decode("utf-8", errors="replace")
                                if isinstance(alert_type, bytes) else None),
                    created_at=_to_datetime(created) if isinstance(created, int) else None,
                    updated_at=_to_datetime(updated) if isinstance(updated, int) else None,
                )
            )
        return alerts


def _parse_trip_updates(feed_message: gtfs_realtime_pb2.FeedMessage) -> list[TripUpdate]:
    trip_updates = []
    for entity in feed_message.entity:
        if not entity.HasField("trip_update"):
            continue
        tu = entity.trip_update
        stop_time_updates = [
            StopTimeUpdate(
                stop_id=stu.stop_id,
                arrival=_to_datetime(stu.arrival.time) if stu.HasField("arrival") else None,
                departure=_to_datetime(stu.departure.time) if stu.HasField("departure") else None,
            )
            for stu in tu.stop_time_update
        ]
        trip_updates.append(
            TripUpdate(
                trip_id=tu.trip.trip_id,
                route_id=tu.trip.route_id,
                start_date=tu.trip.start_date,
                # MTA encodes direction in a proprietary NYCT extension to
                # TripDescriptor, which needs the nyct-gtfs extension proto
                # to decode; not included here.
                direction=None,
                stop_time_updates=stop_time_updates,
            )
        )
    return trip_updates
