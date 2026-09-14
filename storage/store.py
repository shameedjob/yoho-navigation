"""UserStore: the persistence interface web/ and the scheduler use, plus an
in-memory implementation for tests and local runs without Firebase.

Firestore layout (storage/firestore_store.py), keyed by Google `sub`:

  users/{uid}                      email, name, picture, created_at, updated_at,
                                   google {refresh_token_enc, scopes, revoked},
                                   home_enc, token_limit (optional override),
                                   calendar_synced_at
  users/{uid}/usage/{YYYY-MM}      input_tokens, output_tokens, total_tokens, requests
  users/{uid}/calendar_events/{id} summary, start, end, location_enc, updated
  users/{uid}/private/conversation messages (JSON string, recent turns only)
  watch_channels/{channel_id}      uid, resource_id, token_hash, expiration (ms), address
  due_checks/{uid}__{event_id}     uid, event_id, event_start, check_at (Unix s),
                                   status (pending|sent|expired), sent_at, last_error

watch_channels and due_checks are top-level: the webhook knows only the channel
id, and the scheduler asks "what's due" across all users.

Fields ending in `_enc` are storage.crypto ciphertext; the store never sees
their plaintext.
"""

from __future__ import annotations

import copy
import threading
from datetime import datetime, timezone
from typing import Any, Protocol


def usage_period(now: datetime | None = None) -> str:
    """Billing period key: the UTC calendar month, e.g. "2026-09"."""
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m")


class UserStore(Protocol):
    def get_user(self, uid: str) -> dict | None: ...
    def upsert_user(self, uid: str, fields: dict) -> None:
        """Merge `fields` into the user, creating it (with created_at) if new."""
    def update_user(self, uid: str, fields: dict) -> None:
        """Merge `fields` into an existing user. A value of None deletes the field."""
    def delete_user(self, uid: str) -> None:
        """Delete the user and everything under it."""

    def get_usage(self, uid: str, period: str) -> dict: ...
    def add_usage(self, uid: str, period: str, input_tokens: int, output_tokens: int) -> None:
        """Atomically increment the period's counters."""

    def replace_calendar_events(self, uid: str, events: list[dict]) -> None:
        """Make the stored events exactly `events` (each with an `id`)."""
    def list_calendar_events(self, uid: str) -> list[dict]: ...

    def get_conversation(self, uid: str) -> list[dict]: ...
    def set_conversation(self, uid: str, messages: list[dict]) -> None: ...

    def save_watch_channel(self, channel_id: str, fields: dict) -> None: ...
    def get_watch_channel(self, channel_id: str) -> dict | None:
        """The channel with its `id`, or None."""
    def list_watch_channels(self, uid: str | None = None) -> list[dict]:
        """Every channel (with `id`), or one user's."""
    def delete_watch_channel(self, channel_id: str) -> None: ...

    def list_due_checks(self, uid: str) -> list[dict]:
        """One user's due checks, each with its `id`."""
    def replace_due_checks(self, uid: str, checks: list[dict]) -> None:
        """Make the user's due checks exactly `checks` (each with an `id`)."""
    def due_checks_before(self, until: int) -> list[dict]:
        """Every user's checks with check_at <= until, any status."""
    def update_due_check(self, check_id: str, fields: dict) -> None: ...


_EMPTY_USAGE = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "requests": 0}


class MemoryStore:
    """Process-local UserStore. Not shared across workers; tests and dev only."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._users: dict[str, dict] = {}
        self._usage: dict[tuple[str, str], dict] = {}
        self._events: dict[str, dict[str, dict]] = {}
        self._conversations: dict[str, list[dict]] = {}
        self._channels: dict[str, dict] = {}
        self._due_checks: dict[str, dict] = {}

    def get_user(self, uid: str) -> dict | None:
        with self._lock:
            user = self._users.get(uid)
            return copy.deepcopy(user) if user is not None else None

    def upsert_user(self, uid: str, fields: dict) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            user = self._users.setdefault(uid, {"created_at": now})
            _merge(user, fields)
            user["updated_at"] = now

    def update_user(self, uid: str, fields: dict) -> None:
        with self._lock:
            if uid not in self._users:
                raise KeyError(uid)
            _merge(self._users[uid], fields)
            self._users[uid]["updated_at"] = datetime.now(timezone.utc)

    def delete_user(self, uid: str) -> None:
        with self._lock:
            self._users.pop(uid, None)
            self._events.pop(uid, None)
            self._conversations.pop(uid, None)
            for key in [k for k in self._usage if k[0] == uid]:
                del self._usage[key]
            for table in (self._channels, self._due_checks):
                for key in [k for k, v in table.items() if v.get("uid") == uid]:
                    del table[key]

    def get_usage(self, uid: str, period: str) -> dict:
        with self._lock:
            return dict(self._usage.get((uid, period), _EMPTY_USAGE))

    def add_usage(self, uid: str, period: str, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            row = self._usage.setdefault((uid, period), dict(_EMPTY_USAGE))
            row["input_tokens"] += input_tokens
            row["output_tokens"] += output_tokens
            row["total_tokens"] += input_tokens + output_tokens
            row["requests"] += 1

    def replace_calendar_events(self, uid: str, events: list[dict]) -> None:
        with self._lock:
            self._events[uid] = {e["id"]: copy.deepcopy(e) for e in events}

    def list_calendar_events(self, uid: str) -> list[dict]:
        with self._lock:
            events = copy.deepcopy(list(self._events.get(uid, {}).values()))
        return sorted(events, key=lambda e: e.get("start") or "")

    def get_conversation(self, uid: str) -> list[dict]:
        with self._lock:
            return copy.deepcopy(self._conversations.get(uid, []))

    def set_conversation(self, uid: str, messages: list[dict]) -> None:
        with self._lock:
            self._conversations[uid] = copy.deepcopy(messages)

    def save_watch_channel(self, channel_id: str, fields: dict) -> None:
        with self._lock:
            self._channels[channel_id] = copy.deepcopy(fields)

    def get_watch_channel(self, channel_id: str) -> dict | None:
        with self._lock:
            row = self._channels.get(channel_id)
            return {"id": channel_id, **copy.deepcopy(row)} if row is not None else None

    def list_watch_channels(self, uid: str | None = None) -> list[dict]:
        with self._lock:
            return [{"id": k, **copy.deepcopy(v)} for k, v in self._channels.items()
                    if uid is None or v.get("uid") == uid]

    def delete_watch_channel(self, channel_id: str) -> None:
        with self._lock:
            self._channels.pop(channel_id, None)

    def list_due_checks(self, uid: str) -> list[dict]:
        with self._lock:
            return [{"id": k, **copy.deepcopy(v)} for k, v in self._due_checks.items() if v.get("uid") == uid]

    def replace_due_checks(self, uid: str, checks: list[dict]) -> None:
        with self._lock:
            for key in [k for k, v in self._due_checks.items() if v.get("uid") == uid]:
                del self._due_checks[key]
            for c in checks:
                self._due_checks[c["id"]] = {k: copy.deepcopy(v) for k, v in c.items() if k != "id"}

    def due_checks_before(self, until: int) -> list[dict]:
        with self._lock:
            rows = [{"id": k, **copy.deepcopy(v)} for k, v in self._due_checks.items() if v["check_at"] <= until]
        return sorted(rows, key=lambda r: r["check_at"])

    def update_due_check(self, check_id: str, fields: dict) -> None:
        with self._lock:
            if check_id in self._due_checks:
                _merge(self._due_checks[check_id], fields)


def _merge(target: dict, fields: dict[str, Any]) -> None:
    """Shallow-per-level merge matching Firestore set(merge=True); None deletes."""
    for key, value in fields.items():
        if value is None:
            target.pop(key, None)
        elif isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
