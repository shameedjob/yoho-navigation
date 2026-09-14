"""Firestore-backed UserStore (layout documented in storage/store.py).

Credentials: FIREBASE_CREDENTIALS pointing at a service-account JSON, else
Application Default Credentials (`gcloud auth application-default login`
locally, the attached service account on Cloud Run / GCE).

Only the server talks to Firestore -- the browser never gets Firebase
credentials -- so lock client access down entirely in firestore.rules:

    rules_version = '2';
    service cloud.firestore { match /databases/{db}/documents {
      match /{document=**} { allow read, write: if false; }
    } }

The Admin SDK bypasses those rules.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore import DELETE_FIELD, Increment, SERVER_TIMESTAMP

from .store import _EMPTY_USAGE

_BATCH_LIMIT = 400  # Firestore caps a batch at 500 writes


class FirestoreStore:
    def __init__(self, credentials_path: str | None = None, project_id: str | None = None):
        try:
            app = firebase_admin.get_app()
        except ValueError:
            cred = credentials.Certificate(credentials_path) if credentials_path else credentials.ApplicationDefault()
            options = {"projectId": project_id} if project_id else None
            app = firebase_admin.initialize_app(cred, options)
        self._db = firestore.client(app)

    def _user(self, uid: str):
        return self._db.collection("users").document(uid)

    def get_user(self, uid: str) -> dict | None:
        snap = self._user(uid).get()
        return snap.to_dict() if snap.exists else None

    def upsert_user(self, uid: str, fields: dict) -> None:
        ref = self._user(uid)

        @firestore.transactional
        def txn(transaction):
            data = _to_firestore(fields)
            data["updated_at"] = SERVER_TIMESTAMP
            if not ref.get(transaction=transaction).exists:
                data["created_at"] = SERVER_TIMESTAMP
            transaction.set(ref, data, merge=True)

        txn(self._db.transaction())

    def update_user(self, uid: str, fields: dict) -> None:
        data = _flatten(_to_firestore(fields))
        data["updated_at"] = SERVER_TIMESTAMP
        self._user(uid).update(data)

    def delete_user(self, uid: str) -> None:
        self._db.recursive_delete(self._user(uid))
        for name in ("watch_channels", "due_checks"):
            for snap in self._db.collection(name).where("uid", "==", uid).stream():
                snap.reference.delete()

    def get_usage(self, uid: str, period: str) -> dict:
        snap = self._user(uid).collection("usage").document(period).get()
        return {**_EMPTY_USAGE, **(snap.to_dict() or {})}

    def add_usage(self, uid: str, period: str, input_tokens: int, output_tokens: int) -> None:
        self._user(uid).collection("usage").document(period).set({
            "input_tokens": Increment(input_tokens),
            "output_tokens": Increment(output_tokens),
            "total_tokens": Increment(input_tokens + output_tokens),
            "requests": Increment(1),
            "updated_at": SERVER_TIMESTAMP,
        }, merge=True)

    def replace_calendar_events(self, uid: str, events: list[dict]) -> None:
        col = self._user(uid).collection("calendar_events")
        keep = {e["id"] for e in events}
        writes = [("delete", col.document(snap.id), None)
                  for snap in col.select([]).stream() if snap.id not in keep]
        writes += [("set", col.document(e["id"]), {k: v for k, v in e.items() if k != "id"}) for e in events]
        for i in range(0, len(writes), _BATCH_LIMIT):
            batch = self._db.batch()
            for op, ref, data in writes[i:i + _BATCH_LIMIT]:
                batch.delete(ref) if op == "delete" else batch.set(ref, data)
            batch.commit()

    def list_calendar_events(self, uid: str) -> list[dict]:
        col = self._user(uid).collection("calendar_events")
        return [{"id": s.id, **s.to_dict()} for s in col.order_by("start").stream()]

    def get_conversation(self, uid: str) -> list[dict]:
        snap = self._user(uid).collection("private").document("conversation").get()
        return json.loads(snap.get("messages")) if snap.exists else []

    def set_conversation(self, uid: str, messages: list[dict]) -> None:
        # Stored as a JSON string: Strands content blocks nest arrays in arrays,
        # which Firestore maps don't allow.
        self._user(uid).collection("private").document("conversation").set({
            "messages": json.dumps(messages), "updated_at": SERVER_TIMESTAMP,
        })


    def save_watch_channel(self, channel_id: str, fields: dict) -> None:
        self._db.collection("watch_channels").document(channel_id).set({**fields, "updated_at": SERVER_TIMESTAMP})

    def get_watch_channel(self, channel_id: str) -> dict | None:
        snap = self._db.collection("watch_channels").document(channel_id).get()
        return {"id": snap.id, **snap.to_dict()} if snap.exists else None

    def list_watch_channels(self, uid: str | None = None) -> list[dict]:
        query = self._db.collection("watch_channels")
        if uid is not None:
            query = query.where("uid", "==", uid)
        return [{"id": s.id, **s.to_dict()} for s in query.stream()]

    def delete_watch_channel(self, channel_id: str) -> None:
        self._db.collection("watch_channels").document(channel_id).delete()

    def list_due_checks(self, uid: str) -> list[dict]:
        return [{"id": s.id, **s.to_dict()} for s in self._db.collection("due_checks").where("uid", "==", uid).stream()]

    def replace_due_checks(self, uid: str, checks: list[dict]) -> None:
        col = self._db.collection("due_checks")
        keep = {c["id"] for c in checks}
        writes = [("delete", snap.reference, None) for snap in col.where("uid", "==", uid).stream() if snap.id not in keep]
        writes += [("set", col.document(c["id"]), {k: v for k, v in c.items() if k != "id"}) for c in checks]
        for i in range(0, len(writes), _BATCH_LIMIT):
            batch = self._db.batch()
            for op, ref, data in writes[i:i + _BATCH_LIMIT]:
                batch.delete(ref) if op == "delete" else batch.set(ref, data)
            batch.commit()

    def due_checks_before(self, until: int) -> list[dict]:
        # One range filter only, so Firestore's automatic single-field index serves
        # it; status is filtered by the caller. Past checks are dropped on each sync.
        query = self._db.collection("due_checks").where("check_at", "<=", until).order_by("check_at")
        return [{"id": s.id, **s.to_dict()} for s in query.stream()]

    def update_due_check(self, check_id: str, fields: dict) -> None:
        self._db.collection("due_checks").document(check_id).set(_to_firestore(fields), merge=True)


def _to_firestore(fields: dict) -> dict:
    """None -> DELETE_FIELD, recursively (set(merge=True) accepts it nested)."""
    return {k: DELETE_FIELD if v is None else _to_firestore(v) if isinstance(v, dict) else v
            for k, v in fields.items()}


def _flatten(fields: dict, prefix: str = "") -> dict:
    """update() replaces whole maps, so merge nested dicts as dotted field paths."""
    out: dict = {}
    for k, v in fields.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{path}.") if v else {path: v})
        else:
            out[path] = v
    return out
