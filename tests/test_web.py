"""Web service tests: Flask test client over MemoryStore, with Google and the
agent faked. No network, no Firebase, no torch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from accounts.calendar_sync import REFRESH_TOKEN_FIELD
from accounts.home import load_home
from integrations.google import CalendarAccessRevoked, GoogleIdentity, TokenGrant
from integrations.google.oauth import CALENDAR_SCOPE, SCOPES
from storage import DecryptionError, FieldCipher, FileStore, MemoryStore, usage_period
from storage.crypto import generate_key_entry
from web import create_app
from web.config import Settings

UID = "google-sub-123"
WRITE = {"X-Yoho-Request": "1"}


class FakeOAuth:
    def __init__(self):
        self.revoked: list[str] = []
        self.grant = TokenGrant(
            identity=GoogleIdentity(sub=UID, email="rider@example.com", email_verified=True, name="Rider", picture=None),
            refresh_token="refresh-secret", scopes=list(SCOPES))

    def authorization_url(self, login_hint=None):
        return "https://accounts.google.com/o/oauth2/auth?fake", "state-abc", "verifier-xyz"

    def exchange_code(self, code, state, verifier):
        assert (code, state, verifier) == ("the-code", "state-abc", "verifier-xyz")
        return self.grant

    def credentials(self, refresh_token, scopes):
        return SimpleNamespace(refresh_token=refresh_token)

    def revoke(self, refresh_token):
        self.revoked.append(refresh_token)
        return True


class FakeAgent:
    """Stands in for a Strands Agent: records the prompt, reports usage."""

    def __init__(self, history, tokens=(120, 30), reply="Take the 6.", fail=False):
        self.messages = list(history)
        self.event_loop_metrics = SimpleNamespace(accumulated_usage={"inputTokens": 0, "outputTokens": 0})
        self._tokens, self._reply, self._fail = tokens, reply, fail

    def __call__(self, prompt):
        self.event_loop_metrics.accumulated_usage = {"inputTokens": self._tokens[0], "outputTokens": self._tokens[1]}
        if self._fail:
            raise RuntimeError("model error")
        self.messages += [{"role": "user", "content": [{"text": prompt}]},
                          {"role": "assistant", "content": [{"text": self._reply}]}]
        return SimpleNamespace(message=self.messages[-1])


@pytest.fixture
def env(monkeypatch):
    store = MemoryStore()
    oauth = FakeOAuth()
    agent_opts: dict = {}
    settings = Settings(secret_key="test", google_client_id="cid", google_client_secret="secret",
                        google_redirect_uri="http://localhost/cb", data_keys=generate_key_entry("k1"),
                        store="memory", weekly_token_limit=1000, dev=True)
    calendar_items = [
        {"id": "ev1", "status": "confirmed", "summary": "Dentist", "location": "10 Union Sq E, New York",
         "start": {"dateTime": "2026-09-14T10:00:00-04:00"}, "end": {"dateTime": "2026-09-14T11:00:00-04:00"}},
        {"id": "ev2", "status": "cancelled", "start": {"date": "2026-09-15"}, "end": {"date": "2026-09-16"}},
    ]
    fetch = {"items": calendar_items, "error": None}

    def fake_fetch(credentials, **_):
        if fetch["error"]:
            raise fetch["error"]
        assert credentials.refresh_token == "refresh-secret"
        return fetch["items"]

    monkeypatch.setattr("accounts.calendar_sync.fetch_upcoming_events", fake_fetch)
    app = create_app(settings, store=store, oauth=oauth,
                     geocode=lambda addr: (40.7128, -74.0060) if "Water" in addr else None,
                     agent_factory=lambda uid, store, cipher, history, location=None: agent_opts.setdefault("locations", []).append(location) or FakeAgent(history, **{k: v for k, v in agent_opts.items() if k != "locations"}))
    app.testing = True
    return SimpleNamespace(app=app, client=app.test_client(), store=store, oauth=oauth, fetch=fetch,
                           agent_opts=agent_opts, cipher=app.extensions["yoho"].cipher)


def login(env):
    env.client.get("/auth/google/login")
    return env.client.get("/auth/google/callback?state=state-abc&code=the-code")


# --- crypto ---------------------------------------------------------------

def test_cipher_round_trip_and_binding():
    cipher = FieldCipher.from_env_value(generate_key_entry("k1"))
    token = cipher.encrypt("221 Water St", user_id="a", field="home")
    assert "Water" not in token
    assert cipher.decrypt(token, user_id="a", field="home") == "221 Water St"
    with pytest.raises(DecryptionError):
        cipher.decrypt(token, user_id="b", field="home")  # copied to another user
    with pytest.raises(DecryptionError):
        cipher.decrypt(token, user_id="a", field="other")  # copied to another field


def test_cipher_key_rotation():
    old = generate_key_entry("old")
    token = FieldCipher.from_env_value(old).encrypt("x", user_id="u", field="f")
    rotated = FieldCipher.from_env_value(f"{generate_key_entry('new')},{old}")
    assert rotated.decrypt(token, user_id="u", field="f") == "x"
    assert rotated.needs_rotation(token)


# --- auth -----------------------------------------------------------------

def test_login_stores_user_with_encrypted_refresh_token_and_syncs(env):
    resp = login(env)
    assert resp.status_code == 302 and resp.headers["Location"].endswith("/")
    user = env.store.get_user(UID)
    assert user["email"] == "rider@example.com"
    enc = user["google"]["refresh_token_enc"]
    assert "refresh-secret" not in enc
    assert env.cipher.decrypt(enc, user_id=UID, field=REFRESH_TOKEN_FIELD) == "refresh-secret"
    events = env.store.list_calendar_events(UID)
    assert [e["id"] for e in events] == ["ev1"]  # cancelled dropped
    assert "Union" not in events[0]["location_enc"]


def test_callback_rejects_state_mismatch(env):
    env.client.get("/auth/google/login")
    assert env.client.get("/auth/google/callback?state=forged&code=the-code").status_code == 400
    assert env.store.get_user(UID) is None


def test_callback_without_login_start_is_rejected(env):
    assert env.client.get("/auth/google/callback?state=state-abc&code=the-code").status_code == 400


def test_consent_denied_redirects_to_login(env):
    env.client.get("/auth/google/login")
    resp = env.client.get("/auth/google/callback?error=access_denied&state=state-abc")
    assert "/login?error=denied" in resp.headers["Location"]


def test_pages_and_api_require_login(env):
    assert env.client.get("/").headers["Location"].endswith("/login")
    assert env.client.get("/api/me").status_code == 401
    login(env)
    assert env.client.get("/").status_code == 200
    assert env.client.get("/profile").status_code == 200


def test_session_is_permanent_and_survives_restart_with_file_store(env, tmp_path):
    path = tmp_path / "store.pkl"
    store = FileStore(path)
    app = create_app(env.app.extensions["yoho"].settings, store=store, oauth=env.oauth, agent_factory=lambda *a, **k: None)
    client = app.test_client()
    client.get("/auth/google/login")
    resp = client.get("/auth/google/callback?state=state-abc&code=the-code")
    cookie = next(h for h in resp.headers.getlist("Set-Cookie") if h.startswith("yoho_session="))
    assert "Expires=" in cookie  # kept by the browser across closes, not a session-only cookie

    restarted = create_app(app.extensions["yoho"].settings, store=FileStore(path), oauth=env.oauth,
                           agent_factory=lambda *a, **k: None).test_client()
    restarted.set_cookie("yoho_session", client.get_cookie("yoho_session").value)
    assert restarted.get("/").status_code == 200
    assert restarted.get("/api/me").status_code == 200


def test_chat_passes_browser_location_to_agent_only_when_valid(env):
    login(env)
    headers = {"X-Yoho-Request": "1"}
    for location in ({"lat": 40.7359, "lon": -73.9911}, None, {"lat": "x", "lon": 1}, {"lat": 51.5, "lon": -0.12}):
        assert env.client.post("/api/chat", json={"message": "hi", "location": location}, headers=headers).status_code == 200
    assert env.agent_opts["locations"] == [(40.7359, -73.9911), None, None, None]  # bad and out-of-area ignored
    assert "40.7359" not in str(env.store.get_user(UID)) + str(env.store.get_conversation(UID))  # never stored


def test_writes_require_custom_header(env):
    login(env)
    assert env.client.post("/api/chat", json={"message": "hi"}).status_code == 403


# --- home -----------------------------------------------------------------

def test_home_from_address_is_encrypted_and_not_echoed(env):
    login(env)
    resp = env.client.put("/api/me/home", json={"address": "221 Water St"}, headers=WRITE)
    assert resp.get_json() == {"home_set": True}
    stored = env.store.get_user(UID)["home_enc"]
    assert "Water" not in stored and "40.71" not in stored
    home = load_home(env.store, env.cipher, UID)
    assert (home.label, home.lat, home.lon) == ("221 Water St", 40.7128, -74.0060)
    me = env.client.get("/api/me").get_json()
    assert me["home_set"] is True and "Water" not in str(me)


def test_home_validation(env):
    login(env)
    assert env.client.put("/api/me/home", json={"address": "nowhere"}, headers=WRITE).status_code == 422
    assert env.client.put("/api/me/home", json={"lat": 34.05, "lon": -118.24}, headers=WRITE).status_code == 422
    assert env.client.put("/api/me/home", json={"lat": 40.75, "lon": -73.99}, headers=WRITE).status_code == 200
    assert env.client.delete("/api/me/home", headers=WRITE).get_json() == {"home_set": False}
    assert load_home(env.store, env.cipher, UID) is None


def test_home_tool_routes_without_returning_home(env, monkeypatch):
    from agent.agent_interaction import make_user_tools
    login(env)
    env.client.put("/api/me/home", json={"lat": 40.75, "lon": -73.99}, headers=WRITE)
    calls = []
    fake_tools = SimpleNamespace(get_path=lambda start, end, departure_time=None, avoid=None: calls.append((start, end)) or {"steps": []})
    monkeypatch.setitem(__import__("sys").modules, "agent.tools", fake_tools)
    route_from_home = {t.tool_name: t for t in make_user_tools(UID, env.store, env.cipher)}["route_from_home"]
    assert route_from_home(destination=(40.70, -74.01)) == {"steps": []}
    assert calls == [((40.75, -73.99), (40.70, -74.01))]


def test_route_tools_report_departure_and_arrival_times(env, monkeypatch):
    from agent.agent_interaction import RouteLog, make_user_tools
    login(env)
    env.client.put("/api/me/home", json={"lat": 40.75, "lon": -73.99}, headers=WRITE)
    step = lambda stop, lat: {"stop_id": stop, "stop_name": stop, "mode": "subway", "route": "6", "lat": lat, "lon": -73.99}
    route = {"steps": [step("A", 40.74), step("B", 40.73)], "total_time_sec": 1500, "walk_in_sec": 60, "walk_out_sec": 60}
    monkeypatch.setitem(__import__("sys").modules, "agent.tools",
                        SimpleNamespace(get_path=lambda start, end, departure_time=None, avoid=None: route))
    tool = {t.tool_name: t for t in make_user_tools(UID, env.store, env.cipher, RouteLog())}["route_from_home"]
    result = tool(destination=(40.70, -74.01), depart_at="2026-09-14 09:00")
    assert (result["minutes"], result["departs"], result["arrives"]) == (25, "Mon Sep 14, 9:00 AM", "Mon Sep 14, 9:25 AM")


def test_current_location_tools_route_from_here_and_say_when_unknown(env, monkeypatch):
    from agent.agent_interaction import make_user_tools
    login(env)
    env.client.put("/api/me/home", json={"lat": 40.75, "lon": -73.99}, headers=WRITE)
    calls = []
    fake_tools = SimpleNamespace(get_path=lambda start, end, departure_time=None, avoid=None: calls.append((start, end)) or {"steps": []})
    monkeypatch.setitem(__import__("sys").modules, "agent.tools", fake_tools)
    here = (40.7359, -73.9911)
    tools = {t.tool_name: t for t in make_user_tools(UID, env.store, env.cipher, location=here)}
    assert tools["route_from_here"](destination=(40.70, -74.01)) == {"steps": []}
    assert tools["route_to_home"]() == {"steps": []}  # no start: leaves from here
    assert calls == [(here, (40.70, -74.01)), (here, (40.75, -73.99))]

    unknown = {t.tool_name: t for t in make_user_tools(UID, env.store, env.cipher)}
    assert unknown["route_from_here"](destination=(40.70, -74.01))["error"] == "no_location"
    assert unknown["route_to_home"]()["error"] == "no_location"


# --- calendar -------------------------------------------------------------

def test_calendar_sync_endpoint(env):
    login(env)
    env.fetch["items"] = []
    assert env.client.post("/api/calendar/sync", headers=WRITE).get_json() == {"synced": 0}
    assert env.store.list_calendar_events(UID) == []


def test_calendar_revoked_marks_user_and_returns_409(env):
    login(env)
    env.fetch["error"] = CalendarAccessRevoked("invalid_grant")
    resp = env.client.post("/api/calendar/sync", headers=WRITE)
    assert resp.status_code == 409 and resp.get_json()["error"] == "calendar_access_revoked"
    assert env.client.get("/api/me").get_json()["calendar"]["connected"] is False


def test_calendar_scope_unticked_is_not_connected(env):
    env.oauth.grant.scopes = [s for s in SCOPES if s != CALENDAR_SCOPE]
    login(env)
    assert env.client.post("/api/calendar/sync", headers=WRITE).get_json()["error"] == "calendar_not_connected"


# --- chat + quota ---------------------------------------------------------

def test_chat_records_usage_and_history(env):
    login(env)
    resp = env.client.post("/api/chat", json={"message": "How do I get to Union Square?"}, headers=WRITE)
    body = resp.get_json()
    assert body["reply"] == "Take the 6."
    assert body["usage"]["used"] == 150
    assert env.store.get_usage(UID, usage_period())["requests"] == 1
    assert len(env.store.get_conversation(UID)) == 2


def test_chat_bundles_route_with_reply(env, monkeypatch):
    login(env)
    route = {"minutes": 24, "lines": ["the 6 train"], "transfers": 0, "legs": [], "directions": "Take the 6."}
    original = FakeAgent.__call__

    def call_with_route(self, prompt):
        self.route_log = SimpleNamespace(route=route)
        return original(self, prompt)

    monkeypatch.setattr(FakeAgent, "__call__", call_with_route)
    body = env.client.post("/api/chat", json={"message": "Route to Union Square"}, headers=WRITE).get_json()
    assert body["reply"] == "Take the 6. I've highlighted the route on your map." and body["route"] == route


def test_chat_route_is_null_without_a_route(env):
    login(env)
    body = env.client.post("/api/chat", json={"message": "hi"}, headers=WRITE).get_json()
    assert body["route"] is None


def test_chat_blocked_when_over_limit(env):
    login(env)
    env.store.add_usage(UID, usage_period(), 900, 100)
    resp = env.client.post("/api/chat", json={"message": "hi"}, headers=WRITE)
    assert resp.status_code == 429 and resp.get_json()["usage"]["remaining"] == 0


def test_per_user_limit_override(env):
    login(env)
    env.store.update_user(UID, {"token_limit": 5000})
    env.store.add_usage(UID, usage_period(), 900, 100)
    assert env.client.post("/api/chat", json={"message": "hi"}, headers=WRITE).status_code == 200


def test_failed_agent_call_still_charges_tokens(env):
    login(env)
    env.agent_opts.update(fail=True)
    assert env.client.post("/api/chat", json={"message": "hi"}, headers=WRITE).status_code == 502
    assert env.store.get_usage(UID, usage_period())["total_tokens"] == 150


def test_chat_rejects_empty_and_long_messages(env):
    login(env)
    assert env.client.post("/api/chat", json={"message": "  "}, headers=WRITE).status_code == 400
    assert env.client.post("/api/chat", json={"message": "x" * 2001}, headers=WRITE).status_code == 413


def test_trim_history_starts_on_user_text():
    from agent.agent_interaction import trim_history
    msgs = [{"role": "user", "content": [{"text": "a"}]},
            {"role": "assistant", "content": [{"toolUse": {}}]},
            {"role": "user", "content": [{"toolResult": {}}]},
            {"role": "assistant", "content": [{"text": "b"}]},
            {"role": "user", "content": [{"text": "c"}]}]
    assert trim_history(msgs, limit=1) == msgs[4:]


def test_trim_history_keeps_tool_calls_only_from_last_request():
    from agent.agent_interaction import trim_history
    use, result = {"toolUse": {"name": "get_path"}}, {"toolResult": {"content": []}}
    msgs = [{"role": "user", "content": [{"text": "a"}]},
            {"role": "assistant", "content": [{"text": "Let me check."}, use]},
            {"role": "user", "content": [result]},
            {"role": "assistant", "content": [{"text": "b"}]},
            {"role": "user", "content": [{"text": "c"}]},
            {"role": "assistant", "content": [use]},
            {"role": "user", "content": [result]},
            {"role": "assistant", "content": [{"text": "d"}]}]
    assert trim_history(msgs) == [
        {"role": "user", "content": [{"text": "a"}]},
        {"role": "assistant", "content": [{"text": "Let me check."}, {"text": "b"}]},
        *msgs[4:],
    ]


def test_trim_history_keeps_last_request_whole_past_limit():
    from agent.agent_interaction import trim_history
    msgs = [{"role": "user", "content": [{"text": "a"}]}, {"role": "assistant", "content": [{"text": "b"}]},
            {"role": "user", "content": [{"text": "c"}]}]
    for _ in range(3):
        msgs += [{"role": "assistant", "content": [{"toolUse": {}}]}, {"role": "user", "content": [{"toolResult": {}}]}]
    msgs.append({"role": "assistant", "content": [{"text": "d"}]})
    assert trim_history(msgs, limit=4) == msgs[2:]


def test_usage_period_is_the_utc_iso_week():
    from datetime import datetime, timezone
    assert usage_period(datetime(2026, 9, 14, 0, 0, tzinfo=timezone.utc)) == "2026-W38"  # Monday
    assert usage_period(datetime(2026, 9, 20, 23, 59, tzinfo=timezone.utc)) == "2026-W38"  # Sunday
    assert usage_period(datetime(2027, 1, 1, tzinfo=timezone.utc)) == "2026-W53"  # ISO year, not calendar year


# --- account deletion -----------------------------------------------------

def test_delete_account_revokes_and_removes_everything(env):
    login(env)
    env.client.post("/api/chat", json={"message": "hi"}, headers=WRITE)
    assert env.client.delete("/api/me", headers=WRITE).get_json() == {"deleted": True}
    assert env.oauth.revoked == ["refresh-secret"]
    assert env.store.get_user(UID) is None
    assert env.store.list_calendar_events(UID) == []
    assert env.client.get("/api/me").status_code == 401


def test_chat_says_route_is_on_the_map_when_the_model_forgets(env, monkeypatch):
    login(env)
    original = FakeAgent.__call__

    def call_with_route(self, prompt):
        self.route_log = SimpleNamespace(route={"legs": [], "directions": "Take the 6."})
        return original(self, prompt)

    monkeypatch.setattr(FakeAgent, "__call__", call_with_route)
    body = env.client.post("/api/chat", json={"message": "Route to Union Square"}, headers=WRITE).get_json()
    assert body["reply"] == "Take the 6. I've highlighted the route on your map."


def test_dotenv_values_drop_inline_comments():
    from web.__main__ import _dotenv_value
    assert _dotenv_value("memory            # firestore in production") == "memory"
    assert _dotenv_value("            # path to service-account JSON") == ""
    assert _dotenv_value('"keep # this"') == "keep # this"
    assert _dotenv_value("abc#def") == "abc#def"


def test_loading_the_chat_page_starts_a_fresh_conversation(env):
    login(env)
    env.client.post("/api/chat", json={"message": "How do I get to Union Square?"}, headers=WRITE)
    assert len(env.store.get_conversation(UID)) == 2
    assert env.client.get("/").status_code == 200
    assert env.store.get_conversation(UID) == []
    agent_history = []
    env.app.extensions["yoho"].agent_factory = lambda uid, store, cipher, history, location=None: agent_history.append(list(history)) or FakeAgent(history)
    env.client.post("/api/chat", json={"message": "hi again"}, headers=WRITE)
    assert agent_history == [[]]  # the agent after a reload sees no earlier turns


def test_a_page_load_only_clears_that_users_conversation(env):
    login(env)
    other = "google-sub-other"
    env.store.upsert_user(other, {"email": "other@example.com"})
    env.store.set_conversation(other, [{"role": "user", "content": [{"text": "their trip"}]}])
    env.client.post("/api/chat", json={"message": "my trip"}, headers=WRITE)

    other_client = env.app.test_client()
    with other_client.session_transaction() as session:
        session["uid"] = other
    assert other_client.get("/").status_code == 200  # the other user reloads their chat

    assert env.store.get_conversation(other) == []
    assert len(env.store.get_conversation(UID)) == 2  # ours is untouched
