"""Yoho web service: Google sign-in, calendar sync, the chat API, and the
frontend pages, all from one Flask app so the browser talks to one origin
with one session cookie (no CORS, no tokens in JavaScript).

    YOHO_DEV=1 YOHO_STORE=memory python -m web          # local
    gunicorn -w 2 "web:create_app()"                     # production

Dependencies are built once in create_app and hung off app.extensions["yoho"];
tests pass fakes for any of them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

from integrations.google import GoogleOAuth
from storage import FieldCipher, MemoryStore, UserStore

from .config import Settings

Geocoder = Callable[[str], "tuple[float, float] | None"]


@dataclass
class Services:
    settings: Settings
    store: UserStore
    cipher: FieldCipher
    oauth: GoogleOAuth
    geocode: Geocoder
    agent_factory: Callable[..., Any]
    notifier: Any  # integrations.aws.Notifier
    run_background: Callable[[Callable[[], None]], Any]


def _nominatim_geocode(address: str) -> tuple[float, float] | None:
    from geocoding import NominatimClient
    with NominatimClient(user_agent="yoho-navigation-web/0.1") as client:
        return client.geocode(address)


def _default_agent_factory(uid: str, store: UserStore, cipher: FieldCipher, history: list[dict]):
    from agent.agent_interaction import build_agent
    return build_agent(uid, store, cipher, history)


def make_notifier(settings: Settings):
    """SNS when a topic is configured, else a dry run that only logs."""
    from integrations.aws import LogNotifier, SnsNotifier
    return SnsNotifier(settings.sns_topic_arn, settings.aws_region) if settings.sns_topic_arn else LogNotifier()


_background = None


def _run_in_thread_pool(fn: Callable[[], None]):
    """Webhook work off the request thread: Google retries slow responses."""
    global _background
    if _background is None:
        from concurrent.futures import ThreadPoolExecutor
        _background = ThreadPoolExecutor(max_workers=2, thread_name_prefix="yoho-bg")
    return _background.submit(fn)


def create_app(settings: Settings | None = None, *, store: UserStore | None = None, oauth: GoogleOAuth | None = None,
               geocode: Geocoder | None = None, agent_factory: Callable[..., Any] | None = None,
               notifier: Any = None, run_background: Callable[[Callable[[], None]], Any] | None = None) -> Flask:
    settings = settings or Settings.from_env()
    if settings.dev:
        os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

    if store is None:
        if settings.store == "memory":
            store = MemoryStore()
        else:
            from storage.firestore_store import FirestoreStore
            store = FirestoreStore(settings.firebase_credentials, settings.firebase_project_id)

    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=settings.secret_key,
        SESSION_COOKIE_NAME="yoho_session",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",  # Lax, not Strict: the OAuth redirect back from Google must carry it
        SESSION_COOKIE_SECURE=not settings.dev,
        PERMANENT_SESSION_LIFETIME=timedelta(days=30),
        MAX_CONTENT_LENGTH=64 * 1024,
    )
    if not settings.dev:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    app.extensions["yoho"] = Services(
        settings=settings,
        store=store,
        cipher=FieldCipher.from_env_value(settings.data_keys),
        oauth=oauth or GoogleOAuth(settings.google_client_id, settings.google_client_secret, settings.google_redirect_uri),
        geocode=geocode or _nominatim_geocode,
        agent_factory=agent_factory or _default_agent_factory,
        notifier=notifier or make_notifier(settings),
        run_background=run_background or _run_in_thread_pool,
    )

    from . import api, auth, calendar_webhooks, pages
    app.register_blueprint(auth.bp)
    app.register_blueprint(calendar_webhooks.bp)
    app.register_blueprint(api.bp)
    app.register_blueprint(pages.bp)

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        return resp

    return app
