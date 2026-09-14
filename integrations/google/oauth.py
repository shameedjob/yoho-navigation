"""Google OAuth 2.0 / OpenID Connect: one server-side flow covers sign-in and
Calendar access.

Why not Firebase Auth's Google sign-in: it runs in the browser and yields an
ID token plus a short-lived access token, never a refresh token, so nothing
could read the calendar while the user is away (the scheduler's "check before
the event" runs hours later). The authorization-code flow here, with
access_type=offline, returns a refresh token the server keeps -- encrypted --
in Firestore.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Google may grant a subset of the requested scopes (granular consent lets the
# user untick Calendar); without this oauthlib raises instead of letting the
# caller check what was granted.
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

import requests
from google.auth.transport.requests import Request
from google.oauth2 import id_token
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/calendar.readonly",
]
CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar.readonly"
TOKEN_URI = "https://oauth2.googleapis.com/token"
REVOKE_URI = "https://oauth2.googleapis.com/revoke"


@dataclass
class GoogleIdentity:
    sub: str
    email: str
    email_verified: bool
    name: str | None
    picture: str | None


@dataclass
class TokenGrant:
    identity: GoogleIdentity
    refresh_token: str | None  # None when Google doesn't reissue one
    scopes: list[str]


class GoogleOAuth:
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def _flow(self, state: str | None = None) -> Flow:
        config = {"web": {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": TOKEN_URI,
        }}
        return Flow.from_client_config(config, scopes=SCOPES, state=state, redirect_uri=self.redirect_uri,
                                       autogenerate_code_verifier=True)

    def authorization_url(self, login_hint: str | None = None) -> tuple[str, str, str]:
        """Returns (url, state, code_verifier); the caller keeps the last two
        in the session until the callback."""
        flow = self._flow()
        extra = {"login_hint": login_hint} if login_hint else {}
        url, state = flow.authorization_url(
            access_type="offline",
            prompt="consent",  # makes Google return a refresh token every time
            include_granted_scopes="true",
            **extra,
        )
        return url, state, flow.code_verifier

    def exchange_code(self, code: str, state: str, code_verifier: str) -> TokenGrant:
        flow = self._flow(state)
        flow.code_verifier = code_verifier
        flow.fetch_token(code=code)
        creds = flow.credentials
        # Verifies signature, aud == our client id, iss, and exp.
        claims = id_token.verify_oauth2_token(creds.id_token, Request(), self.client_id)
        identity = GoogleIdentity(
            sub=claims["sub"],
            email=claims.get("email", ""),
            email_verified=bool(claims.get("email_verified")),
            name=claims.get("name"),
            picture=claims.get("picture"),
        )
        return TokenGrant(identity=identity, refresh_token=creds.refresh_token, scopes=list(creds.granted_scopes or creds.scopes or []))

    def credentials(self, refresh_token: str, scopes: list[str]) -> Credentials:
        """Credentials that fetch an access token on first use."""
        return Credentials(token=None, refresh_token=refresh_token, token_uri=TOKEN_URI,
                           client_id=self.client_id, client_secret=self.client_secret, scopes=scopes)

    def revoke(self, refresh_token: str) -> bool:
        resp = requests.post(REVOKE_URI, data={"token": refresh_token}, timeout=10)
        return resp.status_code == 200
