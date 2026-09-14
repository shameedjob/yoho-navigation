"""Environment-driven settings for the web service.

  FLASK_SECRET_KEY              signs the session cookie (required)
  GOOGLE_CLIENT_ID              OAuth "Web application" client (required)
  GOOGLE_CLIENT_SECRET          (required)
  GOOGLE_REDIRECT_URI           default http://localhost:5000/auth/google/callback
  YOHO_DATA_KEYS                field-encryption keys, see storage/crypto.py (required)
  YOHO_STORE                    "firestore" (default) or "memory" (dev/tests; lost on restart)
  FIREBASE_CREDENTIALS          service-account JSON path; unset = Application Default Credentials
  FIREBASE_PROJECT_ID           optional
  YOHO_MONTHLY_TOKEN_LIMIT      default per-user monthly tokens, default 200000
  YOHO_AGENT_MODEL              Strands model id; unset = Strands' default
  YOHO_DEV                      "1" allows http:// OAuth redirects and non-Secure cookies
  YOHO_WEBHOOK_BASE_URL         public HTTPS origin Google can reach (e.g. an ngrok URL);
                                unset = no calendar push notifications
  YOHO_SNS_TOPIC_ARN            SNS topic for alert emails; unset = alerts are only logged
  AWS_REGION                    region of the topic; AWS keys come from boto3's usual chain
  YOHO_ALERT_LEAD_MINUTES       alert this long before an event starts, default 60 (accounts/alerts.py)
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    secret_key: str
    google_client_id: str
    google_client_secret: str
    google_redirect_uri: str
    data_keys: str
    store: str = "firestore"
    firebase_credentials: str | None = None
    firebase_project_id: str | None = None
    monthly_token_limit: int = 200_000
    dev: bool = False
    webhook_base_url: str | None = None
    sns_topic_arn: str | None = None
    aws_region: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        missing = [k for k in ("FLASK_SECRET_KEY", "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "YOHO_DATA_KEYS")
                   if not os.environ.get(k)]
        if missing:
            raise RuntimeError(f"missing required environment variables: {', '.join(missing)}")
        return cls(
            secret_key=os.environ["FLASK_SECRET_KEY"],
            google_client_id=os.environ["GOOGLE_CLIENT_ID"],
            google_client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
            google_redirect_uri=os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:5000/auth/google/callback"),
            data_keys=os.environ["YOHO_DATA_KEYS"],
            store=os.environ.get("YOHO_STORE", "firestore"),
            firebase_credentials=os.environ.get("FIREBASE_CREDENTIALS") or None,
            firebase_project_id=os.environ.get("FIREBASE_PROJECT_ID") or None,
            monthly_token_limit=int(os.environ.get("YOHO_MONTHLY_TOKEN_LIMIT", "200000")),
            dev=os.environ.get("YOHO_DEV") == "1",
            webhook_base_url=os.environ.get("YOHO_WEBHOOK_BASE_URL") or None,
            sns_topic_arn=os.environ.get("YOHO_SNS_TOPIC_ARN") or None,
            aws_region=os.environ.get("AWS_REGION") or None,
        )
