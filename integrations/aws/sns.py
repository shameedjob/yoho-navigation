"""Email alerts through Amazon SNS.

One topic for all users (YOHO_SNS_TOPIC_ARN). Each user's email is an "email"
subscription with a filter policy on their uid, and every alert is published
with a `uid` message attribute, so it reaches only that user's inbox.

SNS email specifics:
  - A new subscription stays "PendingConfirmation" until the user clicks the
    link AWS emails them; until then publishes to them are dropped.
  - Messages are plain text, from "AWS Notifications" -- fine for alerts to
    yourself while testing. For branded or HTML email to real users, Amazon SES
    is the service meant for it; Notifier keeps that a drop-in swap.

Credentials come from boto3's usual chain: AWS_ACCESS_KEY_ID /
AWS_SECRET_ACCESS_KEY (or AWS_PROFILE), region from AWS_REGION. The IAM
principal needs sns:Subscribe, sns:Publish and sns:SetSubscriptionAttributes
on the topic.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

log = logging.getLogger(__name__)

SUBJECT_MAX = 100  # SNS rejects longer email subjects


class Notifier(Protocol):
    def subscribe_email(self, uid: str, email: str) -> str | None:
        """Subscribe `email` to `uid`'s alerts; returns the subscription ARN
        (or "pending confirmation")."""
    def send(self, uid: str, subject: str, body: str) -> str:
        """Deliver an alert to `uid`; returns a message id."""


class SnsNotifier:
    def __init__(self, topic_arn: str, region: str | None = None, client=None):
        if client is None:
            import boto3
            client = boto3.client("sns", region_name=region)
        self.topic_arn = topic_arn
        self._sns = client

    def subscribe_email(self, uid: str, email: str) -> str | None:
        resp = self._sns.subscribe(
            TopicArn=self.topic_arn, Protocol="email", Endpoint=email, ReturnSubscriptionArn=True,
            Attributes={"FilterPolicy": json.dumps({"uid": [uid]})},
        )
        return resp.get("SubscriptionArn")

    def send(self, uid: str, subject: str, body: str) -> str:
        resp = self._sns.publish(
            TopicArn=self.topic_arn, Subject=subject[:SUBJECT_MAX], Message=body,
            MessageAttributes={"uid": {"DataType": "String", "StringValue": uid}},
        )
        return resp["MessageId"]


class LogNotifier:
    """Dry run for local development without AWS keys: logs what would be sent."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    def subscribe_email(self, uid: str, email: str) -> str | None:
        log.info("[dry run] would subscribe %s to alerts for %s", email, uid)
        return "dry-run"

    def send(self, uid: str, subject: str, body: str) -> str:
        self.sent.append({"uid": uid, "subject": subject, "body": body})
        log.info("[dry run] alert for %s: %s\n%s", uid, subject, body)
        return f"dry-run-{len(self.sent)}"
