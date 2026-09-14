"""Weekly token budget that gates agent usage.

Each user gets `token_limit` tokens (input + output) per UTC ISO week (Monday to Sunday):
users/{uid}.token_limit if set, else the app default. The check runs before an
agent call and the usage is recorded after it, from Strands' own count.

The check can't know what a request will cost, so a user under the limit can
overshoot it by one request (plus any concurrent ones). Two things bound that:
the chat endpoint lets a user run one agent call at a time, and the model's
max output tokens caps a single reply. If a hard ceiling is ever needed,
reserve an estimate in a transaction before the call and settle after.
"""

from __future__ import annotations

from dataclasses import dataclass

from storage import UserStore, usage_period


@dataclass(frozen=True)
class QuotaStatus:
    period: str
    used: int
    limit: int

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def exceeded(self) -> bool:
        return self.used >= self.limit

    def as_dict(self) -> dict:
        return {"period": self.period, "used": self.used, "limit": self.limit, "remaining": self.remaining}


class QuotaExceeded(Exception):
    def __init__(self, status: QuotaStatus):
        super().__init__(f"weekly token limit reached ({status.used}/{status.limit})")
        self.status = status


def quota_status(store: UserStore, uid: str, default_limit: int, user: dict | None = None) -> QuotaStatus:
    user = user if user is not None else store.get_user(uid)
    limit = int((user or {}).get("token_limit", default_limit))
    period = usage_period()
    return QuotaStatus(period=period, used=int(store.get_usage(uid, period)["total_tokens"]), limit=limit)


def require_quota(store: UserStore, uid: str, default_limit: int) -> QuotaStatus:
    status = quota_status(store, uid, default_limit)
    if status.exceeded:
        raise QuotaExceeded(status)
    return status


def record_usage(store: UserStore, uid: str, input_tokens: int, output_tokens: int) -> None:
    store.add_usage(uid, usage_period(), int(input_tokens), int(output_tokens))
