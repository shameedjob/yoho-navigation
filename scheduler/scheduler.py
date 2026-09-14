"""A lightweight cron-like scheduler: register jobs on standard 5-field
cron expressions, and have them fire a callback -- e.g. an agent call --
when due.

Cron expression format (5 space-separated fields, same as Unix cron):

    minute  hour  day-of-month  month  day-of-week
      *      *         *          *          *

Examples:
    "*/5 * * * *"    every 5 minutes
    "0 * * * *"      the top of every hour
    "0 8 * * *"      8:00 AM every day
    "0 8 * * 1-5"    8:00 AM on weekdays (1=Monday ... 5=Friday)
    "30 6 1 * *"     6:30 AM on the 1st of every month

Cron's minimum granularity is one minute -- there's no way to express
"every 5 seconds" in cron syntax itself.

The scheduler doesn't know or care what a job's callback does; point it at
a function that calls an agent, hits an API, refreshes data, etc.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from croniter import croniter


@dataclass
class Job:
    name: str
    cron_expr: str
    callback: Callable[[], None]
    next_run: datetime


class Scheduler:
    def __init__(self) -> None:
        self._heap: list[tuple[datetime, int, Job]] = []
        self._counter = 0  # heap tie-breaker so Job objects are never compared

    def add_job(self, cron_expr: str, callback: Callable[[], None], name: str | None = None) -> Job:
        next_run = croniter(cron_expr, datetime.now()).get_next(datetime)
        job = Job(name=name or callback.__name__, cron_expr=cron_expr, callback=callback, next_run=next_run)
        self._push(job)
        return job

    def _push(self, job: Job) -> None:
        self._counter += 1
        heapq.heappush(self._heap, (job.next_run, self._counter, job))

    def list_jobs(self) -> list[Job]:
        return [job for _, _, job in self._heap]

    def run_pending(self) -> int:
        """Run any jobs whose scheduled time has passed. Returns how many ran."""
        now = datetime.now()
        ran = 0
        while self._heap and self._heap[0][0] <= now:
            _, _, job = heapq.heappop(self._heap)
            job.callback()
            ran += 1
            # Advance from the job's own scheduled time, not `now`, so a
            # slow callback doesn't cause drift or skipped runs.
            job.next_run = croniter(job.cron_expr, job.next_run).get_next(datetime)
            self._push(job)
        return ran

    def run_forever(self, poll_interval: float = 1.0) -> None:
        """Block, sleeping until jobs come due, running them as they do.
        Runs until interrupted (e.g. Ctrl+C).
        """
        while True:
            if self._heap:
                sleep_for = (self._heap[0][0] - datetime.now()).total_seconds()
                if sleep_for > 0:
                    time.sleep(min(sleep_for, poll_interval))
            else:
                time.sleep(poll_interval)
            self.run_pending()


if __name__ == "__main__":
    def call_agent() -> None:
        # Replace this with whatever "agent call" means for you -- e.g.
        # invoking an LLM, calling MTAClient.get_alerts(), hitting a
        # webhook, etc. The scheduler just calls this with no arguments.
        print(f"[{datetime.now().isoformat(timespec='seconds')}] agent call triggered")

    scheduler = Scheduler()
    scheduler.add_job("* * * * *", call_agent, name="every-minute-agent-call")
    print("scheduled jobs:", [(j.name, j.cron_expr, j.next_run) for j in scheduler.list_jobs()])
    scheduler.run_forever()
