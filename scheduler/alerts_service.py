"""The alert scheduler: one long-running process next to the web app.

    python -m scheduler.alerts_service            # run forever
    python -m scheduler.alerts_service --once     # one pass of each job, then exit

Jobs (scheduler.Scheduler, cron syntax):
  * * * * *   run_due_checks   email every alert that's due (accounts/alerts.py)
  0 * * * *   renew_watches    replace calendar push channels expiring within a day

Run exactly one of these: two would send every alert twice. It reads the same
.env as the web app (web/config.py) and needs the same store -- YOHO_STORE=memory
only works inside one process, so a real deployment uses Firestore. Routing
needs no torch: the schedule graph plus live first-train waits from the snapshot
service (accounts/trips.py schedule_router), or the schedule alone without it.
"""

from __future__ import annotations

import argparse
import logging
import time

from accounts.alerts import renew_watches, run_due_checks
from accounts.trips import schedule_router, transit_geocode
from integrations.google import GoogleOAuth
from scheduler import Scheduler
from storage import FieldCipher

log = logging.getLogger("scheduler.alerts")


def build_services():
    from web import make_notifier
    from web.__main__ import _load_dotenv
    from web.config import Settings

    _load_dotenv()
    settings = Settings.from_env()
    if settings.store == "memory":
        log.warning("YOHO_STORE=memory: this process can't see the web app's users; use firestore")
        from storage import MemoryStore
        store = MemoryStore()
    else:
        from storage.firestore_store import FirestoreStore
        store = FirestoreStore(settings.firebase_credentials, settings.firebase_project_id)
    oauth = GoogleOAuth(settings.google_client_id, settings.google_client_secret, settings.google_redirect_uri)
    return settings, store, FieldCipher.from_env_value(settings.data_keys), oauth, make_notifier(settings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="run each job once and exit")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings, store, cipher, oauth, notifier = build_services()

    def due_checks() -> None:
        for r in run_due_checks(store, cipher, notifier, int(time.time()),
                                geocode=transit_geocode, route=schedule_router,
                                link_base=settings.public_base_url):
            log.info("due check %s: %s%s", r["id"], r["status"], f" ({r['error']})" if r.get("error") else "")

    def watches() -> None:
        if not settings.webhook_base_url:
            return
        renewed = renew_watches(store, cipher, oauth, settings.webhook_base_url, int(time.time()))
        if renewed:
            log.info("renewed calendar channels for %d users", len(renewed))

    def guarded(job):
        # Scheduler.run_forever stops on the first exception; one bad run shouldn't end the service.
        def run() -> None:
            try:
                job()
            except Exception:
                log.exception("%s failed", job.__name__)
        run.__name__ = job.__name__
        return run

    if args.once:
        guarded(due_checks)()
        guarded(watches)()
        return
    scheduler = Scheduler()
    scheduler.add_job("* * * * *", guarded(due_checks), name="run_due_checks")
    scheduler.add_job("0 * * * *", guarded(watches), name="renew_watches")
    log.info("alert scheduler running (notifier: %s)", type(notifier).__name__)
    scheduler.run_forever()


if __name__ == "__main__":
    main()
