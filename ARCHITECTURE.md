# Architecture

## Overview

Yoho Navigation is an agentic navigator for NYC public transit. A user prompt (or a
calendar-driven trigger) asks for a route; an agent decides whether the request is
allowed, plans a path over a graph of transit stops using live MTA data, and returns
directions — either as a direct reply or as a proactive SMS alert ahead of a
calendar event.

**Status:** early design. Several components below (marked *planned*) are not yet
implemented in code.

## Components

### Input layer
| Component | Responsibility |
|---|---|
| User Prompt | Free-text request from a user asking for a route or transit info |
| Scheduled Event | A cron-style job (`scheduler/scheduler.py`) that triggers the Agent on a schedule (e.g. a proactive check), independent of user input |
| Calendar Hook *(planned)* | External webhook/push notification from the Google Calendar API, fired when a user's calendar event is added/updated, so a route can be pre-computed before they need it |
| Weekly Schedule Sync *(planned)* | A separate cron-style job that pulls GTFS static data — specifically bus schedules/routes, which the live MTA API doesn't cover — to rebuild the Graph. System-internal: doesn't go through the Agent |

### Web layer
| Component | Responsibility |
|---|---|
| Web service (`web/`) | Flask app: Google sign-in, on-demand calendar sync, chat API gated by a monthly token budget, and the frontend pages. Users, usage, and events live in Firestore (`storage/`); refresh tokens, home, and event locations are encrypted in the app (`storage/crypto.py`). See `docs/PLAN_FLASK_GOOGLE.md` |

### Prompt processing layer
| Component | Responsibility |
|---|---|
| Prompt Preprocessing | Validates/sanitizes the incoming prompt and checks it's an allowed request before it reaches the agent |

### Processing layer
| Component | Responsibility |
|---|---|
| Agent | Interprets the (pre-processed) request, decides which tools to call, and composes the response |
| Calendar Handler (`scheduler/calendar_hook.py`) | Doesn't call the Agent directly. Manages calendar events (`CalendarEvent`) — add/delete — backed by an indexed priority queue keyed on event time, so the entry due soonest is always cheap to find. When a Calendar Hook fires, it just enqueues "check this event before it starts"; something (the Scheduler, polling for due entries) is what later triggers the Agent |

### Tool layer
| Component | Responsibility |
|---|---|
| Graph pathfinding (`graph/graph.py`) | Dijkstra's algorithm over the transit graph to find the shortest path between stops |
| Geocoding (`geocoding/nominatim.py`) | Resolves a free-text address to (lat, lon) via OpenStreetMap's Nominatim API, so it can be matched to nearby graph nodes |
| Edge-cost models (`ml_model/`) *(built, untrained)* | Predict a high quantile (q=0.9) of per-edge travel time as Dijkstra weights. Two models over the same `FeatureSpec`: a LightGBM baseline (`gbm.py`, one row per edge) and a DCRNN over the subway line graph (`line_graph.py`, `forecaster.py`) forecasting several horizons ahead. The GBM's MODEL_DATA.md feature set and its row prep live in `model_data.py`; the DCRNN has no feature set yet |
| AWS SNS (SMS) *(planned)* | Sends SMS alerts to users — note: the AWS service for this is **SNS** (or Pinpoint), not "AWS SMS" |

### Data layer
| Component | Responsibility |
|---|---|
| Google Calendar API *(planned)* | External source of truth for calendar events — sends the webhook that becomes the Calendar Hook, and is queried for event details (time, location) once a hook fires |
| Calendar Database *(planned)* | Persists calendar events per user ID, linking a user to their Google account/credentials; the in-memory indexed priority queue (`scheduler/indexed_priority_queue.py`) is the runtime structure, this is its durable backing store |
| MTA API (`mta_api/client.py`) | Live GTFS-realtime feeds: trip updates, vehicle positions, service alerts (subway only today — see `docs/MTA_API.md`) |
| GTFS static feeds (bus) | Bus schedules/routes, fetched by the Weekly Schedule Sync job to rebuild the Graph — the live MTA API only covers subway, so this is the only source for bus data |
| Geocoding Cache *(planned)* | Shared (multi-instance) address -> (lat, lon) cache in front of Nominatim. Nominatim's public instance caps usage at 1 request/second (`geocoding/nominatim.py` doesn't rate-limit for you); since an address's coordinates essentially never change, a cache hit avoids that limit entirely instead of just queuing around it |

## Data flow

```
User Prompt   -> Prompt Preprocessing -> Agent -> Tools (Graph, GNN) -> Output

Google Calendar API -> Calendar Hook -> Calendar Handler (enqueue "check before event" entry)
                       |
                       v  (later, when the entry comes due)
                 Scheduler (polls the queue for due entries)
                       |
                       v
                     Agent -> Tools (Graph, GNN) -> AWS SNS (SMS)

Scheduled Event -> Agent -> Tools (Graph, GNN) -> AWS SNS (SMS)

Weekly Schedule Sync -> GTFS static feeds (bus) -> rebuild Graph
    (system-internal; does not touch the Agent or Tool layer)

Tools -> Data Layer (MTA API for live delays, Graph data for pathfinding,
         Geocoding Cache for address lookups, Calendar Database for event lookups)
```

Geocoding sits in front of pathfinding: a route request with a free-text origin/
destination needs 1-2 geocode calls before Dijkstra can even start. Without a cache,
every request pays Nominatim's 1 req/sec limit twice (serialized), which is slow
and doesn't scale past one in-flight request at a time — the cache is what makes
concurrent requests viable, not just a latency nicety.

The Calendar flow is two-stage and asynchronous: the hook only enqueues a future
check, it does not call the Agent itself. The Agent only runs later, once that
entry is due — at which point this collapses into the same shape as the Scheduled
Event flow (a scheduler-driven trigger, not a user-driven one).

The Weekly Schedule Sync is a different kind of "scheduled event" from the one in
the Input layer — it's a data-maintenance job that feeds the Graph directly, not a
trigger that produces a user-facing output. Worth keeping these two conceptually
separate even though both run on `scheduler/scheduler.py`.

All flows converge on the same Agent and Tool layer; they differ only in what
triggers them and what the output channel is (a direct reply vs. an SMS push).
The Tool layer depends on the Data layer for every call — this wasn't shown in the
original flow diagram but is a hard dependency (pathfinding needs live + static
transit data, calendar lookups need the Calendar Database).

## Open questions / known gaps

- **Edge-cost models**: the GBM trains on the MODEL_DATA.md features (`--feature-set
  model_data`); the DCRNN is still feature-agnostic and untrained. Open: a historical per-edge
  feature frame for the DCRNN to train on, and baselines.
  torch and LightGBM can't share a process on macOS (duplicate OpenMP runtime, OMP
  Error #15), so a service serving both needs them in separate processes or one libomp.
- **Calendar Database**: Firestore (`storage/firestore_store.py`), shared with user and usage data.
- **Google Calendar integration**: needs per-user OAuth (consent + token storage/refresh),
  and Google push notification channels expire (max 30 days) and must be renewed —
  worth deciding who's responsible for re-subscribing before a channel lapses.
- **Prompt allow-list logic**: what makes a prompt "allowed" isn't defined — is this a
  content filter, a rate limit, a scope check (transit-only questions), or all three?
- **Failure paths**: none of the flows above show what happens when the MTA API is
  down, SNS delivery fails, or the prompt is rejected. Worth at least a one-line
  fallback behavior per case before this goes further.
- **Privacy**: the Calendar Database maps events to user IDs — this is personal data
  (location + schedule) and should have an access-control note once storage is chosen.
- **Geocoding Cache**: decided to be shared storage (not per-instance), so it also
  helps concurrent/multi-instance requests, not just repeat lookups within one
  process. Since addresses rarely if ever change coordinates, this can likely be a
  long-lived/permanent cache rather than a TTL-based one. Concrete store (Redis?
  a Postgres table?) still open.
