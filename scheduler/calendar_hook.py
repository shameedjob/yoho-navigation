"""CalendarQueue: an IndexedPriorityQueue subclass that stamps each entry
with an `address` -- which time bucket ("day") its priority falls into,
floor(priority / bucket_width) -- alongside the existing `priority` (sort
key) and `key` (dict key) fields.

Ordering is still done by the inherited min-heap, sorted by `priority`
(the event's time) -- this class doesn't change how the minimum is found,
it only changes how entries are built:
  - `address` is added as extra metadata on each entry.
  - `count`, the tie-break for equal-priority entries, is the current
    Unix timestamp at insertion (time.time()) instead of the parent's
    incrementing counter.

Because the heap and the lazy-delete removal scheme don't care about
anything beyond `priority`/`count`/`key`/`data`, `pop`, `peek`, `remove`,
`peek_second`, `__len__`, `__contains__`, and `priority_of` are all
inherited unchanged from IndexedPriorityQueue -- only `push` needs
overriding, to build a `_CalendarEntry` instead of a plain `_Entry`.
"""

from __future__ import annotations

import heapq
import math
import time as time_module
from typing import Any, Hashable
from dataclasses import dataclass, field

from .indexed_priority_queue import _REMOVED, IndexedPriorityQueue
from .indexed_priority_queue import _Entry as _BaseEntry

# How long after an event ends the rider is still assumed to be there.
AFTER_EVENT_SEC = 60 * 60


@dataclass
class CalendarEvent: #to be used as the data for the IPQ
    address: str | None
    lat: float | None  # None until geocoded
    long: float | None
    id: str #<- key
    time: int #<- priority, Unix start
    duration: int  # seconds

    @property
    def end(self) -> int:
        return self.time + self.duration


@dataclass
class CalendarHandler:
    ipq: IndexedPriorityQueue = field(default_factory=IndexedPriorityQueue)

    def add_event(self, id: str, time: int, duration: int, lat: float | None, long: float | None,
                  address: str | None) -> None:
        event = CalendarEvent(address=address, lat=lat, long=long, id=id, duration=duration, time=time)
        self.ipq.push(id, time, event)

    def delete_event(self, id: str) -> bool:
        return self.ipq.remove(id)

    def start_event(self, now: int) -> CalendarEvent | None:
        """The event at the top of the queue if `now` falls within it or the
        hour after it ends -- the rider is taken to be there -- else None.

        Events that ended more than AFTER_EVENT_SEC ago are popped on the way:
        they're past, and left in place they'd hold the top of the queue forever.
        """
        while (top := self.ipq.peek()) is not None:
            event = top[2]
            if event.end + AFTER_EVENT_SEC >= now:
                break
            self.ipq.pop()
        if top is None:
            return None
        event = top[2]
        return event if event.time <= now <= event.end + AFTER_EVENT_SEC else None
