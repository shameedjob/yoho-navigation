"""A priority queue supporting O(1)-initiated removal/reschedule of an
arbitrary entry by key, on top of Python's heapq.

A plain heap gives O(log n) push and O(log n) pop-minimum, but has no way
to find or remove an arbitrary entry without an O(n) scan -- there's no
ordering relationship between sibling subtrees to search with. This adds
a dict mapping key -> heap entry, so removing or rescheduling a specific
key only needs a dict lookup to find it: the entry is marked dead in
place rather than physically removed from the heap array, and gets
skipped and discarded the next time it surfaces at the top of the heap.
This is the pattern described in Python's own heapq docs ("Priority
Queue Implementation Notes").

Trade-off: pop()/peek() are O(log n) amortized, not strict worst-case --
if many entries are removed between pops, they sit as dead weight in the
heap until popped past. Each dead entry is only ever skipped once,
though, so the amortized cost per removal stays bounded over the queue's
lifetime.
"""

from __future__ import annotations

import heapq
from typing import Any, Hashable

_REMOVED = object()  # sentinel marking a dead entry, stored in its .key slot


class _Entry:
    __slots__ = ("priority", "count", "key", "data")

    def __init__(self, priority: Any, count: int, key: Hashable, data: Any):
        self.priority = priority
        self.count = count
        self.key = key
        self.data = data

    def __lt__(self, other: "_Entry") -> bool:
        # count is a tie-breaker so entries with equal priority compare by
        # insertion order, and heapq never has to compare .data directly.
        return (self.priority, self.count) < (other.priority, other.count)


class IndexedPriorityQueue:
    """Min-priority queue keyed by an arbitrary hashable `key`, supporting
    O(1)-initiated removal/reschedule of any key, not just the minimum.
    """

    def __init__(self) -> None:
        self._heap: list[_Entry] = []
        self._entries: dict[Hashable, _Entry] = {}
        self._counter = 0

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._entries

    def priority_of(self, key: Hashable) -> Any | None:
        entry = self._entries.get(key)
        return None if entry is None else entry.priority

    def get_counter(self)->int:
        self._counter+=1
        return self._counter
    
    def push(self, key: Hashable, priority: Any, data: Any = None) -> None:
        """Add a new key, or reschedule it (with a possibly new priority
        and data) if it's already present.
        """
        existing = self._entries.get(key)
        if existing is not None:
            existing.key = _REMOVED  # mark the old entry dead in place

        entry = _Entry(priority=priority, count=self.get_counter(), key=key, data=data)
        self._entries[key] = entry
        heapq.heappush(self._heap, entry)

    def remove(self, key: Hashable) -> bool:
        """Remove a key if present. Returns whether it was present."""
        entry = self._entries.pop(key, None)
        if entry is None:
            return False
        entry.key = _REMOVED
        return True

    def _clean_top(self) -> None:
        while self._heap and self._heap[0].key is _REMOVED:
            heapq.heappop(self._heap)

    def peek(self) -> tuple[Hashable, Any, Any] | None:
        """The (key, priority, data) with the smallest priority, without
        removing it, or None if the queue is empty.
        """
        self._clean_top()
        if not self._heap:
            return None
        entry = self._heap[0]
        return (entry.key, entry.priority, entry.data)

    def pop(self) -> tuple[Hashable, Any, Any] | None:
        """Remove and return the (key, priority, data) with the smallest
        priority, or None if the queue is empty.
        """
        self._clean_top()
        if not self._heap:
            return None
        entry = heapq.heappop(self._heap)
        del self._entries[entry.key]
        return (entry.key, entry.priority, entry.data)

    def peek_second(self) -> tuple[Hashable, Any, Any] | None:
        """The entry with the second-smallest priority, or None if fewer
        than two entries remain.

        In a min-heap the second-smallest is always one of the root's two
        children, which would make this O(1) -- but that shortcut isn't
        safe here since either child could be a dead (removed) entry with
        live descendants further down. This does it correctly instead, as
        pop + peek + push-back: O(log n), but always right regardless of
        how many dead entries are in the way.
        """
        first = self.pop()
        if first is None:
            return None
        second = self.peek()
        self.push(*first)
        return second
