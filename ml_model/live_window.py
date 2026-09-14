"""The graph model's input window, live: which snapshots go in it.

Training (ml_model.train_gat) feeds the model `window` feature snapshots on a
`step_sec` grid, the last one at the grid time it predicts from. Each
snapshot is rebuilt by replay.snapshot_features *at* that grid time, and each
service day's grid starts fresh: replay.day_grid runs from the first step
after local midnight to the next midnight, and the steps before it are
padded with the encoder's "no row" vector. A window never reaches back into
the previous calendar day.

Live has to match on all three counts:

  grid        the snapshot service builds one snapshot per grid time, with
              snapshot.build.build_snapshot(at=grid time) on the first poll at
              or past it -- only data timestamped by the grid time, as replay
  window      the agent takes the last `window` grid times ending at the
              newest one the service has (`window_times`)
  padding     slots before the window end's local day start, and grid times
              the service has no snapshot for (it was down), are "no row"
              (`padded`)

Kept free of torch and LightGBM so the snapshot service can import it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from snapshot.build import TZ

STEP_SEC = 600
WINDOW = 6


def grid_floor(ts: int, step_sec: int = STEP_SEC) -> int:
    """The latest grid time at or before `ts`."""
    return ts // step_sec * step_sec


def window_times(end: int, step_sec: int = STEP_SEC, window: int = WINDOW) -> list[int]:
    """Grid times of a window ending at `end`, oldest first."""
    return [end - step_sec * k for k in range(window - 1, -1, -1)]


def day_start(end: int) -> int:
    """Local midnight starting the day replay.day_grid puts grid time `end` in.

    day_grid(date) runs from midnight + step to the next midnight inclusive, so
    a grid time exactly at midnight is the *previous* day's last step.
    """
    local = datetime.fromtimestamp(end - 1, TZ)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(pd.Timestamp(midnight).timestamp())


def padded(times: list[int], end: int) -> set[int]:
    """Window slots training would have padded: at or before the day start."""
    start = day_start(end)
    return {t for t in times if t <= start}


def _check() -> None:  # a quick self-test of the midnight rule
    midnight = int(pd.Timestamp("2026-05-05", tz=TZ).timestamp())
    assert day_start(midnight + STEP_SEC) == midnight
    assert day_start(midnight) == midnight - 86400
    times = window_times(midnight + 2 * STEP_SEC)
    assert padded(times, times[-1]) == {t for t in times if t <= midnight}
    assert datetime.fromtimestamp(midnight, TZ).hour == 0
    assert timedelta(seconds=STEP_SEC * WINDOW) == timedelta(hours=1)


if __name__ == "__main__":
    _check()
    print("ok")
