"""Which columns a model reads, kept out of the models themselves.

Both edge-cost models (ml_model/gbm.py and the line-graph DCRNN in
ml_model/model.py) take a FeatureSpec instead of hardcoding column names, so
the feature set can change -- lag features, alert encodings, whatever
docs/MODEL_DATA.md settles on -- without touching model code. A spec is saved
next to each trained model, so a checkpoint always knows which columns it
expects at prediction time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# How every per-edge table in the project identifies an edge: training rows,
# snapshot rows (snapshot/build.py) and Graph's per-period cost table all key
# on these three.
EDGE_KEY = ("from_node", "to_node", "is_transfer")


@dataclass
class FeatureSpec:
    """numeric: columns read as floats. Nulls are allowed and stay missing --
    each model decides how to represent absence rather than it being filled
    with a normal-looking value here.
    categorical: columns read as labels, e.g. route or direction.
    target: the column predicted. `edge_sec` by default, the target that
    survives schedule censoring (see docs/MODEL_DATA.md).
    """

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    target: str = "edge_sec"

    def __post_init__(self) -> None:
        columns = self.columns
        if not columns:
            raise ValueError("a FeatureSpec needs at least one feature column")
        duplicated = {c for c in columns if columns.count(c) > 1}
        if duplicated:
            raise ValueError(f"columns listed more than once: {sorted(duplicated)}")
        if self.target in columns:
            raise ValueError(f"target {self.target!r} is also listed as a feature")

    @property
    def columns(self) -> list[str]:
        return [*self.numeric, *self.categorical]

    def check(self, frame, need_target: bool = False) -> None:
        """Raise KeyError naming every column `frame` is missing."""
        wanted = self.columns + ([self.target] if need_target else [])
        missing = [c for c in wanted if c not in frame.columns]
        if missing:
            raise KeyError(f"frame is missing columns: {missing}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSpec":
        return cls(**data)


# Every feature derivable identically from training rows (after
# snapshot.build.add_graph_columns) and from live snapshots. How each side
# computes them is mapped in docs/FEATURE_PARITY.md; change both together.
MODEL_FEATURES = FeatureSpec(
    numeric=["sched_edge_sec", "has_schedule", "prior_delay_sec", "graph_edge_sec",
             "obs_last_edge_sec", "obs_last_age_sec",
             "hour", "minute_of_day", "dow", "is_weekend",
             "station_alert_count", "station_alert_age_sec"],
    categorical=["route", "direction", "service_period", "station_alert_types"],
)
