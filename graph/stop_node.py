from dataclasses import dataclass

@dataclass
class StopNode:
    id:str
    stop_id:str
    vehicle:str
    mode:str  # "bus" or "subway"
    lat:float
    lon:float
    paths:list[tuple[str, int, bool]] #node id, average time to node, is_transfer.
