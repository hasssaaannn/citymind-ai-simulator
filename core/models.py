from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LocationType(str, Enum):
    RESIDENTIAL = "Residential"
    HOSPITAL = "Hospital"
    SCHOOL = "School"
    INDUSTRIAL = "Industrial"
    POWER_PLANT = "PowerPlant"
    AMBULANCE_DEPOT = "AmbulanceDepot"


@dataclass(slots=True)
class CityNode:
    node_id: str
    location_type: LocationType
    population_density: float
    risk_index: int = 0
    accessible: bool = True
    grid_row: int | None = None
    grid_col: int | None = None
    officer_allocation: int = 0


@dataclass(slots=True)
class CityEdge:
    node_a: str
    node_b: str
    base_cost: float = 1.0
    blocked: bool = False

