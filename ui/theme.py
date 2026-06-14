from __future__ import annotations

from core.models import LocationType


BACKGROUND = (16, 18, 24)
PANEL_BG = (28, 33, 46)
PANEL_BORDER = (74, 86, 112)
HUD_TEXT = (220, 225, 235)
MUTED_TEXT = (150, 158, 175)
HIGHLIGHT = (255, 214, 104)

EDGE_DEFAULT = (90, 100, 125)
EDGE_ACTIVE = (255, 210, 90)
EDGE_BLOCKED = (210, 70, 70)

AMBULANCE_MARKER = (80, 220, 180)
POLICE_MARKER = (86, 176, 255)
POLICE_TEXT = (8, 16, 30)
CIVILIAN_PENDING = (230, 180, 90)
CIVILIAN_REACHED = (90, 220, 120)
CIVILIAN_UNREACHABLE = (210, 70, 70)

LOCATION_COLORS: dict[str, tuple[int, int, int]] = {
    LocationType.HOSPITAL.value: (210, 70, 70),
    LocationType.SCHOOL.value: (90, 130, 220),
    LocationType.INDUSTRIAL.value: (230, 140, 60),
    LocationType.RESIDENTIAL.value: (140, 150, 170),
    LocationType.POWER_PLANT.value: (230, 210, 80),
    LocationType.AMBULANCE_DEPOT.value: (90, 200, 120),
}

RISK_COLORS: dict[int, tuple[int, int, int]] = {
    0: (90, 200, 120),
    1: (230, 200, 90),
    2: (210, 80, 80),
}

COVERAGE_PALETTE: list[tuple[int, int, int]] = [
    (80, 150, 220),
    (170, 90, 210),
    (90, 200, 180),
    (230, 130, 90),
    (200, 200, 90),
]


def location_color(location_type: str) -> tuple[int, int, int]:
    return LOCATION_COLORS.get(location_type, (200, 200, 200))


def risk_color(risk_index: int) -> tuple[int, int, int]:
    return RISK_COLORS.get(int(risk_index), (180, 180, 180))


def coverage_color(index: int) -> tuple[int, int, int]:
    return COVERAGE_PALETTE[index % len(COVERAGE_PALETTE)]
