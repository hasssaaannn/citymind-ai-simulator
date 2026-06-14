from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, DefaultDict


ROAD_BLOCKED = "ROAD_BLOCKED"
RISK_UPDATED = "RISK_UPDATED"
RISK_BATCH_UPDATED = "RISK_BATCH_UPDATED"
PLACEMENT_CHANGED = "PLACEMENT_CHANGED"

EventHandler = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class EventBus:
    _subscribers: DefaultDict[str, list[EventHandler]]

    def __init__(self) -> None:
        self._subscribers = defaultdict(list)

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        self._subscribers[event_name].append(handler)

    def publish(self, event_name: str, payload: dict[str, Any]) -> None:
        for handler in self._subscribers[event_name]:
            handler(payload)

