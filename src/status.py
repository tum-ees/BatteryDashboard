from __future__ import annotations

from enum import IntEnum
from collections.abc import Iterable


class Status(IntEnum):
    UNKNOWN = -1
    GREEN = 0
    YELLOW = 1
    RED = 2

    @property
    def label(self) -> str:
        return {
            Status.UNKNOWN: "Unbekannt",
            Status.GREEN: "Grün",
            Status.YELLOW: "Gelb",
            Status.RED: "Rot",
        }[self]

    @property
    def emoji(self) -> str:
        return {
            Status.UNKNOWN: "⚪",
            Status.GREEN: "🟢",
            Status.YELLOW: "🟡",
            Status.RED: "🔴",
        }[self]


def worst_status(statuses: Iterable[Status]) -> Status:
    valid = [status for status in statuses if status != Status.UNKNOWN]
    return max(valid, default=Status.UNKNOWN)
