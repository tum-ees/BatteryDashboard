from __future__ import annotations

import math
from collections.abc import Callable

from src.status import Status


def _finite(value: float) -> bool:
    """Return True only for a real finite numeric measurement."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def humidity_rule(config: dict) -> Callable[[float], Status]:
    green_max = float(config["green_max_pct"])
    yellow_max = float(config["yellow_max_pct"])

    def rule(value: float) -> Status:
        if not _finite(value):
            return Status.UNKNOWN
        value = float(value)
        if value <= green_max:
            return Status.GREEN
        if value <= yellow_max:
            return Status.YELLOW
        return Status.RED

    return rule


def voltage_rule(config: dict) -> Callable[[float], Status]:
    red_low = float(config["red_low_below_v"])
    green_low = float(config["green_low_v"])
    green_high = float(config["green_high_v"])
    red_high = float(config["red_high_above_v"])

    def rule(value: float) -> Status:
        if not _finite(value):
            return Status.UNKNOWN
        value = float(value)
        if value < red_low or value > red_high:
            return Status.RED
        if value < green_low or value > green_high:
            return Status.YELLOW
        return Status.GREEN

    return rule


def current_rule(config: dict) -> Callable[[float], Status]:
    green_abs = float(config["green_abs_max_a"])
    yellow_abs = float(config["yellow_abs_max_a"])

    def rule(value: float) -> Status:
        if not _finite(value):
            return Status.UNKNOWN
        value = abs(float(value))
        if value <= green_abs:
            return Status.GREEN
        if value <= yellow_abs:
            return Status.YELLOW
        return Status.RED

    return rule


def temperature_rule(config: dict) -> Callable[[float], Status]:
    red_low = float(config["red_low_below_c"])
    green_low = float(config["green_low_c"])
    green_high = float(config["green_high_c"])
    red_high = float(config["red_high_above_c"])

    def rule(value: float) -> Status:
        if not _finite(value):
            return Status.UNKNOWN
        value = float(value)
        if value < red_low or value > red_high:
            return Status.RED
        if value < green_low or value > green_high:
            return Status.YELLOW
        return Status.GREEN

    return rule
