from __future__ import annotations

import math
from dataclasses import dataclass

from src.models import AccelerationEvent
from src.status import Status, worst_status


@dataclass
class AccelerationHistoryResult:
    current_status: Status
    worst_status: Status
    series_statuses: list[Status]
    yellow_events: list[AccelerationEvent]
    red_sample_count: int
    reason: str


def _finite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def evaluate_acceleration_history(
    time_s: list[float], acceleration_g: list[float], config: dict
) -> AccelerationHistoryResult:
    """Evaluate acceleration while treating missing/NaN samples as UNKNOWN.

    UNKNOWN samples never count as green/yellow/red and do not re-arm the
    1-second hysteresis. Therefore a data gap cannot accidentally split one
    impact into multiple events or create a false green state.
    """
    green_max = float(config["green_max_g"])
    yellow_max = float(config["yellow_max_g"])
    limit = int(config["yellow_event_limit"])
    hysteresis_s = float(config["rearm_hysteresis_s"])

    statuses: list[Status] = []
    events: list[AccelerationEvent] = []
    red_samples = 0
    armed = True
    event_start: float | None = None
    event_max = 0.0
    green_start: float | None = None
    event_number = 0
    last_valid_t: float | None = None

    for t_raw, a_raw in zip(time_s, acceleration_g):
        if not _finite(t_raw) or not _finite(a_raw):
            statuses.append(Status.UNKNOWN)
            # A missing sample must not contribute to the continuous green
            # hysteresis period.
            green_start = None
            continue

        t = float(t_raw)
        a = abs(float(a_raw))
        last_valid_t = t

        if a > yellow_max:
            status = Status.RED
            red_samples += 1
        elif a > green_max:
            status = Status.YELLOW
        else:
            status = Status.GREEN
        statuses.append(status)

        if green_max < a <= yellow_max:
            green_start = None
            if armed:
                event_number += 1
                event_start = t
                event_max = a
                armed = False
            elif event_start is not None:
                event_max = max(event_max, a)

        elif a <= green_max:
            if not armed:
                if green_start is None:
                    green_start = t
                if t - green_start >= hysteresis_s:
                    if event_start is not None:
                        events.append(
                            AccelerationEvent(
                                index=event_number,
                                start_time_s=event_start,
                                end_time_s=green_start,
                                max_abs_g=event_max,
                            )
                        )
                    event_start = None
                    event_max = 0.0
                    armed = True
                    green_start = None
        else:
            # Direct red sample: do not re-arm a yellow event.
            green_start = None

    if event_start is not None:
        events.append(
            AccelerationEvent(
                index=event_number,
                start_time_s=event_start,
                end_time_s=last_valid_t if last_valid_t is not None else event_start,
                max_abs_g=event_max,
            )
        )

    historical = worst_status(statuses)
    reason = ""
    if historical == Status.UNKNOWN:
        reason = "Messgröße nicht verfügbar"
    elif len(events) >= limit:
        historical = Status.RED
        reason = f"{len(events)} gelbe Beschleunigungsereignisse; Grenzwert {limit} erreicht"
    elif red_samples:
        historical = Status.RED
        reason = "Direkte rote Beschleunigungsgrenze überschritten"
    elif events:
        historical = max(historical, Status.YELLOW)
        reason = f"{len(events)} gelbe Beschleunigungsereignisse"

    current = statuses[-1] if statuses else Status.UNKNOWN

    return AccelerationHistoryResult(
        current_status=current,
        worst_status=historical,
        series_statuses=statuses,
        yellow_events=events,
        red_sample_count=red_samples,
        reason=reason,
    )
