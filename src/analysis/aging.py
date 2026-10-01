from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable

from src.analysis.acceleration import evaluate_acceleration_history
from src.analysis.efc import calculate_equivalent_full_cycles
from src.analysis.rules import current_rule, humidity_rule, temperature_rule, voltage_rule
from src.models import AgingAssessment, AgingData, AgingSegment, MeasurementAssessment
from src.status import Status, worst_status


MEASUREMENT_META = {
    "voltage": ("Spannung", "V"),
    "current": ("Strom", "A"),
    "temperature": ("Temperatur", "°C"),
    "humidity": ("Luftfeuchtigkeit", "%"),
    "acceleration": ("Beschleunigung", "g"),
}


def _finite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def merge_aging_segments(segments: list[AgingSegment]) -> AgingData:
    """Merge Ageing files with block operations instead of sample-wise appends.

    ``local_time_s`` is expected to be monotonic inside each file. Overlapping
    samples at file boundaries are removed with ``bisect_right`` and all
    remaining signal slices are appended in blocks. This is substantially
    faster for long operating histories than a Python loop over every sample.
    """
    if not segments:
        raise ValueError("No Aging segments available")
    segments = sorted(segments, key=lambda s: s.start_time_s)

    t_all: list[float] = []
    u_all: list[float] = []
    i_all: list[float] = []
    temp_all: list[float] = []
    hum_all: list[float] = []
    acc_all: list[float] = []
    files: list[str] = []
    last_time: float | None = None

    available_union: set[str] = set()

    for segment in segments:
        source_file = str(segment.metadata.get("source_file", ""))
        if source_file:
            files.append(source_file)
        available_union.update(segment.metadata.get("available_signals", ()))

        local = segment.local_time_s
        if not local:
            continue

        # Find the first sample strictly newer than the already merged history.
        start_idx = 0
        if last_time is not None:
            local_cutoff = last_time - float(segment.start_time_s)
            start_idx = bisect_right(local, local_cutoff)
            if start_idx >= len(local):
                continue

        offset = float(segment.start_time_s)
        # Only the time axis needs arithmetic; measurement arrays can be sliced
        # and extended directly in C-level list operations.
        t_chunk = [offset + float(t) for t in local[start_idx:]]
        t_all.extend(t_chunk)
        u_all.extend(segment.voltage_v[start_idx:])
        i_all.extend(segment.current_a[start_idx:])
        temp_all.extend(segment.temperature_c[start_idx:])
        hum_all.extend(segment.humidity_pct[start_idx:])
        acc_all.extend(segment.acceleration_g[start_idx:])
        last_time = t_chunk[-1]

    # The datasource already knows which signals were physically present.
    # Avoid scanning millions of values again just to determine availability.
    globally_available = tuple(
        key
        for key in MEASUREMENT_META
        if key in available_union
    )
    globally_missing = tuple(
        key
        for key in MEASUREMENT_META
        if key not in available_union
    )

    first = segments[0]
    return AgingData(
        battery_id=first.battery_id,
        time_s=t_all,
        voltage_v=u_all,
        current_a=i_all,
        temperature_c=temp_all,
        humidity_pct=hum_all,
        acceleration_g=acc_all,
        nominal_capacity_ah=first.nominal_capacity_ah,
        current_positive_is_charge=first.current_positive_is_charge,
        source_files=files,
        metadata={
            "segment_count": len(segments),
            "available_signals": globally_available,
            "missing_signals": globally_missing,
        },
    )


def analyze_aging(data: AgingData, thresholds: dict) -> AgingAssessment:
    """Analyze only sensors that are actually present in the dataset."""
    measurements: dict[str, MeasurementAssessment] = {}
    available = set(data.metadata.get("available_signals", ()))
    n = len(data.time_s)

    measurements["voltage"] = (
        _assess_simple("voltage", data.voltage_v, voltage_rule(thresholds["voltage"]))
        if "voltage" in available
        else _unknown_measurement("voltage", n)
    )
    measurements["current"] = (
        _assess_simple("current", data.current_a, current_rule(thresholds["current"]))
        if "current" in available
        else _unknown_measurement("current", n)
    )
    measurements["humidity"] = (
        _assess_simple("humidity", data.humidity_pct, humidity_rule(thresholds["humidity"]))
        if "humidity" in available
        else _unknown_measurement("humidity", n)
    )
    measurements["temperature"] = (
        _assess_temperature(data, thresholds["temperature"])
        if "temperature" in available
        else _unknown_measurement("temperature", n)
    )
    measurements["acceleration"] = (
        _assess_acceleration(data, thresholds["acceleration"])
        if "acceleration" in available
        else _unknown_acceleration_measurement(n, thresholds["acceleration"])
    )

    overall = worst_status(m.worst_status for m in measurements.values())
    reasons = [
        f"{m.label}: {m.reason or m.worst_status.label}"
        for m in measurements.values()
        if m.worst_status == overall and overall not in (Status.GREEN, Status.UNKNOWN)
    ]

    efc = calculate_equivalent_full_cycles(
        data.time_s, data.current_a, data.nominal_capacity_ah
    )
    return AgingAssessment(overall, measurements, reasons, efc)


def _unknown_measurement(
    key: str,
    n: int,
    extra_details: dict | None = None,
) -> MeasurementAssessment:
    """Create an UNKNOWN assessment without allocating a per-sample status list.

    Missing sensors are not plotted and do not contribute to the overall traffic
    light.  Therefore a giant ``[Status.UNKNOWN] * n`` array would only waste
    memory for long operating histories.
    """
    label, unit = MEASUREMENT_META[key]
    details = {
        "available": False,
        "valid_sample_count": 0,
        "missing_sample_count": n,
    }
    if extra_details:
        details.update(extra_details)

    return MeasurementAssessment(
        key=key,
        label=label,
        unit=unit,
        current_value=None,
        minimum=None,
        maximum=None,
        current_status=Status.UNKNOWN,
        worst_status=Status.UNKNOWN,
        series_statuses=[],
        yellow_sample_count=0,
        red_sample_count=0,
        reason="Messgröße nicht verfügbar",
        details=details,
    )


def _unknown_acceleration_measurement(
    n: int,
    config: dict,
) -> MeasurementAssessment:
    """UNKNOWN acceleration result with the same details contract as a real result."""
    return _unknown_measurement(
        "acceleration",
        n,
        {
            "yellow_event_count": None,
            "yellow_events": [],
            "yellow_event_limit": int(config["yellow_event_limit"]),
            "hysteresis_s": float(config["rearm_hysteresis_s"]),
        },
    )


def infer_operation_modes(data: AgingData, config: dict) -> list[str | None]:
    """Infer charge/discharge mode; None means current is unavailable.

    A real zero-current/rest sample still uses the configured previous-mode
    fallback. A NaN/missing current sample is different: the operating mode is
    genuinely unknown and is therefore represented by None.
    """
    deadband = float(config.get("current_deadband_a", 0.5))
    last_active = str(config.get("rest_fallback_mode", "discharge"))
    modes: list[str | None] = []

    for current in data.current_a:
        if not _finite(current):
            modes.append(None)
            continue

        current = float(current)
        if abs(current) <= deadband:
            modes.append(last_active)
            continue

        positive_is_charge = data.current_positive_is_charge
        if (current > 0 and positive_is_charge) or (
            current < 0 and not positive_is_charge
        ):
            last_active = "charge"
        else:
            last_active = "discharge"
        modes.append(last_active)

    return modes


def _assess_temperature(data: AgingData, config: dict) -> MeasurementAssessment:
    modes = infer_operation_modes(data, config)
    charge = temperature_rule(config["charge"])
    discharge = temperature_rule(config["discharge"])

    statuses: list[Status] = []
    conservative_mode_samples = 0

    for value, mode in zip(data.temperature_c, modes):
        if not _finite(value):
            statuses.append(Status.UNKNOWN)
            continue

        if mode == "charge":
            statuses.append(charge(value))
        elif mode == "discharge":
            statuses.append(discharge(value))
        else:
            # Temperature exists but current does not. Avoid pretending that we
            # know whether charge or discharge limits apply: use the more
            # restrictive result of both rules for this sample.
            statuses.append(worst_status((charge(value), discharge(value))))
            conservative_mode_samples += 1

    worst = worst_status(statuses)
    reason = ""
    if worst == Status.UNKNOWN:
        reason = "Messgröße nicht verfügbar"
    else:
        modes_worst = sorted(
            {
                mode
                for mode, status in zip(modes, statuses)
                if mode is not None and status == worst and worst != Status.GREEN
            }
        )
        if modes_worst:
            reason = "Grenzverletzung im Modus " + "/".join(modes_worst)
        if conservative_mode_samples:
            suffix = (
                f"Betriebsmodus bei {conservative_mode_samples} Samples unbekannt; "
                "konservativ strengere Lade-/Entladegrenze verwendet"
            )
            reason = f"{reason}; {suffix}" if reason else suffix

    return _measurement(
        "temperature",
        data.temperature_c,
        statuses,
        reason,
        {
            "operation_modes": modes,
            "conservative_mode_samples": conservative_mode_samples,
        },
    )


def _assess_acceleration(data: AgingData, config: dict) -> MeasurementAssessment:
    result = evaluate_acceleration_history(data.time_s, data.acceleration_g, config)
    item = _measurement(
        "acceleration",
        data.acceleration_g,
        result.series_statuses,
        result.reason,
    )
    item.worst_status = result.worst_status
    item.current_status = result.current_status
    item.red_sample_count = result.red_sample_count
    item.details = {
        "yellow_event_count": len(result.yellow_events),
        "yellow_events": result.yellow_events,
        "yellow_event_limit": int(config["yellow_event_limit"]),
        "hysteresis_s": float(config["rearm_hysteresis_s"]),
        "available": result.worst_status != Status.UNKNOWN,
    }
    return item


def _assess_simple(
    key: str, values: list[float], rule: Callable[[float], Status]
) -> MeasurementAssessment:
    statuses = [rule(value) for value in values]
    worst = worst_status(statuses)
    reason = ""
    if worst == Status.RED:
        reason = "Mindestens eine rote Grenzwertverletzung"
    elif worst == Status.YELLOW:
        reason = "Mindestens eine gelbe Grenzwertverletzung"
    elif worst == Status.UNKNOWN:
        reason = "Messgröße nicht verfügbar"
    return _measurement(key, values, statuses, reason)


def _measurement(
    key: str,
    values: list[float],
    statuses: list[Status],
    reason: str = "",
    details: dict | None = None,
) -> MeasurementAssessment:
    label, unit = MEASUREMENT_META[key]
    finite_values = [float(value) for value in values if _finite(value)]

    current_value = None
    if values and _finite(values[-1]):
        current_value = float(values[-1])

    worst = worst_status(statuses)
    if worst == Status.UNKNOWN and not reason:
        reason = "Messgröße nicht verfügbar"

    merged_details = dict(details or {})
    merged_details.setdefault("available", bool(finite_values))
    merged_details.setdefault("valid_sample_count", len(finite_values))
    merged_details.setdefault("missing_sample_count", len(values) - len(finite_values))

    return MeasurementAssessment(
        key=key,
        label=label,
        unit=unit,
        current_value=current_value,
        minimum=min(finite_values) if finite_values else None,
        maximum=max(finite_values) if finite_values else None,
        current_status=statuses[-1] if statuses else Status.UNKNOWN,
        worst_status=worst,
        series_statuses=statuses,
        yellow_sample_count=sum(s == Status.YELLOW for s in statuses),
        red_sample_count=sum(s == Status.RED for s in statuses),
        reason=reason,
        details=merged_details,
    )
