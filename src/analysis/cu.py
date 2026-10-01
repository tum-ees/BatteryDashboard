from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.models import CUAnalysisPoint, CUData, DegradationAssessment, PulseResistanceResult


def analyze_cu_record(data: CUData, config: dict) -> CUAnalysisPoint:
    """Extract CU capacity and one comparable DC pulse resistance.

    Pre-evaluated metadata are preferred when available. Missing/null values
    remain missing and never become numerical zero. Raw-signal calculations are
    only used as a fallback for CU files that actually contain time series.
    """
    capacity = calculate_discharge_capacity(data, config)
    pulses = calculate_pulse_resistances(data, config)
    valid_r = [
        p.resistance_ohm
        for p in pulses
        if p.resistance_ohm is not None
        and np.isfinite(p.resistance_ohm)
        and p.resistance_ohm >= 0
    ]
    pulse_r = float(np.median(valid_r)) if valid_r else None

    return CUAnalysisPoint(
        battery_id=data.battery_id,
        cell_id=data.cell_id,
        rpt_index=data.rpt_index,
        efc=data.efc,
        discharge_capacity_ah=capacity,  # None is intentionally allowed at runtime
        pulse_resistance_ohm=pulse_r,
        pulse_results=pulses,
        source_file=data.source_file,
    )


def calculate_discharge_capacity(data: CUData, config: dict | None = None) -> float | None:
    config = config or {}

    if data.metadata.get("source_format") == "precomputed_cu_metadata":
        value = _capacity_from_precomputed_metadata(data, config)
        if value is not None:
            return value
        # Metadata-only exports have no raw signal fallback.
        if not data.time_s:
            return None

    if data.metadata.get("source_format") == "basytec_cu":
        value = _calculate_basytec_capacity(data, config)
        if value is not None:
            return value

    return _calculate_generic_discharge_capacity(
        data,
        deadband_a=float(config.get("capacity_current_deadband_a", 0.05)),
    )


def _capacity_from_precomputed_metadata(data: CUData, config: dict) -> float | None:
    candidates = data.metadata.get("capacity_candidates")
    if not isinstance(candidates, dict):
        return None

    key_order = config.get(
        "capacity_metadata_keys",
        ["Capacity_CCCV_Ah", "Capacity_Ah"],
    )
    if not isinstance(key_order, list):
        key_order = ["Capacity_CCCV_Ah", "Capacity_Ah"]

    for key in key_order:
        value = _finite_float_or_none(candidates.get(str(key)))
        if value is not None and value >= 0:
            return value
    return None


def _calculate_basytec_capacity(data: CUData, config: dict) -> float | None:
    """Integrate the configured discharge step, e.g. label ``CC_DIS``.

    The example CU plan explicitly evaluates the final capacity discharge via the
    ``CC_DIS`` step.  Restricting the integration to that line avoids summing all
    discharge events/pulses of a complete RPT.
    """
    line_values = data.metadata.get("line")
    if not isinstance(line_values, list) or len(line_values) != len(data.time_s):
        return None

    label = str(config.get("capacity_discharge_label", "CC_DIS"))
    line_number = _line_number_for_label(data.metadata.get("test_table"), label)
    if line_number is None:
        return None

    lines = np.asarray(line_values, dtype=float)
    mask = np.isfinite(lines) & (lines.astype(int) == int(line_number))
    indices = np.flatnonzero(mask)
    if len(indices) < 2:
        return None

    t = np.asarray(data.time_s, dtype=float)[indices]
    current = np.asarray(data.current_a, dtype=float)[indices]
    finite = np.isfinite(t) & np.isfinite(current)
    t = t[finite]
    current = current[finite]
    if len(t) < 2:
        return None

    deadband_a = float(config.get("capacity_current_deadband_a", 0.05))
    if data.current_positive_is_charge:
        discharge_current = np.maximum(-current, 0.0)
    else:
        discharge_current = np.maximum(current, 0.0)
    discharge_current[discharge_current < deadband_a] = 0.0

    capacity_ah = float(np.trapezoid(discharge_current, t) / 3600.0)
    return capacity_ah if math.isfinite(capacity_ah) else None


def _calculate_generic_discharge_capacity(data: CUData, deadband_a: float = 0.05) -> float | None:
    t = np.asarray(data.time_s, dtype=float)
    current = np.asarray(data.current_a, dtype=float)
    if len(t) < 2 or len(current) != len(t):
        return None

    finite = np.isfinite(t) & np.isfinite(current)
    t = t[finite]
    current = current[finite]
    if len(t) < 2:
        return None

    if data.current_positive_is_charge:
        discharge_current = np.maximum(-current, 0.0)
    else:
        discharge_current = np.maximum(current, 0.0)
    discharge_current[discharge_current < deadband_a] = 0.0
    return float(np.trapezoid(discharge_current, t) / 3600.0)


def calculate_pulse_resistances(data: CUData, config: dict) -> list[PulseResistanceResult]:
    """Return one configured reference resistance for one CU.

    For pre-evaluated CU metadata the default reference is
    ``SoC50 / 0.5C / mean / 10s`` because that combination is present in both
    supplied example formats. The exact path is configurable. Missing entries
    return no resistance point instead of zero or an exception.
    """
    if data.metadata.get("source_format") == "precomputed_cu_metadata":
        result = _reference_resistance_from_metadata(data, config)
        return [] if result is None else [result]

    if data.metadata.get("source_format") == "basytec_cu":
        result = _calculate_basytec_reference_pulse(data, config)
        if result is not None:
            return [result]

    return _calculate_generic_pulse_resistances(data, config)


def _reference_resistance_from_metadata(
    data: CUData,
    config: dict,
) -> PulseResistanceResult | None:
    r_dc = data.metadata.get("r_dc")
    if not isinstance(r_dc, dict):
        return None

    soc = str(config.get("metadata_resistance_soc", "SoC50"))
    c_rate = str(config.get("metadata_resistance_c_rate", "0.5C"))
    mode = str(config.get("metadata_resistance_mode", "mean"))
    delay_key = str(config.get("metadata_resistance_delay", "10s"))

    rate_node = _nested_dict(r_dc, soc, c_rate)
    if rate_node is None:
        return None

    value = None
    mode_node = rate_node.get(mode)
    if isinstance(mode_node, dict):
        value = _finite_float_or_none(mode_node.get(delay_key))

    # If a stored mean is empty, recompute it only from the same SoC/C-rate/
    # delay using whichever charge/discharge values are actually available.
    # We deliberately do NOT switch to another SoC or C-rate because that would
    # break comparability across RPTs.
    if value is None and mode.casefold() == "mean":
        available: list[float] = []
        for branch in ("discharge", "charge"):
            node = rate_node.get(branch)
            if isinstance(node, dict):
                candidate = _finite_float_or_none(node.get(delay_key))
                if candidate is not None:
                    available.append(candidate)
        if available:
            value = float(np.mean(available))

    if value is None or value < 0:
        return None

    delay_s = _duration_key_to_seconds(delay_key)
    return PulseResistanceResult(
        start_time_s=float("nan"),
        delta_i_a=float("nan"),
        resistance_ohm=value,
        delay_s=delay_s,
    )


def get_r_dc_dimensions(
    records: list[CUData],
    *,
    soc: str | None = None,
    c_rate: str | None = None,
    mode: str | None = None,
) -> dict[str, list[str]]:
    """Return the available axes of the pre-evaluated ``MetaData.R_DC`` tree.

    The returned choices are the union across all supplied CU records.  Choices
    are filtered by the already selected parent axes, so the UI can build
    dependent selectors without offering combinations that occur nowhere in the
    available CU files.
    """
    socs: set[str] = set()
    c_rates: set[str] = set()
    modes: set[str] = set()
    delays: set[str] = set()

    for data in records:
        r_dc = data.metadata.get("r_dc")
        if not isinstance(r_dc, dict):
            continue

        for soc_key, soc_node in r_dc.items():
            if not isinstance(soc_node, dict):
                continue
            soc_key = str(soc_key)
            socs.add(soc_key)
            if soc is not None and soc_key != soc:
                continue

            for rate_key, rate_node in soc_node.items():
                if not isinstance(rate_node, dict):
                    continue
                rate_key = str(rate_key)
                c_rates.add(rate_key)
                if c_rate is not None and rate_key != c_rate:
                    continue

                for mode_key, mode_node in rate_node.items():
                    if not isinstance(mode_node, dict):
                        continue
                    mode_key = str(mode_key)
                    modes.add(mode_key)
                    if mode is not None and mode_key != mode:
                        continue
                    delays.update(str(key) for key in mode_node.keys())

                # A mean can be reconstructed from charge/discharge at the same
                # SoC, C-rate and delay when the stored mean is empty.  Include
                # those delay keys in the selector as well.
                if mode is not None and mode.casefold() == "mean":
                    for branch in ("discharge", "charge"):
                        branch_node = rate_node.get(branch)
                        if isinstance(branch_node, dict):
                            delays.update(str(key) for key in branch_node.keys())

    return {
        "socs": sorted(socs, key=_soc_sort_key),
        "c_rates": sorted(c_rates, key=_c_rate_sort_key),
        "modes": sorted(modes, key=_mode_sort_key),
        "delays": sorted(delays, key=_delay_sort_key),
    }


def get_r_dc_value(
    data: CUData,
    soc: str,
    c_rate: str,
    mode: str,
    delay_key: str,
) -> float | None:
    """Read one resistance from ``MetaData.R_DC``.

    ``None``, empty strings, NaN and missing branches stay missing.  The only
    fallback is for ``mean``: if the stored mean is unavailable, the mean is
    computed from whichever charge/discharge values exist at exactly the same
    SoC, C-rate and delay.  No cross-SoC or cross-C-rate substitution is done.
    """
    value, _ = _get_r_dc_value_and_source(data, soc, c_rate, mode, delay_key)
    return value


def flatten_r_dc_metadata(data: CUData) -> list[dict[str, Any]]:
    """Flatten the complete R_DC hierarchy of one CU into table rows.

    Empty entries are retained as ``None`` so they remain visible as missing in
    the dashboard.  For a missing stored ``mean`` with valid charge/discharge
    values, ``value_source`` is marked ``derived_mean``.
    """
    r_dc = data.metadata.get("r_dc")
    if not isinstance(r_dc, dict):
        return []

    rows: list[dict[str, Any]] = []
    for soc in sorted((str(k) for k in r_dc.keys()), key=_soc_sort_key):
        soc_node = r_dc.get(soc)
        if not isinstance(soc_node, dict):
            continue
        for c_rate in sorted((str(k) for k in soc_node.keys()), key=_c_rate_sort_key):
            rate_node = soc_node.get(c_rate)
            if not isinstance(rate_node, dict):
                continue
            for mode in sorted((str(k) for k in rate_node.keys()), key=_mode_sort_key):
                mode_node = rate_node.get(mode)
                if not isinstance(mode_node, dict):
                    continue
                for delay_key in sorted((str(k) for k in mode_node.keys()), key=_delay_sort_key):
                    value, value_source = _get_r_dc_value_and_source(
                        data, soc, c_rate, mode, delay_key
                    )
                    rows.append(
                        {
                            "soc": soc,
                            "c_rate": c_rate,
                            "mode": mode,
                            "delay": delay_key,
                            "resistance_ohm": value,
                            "value_source": value_source,
                        }
                    )
    return rows


def _get_r_dc_value_and_source(
    data: CUData,
    soc: str,
    c_rate: str,
    mode: str,
    delay_key: str,
) -> tuple[float | None, str]:
    r_dc = data.metadata.get("r_dc")
    if not isinstance(r_dc, dict):
        return None, "missing"

    rate_node = _nested_dict(r_dc, soc, c_rate)
    if rate_node is None:
        return None, "missing"

    mode_node = rate_node.get(mode)
    if isinstance(mode_node, dict):
        value = _finite_float_or_none(mode_node.get(delay_key))
        if value is not None and value >= 0:
            return value, "metadata"

    if mode.casefold() == "mean":
        available: list[float] = []
        for branch in ("discharge", "charge"):
            node = rate_node.get(branch)
            if not isinstance(node, dict):
                continue
            candidate = _finite_float_or_none(node.get(delay_key))
            if candidate is not None and candidate >= 0:
                available.append(candidate)
        if available:
            return float(np.mean(available)), "derived_mean"

    return None, "missing"


def _first_number(text: str) -> float:
    import re

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text))
    if match is None:
        return float("inf")
    try:
        return float(match.group(0))
    except ValueError:
        return float("inf")


def _soc_sort_key(value: str) -> tuple[float, str]:
    number = _first_number(value)
    # High SoC first: 90, 50, 10.
    return (-number if math.isfinite(number) else float("inf"), value)


def _c_rate_sort_key(value: str) -> tuple[float, str]:
    return (_first_number(value), value)


def _delay_sort_key(value: str) -> tuple[float, str]:
    seconds = _duration_key_to_seconds(value)
    return (seconds if math.isfinite(seconds) else float("inf"), value)


def _mode_sort_key(value: str) -> tuple[int, str]:
    order = {"discharge": 0, "charge": 1, "mean": 2}
    return (order.get(value.casefold(), 99), value)


def _nested_dict(root: dict, *keys: str) -> dict | None:
    node: Any = root
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node if isinstance(node, dict) else None


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _duration_key_to_seconds(value: str) -> float:
    text = value.strip().lower()
    try:
        if text.endswith("ms"):
            return float(text[:-2]) / 1000.0
        if text.endswith("s"):
            return float(text[:-1])
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def _calculate_basytec_reference_pulse(
    data: CUData,
    config: dict,
) -> PulseResistanceResult | None:
    line_values = data.metadata.get("line")
    r_values = data.metadata.get("r_ohm")
    dt_values = data.metadata.get("dt_pulse_s")
    di_values = data.metadata.get("delta_i_pulse_a")

    n = len(data.time_s)
    if not (
        isinstance(line_values, list)
        and isinstance(r_values, list)
        and isinstance(dt_values, list)
        and len(line_values) == n
        and len(r_values) == n
        and len(dt_values) == n
    ):
        return None

    label = str(
        config.get(
            "reference_pulse_label",
            "PULSE_DIS_MEDIUM_START_50",
        )
    )
    line_number = _line_number_for_label(data.metadata.get("test_table"), label)
    if line_number is None:
        return None

    delay_s = float(config.get("pulse_resistance_delay_s", 10.0))
    tolerance_s = float(config.get("pulse_delay_tolerance_s", 0.75))

    lines = np.asarray(line_values, dtype=float)
    r = np.asarray(r_values, dtype=float)
    dt = np.asarray(dt_values, dtype=float)
    time_s = np.asarray(data.time_s, dtype=float)

    candidate = (
        np.isfinite(lines)
        & (lines.astype(int) == int(line_number))
        & np.isfinite(r)
        & (r > 0.0)
        & np.isfinite(dt)
        & (dt > 0.0)
    )
    indices = np.flatnonzero(candidate)
    if len(indices) == 0:
        return None

    nearest_local = int(np.argmin(np.abs(dt[indices] - delay_s)))
    idx = int(indices[nearest_local])
    actual_delay = float(dt[idx])
    if abs(actual_delay - delay_s) > tolerance_s:
        return None

    delta_i = float("nan")
    if isinstance(di_values, list) and len(di_values) == n:
        try:
            delta_i = float(di_values[idx])
        except (TypeError, ValueError):
            pass
    if not math.isfinite(delta_i):
        delta_i = float(data.current_a[idx])

    start_time = float(time_s[idx] - actual_delay)
    return PulseResistanceResult(
        start_time_s=start_time,
        delta_i_a=delta_i,
        resistance_ohm=float(r[idx]),
        delay_s=actual_delay,
    )


def _calculate_generic_pulse_resistances(data: CUData, config: dict) -> list[PulseResistanceResult]:
    t = np.asarray(data.time_s, dtype=float)
    i = np.asarray(data.current_a, dtype=float)
    u = np.asarray(data.voltage_v, dtype=float)
    if len(t) < 3 or len(i) != len(t) or len(u) != len(t):
        return []

    delta_i_min = float(config.get("pulse_delta_i_min_a", 1.0))
    delay_s = float(config.get("pulse_resistance_delay_s", 10.0))
    pre_window_s = float(config.get("pulse_pre_window_s", 5.0))
    min_pulse_duration_s = float(config.get("min_pulse_duration_s", delay_s))

    di = np.diff(i, prepend=i[0])
    starts = np.where(np.abs(di) >= delta_i_min)[0]
    results: list[PulseResistanceResult] = []

    for idx in starts:
        if idx == 0:
            continue
        i_before = i[idx - 1]
        i_after = i[idx]
        delta_i = i_after - i_before
        if abs(delta_i) < delta_i_min:
            continue
        if abs(i_after) <= abs(i_before) + 0.5 * delta_i_min:
            continue

        target_t = t[idx] + delay_s
        target_idx = int(np.searchsorted(t, target_t, side="left"))
        if target_idx >= len(t):
            continue
        segment = i[idx : target_idx + 1]
        if t[target_idx] - t[idx] < min_pulse_duration_s:
            continue
        tolerance = max(0.05 * abs(i_after), 0.2)
        if np.any(np.abs(segment - i_after) > tolerance):
            continue

        pre_start_t = t[idx] - pre_window_s
        pre_start_idx = int(np.searchsorted(t, pre_start_t, side="left"))
        pre_slice = slice(pre_start_idx, idx)
        if idx - pre_start_idx < 1:
            continue
        v0 = float(np.mean(u[pre_slice]))
        i0 = float(np.mean(i[pre_slice]))
        v1 = float(u[target_idx])
        i1 = float(np.mean(i[max(idx, target_idx - 1) : target_idx + 1]))
        delta = i1 - i0
        if abs(delta) < delta_i_min:
            continue
        resistance = abs((v1 - v0) / delta)
        results.append(
            PulseResistanceResult(
                start_time_s=float(t[idx]),
                delta_i_a=float(delta),
                resistance_ohm=float(resistance),
                delay_s=float(t[target_idx] - t[idx]),
            )
        )

    return results


def _line_number_for_label(test_table: Any, label: str) -> int | None:
    if not isinstance(test_table, list):
        return None
    target = label.strip().casefold()
    for row in test_table:
        if not isinstance(row, dict):
            continue
        row_label = str(row.get("Label", "")).strip().casefold()
        if row_label != target:
            continue
        try:
            return int(row.get("Line"))
        except (TypeError, ValueError):
            return None
    return None


# Kept for API compatibility. It is not used in the CU-only dashboard stage.
def assess_degradation(points: list[CUAnalysisPoint]) -> DegradationAssessment | None:
    return None
