from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from src.analysis.cu import get_r_dc_value
from src.models import CUAnalysisPoint, CUData


@dataclass
class FischerPredictionPoint:
    rpt_index: int
    source_file: str
    capacity_ah: float
    resistance_ohm: float
    usage: float | None
    soh_measured: float | None = None
    resistance_increase: float | None = None
    soh_predicted: float | None = None
    is_hampel_outlier: bool = False
    excluded_before_reference: bool = False


@dataclass
class FischerPredictionResult:
    ready: bool
    message: str
    feature_label: str
    points: list[FischerPredictionPoint] = field(default_factory=list)
    used_points: list[FischerPredictionPoint] = field(default_factory=list)
    beta1: float | None = None
    beta2: float | None = None
    reference_capacity_ah: float | None = None
    reference_resistance_ohm: float | None = None
    reference_rpt_index: int | None = None
    rmse_pct: float | None = None
    r2_adj: float | None = None
    pearson_r: float | None = None
    cv_rmse_mean_pct: float | None = None
    cv_rmse_std_pct: float | None = None
    current_soh_measured_pct: float | None = None
    current_soh_predicted_pct: float | None = None
    capacity_fade_pct: float | None = None
    usage_label: str | None = None
    usage_forecast: list[float] = field(default_factory=list)
    soh_forecast_pct: list[float] = field(default_factory=list)
    resistance_increase_forecast_pct: list[float] = field(default_factory=list)
    estimated_eol_usage: float | None = None
    eol_soh_pct: float | None = None
    notes: list[str] = field(default_factory=list)


def _finite_float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _power_law(r_incr: np.ndarray | float, beta1: float, beta2: float):
    r = np.maximum(np.asarray(r_incr, dtype=float), 0.0)
    return 1.0 - beta1 * np.power(r, beta2)


def _fit_power_law(r_incr: np.ndarray, soh: np.ndarray) -> tuple[float, float]:
    if len(r_incr) < 3:
        raise ValueError("Mindestens drei Punkte werden für den Power-Law-Fit benötigt.")

    positive = r_incr > 1e-12
    if np.any(positive):
        x_last = float(np.max(r_incr[positive]))
        y_last = float(soh[np.argmax(r_incr)])
        b1_guess = max((1.0 - y_last) / max(x_last, 1e-6), 0.1)
    else:
        b1_guess = 1.0

    popt, _ = curve_fit(
        _power_law,
        r_incr,
        soh,
        p0=[b1_guess, 1.0],
        bounds=([0.0, 0.05], [1000.0, 5.0]),
        maxfev=20000,
    )
    return float(popt[0]), float(popt[1])


def _hampel_outlier_mask(values: np.ndarray, half_window: int, threshold: float) -> np.ndarray:
    n = len(values)
    mask = np.zeros(n, dtype=bool)
    if n < 3 or half_window <= 0:
        return mask

    scale = 1.4826
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        window = values[lo:hi]
        finite = window[np.isfinite(window)]
        if len(finite) < 3:
            continue
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        scaled_mad = scale * mad
        if scaled_mad <= 0.0:
            continue
        if abs(float(values[i]) - median) > threshold * scaled_mad:
            mask[i] = True
    return mask


def _adjusted_r2(y: np.ndarray, y_hat: np.ndarray, n_params: int) -> float | None:
    n = len(y)
    if n <= n_params or n < 2:
        return None
    sse = float(np.sum((y - y_hat) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    if sst <= 0.0:
        return None
    r2 = 1.0 - sse / sst
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - n_params))


def _cross_validate(
    r_incr: np.ndarray,
    soh: np.ndarray,
    repeats: int,
    train_fraction: float,
    seed: int,
) -> tuple[float | None, float | None]:
    n = len(r_incr)
    if n < 5 or repeats <= 0:
        return None, None

    train_n = max(3, int(round(train_fraction * n)))
    train_n = min(train_n, n - 1)
    if train_n < 3 or n - train_n < 1:
        return None, None

    rng = np.random.default_rng(seed)
    rmses: list[float] = []
    indices = np.arange(n)

    for _ in range(repeats):
        perm = rng.permutation(indices)
        train_idx = perm[:train_n]
        val_idx = perm[train_n:]
        try:
            b1, b2 = _fit_power_law(r_incr[train_idx], soh[train_idx])
        except Exception:
            continue
        pred = _power_law(r_incr[val_idx], b1, b2)
        rmse_pct = 100.0 * float(np.sqrt(np.mean((soh[val_idx] - pred) ** 2)))
        if math.isfinite(rmse_pct):
            rmses.append(rmse_pct)

    if not rmses:
        return None, None
    return float(np.mean(rmses)), float(np.std(rmses, ddof=0))


def _usage_value(item: CUData, reference_capacity_ah: float | None) -> tuple[float | None, str | None]:
    efc = _finite_float_or_none(item.efc)
    if efc is not None and efc >= 0:
        return efc, "EFC"

    throughput = _finite_float_or_none(item.metadata.get("charge_throughput_ah"))
    if throughput is not None and throughput >= 0 and reference_capacity_ah and reference_capacity_ah > 0:
        return throughput / (2.0 * reference_capacity_ah), "EFC*"
    return None, None


def _build_usage_forecast(
    result: FischerPredictionResult,
    config: dict,
) -> None:
    if not bool(config.get("enable_usage_projection", True)):
        return
    if result.beta1 is None or result.beta2 is None or not result.used_points:
        return

    usage_pairs = [
        (p.usage, p.resistance_increase)
        for p in result.used_points
        if p.usage is not None and p.resistance_increase is not None
    ]
    if len(usage_pairs) < 3:
        result.notes.append("Keine Nutzungsprognose: weniger als drei gültige EFC-/Durchsatzpunkte.")
        return

    usage = np.asarray([float(x) for x, _ in usage_pairs], dtype=float)
    r_incr = np.asarray([float(y) for _, y in usage_pairs], dtype=float)
    order = np.argsort(usage)
    usage = usage[order]
    r_incr = r_incr[order]

    x0 = float(usage[0])
    dx = usage - x0
    denom = float(np.sum(dx * dx))
    if denom <= 0.0:
        return

    # Dashboard extension, deliberately simple and transparent: linear trend of
    # resistance increase versus usage, passed through the Fischer power-law map.
    slope = float(np.sum(dx * r_incr) / denom)
    if not math.isfinite(slope) or slope <= 0.0:
        result.notes.append("Keine EOL-Projektion: der Widerstandstrend über die Nutzung ist nicht positiv.")
        return

    eol_soh = float(config.get("eol_soh", 0.80))
    eol_soh = min(max(eol_soh, 0.05), 0.99)
    result.eol_soh_pct = 100.0 * eol_soh

    if result.beta1 <= 0.0 or result.beta2 <= 0.0:
        return
    target_r_incr = ((1.0 - eol_soh) / result.beta1) ** (1.0 / result.beta2)
    if not math.isfinite(target_r_incr) or target_r_incr <= 0.0:
        return

    estimated_eol_usage = x0 + target_r_incr / slope
    if not math.isfinite(estimated_eol_usage):
        return
    result.estimated_eol_usage = float(estimated_eol_usage)

    last_usage = float(usage[-1])
    forecast_factor = max(float(config.get("forecast_factor", 1.5)), 1.0)
    end_usage = max(estimated_eol_usage, last_usage + forecast_factor * max(last_usage - x0, 1.0))
    n_forecast = max(int(config.get("n_forecast", 120)), 20)
    x_forecast = np.linspace(x0, end_usage, n_forecast)
    r_forecast = np.maximum(slope * (x_forecast - x0), 0.0)
    soh_forecast = _power_law(r_forecast, result.beta1, result.beta2)

    result.usage_forecast = x_forecast.tolist()
    result.resistance_increase_forecast_pct = (100.0 * r_forecast).tolist()
    result.soh_forecast_pct = (100.0 * soh_forecast).tolist()
    result.notes.append(
        "Die EOL-Projektion ist eine Dashboard-Erweiterung: R_incr wird linear über die Nutzung extrapoliert und anschließend über die Fischer-Power-Law-Beziehung in SOH_C abgebildet."
    )


def analyze_fischer_prediction(
    cu_data: list[CUData],
    cu_points: list[CUAnalysisPoint],
    config: dict,
) -> FischerPredictionResult:
    soc = str(config.get("soc", "SoC50"))
    c_rate = str(config.get("c_rate", "1C"))
    mode = str(config.get("mode", "mean"))
    delay = str(config.get("delay", "10s"))
    feature_label = f"{soc} | {c_rate} | {mode} | {delay}"

    point_by_source = {point.source_file: point for point in cu_points}
    provisional: list[tuple[CUData, CUAnalysisPoint, float, float]] = []

    for item in cu_data:
        analysis_point = point_by_source.get(item.source_file)
        if analysis_point is None:
            continue
        capacity = _finite_float_or_none(analysis_point.discharge_capacity_ah)
        resistance = get_r_dc_value(item, soc, c_rate, mode, delay)
        resistance = _finite_float_or_none(resistance)
        if capacity is None or capacity <= 0 or resistance is None or resistance <= 0:
            continue
        provisional.append((item, analysis_point, capacity, resistance))

    result = FischerPredictionResult(
        ready=False,
        message="",
        feature_label=feature_label,
    )

    if not provisional:
        result.message = f"Keine gemeinsamen Kapazitäts- und Widerstandswerte für {feature_label} gefunden."
        return result

    # First pass: use the first valid capacity only to derive EFC* when native EFC is absent.
    capacity_for_usage = provisional[0][2]
    for item, analysis_point, capacity, resistance in provisional:
        usage, usage_label = _usage_value(item, capacity_for_usage)
        if result.usage_label is None and usage_label is not None:
            result.usage_label = usage_label
        result.points.append(
            FischerPredictionPoint(
                rpt_index=int(analysis_point.rpt_index),
                source_file=analysis_point.source_file,
                capacity_ah=float(capacity),
                resistance_ohm=float(resistance),
                usage=usage,
            )
        )

    max_resistance = float(config.get("max_resistance_ohm", 0.5))
    resistances = np.asarray([p.resistance_ohm for p in result.points], dtype=float)
    finite_mask = np.isfinite(resistances) & (resistances > 0.0) & (resistances <= max_resistance)

    half_window = int(config.get("hampel_half_window", 2))
    hampel_threshold = float(config.get("hampel_threshold", 2.0))
    hampel_mask = _hampel_outlier_mask(resistances, half_window, hampel_threshold)

    filtered: list[FischerPredictionPoint] = []
    for idx, p in enumerate(result.points):
        if not finite_mask[idx]:
            p.is_hampel_outlier = True
            continue
        if hampel_mask[idx]:
            p.is_hampel_outlier = True
            continue
        filtered.append(p)

    min_rpts = int(config.get("min_rpts", 5))
    if len(filtered) < min_rpts:
        result.message = (
            f"Nur {len(filtered)} gültige RPT-Paare vorhanden. Für die Fischer-orientierte Auswertung "
            f"werden mindestens {min_rpts} benötigt."
        )
        return result

    # Fischer preprocessing: ignore all earlier RPTs until the global resistance minimum.
    resistance_filtered = np.asarray([p.resistance_ohm for p in filtered], dtype=float)
    ref_idx = int(np.argmin(resistance_filtered))
    reference_point = filtered[ref_idx]
    for p in filtered[:ref_idx]:
        p.excluded_before_reference = True

    used = filtered[ref_idx:]
    if len(used) < 3:
        result.message = "Nach der Referenzierung am globalen Widerstandsminimum bleiben zu wenige Punkte übrig."
        return result

    c_ref = float(reference_point.capacity_ah)
    r_ref = float(reference_point.resistance_ohm)
    result.reference_capacity_ah = c_ref
    result.reference_resistance_ohm = r_ref
    result.reference_rpt_index = int(reference_point.rpt_index)

    # Recompute the usage axis consistently with the actual reference capacity.
    # Never mix native EFC and the throughput-based approximation in one trajectory.
    item_by_source = {item.source_file: item for item in cu_data}
    used_items = [item_by_source.get(p.source_file) for p in used]
    native_efc_values = [
        None if item is None else _finite_float_or_none(item.efc)
        for item in used_items
    ]
    use_native_efc = all(value is not None and value >= 0.0 for value in native_efc_values)
    result.usage_label = "EFC" if use_native_efc else "EFC*"

    for p, item, native_efc in zip(used, used_items, native_efc_values):
        p.soh_measured = float(p.capacity_ah / c_ref)
        p.resistance_increase = float(p.resistance_ohm / r_ref - 1.0)
        if item is None:
            p.usage = None
        elif use_native_efc:
            p.usage = float(native_efc)
        else:
            throughput = _finite_float_or_none(item.metadata.get("charge_throughput_ah"))
            p.usage = (
                None
                if throughput is None or throughput < 0.0
                else float(throughput / (2.0 * c_ref))
            )

    if not any(p.usage is not None for p in used):
        result.usage_label = None

    result.used_points = used
    soh = np.asarray([p.soh_measured for p in used], dtype=float)
    r_incr = np.asarray([max(float(p.resistance_increase), 0.0) for p in used], dtype=float)

    capacity_fade = max(0.0, 1.0 - float(np.min(soh)))
    result.capacity_fade_pct = 100.0 * capacity_fade
    min_capacity_fade = float(config.get("min_capacity_fade_fraction", 0.03))
    strict_fade = bool(config.get("require_min_capacity_fade", True))
    if strict_fade and capacity_fade < min_capacity_fade:
        result.message = (
            f"Kapazitätsabnahme nur {100.0 * capacity_fade:.2f} %. Die Fischer-Auswertung "
            f"setzt für die Korrelation mindestens {100.0 * min_capacity_fade:.1f} % voraus."
        )
        return result

    try:
        beta1, beta2 = _fit_power_law(r_incr, soh)
    except Exception as exc:
        result.message = f"Power-Law-Fit fehlgeschlagen: {exc}"
        return result

    pred = np.asarray(_power_law(r_incr, beta1, beta2), dtype=float)
    for p, y_hat in zip(used, pred):
        p.soh_predicted = float(y_hat)

    result.beta1 = beta1
    result.beta2 = beta2
    result.rmse_pct = 100.0 * float(np.sqrt(np.mean((soh - pred) ** 2)))
    result.r2_adj = _adjusted_r2(soh, pred, n_params=2)
    result.pearson_r = float(np.corrcoef(soh, r_incr)[0, 1]) if len(soh) >= 2 else None

    cv_mean, cv_std = _cross_validate(
        r_incr,
        soh,
        repeats=int(config.get("cv_repeats", 10)),
        train_fraction=float(config.get("cv_train_fraction", 0.70)),
        seed=int(config.get("cv_seed", 42)),
    )
    result.cv_rmse_mean_pct = cv_mean
    result.cv_rmse_std_pct = cv_std
    result.current_soh_measured_pct = 100.0 * float(soh[-1])
    result.current_soh_predicted_pct = 100.0 * float(pred[-1])
    result.ready = True
    result.message = "Fischer-Power-Law-Fit erfolgreich."

    if result.usage_label == "EFC*":
        result.notes.append(
            "EFC* wird aus kumuliertem Ladungsdurchsatz /(2·C_ref) angenähert, da in den CU-Metadaten kein nativer EFC-Wert vorliegt."
        )

    _build_usage_forecast(result, config)
    return result
