from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.models import CUData


@dataclass
class DMAStudyResult:
    library_available: bool
    results_by_cu: dict[str, Any] = field(default_factory=dict)
    study_results: Any | None = None
    skipped: dict[str, str] = field(default_factory=dict)
    message: str = ""
    pydma_version: str | None = None


def _load_pydma():
    try:
        import pydma  # type: ignore
    except ImportError as exc:
        return None, f"pyDMA konnte nicht importiert werden: {exc}"
    return pydma, ""


def pydma_status() -> tuple[bool, str | None, str]:
    pydma, message = _load_pydma()
    if pydma is None:
        return False, None, message
    version = getattr(pydma, "__version__", None)
    return True, None if version is None else str(version), ""


def _cu_label(item: CUData) -> str:
    if int(item.rpt_index or 0) > 0:
        return f"CU{int(item.rpt_index)}"
    source = Path(item.source_file).stem if item.source_file else "CU"
    return source


def _extract_pocv(item: CUData, direction: str) -> tuple[np.ndarray, np.ndarray] | None:
    key = f"pocv_{direction.casefold()}"
    pocv = item.metadata.get(key)
    if not isinstance(pocv, dict):
        return None

    capacity = pocv.get("capacity_ah")
    voltage = pocv.get("voltage_v")
    if not isinstance(capacity, list) or not isinstance(voltage, list):
        return None
    if len(capacity) == 0 or len(capacity) != len(voltage):
        return None

    try:
        q = np.asarray(capacity, dtype=float)
        u = np.asarray(voltage, dtype=float)
    except (TypeError, ValueError):
        return None

    valid = np.isfinite(q) & np.isfinite(u)
    q = q[valid]
    u = u[valid]
    if len(q) < 2:
        return None
    return q, u


def count_dma_inputs(cu_data: list[CUData], direction: str = "charge", min_points: int = 20) -> int:
    count = 0
    for item in cu_data:
        extracted = _extract_pocv(item, direction)
        if extracted is not None and len(extracted[0]) >= int(min_points):
            count += 1
    return count


def _make_dma_config(pydma, config: dict[str, Any]):
    return pydma.DMAConfig(
        direction=str(config.get("direction", "charge")),
        speed_preset=str(config.get("speed_preset", "fast")),
        max_tries_overall=int(config.get("max_tries_overall", 5)),
        weight_ocv=float(config.get("weight_ocv", 80.0)),
        weight_dva=float(config.get("weight_dva", 20.0)),
        weight_ica=float(config.get("weight_ica", 0.0)),
        roi_dva_min=float(config.get("roi_dva_min", 0.1)),
        roi_dva_max=float(config.get("roi_dva_max", 0.8)),
        roi_ocv_min=float(config.get("roi_ocv_min", 0.0)),
        roi_ocv_max=float(config.get("roi_ocv_max", 0.1)),
        allow_anode_inhomogeneity=bool(config.get("allow_anode_inhomogeneity", True)),
    )


def run_dma_study(
    cu_data: list[CUData],
    config: dict[str, Any],
    anode_ocp_path: str | Path,
    cathode_ocp_path: str | Path,
) -> DMAStudyResult:
    pydma, import_message = _load_pydma()
    if pydma is None:
        return DMAStudyResult(library_available=False, message=import_message)

    anode_path = Path(anode_ocp_path)
    cathode_path = Path(cathode_ocp_path)
    if not anode_path.is_file():
        return DMAStudyResult(
            library_available=True,
            message=f"Anoden-Halbzellkurve nicht gefunden: {anode_path}",
            pydma_version=str(getattr(pydma, "__version__", "")) or None,
        )
    if not cathode_path.is_file():
        return DMAStudyResult(
            library_available=True,
            message=f"Kathoden-Halbzellkurve nicht gefunden: {cathode_path}",
            pydma_version=str(getattr(pydma, "__version__", "")) or None,
        )

    try:
        anode_ocp = pydma.load_ocp(str(anode_path))
        cathode_ocp = pydma.load_ocp(str(cathode_path), electrode_type="cathode")
        analyzer = pydma.DMAAnalyzer(config=_make_dma_config(pydma, config))
        analyzer.set_anode(anode_ocp)
        analyzer.set_cathode(cathode_ocp)
        study_results = pydma.AgingStudyResults()
    except Exception as exc:
        return DMAStudyResult(
            library_available=True,
            message=f"pyDMA konnte nicht initialisiert werden: {exc}",
            pydma_version=str(getattr(pydma, "__version__", "")) or None,
        )

    direction = str(config.get("direction", "charge")).casefold()
    min_points = int(config.get("min_pocv_points", 20))
    results_by_cu: dict[str, Any] = {}
    skipped: dict[str, str] = {}

    ordered = sorted(
        cu_data,
        key=lambda item: (
            item.rpt_index if int(item.rpt_index or 0) > 0 else 10**9,
            item.source_file,
        ),
    )

    for item in ordered:
        label = _cu_label(item)
        extracted = _extract_pocv(item, direction)
        if extracted is None:
            skipped[label] = f"pOCV_{direction} nicht verfügbar oder ungültig"
            continue

        capacity, voltage = extracted
        if len(capacity) < min_points:
            skipped[label] = f"nur {len(capacity)} gültige pOCV-Punkte"
            continue

        try:
            result = analyzer.analyze(
                measured_capacity=capacity,
                measured_voltage=voltage,
            )
            try:
                result.cu_name = label
            except Exception:
                pass
            results_by_cu[label] = result
            study_results.add_result(result)
        except Exception as exc:
            skipped[label] = str(exc)

    message = ""
    if not results_by_cu:
        message = "Keine CU-Datei konnte mit pyDMA ausgewertet werden."

    return DMAStudyResult(
        library_available=True,
        results_by_cu=results_by_cu,
        study_results=study_results if results_by_cu else None,
        skipped=skipped,
        message=message,
        pydma_version=str(getattr(pydma, "__version__", "")) or None,
    )


def make_dma_result_figure(result: Any):
    pydma, message = _load_pydma()
    if pydma is None:
        raise RuntimeError(message)
    return pydma.plot_ocv_model_param_show(result)


def make_dma_study_figure(study_results: Any):
    pydma, message = _load_pydma()
    if pydma is None:
        raise RuntimeError(message)
    return pydma.plot_aging_study(study_results)
