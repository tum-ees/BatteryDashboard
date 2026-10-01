from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.status import Status


@dataclass(frozen=True)
class BatteryInfo:
    battery_id: str
    aging_files: int
    cu_files: int
    geis_files: int
    cell_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DataFileInfo:
    path: Path
    battery_id: str
    file_type: str
    cell_id: str | None = None
    rpt_index: int | None = None
    efc: float | None = None
    start_time_s: float | None = None


@dataclass
class AgingSegment:
    battery_id: str
    start_time_s: float
    local_time_s: list[float]
    voltage_v: list[float]
    current_a: list[float]
    temperature_c: list[float]
    humidity_pct: list[float]
    acceleration_g: list[float]
    nominal_capacity_ah: float
    current_positive_is_charge: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgingData:
    battery_id: str
    time_s: list[float]
    voltage_v: list[float]
    current_a: list[float]
    temperature_c: list[float]
    humidity_pct: list[float]
    acceleration_g: list[float]
    nominal_capacity_ah: float
    current_positive_is_charge: bool = True
    source_files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CUData:
    battery_id: str
    cell_id: str
    rpt_index: int
    efc: float
    time_s: list[float]
    voltage_v: list[float]
    current_a: list[float]
    nominal_capacity_ah: float
    current_positive_is_charge: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""


@dataclass
class GEISData:
    battery_id: str
    cell_id: str
    rpt_index: int
    efc: float
    frequency_hz: list[float]
    z_real_ohm: list[float]
    z_imag_ohm: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""


@dataclass
class AccelerationEvent:
    index: int
    start_time_s: float
    end_time_s: float
    max_abs_g: float


@dataclass
class MeasurementAssessment:
    key: str
    label: str
    unit: str
    current_value: float | None
    minimum: float | None
    maximum: float | None
    current_status: Status
    worst_status: Status
    series_statuses: list[Status]
    yellow_sample_count: int = 0
    red_sample_count: int = 0
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgingAssessment:
    overall_status: Status
    measurements: dict[str, MeasurementAssessment]
    reasons: list[str]
    equivalent_full_cycles: list[float]


@dataclass
class PulseResistanceResult:
    start_time_s: float
    delta_i_a: float
    resistance_ohm: float
    delay_s: float


@dataclass
class CUAnalysisPoint:
    battery_id: str
    cell_id: str
    rpt_index: int
    efc: float
    discharge_capacity_ah: float
    pulse_resistance_ohm: float | None
    pulse_results: list[PulseResistanceResult]
    source_file: str


@dataclass
class DRTResult:
    tau_s: list[float]
    frequency_hz: list[float]
    gamma_ohm: list[float]
    peak_frequency_hz: float | None
    peak_gamma_ohm: float | None
    integral_ohm: float | None
    lambda_reg: float | None = None
    inductance_h: float | None = None
    r_inf_ohm: float | None = None
    measurement_frequency_hz: list[float] = field(default_factory=list)
    reconstructed_real_ohm: list[float] = field(default_factory=list)
    reconstructed_imag_ohm: list[float] = field(default_factory=list)
    rmse_real_ohm: float | None = None
    rmse_imag_ohm: float | None = None
    peak_frequencies_hz: list[float] = field(default_factory=list)
    peak_gammas_ohm: list[float] = field(default_factory=list)
    message: str = ""


@dataclass
class GEISAnalysisPoint:
    battery_id: str
    cell_id: str
    rpt_index: int
    efc: float
    r_inf_ohm: float
    drt: DRTResult
    source_file: str


@dataclass
class RegressionResult:
    x: list[float]
    y: list[float]
    x_forecast: list[float]
    y_forecast: list[float]
    slope: float | None
    intercept: float | None
    r_squared: float | None


@dataclass
class DegradationAssessment:
    label: str
    capacity_loss_pct: float | None
    resistance_growth_pct: float | None
    note: str


@dataclass
class RPTAssessment:
    cu_points: list[CUAnalysisPoint]
    geis_points: list[GEISAnalysisPoint]
    capacity_regression: RegressionResult | None
    resistance_regression: RegressionResult | None
    drt_integral_regression: RegressionResult | None
    degradation: DegradationAssessment | None
