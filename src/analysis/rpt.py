from __future__ import annotations

from src.analysis.cu import analyze_cu_record, assess_degradation
from src.analysis.drt import calculate_drt
from src.analysis.regression import linear_forecast
from src.models import CUData, GEISAnalysisPoint, GEISData, RPTAssessment


def analyze_rpt(cu_data: list[CUData], geis_data: list[GEISData], config: dict) -> RPTAssessment:
    cu_points = [analyze_cu_record(item, config.get("cu", {})) for item in cu_data]

    geis_points: list[GEISAnalysisPoint] = []
    for item in geis_data:
        drt = calculate_drt(item, config.get("geis", {}))
        r_inf = min(item.z_real_ohm) if item.z_real_ohm else float("nan")
        geis_points.append(
            GEISAnalysisPoint(
                battery_id=item.battery_id,
                cell_id=item.cell_id,
                rpt_index=item.rpt_index,
                efc=item.efc,
                r_inf_ohm=float(r_inf),
                drt=drt,
                source_file=item.source_file,
            )
        )

    capacity_reg = linear_forecast(
        [p.efc for p in cu_points],
        [p.discharge_capacity_ah for p in cu_points],
        **config.get("regression", {}),
    ) if len(cu_points) >= 2 else None

    r_points = [p for p in cu_points if p.pulse_resistance_ohm is not None]
    resistance_reg = linear_forecast(
        [p.efc for p in r_points],
        [float(p.pulse_resistance_ohm) for p in r_points],
        **config.get("regression", {}),
    ) if len(r_points) >= 2 else None

    drt_points = [p for p in geis_points if p.drt.integral_ohm is not None]
    drt_reg = linear_forecast(
        [p.efc for p in drt_points],
        [float(p.drt.integral_ohm) for p in drt_points],
        **config.get("regression", {}),
    ) if len(drt_points) >= 2 else None

    return RPTAssessment(
        cu_points=cu_points,
        geis_points=geis_points,
        capacity_regression=capacity_reg,
        resistance_regression=resistance_reg,
        drt_integral_regression=drt_reg,
        degradation=assess_degradation(cu_points),
    )
