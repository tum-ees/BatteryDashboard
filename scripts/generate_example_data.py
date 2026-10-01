from __future__ import annotations

"""Generate the small synthetic data set shipped with the repository.

Only ``data/BAT_EXAMPLE`` is replaced. Other battery folders in ``data/`` are
never touched, so the script is safe to run next to user data.
"""

import json
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
BATTERY_ID = "BAT_EXAMPLE"
BATTERY_ROOT = DATA_ROOT / BATTERY_ID
RNG = np.random.default_rng(7)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def generate_aging() -> None:
    segment_duration_s = 1800
    dt = 1.0

    for segment_idx in range(3):
        t = np.arange(0.0, segment_duration_s, dt)
        absolute_t = segment_idx * segment_duration_s + t

        current = 35.0 * np.sin(2.0 * np.pi * absolute_t / 900.0)
        voltage = 26.0 + 1.2 * np.sin(2.0 * np.pi * absolute_t / 1400.0) + 0.008 * current
        temperature = (
            25.0
            + 8.0 * np.maximum(current, 0.0) / 35.0
            + 3.0 * np.maximum(-current, 0.0) / 35.0
        )
        humidity = 50.0 + 5.0 * np.sin(2.0 * np.pi * absolute_t / 2700.0)
        acceleration = 0.2 + 0.05 * RNG.normal(size=len(t))

        # One small yellow-range interval makes the example dashboard visibly
        # demonstrate the threshold/status logic without creating a red state.
        if segment_idx == 1:
            humidity[500:650] = 90.0
            current[900:930] = 72.0

        payload = {
            "battery_id": BATTERY_ID,
            "file_type": "Aging",
            "segment_index": segment_idx + 1,
            "start_time_s": float(segment_idx * segment_duration_s),
            "metadata": {
                "nominal_capacity_ah": 2.0,
                "current_positive_is_charge": True,
                "description": "Synthetic operating-data example.",
            },
            "signals": {
                "time_s": t.tolist(),
                "voltage_v": voltage.tolist(),
                "current_a": current.tolist(),
                "temperature_c": temperature.tolist(),
                "humidity_pct": humidity.tolist(),
                "acceleration_g": acceleration.tolist(),
            },
        }
        write_json(BATTERY_ROOT / "Aging" / f"Aging{segment_idx + 1}.json", payload)


def _rdc_block(resistance_ohm: float) -> dict:
    """Create a compact R_DC hierarchy compatible with the dashboard."""
    result: dict[str, dict] = {}
    for soc, soc_factor in (("SoC90", 1.08), ("SoC50", 1.00), ("SoC10", 1.15)):
        rates: dict[str, dict] = {}
        for c_rate, rate_factor in (("0.5C", 0.97), ("1C", 1.00)):
            r10 = resistance_ohm * soc_factor * rate_factor
            discharge = {
                "0.1s": 0.82 * r10,
                "1s": 0.90 * r10,
                "10s": r10,
            }
            charge = {
                "0.1s": 0.81 * r10,
                "1s": 0.89 * r10,
                "10s": 0.99 * r10,
            }
            mean = {
                key: 0.5 * (discharge[key] + charge[key])
                for key in discharge
            }
            rates[c_rate] = {
                "discharge": discharge,
                "charge": charge,
                "mean": mean,
            }
        result[soc] = rates
    return result


def generate_cu() -> None:
    nominal_capacity = 2.0
    efcs = [0, 200, 400, 600, 800]
    capacity_fade_per_rpt = 0.012
    resistance_growth_per_rpt = 0.06

    for rpt_index, efc in enumerate(efcs, start=1):
        age = rpt_index - 1
        target_capacity = nominal_capacity * (1.0 - capacity_fade_per_rpt * age)
        resistance = 0.020 * (1.0 + resistance_growth_per_rpt * age)

        discharge_current = -6.0
        discharge_duration = max(400, int(target_capacity * 3600.0 / abs(discharge_current)))
        total_time = 180 + discharge_duration + 60
        t = np.arange(0.0, total_time + 1.0, 1.0)
        current = np.zeros_like(t)

        # Two diagnostic pulses and one complete discharge.
        current[(t >= 60) & (t < 80)] = -2.0
        current[(t >= 120) & (t < 140)] = -4.0
        discharge_start = 180
        discharge_end = discharge_start + discharge_duration
        mask_dis = (t >= discharge_start) & (t < discharge_end)
        current[mask_dis] = discharge_current

        q_fraction = np.zeros_like(t, dtype=float)
        q_fraction[mask_dis] = (t[mask_dis] - discharge_start) / max(discharge_duration, 1)
        q_fraction[t >= discharge_end] = 1.0
        ocv = 4.15 - 1.15 * q_fraction - 0.12 * q_fraction**2
        voltage = ocv + current * resistance
        voltage += 0.001 * RNG.normal(size=len(t))

        payload = {
            "battery_id": BATTERY_ID,
            "cell_id": "CELL_01",
            "file_type": "CU",
            "rpt_index": rpt_index,
            "efc": float(efc),
            "metadata": {
                "nominal_capacity_ah": nominal_capacity,
                "current_positive_is_charge": True,
                "charge_throughput_ah": float(2.0 * nominal_capacity * efc),
                "r_dc": _rdc_block(resistance),
                "description": (
                    "Synthetic CU example with capacity discharge, diagnostic pulses "
                    "and pre-evaluated R_DC metadata."
                ),
            },
            "signals": {
                "time_s": t.tolist(),
                "voltage_v": voltage.tolist(),
                "current_a": current.tolist(),
            },
        }
        write_json(BATTERY_ROOT / "CU" / f"CU{rpt_index}.json", payload)


def generate_geis() -> None:
    frequencies = np.logspace(4, -2, 70)
    efcs = [0, 200, 400, 600, 800]
    capacity_fade_per_rpt = 0.012
    resistance_growth_per_rpt = 0.06

    for rpt_index, efc in enumerate(efcs, start=1):
        age = rpt_index - 1
        r0 = 0.008 * (1.0 + 0.25 * resistance_growth_per_rpt * age)
        r1 = 0.012 * (1.0 + resistance_growth_per_rpt * age)
        r2 = 0.007 * (1.0 + 0.5 * capacity_fade_per_rpt * age)
        tau1 = 0.010 * (1.0 + 0.08 * age)
        tau2 = 0.35 * (1.0 + 0.06 * age)
        inductance_h = 35e-9

        omega = 2.0 * np.pi * frequencies
        z = (
            r0
            + r1 / (1.0 + 1j * omega * tau1)
            + r2 / (1.0 + 1j * omega * tau2)
            + 1j * omega * inductance_h
        )
        z += RNG.normal(0.0, 2e-5, size=len(z)) + 1j * RNG.normal(0.0, 2e-5, size=len(z))

        payload = {
            "battery_id": BATTERY_ID,
            "cell_id": "CELL_01",
            "file_type": "GEIS",
            "rpt_index": rpt_index,
            "efc": float(efc),
            "metadata": {"temperature_c": 25.0, "soc": 0.5, "description": "Synthetic GEIS example."},
            "eis": {
                "frequency_hz": frequencies.tolist(),
                "z_real_ohm": z.real.tolist(),
                "z_imag_ohm": z.imag.tolist(),
            },
        }
        write_json(BATTERY_ROOT / "GEIS" / f"GEIS{rpt_index}.json", payload)


def main() -> None:
    if BATTERY_ROOT.exists():
        shutil.rmtree(BATTERY_ROOT)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    generate_aging()
    generate_cu()
    generate_geis()
    print(f"Example data written to {BATTERY_ROOT}")


if __name__ == "__main__":
    main()
