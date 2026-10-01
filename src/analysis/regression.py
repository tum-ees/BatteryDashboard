from __future__ import annotations

import numpy as np

from src.models import RegressionResult


def linear_forecast(
    x: list[float],
    y: list[float],
    forecast_fraction: float = 0.25,
    min_forecast_span: float = 100.0,
    n_forecast: int = 40,
) -> RegressionResult | None:
    pairs = [(float(xi), float(yi)) for xi, yi in zip(x, y) if np.isfinite(xi) and np.isfinite(yi)]
    if len(pairs) < 2:
        return None
    x_arr = np.asarray([p[0] for p in pairs], dtype=float)
    y_arr = np.asarray([p[1] for p in pairs], dtype=float)

    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]

    slope, intercept = np.polyfit(x_arr, y_arr, 1)
    y_fit = slope * x_arr + intercept
    ss_res = float(np.sum((y_arr - y_fit) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    span = max(float(x_arr[-1] - x_arr[0]), min_forecast_span)
    x_end = float(x_arr[-1] + forecast_fraction * span)
    x_forecast = np.linspace(float(x_arr[0]), x_end, n_forecast)
    y_forecast = slope * x_forecast + intercept

    return RegressionResult(
        x=x_arr.tolist(),
        y=y_arr.tolist(),
        x_forecast=x_forecast.tolist(),
        y_forecast=y_forecast.tolist(),
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r2),
    )
