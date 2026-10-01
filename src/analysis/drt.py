from __future__ import annotations

import math

import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import find_peaks

from src.models import DRTResult, GEISData


def _empty_result(message: str) -> DRTResult:
    return DRTResult(
        tau_s=[],
        frequency_hz=[],
        gamma_ohm=[],
        peak_frequency_hz=None,
        peak_gamma_ohm=None,
        integral_ohm=None,
        message=message,
    )


def _fit_inductance(
    frequency_hz: np.ndarray,
    z_imag_ohm: np.ndarray,
    n_points: int,
) -> float:
    """Estimate series inductance from the highest-frequency points.

    A straight line ``Im(Z) = omega * L + c`` is fitted.  The intercept absorbs
    small non-inductive contributions; only the non-negative slope is kept as L.
    """
    if len(frequency_hz) < 3:
        return 0.0

    n = max(3, min(int(n_points), len(frequency_hz)))
    omega = 2.0 * np.pi * frequency_hz[:n]
    y = z_imag_ohm[:n]
    valid = np.isfinite(omega) & np.isfinite(y)
    if np.count_nonzero(valid) < 3:
        return 0.0

    a = np.column_stack([omega[valid], np.ones(np.count_nonzero(valid))])
    slope, _intercept = np.linalg.lstsq(a, y[valid], rcond=None)[0]
    if not math.isfinite(float(slope)):
        return 0.0
    return max(float(slope), 0.0)


def _second_difference_matrix(n: int) -> np.ndarray:
    if n < 3:
        return np.empty((0, n), dtype=float)
    d2 = np.zeros((n - 2, n), dtype=float)
    for idx in range(n - 2):
        d2[idx, idx : idx + 3] = (1.0, -2.0, 1.0)
    return d2


def _solve_regularized_nnls(
    a_data: np.ndarray,
    b_data: np.ndarray,
    d_full: np.ndarray,
    lambda_reg: float,
) -> tuple[np.ndarray, float, float]:
    if d_full.size:
        a = np.vstack([a_data, np.sqrt(lambda_reg) * d_full])
        b = np.concatenate([b_data, np.zeros(d_full.shape[0])])
    else:
        a, b = a_data, b_data

    solution = lsq_linear(
        a,
        b,
        bounds=(0.0, np.inf),
        method="trf",
        lsmr_tol="auto",
        max_iter=500,
    )
    x = np.maximum(solution.x, 0.0)
    residual_norm = float(np.linalg.norm(a_data @ x - b_data))
    roughness_norm = float(np.linalg.norm(d_full @ x)) if d_full.size else 0.0
    return x, residual_norm, roughness_norm


def _choose_lambda_lcurve(
    a_data: np.ndarray,
    b_data: np.ndarray,
    d_full: np.ndarray,
    config: dict,
) -> tuple[float, np.ndarray]:
    n_candidates = max(7, int(config.get("lambda_candidates", 17)))
    lambda_min = float(config.get("lambda_min", 1e-8))
    lambda_max = float(config.get("lambda_max", 1e2))
    if lambda_min <= 0 or lambda_max <= lambda_min:
        lambda_min, lambda_max = 1e-8, 1e2

    candidates = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_candidates)
    solutions: list[np.ndarray] = []
    residuals: list[float] = []
    roughness: list[float] = []

    for value in candidates:
        x, res, rough = _solve_regularized_nnls(a_data, b_data, d_full, float(value))
        solutions.append(x)
        residuals.append(max(res, np.finfo(float).tiny))
        roughness.append(max(rough, np.finfo(float).tiny))

    # Curvature of the L-curve in log(residual)-log(roughness) space.
    t = np.log(candidates)
    x_curve = np.log(np.asarray(residuals))
    y_curve = np.log(np.asarray(roughness))
    dx = np.gradient(x_curve, t)
    dy = np.gradient(y_curve, t)
    ddx = np.gradient(dx, t)
    ddy = np.gradient(dy, t)
    denominator = np.power(dx * dx + dy * dy, 1.5)
    curvature = np.divide(
        np.abs(dx * ddy - dy * ddx),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    # End points are numerically unreliable; select from the interior where possible.
    if len(candidates) >= 7:
        interior = curvature[2:-2]
        chosen_idx = 2 + int(np.nanargmax(interior))
    else:
        chosen_idx = int(np.nanargmax(curvature))

    return float(candidates[chosen_idx]), solutions[chosen_idx]


def calculate_drt(data: GEISData, config: dict) -> DRTResult:
    """Calculate a non-negative DRT from one GEIS spectrum.

    Method
    ------
    * invalid samples are removed and frequencies sorted high -> low
    * a high-frequency series inductance is fitted and subtracted from Im(Z)
    * real and corrected imaginary parts are fitted jointly
    * gamma >= 0 is enforced
    * second-order Tikhonov regularization smooths the DRT
    * lambda can be selected automatically from the L-curve
    * the original impedance is reconstructed for quality control
    """
    f = np.asarray(data.frequency_hz, dtype=float)
    z_re = np.asarray(data.z_real_ohm, dtype=float)
    z_im = np.asarray(data.z_imag_ohm, dtype=float)

    valid = np.isfinite(f) & np.isfinite(z_re) & np.isfinite(z_im) & (f > 0.0)
    f, z_re, z_im = f[valid], z_re[valid], z_im[valid]
    min_points = int(config.get("min_points", 20))
    if len(f) < min_points:
        return _empty_result(
            f"Zu wenige GEIS-Punkte für eine robuste DRT ({len(f)} < {min_points})."
        )

    order = np.argsort(f)[::-1]
    f, z_re, z_im = f[order], z_re[order], z_im[order]
    omega = 2.0 * np.pi * f

    inductance_h = _fit_inductance(
        f,
        z_im,
        int(config.get("inductance_fit_points", 8)),
    )
    z_im_corr = z_im - omega * inductance_h

    n_tau = max(30, int(config.get("drt_points", 120)))
    tau_min_factor = float(config.get("tau_min_factor", 0.5))
    tau_max_factor = float(config.get("tau_max_factor", 2.0))
    if tau_min_factor <= 0:
        tau_min_factor = 0.5
    if tau_max_factor <= 0:
        tau_max_factor = 2.0

    tau_min = tau_min_factor / (2.0 * np.pi * np.max(f))
    tau_max = tau_max_factor / (2.0 * np.pi * np.min(f))
    if not math.isfinite(tau_min) or not math.isfinite(tau_max) or tau_max <= tau_min:
        return _empty_result("Ungültiger Relaxationszeitbereich.")

    tau = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)
    ln_tau = np.log(tau)
    dln = float(np.mean(np.diff(ln_tau)))

    wt = omega[:, None] * tau[None, :]
    kernel_re = (1.0 / (1.0 + wt**2)) * dln
    kernel_im = (wt / (1.0 + wt**2)) * dln

    # Unknown vector x = [R_inf, gamma_1, ..., gamma_N].
    # After removing L: Re(Z) = R_inf + K_re gamma
    #                   -Im(Z_corr) = K_im gamma
    a_data = np.block(
        [
            [np.ones((len(f), 1)), kernel_re],
            [np.zeros((len(f), 1)), kernel_im],
        ]
    )
    b_data = np.concatenate([z_re, -z_im_corr])

    # Scale only the data equations. This makes automatic lambda selection less
    # sensitive to whether impedances are expressed in ohm or milliohm.
    scale = float(np.sqrt(np.mean(b_data**2)))
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    a_scaled = a_data / scale
    b_scaled = b_data / scale

    d2 = _second_difference_matrix(n_tau)
    d_full = np.hstack([np.zeros((d2.shape[0], 1)), d2]) if d2.size else np.empty((0, n_tau + 1))

    lambda_mode = str(config.get("lambda_mode", "auto_lcurve")).casefold()
    if lambda_mode in {"auto", "auto_lcurve", "lcurve"}:
        lambda_reg, x = _choose_lambda_lcurve(a_scaled, b_scaled, d_full, config)
    else:
        lambda_reg = float(config.get("drt_lambda", 1e-3))
        if lambda_reg <= 0:
            lambda_reg = 1e-3
        x, _residual, _roughness = _solve_regularized_nnls(
            a_scaled,
            b_scaled,
            d_full,
            lambda_reg,
        )

    r_inf = float(x[0])
    gamma = np.maximum(np.asarray(x[1:], dtype=float), 0.0)

    reconstructed_re = r_inf + kernel_re @ gamma
    reconstructed_im = omega * inductance_h - kernel_im @ gamma

    rmse_re = float(np.sqrt(np.mean((z_re - reconstructed_re) ** 2)))
    rmse_im = float(np.sqrt(np.mean((z_im - reconstructed_im) ** 2)))
    integral = float(np.sum(gamma) * dln) if gamma.size else None

    freq_tau = 1.0 / (2.0 * np.pi * tau)

    peak_frequency: float | None = None
    peak_gamma: float | None = None
    peak_frequencies: list[float] = []
    peak_gammas: list[float] = []
    if gamma.size and np.any(gamma > 0.0):
        prominence_fraction = float(config.get("peak_prominence_fraction", 0.05))
        prominence = max(0.0, prominence_fraction) * float(np.max(gamma))
        peak_indices, _properties = find_peaks(gamma, prominence=prominence)
        if peak_indices.size:
            # Report peaks from high to low relaxation frequency for reproducibility.
            peak_indices = peak_indices[np.argsort(freq_tau[peak_indices])[::-1]]
            peak_frequencies = [float(freq_tau[idx]) for idx in peak_indices]
            peak_gammas = [float(gamma[idx]) for idx in peak_indices]
            largest = int(peak_indices[np.argmax(gamma[peak_indices])])
            peak_frequency = float(freq_tau[largest])
            peak_gamma = float(gamma[largest])

    return DRTResult(
        tau_s=tau.tolist(),
        frequency_hz=freq_tau.tolist(),
        gamma_ohm=gamma.tolist(),
        peak_frequency_hz=peak_frequency,
        peak_gamma_ohm=peak_gamma,
        integral_ohm=integral,
        lambda_reg=lambda_reg,
        inductance_h=inductance_h,
        r_inf_ohm=r_inf,
        measurement_frequency_hz=f.tolist(),
        reconstructed_real_ohm=reconstructed_re.tolist(),
        reconstructed_imag_ohm=reconstructed_im.tolist(),
        rmse_real_ohm=rmse_re,
        rmse_imag_ohm=rmse_im,
        peak_frequencies_hz=peak_frequencies,
        peak_gammas_ohm=peak_gammas,
        message="",
    )
