from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from src.models import AgingAssessment, AgingData, DRTResult, GEISData, RegressionResult
from src.status import Status


STATUS_COLORS = {
    Status.GREEN: "#2e7d32",
    Status.YELLOW: "#f9a825",
    Status.RED: "#c62828",
    Status.UNKNOWN: "#9ca3af",
}


# Plotting millions of raw samples is neither visually useful nor browser-friendly.
# The analysis still uses every raw sample; only the rendering is decimated.
AGING_MAX_LINE_POINTS = 12_000
AGING_MAX_MARKERS_PER_STATUS = 2_000


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _minmax_downsample_indices(y: np.ndarray, max_points: int) -> np.ndarray:
    """Return temporal min/max representatives while preserving spikes.

    Each bucket contributes its local minimum and maximum in temporal order.
    Compared with a naive ``data[::k]`` stride this keeps short excursions much
    more reliably. First/last sample are always retained.
    """
    n = int(y.size)
    if n <= max_points or max_points < 4:
        return np.arange(n, dtype=np.int64)

    bucket_count = max(1, (max_points - 2) // 2)
    interior = n - 2
    bucket_size = max(1, int(np.ceil(interior / bucket_count)))

    out: list[int] = [0]
    for start in range(1, n - 1, bucket_size):
        end = min(n - 1, start + bucket_size)
        chunk = y[start:end]
        if chunk.size == 0:
            continue
        finite = np.isfinite(chunk)
        if not np.any(finite):
            # Preserve a representative NaN so visual gaps are not bridged.
            out.append(start)
            continue
        local_valid = np.flatnonzero(finite)
        valid_values = chunk[finite]
        imin = start + int(local_valid[int(np.argmin(valid_values))])
        imax = start + int(local_valid[int(np.argmax(valid_values))])
        if imin <= imax:
            out.append(imin)
            if imax != imin:
                out.append(imax)
        else:
            out.append(imax)
            if imin != imax:
                out.append(imin)

    out.append(n - 1)
    # Duplicates can occur for tiny buckets. Preserve temporal order.
    return np.asarray(list(dict.fromkeys(out)), dtype=np.int64)


def _representative_violation_indices(
    status_codes: np.ndarray, target: int, max_markers: int
) -> np.ndarray:
    """Keep all sparse violations; summarize very dense violation regions."""
    if status_codes.size == 0:
        return np.empty(0, dtype=np.int64)
    mask = status_codes == target
    count = int(np.count_nonzero(mask))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count <= max_markers:
        return np.flatnonzero(mask).astype(np.int64, copy=False)

    # Prefer boundaries of contiguous violation runs so short events remain visible.
    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    boundaries = np.unique(np.concatenate((starts, ends))).astype(np.int64, copy=False)

    if boundaries.size >= max_markers:
        pick = np.linspace(0, boundaries.size - 1, max_markers, dtype=np.int64)
        return boundaries[pick]

    # Fill remaining budget with approximately uniform violation samples.
    viol = np.flatnonzero(mask).astype(np.int64, copy=False)
    remaining = max_markers - boundaries.size
    pick = np.linspace(0, viol.size - 1, remaining, dtype=np.int64)
    return np.unique(np.concatenate((boundaries, viol[pick])))


def _take_by_index(values, indices: np.ndarray, *, scale: float = 1.0) -> np.ndarray:
    # Avoid converting a 9M-element Python list just to retain ~12k x values.
    return np.fromiter((float(values[int(i)]) * scale for i in indices), dtype=float, count=len(indices))


def make_aging_figure(
    data: AgingData,
    assessment: AgingAssessment,
    key: str,
    x_axis: str,
) -> go.Figure:
    values = {
        "voltage": data.voltage_v,
        "current": data.current_a,
        "temperature": data.temperature_c,
        "humidity": data.humidity_pct,
        "acceleration": data.acceleration_g,
    }[key]
    item = assessment.measurements[key]

    y_full = _as_float_array(values)
    n = int(y_full.size)
    display_idx = _minmax_downsample_indices(y_full, AGING_MAX_LINE_POINTS)
    y_display = y_full[display_idx]

    if x_axis == "EFC":
        x_source = assessment.equivalent_full_cycles
        x_title = "Equivalent Full Cycles / -"
        x_display = _take_by_index(x_source, display_idx)
    else:
        x_source = data.time_s
        x_title = "Zeit seit Messbeginn / h"
        x_display = _take_by_index(x_source, display_idx, scale=1.0 / 3600.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x_display,
            y=y_display,
            mode="lines",
            name=item.label,
            line=dict(color="#344054", width=1.5),
            hovertemplate=f"{item.label}: %{{y:.5g}} {item.unit}<br>x: %{{x:.5g}}<extra></extra>",
        )
    )

    if item.series_statuses:
        m = min(n, len(item.series_statuses))
        status_codes = np.asarray(item.series_statuses[:m], dtype=np.int8)
        for status in (Status.YELLOW, Status.RED):
            indices = _representative_violation_indices(
                status_codes, int(status), AGING_MAX_MARKERS_PER_STATUS
            )
            if indices.size:
                if x_axis == "EFC":
                    x_mark = _take_by_index(assessment.equivalent_full_cycles, indices)
                else:
                    x_mark = _take_by_index(data.time_s, indices, scale=1.0 / 3600.0)
                y_mark = y_full[indices]
                fig.add_trace(
                    go.Scattergl(
                        x=x_mark,
                        y=y_mark,
                        mode="markers",
                        name=f"{status.label}e Grenzverletzung",
                        marker=dict(color=STATUS_COLORS[status], size=7),
                        hovertemplate=(
                            f"{status.label}: %{{y:.5g}} {item.unit}<br>x: %{{x:.5g}}<extra></extra>"
                        ),
                    )
                )

    if n > len(display_idx):
        fig.add_annotation(
            x=1.0,
            y=1.0,
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="bottom",
            text=f"Darstellung: {len(display_idx):,} von {n:,} Punkten".replace(",", "."),
            showarrow=False,
            font=dict(size=11),
        )

    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=f"{item.label} / {item.unit}",
        hovermode="closest",
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(orientation="h"),
        uirevision=f"aging-{data.battery_id}-{key}-{x_axis}",
    )
    return fig


def make_regression_figure(
    regression: RegressionResult | None,
    title: str,
    y_title: str,
) -> go.Figure:
    fig = go.Figure()
    if regression is None:
        fig.update_layout(title=f"{title} – zu wenige Datenpunkte")
        return fig
    fig.add_trace(go.Scatter(x=regression.x, y=regression.y, mode="markers+lines", name="RPT Daten"))
    fig.add_trace(
        go.Scatter(
            x=regression.x_forecast,
            y=regression.y_forecast,
            mode="lines",
            name=f"Lineare Regression (R²={regression.r_squared:.3f})",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Equivalent Full Cycles / -",
        yaxis_title=y_title,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


def _geis_display_label(data: GEISData) -> str:
    if data.rpt_index > 0:
        return f"GEIS{int(data.rpt_index)}"
    source = str(getattr(data, "source_file", "") or "")
    if source:
        stem = source.split("/")[-1].split("\\")[-1].rsplit(".", 1)[0]
        return stem
    return "GEIS"


def _target_frequency_index(frequency_hz: np.ndarray, target_hz: float, *, max_log10_distance: float = 0.25) -> int | None:
    valid = np.isfinite(frequency_hz) & (frequency_hz > 0.0)
    if not np.any(valid):
        return None
    valid_idx = np.flatnonzero(valid)
    distances = np.abs(np.log10(frequency_hz[valid]) - np.log10(float(target_hz)))
    nearest_local = int(np.argmin(distances))
    if float(distances[nearest_local]) > float(max_log10_distance):
        return None
    return int(valid_idx[nearest_local])


def _geis_color_sequence(n: int) -> list[str]:
    if n <= 0:
        return []
    if n == 1:
        return [sample_colorscale("Viridis", [0.55])[0]]
    positions = np.linspace(0.08, 0.92, n)
    return list(sample_colorscale("Viridis", positions))


def make_multi_nyquist_figure(
    data_list: list[GEISData],
    highlight_frequencies_hz: tuple[float, ...] = (1000.0, 100.0, 10.0, 1.0, 0.1),
) -> go.Figure:
    fig = go.Figure()
    if not data_list:
        fig.update_layout(title="Nyquist-Plot – keine GEIS-Daten verfügbar")
        return fig

    marker_symbols = {
        1000.0: "circle",
        100.0: "square",
        10.0: "diamond",
        1.0: "x",
        0.1: "triangle-down",
    }

    # Legend entries for marker meaning
    for target in highlight_frequencies_hz:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"{target:g} Hz",
                marker=dict(symbol=marker_symbols.get(float(target), "circle"), size=11, color="#111827", line=dict(color="#111827", width=1)),
                hoverinfo="skip",
            )
        )

    colors = _geis_color_sequence(len(data_list))

    for idx, (data, color) in enumerate(zip(data_list, colors, strict=False)):
        f = np.asarray(data.frequency_hz, dtype=float)
        z_re = np.asarray(data.z_real_ohm, dtype=float)
        z_im = np.asarray(data.z_imag_ohm, dtype=float)
        valid = np.isfinite(f) & np.isfinite(z_re) & np.isfinite(z_im) & (f > 0.0)
        if not np.any(valid):
            continue

        f = f[valid]
        z_re = z_re[valid]
        z_im = z_im[valid]
        order = np.argsort(f)[::-1]
        f = f[order]
        x_mohm = 1000.0 * z_re[order]
        y_mohm = 1000.0 * z_im[order]

        label = _geis_display_label(data)
        source_format = str(data.metadata.get("source_format", "")).strip()
        if source_format == "summary_only":
            label += " (char. Punkte)"

        fig.add_trace(
            go.Scattergl(
                x=x_mohm,
                y=y_mohm,
                mode="lines+markers",
                name=label,
                legendgroup=f"spec_{idx}",
                line=dict(color=color, width=2),
                marker=dict(color=color, size=5),
                customdata=np.column_stack([f]),
                hovertemplate=(
                    f"{label}<br>Re(Z) %{{x:.4f}} mΩ<br>−Im(Z) %{{y:.4f}} mΩ"
                    "<br>f %{customdata[0]:.4g} Hz<extra></extra>"
                ),
            )
        )

        for target in highlight_frequencies_hz:
            match_idx = _target_frequency_index(f, float(target))
            if match_idx is None:
                continue
            actual_frequency = float(f[match_idx])
            fig.add_trace(
                go.Scatter(
                    x=[float(x_mohm[match_idx])],
                    y=[float(y_mohm[match_idx])],
                    mode="markers",
                    name=f"{target:g} Hz",
                    legendgroup=f"spec_{idx}",
                    showlegend=False,
                    marker=dict(
                        symbol=marker_symbols.get(float(target), "circle"),
                        size=12,
                        color=color,
                        line=dict(color="#111827", width=1.2),
                    ),
                    hovertemplate=(
                        f"{label}<br>Zielmarke {target:g} Hz"
                        f"<br>Nächster Messpunkt {actual_frequency:.4g} Hz"
                        "<br>Re(Z) %{x:.4f} mΩ<br>Im(Z) %{y:.4f} mΩ<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title="GEIS Nyquist-Plot",
        xaxis_title="Re(Z) / mΩ",
        yaxis_title="Im(Z) / mΩ (invertierte Y-Achse)",
        yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1.0),
        hovermode="closest",
        margin=dict(l=55, r=20, t=50, b=45),
        legend=dict(orientation="h"),
    )
    return fig


def make_multi_drt_figure(
    data_list: list[GEISData],
    results: list[DRTResult],
) -> go.Figure:
    """Overlay valid DRTs on the relaxation-time axis τ."""
    fig = go.Figure()
    colors = _geis_color_sequence(len(data_list))

    for data, result, color in zip(data_list, results, colors, strict=False):
        if not result.tau_s or not result.gamma_ohm:
            continue

        tau = np.asarray(result.tau_s, dtype=float)
        gamma = 1000.0 * np.asarray(result.gamma_ohm, dtype=float)
        valid = np.isfinite(tau) & np.isfinite(gamma) & (tau > 0.0)
        if not np.any(valid):
            continue

        tau, gamma = tau[valid], gamma[valid]
        order = np.argsort(tau)
        tau, gamma = tau[order], gamma[order]
        equivalent_frequency = 1.0 / (2.0 * np.pi * tau)

        label = _geis_display_label(data)
        fig.add_trace(
            go.Scattergl(
                x=tau,
                y=gamma,
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
                customdata=np.column_stack([equivalent_frequency]),
                hovertemplate=(
                    f"{label}<br>τ %{{x:.4g}} s"
                    "<br>fτ %{customdata[0]:.4g} Hz"
                    "<br>γ %{y:.4f} mΩ<extra></extra>"
                ),
            )
        )

        if result.peak_frequencies_hz and result.peak_gammas_ohm:
            peak_f = np.asarray(result.peak_frequencies_hz, dtype=float)
            peak_gamma = 1000.0 * np.asarray(result.peak_gammas_ohm, dtype=float)
            peak_valid = np.isfinite(peak_f) & np.isfinite(peak_gamma) & (peak_f > 0.0)
            if np.any(peak_valid):
                peak_f = peak_f[peak_valid]
                peak_gamma = peak_gamma[peak_valid]
                peak_tau = 1.0 / (2.0 * np.pi * peak_f)
                fig.add_trace(
                    go.Scatter(
                        x=peak_tau,
                        y=peak_gamma,
                        mode="markers",
                        name=f"{label} Peaks",
                        showlegend=False,
                        marker=dict(
                            color=color,
                            size=9,
                            symbol="diamond",
                            line=dict(color="#111827", width=1),
                        ),
                        customdata=np.column_stack([peak_f]),
                        hovertemplate=(
                            f"{label}<br>DRT-Peak τ %{{x:.4g}} s"
                            "<br>fτ %{customdata[0]:.4g} Hz"
                            "<br>γ %{y:.4f} mΩ<extra></extra>"
                        ),
                    )
                )

    fig.update_xaxes(type="log")
    fig.update_layout(
        title="Distribution of Relaxation Times",
        xaxis_title="Relaxationszeit τ / s",
        yaxis_title="γ / mΩ",
        hovermode="closest",
        margin=dict(l=55, r=20, t=50, b=45),
        legend=dict(orientation="h"),
    )
    return fig


def make_drt_reconstruction_figure(data: GEISData, result: DRTResult) -> go.Figure:
    """Nyquist quality-control plot: measured spectrum vs DRT reconstruction."""
    fig = go.Figure()
    if not result.measurement_frequency_hz or not result.reconstructed_real_ohm:
        fig.update_layout(title="DRT-Rekonstruktion – nicht verfügbar")
        return fig

    f_model = np.asarray(result.measurement_frequency_hz, dtype=float)
    re_model = 1000.0 * np.asarray(result.reconstructed_real_ohm, dtype=float)
    im_model = 1000.0 * np.asarray(result.reconstructed_imag_ohm, dtype=float)

    f_raw = np.asarray(data.frequency_hz, dtype=float)
    re_raw = np.asarray(data.z_real_ohm, dtype=float)
    im_raw = np.asarray(data.z_imag_ohm, dtype=float)
    valid = np.isfinite(f_raw) & np.isfinite(re_raw) & np.isfinite(im_raw) & (f_raw > 0.0)
    f_raw, re_raw, im_raw = f_raw[valid], re_raw[valid], im_raw[valid]
    order = np.argsort(f_raw)[::-1]
    f_raw, re_raw, im_raw = f_raw[order], re_raw[order], im_raw[order]

    label = _geis_display_label(data)
    fig.add_trace(
        go.Scatter(
            x=1000.0 * re_raw,
            y=1000.0 * im_raw,
            mode="markers",
            name="Messung",
            customdata=np.column_stack([f_raw]),
            hovertemplate=(
                "Messung<br>Re(Z) %{x:.4f} mΩ<br>Im(Z) %{y:.4f} mΩ"
                "<br>f %{customdata[0]:.4g} Hz<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=re_model,
            y=im_model,
            mode="lines",
            name="DRT-Rekonstruktion",
            customdata=np.column_stack([f_model]),
            hovertemplate=(
                "DRT-Rekonstruktion<br>Re(Z) %{x:.4f} mΩ<br>Im(Z) %{y:.4f} mΩ"
                "<br>f %{customdata[0]:.4g} Hz<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"{label}: Messung vs. DRT-Rekonstruktion",
        xaxis_title="Re(Z) / mΩ",
        yaxis_title="Im(Z) / mΩ (invertierte Y-Achse)",
        yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1.0),
        hovermode="closest",
        margin=dict(l=55, r=20, t=50, b=45),
        legend=dict(orientation="h"),
    )
    return fig


def make_drt_figure(frequency_hz: list[float], gamma_ohm: list[float]) -> go.Figure:
    """Single DRT helper, displayed over relaxation time τ for consistency."""
    fig = go.Figure()
    if frequency_hz and gamma_ohm:
        f = np.asarray(frequency_hz, dtype=float)
        gamma = np.asarray(gamma_ohm, dtype=float) * 1000.0
        valid = np.isfinite(f) & np.isfinite(gamma) & (f > 0.0)
        if np.any(valid):
            f, gamma = f[valid], gamma[valid]
            tau = 1.0 / (2.0 * np.pi * f)
            order = np.argsort(tau)
            tau, gamma = tau[order], gamma[order]
            fig.add_trace(go.Scatter(x=tau, y=gamma, mode="lines", name="DRT"))
            fig.update_xaxes(type="log")
    fig.update_layout(
        xaxis_title="Relaxationszeit τ / s",
        yaxis_title="γ / mΩ",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig
