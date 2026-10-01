from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis.aging import analyze_aging, merge_aging_segments
from src.analysis.config import load_json_config
from src.analysis.cu import (
    analyze_cu_record,
    flatten_r_dc_metadata,
    get_r_dc_dimensions,
    get_r_dc_value,
)
from src.analysis.drt import calculate_drt
from src.analysis.prediction import analyze_fischer_prediction
from src.analysis.dma import (
    count_dma_inputs,
    make_dma_result_figure,
    make_dma_study_figure,
    pydma_status,
    run_dma_study,
)
from src.datasource.local_folder_tree import LocalFolderTreeDataSource
from src.ui.components import measurement_card_html, status_card_html
from src.ui.plotting import (
    make_aging_figure,
    make_drt_reconstruction_figure,
    make_multi_drt_figure,
    make_multi_nyquist_figure,
)


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
THRESHOLDS = ROOT / "config" / "thresholds.json"
RPT_CONFIG = ROOT / "config" / "rpt_analysis.json"

st.set_page_config(page_title="HealthBatt Dashboard", page_icon="🔋", layout="wide")


@st.cache_data(show_spinner=False)
def scan_batteries():
    return LocalFolderTreeDataSource(DATA_ROOT, discover_cell_ids=False).scan()


@st.cache_resource(show_spinner=False)
def load_aging_analysis(battery_id: str):
    source = LocalFolderTreeDataSource(DATA_ROOT, discover_cell_ids=False)
    source.scan()
    segments = source.load_aging_segments(battery_id)
    merged = merge_aging_segments(segments)
    thresholds = load_json_config(THRESHOLDS)
    return merged, analyze_aging(merged, thresholds), thresholds


@st.cache_data(show_spinner=False)
def load_rpt_analysis(battery_id: str):
    source = LocalFolderTreeDataSource(DATA_ROOT, discover_cell_ids=False)
    source.scan()
    cu = source.load_cu_data(battery_id)
    geis = source.load_geis_data(battery_id)
    config = load_json_config(RPT_CONFIG)
    cu_config = config.get("cu", {})
    geis_config = config.get("geis", {})
    dma_config = config.get("dma", {})
    prediction_config = config.get("prediction", {})
    points = [analyze_cu_record(item, cu_config) for item in cu]
    drt_results = [calculate_drt(item, geis_config) for item in geis]
    return cu, points, cu_config, geis, drt_results, geis_config, dma_config, prediction_config


@st.cache_resource(show_spinner=False)
def load_dma_analysis(
    battery_id: str,
    anode_ocp_path: str,
    cathode_ocp_path: str,
    dma_config_json: str,
):
    """Run pyDMA only when the DMA section is explicitly activated."""
    source = LocalFolderTreeDataSource(DATA_ROOT, discover_cell_ids=False)
    source.scan()
    cu = source.load_cu_data(battery_id)
    dma_config = json.loads(dma_config_json)
    return run_dma_study(cu, dma_config, anode_ocp_path, cathode_ocp_path)


def _resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def _render_pydma_figure(fig, *, key: str) -> None:
    """Render Plotly or Matplotlib figures returned by pyDMA."""
    if fig is None:
        st.caption("pyDMA hat für diese Ansicht keine Figure zurückgegeben.")
        return
    if hasattr(fig, "to_plotly_json"):
        st.plotly_chart(fig, use_container_width=True, key=key)
        return
    if hasattr(fig, "savefig"):
        st.pyplot(fig, use_container_width=True)
        return
    st.write(fig)


def reset_analysis_if_battery_changed(battery_id: str) -> None:
    if st.session_state.get("last_battery") != battery_id:
        st.session_state["last_battery"] = battery_id
        st.session_state["active_analysis"] = None


def render_aging(battery_id: str) -> None:
    with st.spinner("Betriebsdaten werden zusammengeführt und analysiert …"):
        data, assessment, thresholds = load_aging_analysis(battery_id)

    st.markdown("## 1. Betriebsdatenanalyse")
    st.caption(
        f"{len(data.source_files)} Aging-Dateien wurden zu einer kontinuierlichen Zeitreihe "
        f"mit {len(data.time_s):,} Messpunkten zusammengesetzt.".replace(",", ".")
    )
    st.markdown(
        status_card_html(
            "Gesamtzustand",
            assessment.overall_status,
            "Schlechtester Zustand seit Messbeginn",
        ),
        unsafe_allow_html=True,
    )

    if assessment.reasons:
        with st.expander("Grund für die Gesamtbewertung", expanded=True):
            for reason in assessment.reasons:
                st.write(f"• {reason}")

    order = ["voltage", "current", "temperature", "humidity", "acceleration"]
    cols = st.columns(len(order))
    for col, key in zip(cols, order):
        with col:
            st.markdown(measurement_card_html(assessment.measurements[key]), unsafe_allow_html=True)

    accel = assessment.measurements["acceleration"]
    accel_available = bool(accel.details.get("available", False))

    if accel_available:
        yellow_event_count = accel.details.get("yellow_event_count")
        yellow_event_limit = accel.details.get("yellow_event_limit")
        hysteresis_s = accel.details.get("hysteresis_s")

        if yellow_event_count is not None and yellow_event_limit is not None:
            st.info(
                f"Beschleunigungsereignisse im gelben Bereich: "
                f"{yellow_event_count} / {yellow_event_limit}. "
                f"Ein neues Ereignis wird erst nach {hysteresis_s:g} s kontinuierlich "
                f"≤ {thresholds['acceleration']['green_max_g']:g} g gezählt."
            )
    else:
        st.caption(
            "Beschleunigungsdaten nicht verfügbar – die Ereigniszählung wird für "
            "diesen Datensatz nicht durchgeführt."
        )

    st.markdown("### Zeitreihen und Grenzverletzungen")
    available_order = [
        key
        for key in order
        if bool(assessment.measurements[key].details.get("available", False))
    ]

    if available_order:
        label_to_key = {assessment.measurements[key].label: key for key in available_order}
        c1, c2 = st.columns([2, 1])
        with c1:
            selected_label = st.selectbox("Messgröße", list(label_to_key), key="aging_signal")
        with c2:
            x_axis = st.radio("X-Achse", ["Zeit", "EFC"], horizontal=True, key="aging_x")
        selected_key = label_to_key[selected_label]
        st.plotly_chart(
            make_aging_figure(data, assessment, selected_key, x_axis),
            use_container_width=True,
        )
    else:
        selected_key = None
        st.warning("Keine auswertbaren Betriebsdatensignale für die Darstellung vorhanden.")

    if (
        selected_key == "acceleration"
        and accel_available
        and accel.details.get("yellow_events")
    ):
        st.markdown("#### Erkannte Beschleunigungsereignisse")
        event_rows = []
        for event in accel.details["yellow_events"]:
            event_rows.append(
                {
                    "Nr.": event.index,
                    "Start / s": round(event.start_time_s, 2),
                    "Ende / s": round(event.end_time_s, 2),
                    "Max. |a| / g": round(event.max_abs_g, 2),
                    "Bewertung": (
                        "Rot ab hier"
                        if event.index >= accel.details.get("yellow_event_limit", 10)
                        else "Gelb"
                    ),
                }
            )
        st.dataframe(event_rows, use_container_width=True, hide_index=True)

    with st.expander("Detailübersicht Betriebsdaten"):
        rows = []
        for key in order:
            item = assessment.measurements[key]
            rows.append(
                {
                    "Messgröße": item.label,
                    "Daten": "vorhanden" if item.details.get("available", False) else "nicht verfügbar",
                    "Aktuell": f"{item.current_status.emoji} {item.current_status.label}",
                    "Historisch": f"{item.worst_status.emoji} {item.worst_status.label}",
                    "Aktueller Wert": item.current_value,
                    "Minimum": item.minimum,
                    "Maximum": item.maximum,
                    "Gelbe Samples": item.yellow_sample_count,
                    "Rote Samples": item.red_sample_count,
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _select_rdc_option(
    label: str,
    options: list[str],
    preferred: str,
    key: str,
    *,
    format_func=None,
) -> str:
    if not options:
        raise ValueError(f"No options available for {label}")
    default = preferred if preferred in options else options[0]
    if st.session_state.get(key) not in options:
        st.session_state[key] = default
    kwargs = {"key": key}
    if format_func is not None:
        kwargs["format_func"] = format_func
    return st.selectbox(label, options, **kwargs)


def _geis_label(item) -> str:
    idx = int(getattr(item, "rpt_index", 0) or 0)
    if idx > 0:
        return f"GEIS{idx}"
    return Path(item.source_file).stem


def render_rpt(battery_id: str, battery_info) -> None:
    st.markdown("## 2. RPT-Analyse – CU & GEIS")

    if battery_info.cu_files <= 0 and battery_info.geis_files <= 0:
        st.warning("Für diese Batterie wurden weder CU- noch GEIS-Dateien gefunden.")
        return

    with st.spinner("RPT-Daten werden geladen und ausgewertet …"):
        cu_raw, points, cu_config, geis_raw, drt_raw, geis_config, dma_config, prediction_config = load_rpt_analysis(battery_id)

    st.caption(
        "Aktueller Ausbaustand: CU-Kapazität, vollständige `MetaData.R_DC`-Auswertung, "
        "GEIS-Nyquist/DRT, eine minimalistische pyDMA-Auswertung der pOCV-Daten sowie "
        "eine Fischer-orientierte SOH-Prädiktion aus R_DC,10s."
    )

    if not points and not geis_raw:
        st.warning(
            "RPT-Dateien wurden gefunden, konnten aber weder als CU noch als GEIS auswertbar geladen werden. "
            "Prüfe das JSON-Format."
        )
        return

    if points:
        x_values = [p.rpt_index if p.rpt_index > 0 else idx + 1 for idx, p in enumerate(points)]
        raw_by_source = {item.source_file: item for item in cu_raw}

        latest = points[-1]
        st.markdown("### CU – Kapazität und R_DC")
        m1, m2 = st.columns(2)
        m1.metric("Ausgewertete CU-Dateien", len(points))
        m2.metric(
            "Letzte Kapazität",
            "–" if latest.discharge_capacity_ah is None else f"{latest.discharge_capacity_ah:.3f} Ah",
        )

        st.markdown("#### Kapazität")
        fig_cap = go.Figure()
        fig_cap.add_trace(
            go.Scatter(
                x=x_values,
                y=[p.discharge_capacity_ah for p in points],
                connectgaps=False,
                mode="lines+markers",
                name="Kapazität",
                customdata=[Path(p.source_file).name for p in points],
                hovertemplate=(
                    "CU %{x}<br>Kapazität %{y:.4f} Ah<br>Datei %{customdata}<extra></extra>"
                ),
            )
        )
        fig_cap.update_layout(
            title="Kapazität aus CU-Metadaten",
            xaxis_title="CU / RPT-Index",
            yaxis_title="Kapazität / Ah",
            hovermode="closest",
            margin=dict(l=55, r=20, t=55, b=50),
        )
        st.plotly_chart(fig_cap, use_container_width=True)

        st.markdown("#### DC-Widerstand aus `MetaData.R_DC`")
        structured_records = [
            item
            for item in cu_raw
            if isinstance(item.metadata.get("r_dc"), dict) and item.metadata.get("r_dc")
        ]

        if not structured_records:
            st.info(
                "In den geladenen CU-Dateien wurde keine strukturierte `MetaData.R_DC`-Auswertung gefunden."
            )
        else:
            defaults = {
                "soc": str(cu_config.get("metadata_resistance_soc", "SoC50")),
                "c_rate": str(cu_config.get("metadata_resistance_c_rate", "0.5C")),
                "mode": str(cu_config.get("metadata_resistance_mode", "mean")),
                "delay": str(cu_config.get("metadata_resistance_delay", "10s")),
            }

            all_dims = get_r_dc_dimensions(structured_records)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                selected_soc = _select_rdc_option(
                    "SoC",
                    all_dims["socs"],
                    defaults["soc"],
                    "rpt_rdc_soc",
                )
            rate_dims = get_r_dc_dimensions(structured_records, soc=selected_soc)
            with c2:
                selected_rate = _select_rdc_option(
                    "C-Rate",
                    rate_dims["c_rates"],
                    defaults["c_rate"],
                    "rpt_rdc_rate",
                )
            mode_dims = get_r_dc_dimensions(
                structured_records,
                soc=selected_soc,
                c_rate=selected_rate,
            )
            mode_labels = {
                "discharge": "Entladen",
                "charge": "Laden",
                "mean": "Mittelwert",
            }
            with c3:
                selected_mode = _select_rdc_option(
                    "Richtung / Aggregation",
                    mode_dims["modes"],
                    defaults["mode"],
                    "rpt_rdc_mode",
                    format_func=lambda x: mode_labels.get(x.casefold(), x),
                )
            delay_dims = get_r_dc_dimensions(
                structured_records,
                soc=selected_soc,
                c_rate=selected_rate,
                mode=selected_mode,
            )
            with c4:
                selected_delay = _select_rdc_option(
                    "Auswertezeit",
                    delay_dims["delays"],
                    defaults["delay"],
                    "rpt_rdc_delay",
                )

            selected_values: list[float | None] = []
            for point in points:
                raw_item = raw_by_source.get(point.source_file)
                if raw_item is None:
                    selected_values.append(None)
                else:
                    selected_values.append(
                        get_r_dc_value(
                            raw_item,
                            selected_soc,
                            selected_rate,
                            selected_mode,
                            selected_delay,
                        )
                    )

            latest_selected = selected_values[-1] if selected_values else None
            valid_count = sum(value is not None for value in selected_values)
            metric1, metric2, metric3 = st.columns(3)
            metric1.metric(
                "Widerstand im letzten CU",
                "–" if latest_selected is None else f"{1000.0 * latest_selected:.3f} mΩ",
            )
            metric2.metric("Verfügbare Werte", f"{valid_count} / {len(selected_values)}")
            metric3.metric(
                "Auswahl",
                f"{selected_soc} | {selected_rate} | {selected_delay}",
            )

            fig_r = go.Figure()
            fig_r.add_trace(
                go.Scatter(
                    x=x_values,
                    y=[None if value is None else 1000.0 * value for value in selected_values],
                    connectgaps=False,
                    mode="lines+markers",
                    name="R_DC",
                    customdata=[Path(p.source_file).name for p in points],
                    hovertemplate=(
                        "CU %{x}<br>R_DC %{y:.4f} mΩ<br>Datei %{customdata}<extra></extra>"
                    ),
                )
            )
            fig_r.update_layout(
                title=(
                    f"R_DC: {selected_soc} / {selected_rate} / "
                    f"{mode_labels.get(selected_mode.casefold(), selected_mode)} / {selected_delay}"
                ),
                xaxis_title="CU / RPT-Index",
                yaxis_title=f"R({selected_delay}) / mΩ",
                hovermode="closest",
                margin=dict(l=55, r=20, t=55, b=50),
            )
            st.plotly_chart(fig_r, use_container_width=True)

            if valid_count < len(selected_values):
                st.caption(
                    "Fehlende `R_DC`-Einträge bleiben Lücken im Verlauf. Sie werden weder als 0 Ω "
                    "interpretiert noch durch Werte einer anderen C-Rate oder eines anderen SoC ersetzt."
                )

            st.markdown("##### Ausgewählte R_DC-Kombination je CU")
            selected_rows = []
            for idx, (point, value) in enumerate(zip(points, selected_values)):
                raw_item = raw_by_source.get(point.source_file)
                selected_rows.append(
                    {
                        "CU / RPT": x_values[idx],
                        "dateStart": None if raw_item is None else raw_item.metadata.get("dateStart"),
                        "SoC": selected_soc,
                        "C-Rate": selected_rate,
                        "Richtung": mode_labels.get(selected_mode.casefold(), selected_mode),
                        "Zeit": selected_delay,
                        "R_DC / mΩ": None if value is None else round(1000.0 * value, 4),
                        "Datei": Path(point.source_file).name,
                    }
                )
            st.dataframe(selected_rows, use_container_width=True, hide_index=True)

            with st.expander("Vollständige R_DC-Struktur aller CU-Dateien"):
                full_rows = []
                for idx, point in enumerate(points):
                    raw_item = raw_by_source.get(point.source_file)
                    if raw_item is None:
                        continue
                    for row in flatten_r_dc_metadata(raw_item):
                        source_label = {
                            "metadata": "Metadaten",
                            "derived_mean": "Mittelwert berechnet",
                            "missing": "fehlend",
                        }.get(row["value_source"], row["value_source"])
                        full_rows.append(
                            {
                                "CU / RPT": x_values[idx],
                                "dateStart": raw_item.metadata.get("dateStart"),
                                "SoC": row["soc"],
                                "C-Rate": row["c_rate"],
                                "Richtung": mode_labels.get(str(row["mode"]).casefold(), row["mode"]),
                                "Zeit": row["delay"],
                                "R_DC / mΩ": (
                                    None
                                    if row["resistance_ohm"] is None
                                    else round(1000.0 * float(row["resistance_ohm"]), 4)
                                ),
                                "Wertquelle": source_label,
                                "Datei": Path(point.source_file).name,
                            }
                        )
                if full_rows:
                    st.dataframe(full_rows, use_container_width=True, hide_index=True)
                else:
                    st.caption("Keine strukturierten R_DC-Werte vorhanden.")

        st.markdown("#### CU-Kapazitätswerte")
        capacity_rows = []
        for idx, point in enumerate(points):
            raw_item = raw_by_source.get(point.source_file)
            capacity_rows.append(
                {
                    "CU / RPT": x_values[idx],
                    "dateStart": None if raw_item is None else raw_item.metadata.get("dateStart"),
                    "Kapazität / Ah": (
                        None if point.discharge_capacity_ah is None else round(point.discharge_capacity_ah, 4)
                    ),
                    "Datei": Path(point.source_file).name,
                }
            )
        st.dataframe(capacity_rows, use_container_width=True, hide_index=True)

        with st.expander("Technische Details zur CU-Auswertung"):
            st.write(
                "Die Dropdowns werden dynamisch aus der Vereinigung der tatsächlich vorhandenen "
                "`R_DC`-Strukturen aufgebaut. Dadurch können sich z. B. 0.1C, 0.5C und 1C in der "
                "Auswahl befinden, auch wenn nicht jede CU-Datei jede C-Rate enthält."
            )
            st.write(
                "`null`, leere Strings, NaN oder fehlende Einträge werden als fehlend behandelt. "
                "Nur bei `mean` darf ein fehlender gespeicherter Mittelwert aus den vorhandenen "
                "Lade-/Entladewerten derselben SoC-, C-Rate- und Zeitkombination rekonstruiert werden."
            )
    else:
        st.info(
            "Es wurden keine auswertbaren CU-Dateien gefunden. Der GEIS-Teil kann dennoch dargestellt werden, "
            "sofern GEIS-Daten vorliegen."
        )

    st.markdown("### Prädiktion – Fischer et al. (2025)")
    if not points:
        st.info("Für die Fischer-orientierte Prädiktion werden CU-Kapazitäts- und R_DC-Werte benötigt.")
    else:
        prediction_run_config = dict(prediction_config)
        st.caption(
            "Die Auswertung folgt der in Fischer et al. (Journal of Power Sources, 2025, 237921) "
            "beschriebenen Korrelation zwischen relativem 10-s-DC-Pulswiderstandsanstieg und SOH_C. "
            "Als papernahe Voreinstellung wird SoC50 | 1C | Mittelwert | 10s verwendet."
        )

        if structured_records:
            with st.expander("Widerstandsfeature für Prädiktion", expanded=False):
                pred_defaults = {
                    "soc": str(prediction_config.get("soc", "SoC50")),
                    "c_rate": str(prediction_config.get("c_rate", "1C")),
                    "mode": str(prediction_config.get("mode", "mean")),
                    "delay": str(prediction_config.get("delay", "10s")),
                }
                pred_all = get_r_dc_dimensions(structured_records)
                pc1, pc2, pc3, pc4 = st.columns(4)
                with pc1:
                    pred_soc = _select_rdc_option(
                        "SoC", pred_all["socs"], pred_defaults["soc"], "prediction_rdc_soc"
                    )
                pred_rate_dims = get_r_dc_dimensions(structured_records, soc=pred_soc)
                with pc2:
                    pred_rate = _select_rdc_option(
                        "C-Rate", pred_rate_dims["c_rates"], pred_defaults["c_rate"], "prediction_rdc_rate"
                    )
                pred_mode_dims = get_r_dc_dimensions(
                    structured_records, soc=pred_soc, c_rate=pred_rate
                )
                pred_mode_labels = {"discharge": "Entladen", "charge": "Laden", "mean": "Mittelwert"}
                with pc3:
                    pred_mode = _select_rdc_option(
                        "Richtung / Aggregation",
                        pred_mode_dims["modes"],
                        pred_defaults["mode"],
                        "prediction_rdc_mode",
                        format_func=lambda x: pred_mode_labels.get(x.casefold(), x),
                    )
                pred_delay_dims = get_r_dc_dimensions(
                    structured_records, soc=pred_soc, c_rate=pred_rate, mode=pred_mode
                )
                with pc4:
                    pred_delay = _select_rdc_option(
                        "Auswertezeit",
                        pred_delay_dims["delays"],
                        pred_defaults["delay"],
                        "prediction_rdc_delay",
                    )
                prediction_run_config.update(
                    {"soc": pred_soc, "c_rate": pred_rate, "mode": pred_mode, "delay": pred_delay}
                )
                st.caption(
                    "Für die engste Anlehnung an Fischer et al. sollte – sofern in den Daten vorhanden – "
                    "SoC50, 1C und 10s verwendet werden. Andere Kombinationen sind als explorative Variante möglich."
                )

        prediction = analyze_fischer_prediction(cu_raw, points, prediction_run_config)
        st.caption(f"Verwendetes Widerstandsfeature: `{prediction.feature_label}`")

        if not prediction.ready:
            st.warning(prediction.message)
            if prediction.points:
                with st.expander("Verfügbare Punkte / Vorverarbeitung"):
                    rows = []
                    for p in prediction.points:
                        rows.append(
                            {
                                "CU / RPT": p.rpt_index,
                                "Kapazität / Ah": round(p.capacity_ah, 5),
                                "R_DC / mΩ": round(1000.0 * p.resistance_ohm, 5),
                                "Hampel/ungültig": bool(p.is_hampel_outlier),
                                "Vor R_min ausgeschlossen": bool(p.excluded_before_reference),
                                "Datei": Path(p.source_file).name,
                            }
                        )
                    st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "SOH_C gemessen",
                "–" if prediction.current_soh_measured_pct is None else f"{prediction.current_soh_measured_pct:.2f} %",
            )
            m2.metric(
                "SOH_C aus R_DC",
                "–" if prediction.current_soh_predicted_pct is None else f"{prediction.current_soh_predicted_pct:.2f} %",
            )
            m3.metric(
                "Fit-RMSE",
                "–" if prediction.rmse_pct is None else f"{prediction.rmse_pct:.2f} %-Pkt.",
            )
            if prediction.cv_rmse_mean_pct is None:
                cv_text = "–"
            elif prediction.cv_rmse_std_pct is None:
                cv_text = f"{prediction.cv_rmse_mean_pct:.2f} %-Pkt."
            else:
                cv_text = f"{prediction.cv_rmse_mean_pct:.2f} ± {prediction.cv_rmse_std_pct:.2f} %-Pkt."
            m4.metric("70/30-CV RMSE", cv_text)

            st.markdown("#### SOH_C als Funktion des Widerstandsanstiegs")
            r_incr_pct = [100.0 * float(p.resistance_increase) for p in prediction.used_points]
            soh_meas_pct = [100.0 * float(p.soh_measured) for p in prediction.used_points]
            soh_pred_pct = [100.0 * float(p.soh_predicted) for p in prediction.used_points]
            order = sorted(range(len(r_incr_pct)), key=lambda i: r_incr_pct[i])

            fig_corr = go.Figure()
            fig_corr.add_trace(
                go.Scatter(
                    x=r_incr_pct,
                    y=soh_meas_pct,
                    mode="markers",
                    name="Gemessener SOH_C",
                    customdata=[Path(p.source_file).name for p in prediction.used_points],
                    hovertemplate=(
                        "R_incr %{x:.3f} %<br>SOH_C %{y:.3f} %<br>%{customdata}<extra></extra>"
                    ),
                )
            )
            fig_corr.add_trace(
                go.Scatter(
                    x=[r_incr_pct[i] for i in order],
                    y=[soh_pred_pct[i] for i in order],
                    mode="lines",
                    name="Power-Law-Fit",
                )
            )
            fig_corr.update_layout(
                xaxis_title="Relativer Widerstandsanstieg R_incr / %",
                yaxis_title="SOH_C / %",
                hovermode="closest",
                margin=dict(l=55, r=20, t=25, b=50),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            st.latex(r"SOH_C = 1 - \beta_1 \cdot (R_{incr.})^{\beta_2}")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("β₁", f"{prediction.beta1:.5g}" if prediction.beta1 is not None else "–")
            p2.metric("β₂", f"{prediction.beta2:.5g}" if prediction.beta2 is not None else "–")
            p3.metric(
                "Pearson r",
                "–" if prediction.pearson_r is None else f"{prediction.pearson_r:.4f}",
            )
            p4.metric(
                "R²_adj",
                "–" if prediction.r2_adj is None else f"{prediction.r2_adj:.4f}",
            )

            st.markdown("#### Verlauf über die CU/RPT-Reihenfolge")
            fig_soh = go.Figure()
            fig_soh.add_trace(
                go.Scatter(
                    x=[p.rpt_index for p in prediction.used_points],
                    y=soh_meas_pct,
                    mode="lines+markers",
                    name="SOH_C gemessen",
                )
            )
            fig_soh.add_trace(
                go.Scatter(
                    x=[p.rpt_index for p in prediction.used_points],
                    y=soh_pred_pct,
                    mode="lines+markers",
                    name="SOH_C aus R_DC",
                    line=dict(dash="dash"),
                )
            )
            fig_soh.add_hline(y=80.0, line_dash="dot", annotation_text="80 % SOH_C")
            fig_soh.update_layout(
                xaxis_title="CU / RPT-Index",
                yaxis_title="SOH_C / %",
                hovermode="closest",
                margin=dict(l=55, r=20, t=25, b=50),
            )
            st.plotly_chart(fig_soh, use_container_width=True)

            if prediction.usage_forecast and prediction.usage_label:
                st.markdown("#### EOL-Projektion über Nutzung")
                st.caption(
                    "Diese EOL-Projektion ist eine Dashboard-Erweiterung und nicht Teil des Fischer-Power-Law-Modells selbst: "
                    "Der gemessene Widerstandsanstieg wird linear über die Nutzung extrapoliert und anschließend über die "
                    "Power-Law-Beziehung in SOH_C übersetzt."
                )
                fig_forecast = go.Figure()
                usage_meas = [p.usage for p in prediction.used_points if p.usage is not None]
                soh_usage_meas = [
                    100.0 * float(p.soh_measured)
                    for p in prediction.used_points
                    if p.usage is not None and p.soh_measured is not None
                ]
                fig_forecast.add_trace(
                    go.Scatter(
                        x=usage_meas,
                        y=soh_usage_meas,
                        mode="markers+lines",
                        name="Gemessener SOH_C",
                    )
                )
                fig_forecast.add_trace(
                    go.Scatter(
                        x=prediction.usage_forecast,
                        y=prediction.soh_forecast_pct,
                        mode="lines",
                        name="Projektion",
                        line=dict(dash="dash"),
                    )
                )
                if prediction.eol_soh_pct is not None:
                    fig_forecast.add_hline(
                        y=prediction.eol_soh_pct,
                        line_dash="dot",
                        annotation_text=f"EOL {prediction.eol_soh_pct:.0f} %",
                    )
                if prediction.estimated_eol_usage is not None:
                    fig_forecast.add_vline(
                        x=prediction.estimated_eol_usage,
                        line_dash="dot",
                        annotation_text=f"EOL ≈ {prediction.estimated_eol_usage:.1f} {prediction.usage_label}",
                    )
                fig_forecast.update_layout(
                    xaxis_title=prediction.usage_label,
                    yaxis_title="SOH_C / %",
                    hovermode="closest",
                    margin=dict(l=55, r=20, t=25, b=50),
                )
                st.plotly_chart(fig_forecast, use_container_width=True)

                if prediction.estimated_eol_usage is not None:
                    st.metric(
                        f"Geschätzte Nutzung bis {prediction.eol_soh_pct:.0f} % SOH_C",
                        f"{prediction.estimated_eol_usage:.1f} {prediction.usage_label}",
                    )

            with st.expander("Methodik / Vorverarbeitung"):
                st.write(
                    f"Referenzierung ab globalem Widerstandsminimum: CU/RPT {prediction.reference_rpt_index}; "
                    f"C_ref = {prediction.reference_capacity_ah:.5g} Ah; "
                    f"R_ref = {1000.0 * prediction.reference_resistance_ohm:.5g} mΩ."
                )
                st.write(
                    "Vor dem Fit werden Widerstandsausreißer mit einem Hampel-Filter behandelt. "
                    "Die Power-Law-Parameter werden anschließend auf den verbleibenden Punkten bestimmt."
                )
                st.write(
                    "Die 70/30-Validierung wird zehnmal mit zufälligen Splits wiederholt, sofern genügend Punkte vorliegen. "
                    "Bei nur einer Batterietrajektorie ist dies eine interne Trajektorienvalidierung und nicht identisch mit der "
                    "zellübergreifenden Global-Fit-Validierung aus Fischer et al."
                )
                for note in prediction.notes:
                    st.write(f"• {note}")

    st.markdown("### DMA – pyDMA")
    with st.expander("Differential Model Analysis aus CU-pOCV", expanded=False):
        dma_direction = str(dma_config.get("direction", "charge")).casefold()
        dma_min_points = int(dma_config.get("min_pocv_points", 20))
        available_dma_inputs = count_dma_inputs(cu_raw, dma_direction, dma_min_points)
        lib_ok, lib_version, lib_message = pydma_status()

        d1, d2, d3 = st.columns(3)
        d1.metric("CU-Dateien mit pOCV", f"{available_dma_inputs} / {len(cu_raw)}")
        d2.metric("pOCV-Richtung", dma_direction)
        d3.metric("pyDMA", f"verfügbar{f' ({lib_version})' if lib_version else ''}" if lib_ok else "nicht verfügbar")

        st.caption(
            "Die DMA verwendet direkt die bereits ausgewerteten `MetaData.pOCV_charge`-Daten "
            "(`AhStep_Ah`, `U_V`) aus den CU-Dateien. Die pyDMA-Parameter entsprechen zunächst "
            "dem bereitgestellten Jupyter-Notebook."
        )

        default_anode = str(dma_config.get("anode_ocp_path", "data/halfcell/anodeOCP.mat"))
        default_cathode = str(dma_config.get("cathode_ocp_path", "data/halfcell/cathodeOCP.mat"))
        c1, c2 = st.columns(2)
        with c1:
            anode_text = st.text_input("Anoden-Halbzellkurve", value=default_anode, key="dma_anode_ocp_path")
        with c2:
            cathode_text = st.text_input("Kathoden-Halbzellkurve", value=default_cathode, key="dma_cathode_ocp_path")

        anode_path = _resolve_project_path(anode_text)
        cathode_path = _resolve_project_path(cathode_text)
        st.caption(
            f"Aufgelöst: Anode `{anode_path}` | Kathode `{cathode_path}`. "
            "Die Pfade können später einfach in `config/rpt_analysis.json` geändert oder hier überschrieben werden."
        )

        dma_state_key = f"dma_active_{battery_id}"
        if st.button(
            "DMA ausführen",
            key=f"dma_run_{battery_id}",
            disabled=(available_dma_inputs == 0 or not lib_ok),
        ):
            st.session_state[dma_state_key] = True

        if not lib_ok:
            st.warning(lib_message or "pyDMA ist in der aktiven Python-Umgebung nicht verfügbar.")
        elif available_dma_inputs == 0:
            st.info(f"Keine gültigen pOCV_{dma_direction}-Daten für die DMA gefunden.")

        if st.session_state.get(dma_state_key, False) and lib_ok and available_dma_inputs > 0:
            with st.spinner("pyDMA wertet die pOCV-Kurven aus …"):
                study = load_dma_analysis(
                    battery_id,
                    str(anode_path),
                    str(cathode_path),
                    json.dumps(dma_config, sort_keys=True),
                )

            if study.message and not study.results_by_cu:
                st.error(study.message)
            else:
                r1, r2 = st.columns(2)
                r1.metric("Erfolgreiche DMA-Fits", len(study.results_by_cu))
                r2.metric("Übersprungene CU", len(study.skipped))

                if study.results_by_cu:
                    labels = list(study.results_by_cu)
                    selected_cu = st.selectbox(
                        "DMA-Detailansicht",
                        labels,
                        key=f"dma_selected_cu_{battery_id}",
                    )
                    try:
                        detail_fig = make_dma_result_figure(study.results_by_cu[selected_cu])
                        _render_pydma_figure(detail_fig, key=f"dma_detail_fig_{battery_id}_{selected_cu}")
                    except Exception as exc:
                        st.warning(f"pyDMA-Detailplot konnte nicht dargestellt werden: {exc}")

                    if study.study_results is not None and len(study.results_by_cu) >= 2:
                        st.markdown("#### DMA-Alterungsverlauf")
                        try:
                            aging_fig = make_dma_study_figure(study.study_results)
                            _render_pydma_figure(aging_fig, key=f"dma_study_fig_{battery_id}")
                        except Exception as exc:
                            st.warning(f"pyDMA-Aging-Plot konnte nicht dargestellt werden: {exc}")

                if study.skipped:
                    with st.expander("Nicht ausgewertete CU-Dateien"):
                        for label, reason in study.skipped.items():
                            st.write(f"**{label}:** {reason}")

        with st.expander("pyDMA-Konfiguration"):
            st.json(dma_config)

    st.markdown("### GEIS – Nyquist")
    if not geis_raw:
        st.info("Für diese Batterie wurden keine GEIS-Dateien gefunden oder es konnten keine GEIS-Daten geladen werden.")
    else:
        geis_sorted = sorted(
            geis_raw,
            key=lambda item: (int(getattr(item, "rpt_index", 0) or 0), Path(item.source_file).name.lower()),
        )
        plot_ready = [
            item
            for item in geis_sorted
            if len(item.frequency_hz) >= 2
            and len(item.z_real_ohm) == len(item.frequency_hz)
            and len(item.z_imag_ohm) == len(item.frequency_hz)
        ]
        skipped = [item for item in geis_sorted if item not in plot_ready]

        c1, c2, c3 = st.columns(3)
        c1.metric("GEIS-Dateien gesamt", len(geis_sorted))
        c2.metric("Darstellbare Spektren", len(plot_ready))
        c3.metric("Übersprungene Dateien", len(skipped))

        st.caption(
            "Die Linienfarben codieren die zeitliche Reihenfolge der GEIS-Dateien. "
            "Die Reihenfolge wird primär aus dem Dateinamen bzw. GEIS-Index abgeleitet, z. B. GEIS2 → GEIS4 → GEIS6. "
            "Markierte Punkte kennzeichnen die nächstliegenden Messpunkte zu 1000, 100, 10, 1 und 0.1 Hz. "
            "Die Y-Achse zeigt Im(Z) und ist invertiert dargestellt, sodass der kapazitive Anteil wie im üblichen Batterie-Nyquist-Plot nach oben erscheint."
        )

        if plot_ready:
            st.plotly_chart(make_multi_nyquist_figure(plot_ready), use_container_width=True)
        else:
            st.warning(
                "Es wurden zwar GEIS-Dateien gefunden, aber keine davon enthält ein darstellbares Frequenzspektrum. "
                "Für einen Nyquist-Plot werden Frequenz-, Realteil- und Imaginärteil-Daten benötigt."
            )

        geis_rows = []
        for item in geis_sorted:
            geis_rows.append(
                {
                    "GEIS": _geis_label(item),
                    "Punkte": len(item.frequency_hz),
                    "Format": str(item.metadata.get("source_format", "")).strip() or "unbekannt",
                    "Datei": Path(item.source_file).name,
                }
            )
        st.dataframe(geis_rows, use_container_width=True, hide_index=True)

        if any(str(item.metadata.get("source_format", "")) == "summary_only" for item in geis_sorted):
            st.info(
                "Mindestens eine ältere GEIS-Datei enthält nur bereits ausgewertete charakteristische Punkte. "
                "Diese Dateien werden weiterhin als sparse Nyquist-Darstellung unterstützt; vollständige Dateien mit f_Hz/ReZ_Ohm/ImZ_Ohm werden als komplettes Spektrum geplottet."
            )

        st.markdown("### GEIS – Distribution of Relaxation Times")
        drt_by_source = {item.source_file: result for item, result in zip(geis_raw, drt_raw)}
        drt_data = []
        drt_results = []
        for item in geis_sorted:
            result = drt_by_source.get(item.source_file)
            if result is not None and result.gamma_ohm and result.frequency_hz:
                drt_data.append(item)
                drt_results.append(result)

        if drt_data:
            st.caption(
                "Die DRT wird aus Real- und Imaginärteil gemeinsam mittels nichtnegativer, "
                "Tikhonov-regularisierter Inversion bestimmt. Vor der Inversion wird die "
                "hochfrequente Serieninduktivität gefittet und berücksichtigt. λ wird standardmäßig "
                "automatisch über die L-Kurve gewählt. Die Linienfarben entsprechen exakt der "
                "zeitlichen Farbcodierung des Nyquist-Plots."
            )
            full_drt_results = [drt_by_source[item.source_file] for item in geis_sorted]
            st.plotly_chart(
                make_multi_drt_figure(geis_sorted, full_drt_results),
                use_container_width=True,
            )

            st.markdown("#### DRT-Rekonstruktionskontrolle")
            source_to_data = {item.source_file: item for item in drt_data}
            source_to_result = {item.source_file: result for item, result in zip(drt_data, drt_results)}
            choices = [item.source_file for item in drt_data]
            selected_source = st.selectbox(
                "GEIS für Rekonstruktionsvergleich",
                choices,
                format_func=lambda value: _geis_label(source_to_data[value]),
                key="rpt_drt_reconstruction_geis",
            )
            selected_data = source_to_data[selected_source]
            selected_result = source_to_result[selected_source]
            st.plotly_chart(
                make_drt_reconstruction_figure(selected_data, selected_result),
                use_container_width=True,
            )

            q1, q2, q3, q4 = st.columns(4)
            q1.metric(
                "L",
                "–" if selected_result.inductance_h is None else f"{selected_result.inductance_h * 1e9:.2f} nH",
            )
            q2.metric(
                "R∞",
                "–" if selected_result.r_inf_ohm is None else f"{selected_result.r_inf_ohm * 1e3:.4f} mΩ",
            )
            q3.metric(
                "RMSE Re(Z)",
                "–" if selected_result.rmse_real_ohm is None else f"{selected_result.rmse_real_ohm * 1e3:.4f} mΩ",
            )
            q4.metric(
                "RMSE Im(Z)",
                "–" if selected_result.rmse_imag_ohm is None else f"{selected_result.rmse_imag_ohm * 1e3:.4f} mΩ",
            )

            st.markdown("#### DRT-Qualitätsübersicht")
            quality_rows = []
            for item in geis_sorted:
                result = drt_by_source.get(item.source_file)
                if result is None:
                    continue
                quality_rows.append(
                    {
                        "GEIS": _geis_label(item),
                        "Punkte": len(item.frequency_hz),
                        "DRT": "verfügbar" if result.gamma_ohm else "nicht verfügbar",
                        "L / nH": None if result.inductance_h is None else round(result.inductance_h * 1e9, 3),
                        "R∞ / mΩ": None if result.r_inf_ohm is None else round(result.r_inf_ohm * 1e3, 5),
                        "λ": result.lambda_reg,
                        "RMSE Re / mΩ": None if result.rmse_real_ohm is None else round(result.rmse_real_ohm * 1e3, 5),
                        "RMSE Im / mΩ": None if result.rmse_imag_ohm is None else round(result.rmse_imag_ohm * 1e3, 5),
                        "DRT-Integral / mΩ": None if result.integral_ohm is None else round(result.integral_ohm * 1e3, 5),
                        "Hauptpeak / Hz": result.peak_frequency_hz,
                        "Peaks": len(result.peak_frequencies_hz),
                        "Hinweis": result.message or "",
                    }
                )
            st.dataframe(quality_rows, use_container_width=True, hide_index=True)

            with st.expander("Technische DRT-Einstellungen"):
                st.json(geis_config)
                st.caption(
                    "DRT-Peaks an den Grenzen des ausgewerteten Relaxationszeitfensters werden "
                    "nicht automatisch als lokale Peaks gezählt. Das reduziert die Gefahr, einen "
                    "reinen Rand-/Diffusionseffekt als diskreten Relaxationsprozess zu interpretieren."
                )
        else:
            st.info(
                "Für die vorhandenen GEIS-Dateien konnte noch keine robuste DRT berechnet werden. "
                "Standardmäßig sind mindestens 20 vollständige Frequenzpunkte erforderlich; "
                "summary-only Dateien werden daher bewusst nicht invertiert."
            )

        failed_drt = [
            (_geis_label(item), drt_by_source[item.source_file].message)
            for item in geis_sorted
            if item.source_file in drt_by_source
            and not drt_by_source[item.source_file].gamma_ohm
            and drt_by_source[item.source_file].message
        ]
        if failed_drt:
            with st.expander("Nicht ausgewertete GEIS-Dateien"):
                for label, message in failed_drt:
                    st.write(f"**{label}:** {message}")


st.title("🔋 HealthBatt Dashboard")
st.caption("Lokaler Plattform-Prototyp: Dateierkennung → Batterie-ID → Betriebs- oder RPT-Analyse")

batteries = scan_batteries()
if not batteries:
    st.error("Keine Daten im lokalen Verzeichnis 'data/' gefunden.")
    st.stop()

by_id = {item.battery_id: item for item in batteries}
selected_id = st.sidebar.selectbox("Batterie-ID", list(by_id))
info = by_id[selected_id]
reset_analysis_if_battery_changed(selected_id)

st.sidebar.markdown("### Erkannte Dateien")
st.sidebar.write(f"Aging: **{info.aging_files}**")
st.sidebar.write(f"CU: **{info.cu_files}**")
st.sidebar.write(f"GEIS: **{info.geis_files}**")
if st.sidebar.button("Daten neu einlesen"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state["active_analysis"] = None
    st.rerun()

st.markdown("### Analyse auswählen")
left, right = st.columns(2)
with left:
    st.markdown("**Betriebsdaten**  \nAging-Dateien zusammenführen, Grenzwerte prüfen, Ampel und Zeitreihen darstellen.")
    if st.button("Betriebsdatenanalyse starten", use_container_width=True, disabled=info.aging_files == 0):
        st.session_state["active_analysis"] = "AGING"
with right:
    st.markdown("**RPT / Alterung**  \nCU: Kapazität/Widerstand; GEIS: Nyquist-Plot; anschließend weitere Analysen.")
    if st.button(
        "RPT-Analyse starten",
        use_container_width=True,
        disabled=(info.cu_files == 0 and info.geis_files == 0),
    ):
        st.session_state["active_analysis"] = "RPT"

st.divider()
active = st.session_state.get("active_analysis")
if active == "AGING":
    render_aging(selected_id)
elif active == "RPT":
    render_rpt(selected_id, info)
else:
    st.info("Wähle oben eine der beiden Analysen aus.")
