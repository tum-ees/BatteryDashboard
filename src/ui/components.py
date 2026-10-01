from __future__ import annotations

from html import escape

from src.models import MeasurementAssessment
from src.status import Status


COLORS = {
    Status.UNKNOWN: "#9ca3af",
    Status.GREEN: "#2e7d32",
    Status.YELLOW: "#f9a825",
    Status.RED: "#c62828",
}


def status_card_html(title: str, status: Status, subtitle: str = "") -> str:
    color = COLORS[status]
    return f"""
    <div style="border:1px solid #d8dde6;border-left:10px solid {color};border-radius:10px;
                padding:16px 18px;margin-bottom:10px;background:#ffffff;">
      <div style="font-size:0.9rem;color:#667085;">{escape(subtitle)}</div>
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
        <div style="font-size:1.15rem;font-weight:650;">{escape(title)}</div>
        <div style="font-size:1.35rem;font-weight:750;color:{color};">{status.emoji} {status.label}</div>
      </div>
    </div>
    """


def measurement_card_html(item: MeasurementAssessment) -> str:
    historical_color = COLORS[item.worst_status]
    current = "–" if item.current_value is None else f"{item.current_value:.3g} {item.unit}"
    return f"""
    <div style="border:1px solid #d8dde6;border-top:6px solid {historical_color};border-radius:10px;
                padding:14px 16px;background:#ffffff;min-height:145px;">
      <div style="font-weight:700;font-size:1.05rem;">{escape(item.label)}</div>
      <div style="font-size:1.35rem;margin:6px 0;">{escape(current)}</div>
      <div>Historisch: <b>{item.worst_status.emoji} {item.worst_status.label}</b></div>
      <div>Aktuell: {item.current_status.emoji} {item.current_status.label}</div>
    </div>
    """
