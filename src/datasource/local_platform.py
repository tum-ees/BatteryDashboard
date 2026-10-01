from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.datasource.base import BatteryDataSource
from src.models import AgingSegment, BatteryInfo, CUData, DataFileInfo, GEISData


_TYPE_PATTERN = re.compile(r"(?:^|[_\-.])(Aging|CU|GEIS)(?:[_\-.]|$)", re.IGNORECASE)
_ID_FALLBACK = re.compile(r"^(.*?)(?:[_\-.](?:Aging|CU|GEIS)(?:[_\-.]|$))", re.IGNORECASE)


class LocalPlatformDataSource(BatteryDataSource):
    """Local directory implementation that mimics an online file platform.

    Files are found recursively. The preferred source of truth is metadata inside
    the JSON (battery_id, file_type, cell_id, rpt_index, efc). If these fields are
    missing, file_type and battery_id are inferred from the file name.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._index: dict[str, list[DataFileInfo]] = defaultdict(list)

    def scan(self) -> list[BatteryInfo]:
        self._index.clear()
        for path in sorted(self.root.rglob("*.json")):
            raw = self._safe_load(path)
            if raw is None:
                continue
            info = self._classify(path, raw)
            if info is not None:
                self._index[info.battery_id].append(info)

        result: list[BatteryInfo] = []
        for battery_id, files in sorted(self._index.items()):
            cells = sorted({f.cell_id for f in files if f.cell_id})
            result.append(
                BatteryInfo(
                    battery_id=battery_id,
                    aging_files=sum(f.file_type == "AGING" for f in files),
                    cu_files=sum(f.file_type == "CU" for f in files),
                    geis_files=sum(f.file_type == "GEIS" for f in files),
                    cell_ids=tuple(cells),
                )
            )
        return result

    def load_aging_segments(self, battery_id: str) -> list[AgingSegment]:
        self._ensure_scanned()
        segments: list[AgingSegment] = []
        for info in self._files(battery_id, "AGING"):
            raw = self._require_load(info.path)
            signals = raw.get("signals", {})
            metadata = raw.get("metadata", {})
            t = _float_list(signals.get("time_s"), f"{info.path.name}: signals.time_s")
            n = len(t)
            if n == 0:
                continue
            segments.append(
                AgingSegment(
                    battery_id=battery_id,
                    start_time_s=float(raw.get("start_time_s", info.start_time_s or 0.0)),
                    local_time_s=t,
                    voltage_v=_required_series(signals, "voltage_v", n, info.path.name),
                    current_a=_required_series(signals, "current_a", n, info.path.name),
                    temperature_c=_required_series(signals, "temperature_c", n, info.path.name),
                    humidity_pct=_required_series(signals, "humidity_pct", n, info.path.name),
                    acceleration_g=_required_series(signals, "acceleration_g", n, info.path.name),
                    nominal_capacity_ah=float(metadata.get("nominal_capacity_ah", 1.0)),
                    current_positive_is_charge=bool(metadata.get("current_positive_is_charge", True)),
                    metadata={**metadata, "source_file": str(info.path)},
                )
            )
        return sorted(segments, key=lambda s: s.start_time_s)

    def load_cu_data(self, battery_id: str) -> list[CUData]:
        self._ensure_scanned()
        result: list[CUData] = []
        for info in self._files(battery_id, "CU"):
            raw = self._require_load(info.path)
            signals = raw.get("signals", {})
            metadata = raw.get("metadata", {})
            t = _float_list(signals.get("time_s"), f"{info.path.name}: signals.time_s")
            n = len(t)
            if n == 0:
                continue
            result.append(
                CUData(
                    battery_id=battery_id,
                    cell_id=str(raw.get("cell_id", info.cell_id or "cell_1")),
                    rpt_index=int(raw.get("rpt_index", info.rpt_index or 0)),
                    efc=float(raw.get("efc", info.efc or 0.0)),
                    time_s=t,
                    voltage_v=_required_series(signals, "voltage_v", n, info.path.name),
                    current_a=_required_series(signals, "current_a", n, info.path.name),
                    nominal_capacity_ah=float(metadata.get("nominal_capacity_ah", 1.0)),
                    current_positive_is_charge=bool(metadata.get("current_positive_is_charge", True)),
                    metadata=metadata,
                    source_file=str(info.path),
                )
            )
        return sorted(result, key=lambda x: (x.efc, x.rpt_index, x.cell_id))

    def load_geis_data(self, battery_id: str) -> list[GEISData]:
        self._ensure_scanned()
        result: list[GEISData] = []
        for info in self._files(battery_id, "GEIS"):
            raw = self._require_load(info.path)
            eis = raw.get("eis", {})
            f = _float_list(eis.get("frequency_hz"), f"{info.path.name}: eis.frequency_hz")
            n = len(f)
            if n == 0:
                continue
            z_re = _float_list(eis.get("z_real_ohm"), f"{info.path.name}: eis.z_real_ohm")
            z_im = _float_list(eis.get("z_imag_ohm"), f"{info.path.name}: eis.z_imag_ohm")
            if len(z_re) != n or len(z_im) != n:
                raise ValueError(f"{info.path.name}: EIS arrays must have equal length")
            result.append(
                GEISData(
                    battery_id=battery_id,
                    cell_id=str(raw.get("cell_id", info.cell_id or "cell_1")),
                    rpt_index=int(raw.get("rpt_index", info.rpt_index or 0)),
                    efc=float(raw.get("efc", info.efc or 0.0)),
                    frequency_hz=f,
                    z_real_ohm=z_re,
                    z_imag_ohm=z_im,
                    metadata=raw.get("metadata", {}),
                    source_file=str(info.path),
                )
            )
        return sorted(result, key=lambda x: (x.efc, x.rpt_index, x.cell_id))

    def _files(self, battery_id: str, file_type: str) -> list[DataFileInfo]:
        return [f for f in self._index.get(battery_id, []) if f.file_type == file_type]

    def _ensure_scanned(self) -> None:
        if not self._index:
            self.scan()

    def _safe_load(self, path: Path) -> dict[str, Any] | None:
        try:
            return self._require_load(path)
        except (OSError, json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _require_load(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"{path.name}: JSON root must be an object")
        return raw

    def _classify(self, path: Path, raw: dict[str, Any]) -> DataFileInfo | None:
        file_type = str(raw.get("file_type", "")).strip().upper()
        if file_type not in {"AGING", "CU", "GEIS"}:
            match = _TYPE_PATTERN.search(path.stem)
            if not match:
                return None
            file_type = match.group(1).upper()

        battery_id = str(raw.get("battery_id", "")).strip()
        if not battery_id:
            match = _ID_FALLBACK.search(path.stem)
            battery_id = match.group(1).rstrip("_.-") if match else path.stem

        return DataFileInfo(
            path=path,
            battery_id=battery_id,
            file_type=file_type,
            cell_id=_optional_str(raw.get("cell_id")),
            rpt_index=_optional_int(raw.get("rpt_index")),
            efc=_optional_float(raw.get("efc")),
            start_time_s=_optional_float(raw.get("start_time_s")),
        )


def _required_series(signals: dict[str, Any], key: str, n: int, file_name: str) -> list[float]:
    values = _float_list(signals.get(key), f"{file_name}: signals.{key}")
    if len(values) != n:
        raise ValueError(f"{file_name}: signals.{key} has {len(values)} values, expected {n}")
    return values


def _float_list(value: Any, path: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} contains non-numeric values") from exc


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
