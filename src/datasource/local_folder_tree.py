from __future__ import annotations

# >>> FAST_LAZY_V6_DATESTART_ORJSON: lazy scan + dateStart stitching + fast JSON <<<

"""FAST_LAZY_V6_DATESTART_ORJSON local data source for the fixed battery folder structure.

Expected layout::

    <root>/
    ├── <BATTERY_ID_1>/
    │   ├── Aging/
    │   │   └── *.json
    │   ├── CU/
    │   │   └── *.json
    │   └── GEIS/
    │       └── *.json
    └── <BATTERY_ID_2>/
        └── ...

Important performance guarantee
-------------------------------
``scan()`` NEVER calls ``json.load()`` and NEVER loads measurement arrays.
It only inspects the directory tree and stores file paths.

For backwards compatibility with the existing dashboard, ``cell_id`` can be
optionally discovered from a small prefix of CU/GEIS files.  This is a plain
text probe, not JSON parsing.  Set ``discover_cell_ids=False`` if you want
strictly zero file-content reads during ``scan()``.

Full JSON parsing happens only in:
- ``load_aging_segments()``
- ``load_cu_data()``
- ``load_geis_data()``
"""

import json
import math
import os
import re

try:
    import orjson  # optional fast JSON parser
except ImportError:  # pragma: no cover - standard-library fallback
    orjson = None
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

from src.datasource.base import BatteryDataSource
from src.models import AgingSegment, BatteryInfo, CUData, DataFileInfo, GEISData


@dataclass(frozen=True)
class _ScanCacheEntry:
    index: dict[str, tuple[DataFileInfo, ...]]
    batteries: tuple[BatteryInfo, ...]
    watched_dirs: tuple[Path, ...]
    signature: tuple[tuple[str, int], ...]


class LocalFolderTreeDataSource(BatteryDataSource):
    """Data source for ``<root>/<battery_id>/{Aging,CU,GEIS}/*.json``.

    Parameters
    ----------
    root:
        Directory containing one subfolder per battery ID.
    recursive:
        Search recursively below Aging/CU/GEIS.  The default ``False`` is
        fastest and matches the specified directory structure.
    skip_invalid_json:
        Skip invalid JSON when a selected dataset is loaded.
    discover_cell_ids:
        If ``True`` (default), scan a small text prefix of CU/GEIS files for
        ``cell_id`` so the existing dashboard can build its cell dropdown.
        No ``json.load()`` occurs.  Use ``False`` for absolutely no file-content
        reads during ``scan()``.
    metadata_probe_bytes:
        Maximum prefix size used for ``cell_id`` discovery.
    use_scan_cache:
        Cache the file tree in-process between Streamlit reruns/instances.
    """

    VERSION = "FAST_LAZY_V6_DATESTART_ORJSON"

    _FOLDER_TYPES = {
        "aging": "AGING",
        "ageing": "AGING",
        "cu": "CU",
        "geis": "GEIS",
    }

    _CELL_ID_RE = re.compile(r'"cell_id"\s*:\s*"((?:\\.|[^"\\])*)"')

    _SCAN_CACHE: dict[tuple[str, bool, bool, int], _ScanCacheEntry] = {}
    _CACHE_LOCK = RLock()

    def __init__(
        self,
        root: str | Path,
        *,
        recursive: bool = False,
        skip_invalid_json: bool = True,
        discover_cell_ids: bool = True,
        metadata_probe_bytes: int = 64 * 1024,
        use_scan_cache: bool = True,
    ) -> None:
        self.root = Path(root)
        self.recursive = bool(recursive)
        self.skip_invalid_json = bool(skip_invalid_json)
        self.discover_cell_ids = bool(discover_cell_ids)
        self.metadata_probe_bytes = max(0, int(metadata_probe_bytes))
        self.use_scan_cache = bool(use_scan_cache)

        self._index: dict[str, list[DataFileInfo]] = defaultdict(list)
        self._batteries: list[BatteryInfo] = []
        self._has_scanned = False

    # ------------------------------------------------------------------
    # BatteryDataSource interface
    # ------------------------------------------------------------------

    def scan(self) -> list[BatteryInfo]:
        """Index batteries and files without fully loading any JSON file."""
        self._validate_root()

        cache_key = self._cache_key()
        if self.use_scan_cache:
            with self._CACHE_LOCK:
                cached = self._SCAN_CACHE.get(cache_key)
            if cached is not None and self._cache_is_current(cached):
                self._restore_cache(cached)
                self._has_scanned = True
                return list(self._batteries)

        # IMPORTANT: _build_path_index() does not call json.load().
        index, watched_dirs = self._build_path_index()
        batteries = self._build_battery_info(index)

        self._index = defaultdict(
            list,
            {battery_id: list(files) for battery_id, files in index.items()},
        )
        self._batteries = list(batteries)
        self._has_scanned = True

        if self.use_scan_cache:
            entry = _ScanCacheEntry(
                index={battery_id: tuple(files) for battery_id, files in index.items()},
                batteries=tuple(batteries),
                watched_dirs=tuple(watched_dirs),
                signature=self._directory_signature(watched_dirs),
            )
            with self._CACHE_LOCK:
                self._SCAN_CACHE[cache_key] = entry

        return list(self._batteries)

    def load_aging_segments(self, battery_id: str) -> list[AgingSegment]:
        """Load the selected battery's Aging files efficiently.

        For the real BasyTec/platform format the absolute segment ordering is
        derived from the single ``TestInfo.dateStart`` value per file. The
        large ``TestData.Time_1`` string array is intentionally ignored.

        Only the channels needed by the operating-data dashboard are copied
        into ``AgingSegment``: ``Time_s``, ``I_A``, ``U_V`` and ``T1`` plus
        optional humidity / acceleration channels when present. Unrelated
        TestData arrays are discarded immediately after parsing.
        """
        self._ensure_scanned()

        parsed: list[tuple[float | None, AgingSegment]] = []

        for info in self._files(battery_id, "AGING"):
            raw = self._load_json_or_none(info.path)
            if raw is None:
                continue

            try:
                if isinstance(raw.get("TestData"), dict):
                    absolute_start_s, segment = self._parse_basytec_aging_fast(
                        battery_id, info, raw
                    )
                else:
                    absolute_start_s, segment = self._parse_normalized_aging(
                        battery_id, info, raw
                    )
            except (TypeError, ValueError, KeyError):
                if self.skip_invalid_json:
                    continue
                raise

            if segment is not None:
                parsed.append((absolute_start_s, segment))

        if not parsed:
            return []

        # Use one dateStart scalar per file as absolute anchor. This avoids
        # parsing/sorting the Time_1 timestamp array completely.
        absolute_starts = [start for start, _ in parsed if start is not None]
        origin_s = min(absolute_starts) if absolute_starts else None

        segments: list[AgingSegment] = []
        for absolute_start_s, segment in parsed:
            if absolute_start_s is not None and origin_s is not None:
                segment.start_time_s = float(absolute_start_s - origin_s)
            segments.append(segment)

        return sorted(
            segments,
            key=lambda item: (
                item.start_time_s,
                item.metadata.get("source_file", ""),
            ),
        )

    def _parse_basytec_aging_fast(
        self,
        battery_id: str,
        info: DataFileInfo,
        raw: dict[str, Any],
    ) -> tuple[float | None, AgingSegment | None]:
        """Parse only the relevant BasyTec channels after one JSON decode."""
        test_data = _mapping(raw.get("TestData"), f"{info.path.name}: TestData")
        test_info = _optional_mapping(raw.get("TestInfo"))
        battery_info = _optional_mapping(test_info.get("BatteryInfo"))

        # Time_s is the only timeline array needed. Time_1 is intentionally not
        # touched because dateStart is sufficient for file ordering/stitching.
        time_s_raw = test_data.get("Time_s")
        if not isinstance(time_s_raw, list) or not time_s_raw:
            raise ValueError(f"{info.path.name}: TestData.Time_s missing or empty")
        local_time_s = _numeric_list_fast(
            time_s_raw, f"{info.path.name}: TestData.Time_s"
        )
        n = len(local_time_s)

        start_iso = test_info.get("dateStart")
        absolute_start_s = None
        if isinstance(start_iso, str) and start_iso.strip():
            absolute_start_s = _parse_iso_timestamp(start_iso)

        channel_specs = {
            "voltage": ("voltage_v", ("U_V", "voltage_v")),
            "current": ("current_a", ("I_A", "current_a")),
            "temperature": ("temperature_c", ("T1", "temperature_c")),
            "humidity": ("humidity_pct", ("humidity_pct", "Humidity", "RH", "rH")),
            "acceleration": ("acceleration_g", ("acceleration_g", "Acceleration_g", "a_g")),
        }

        series: dict[str, list[float]] = {}
        available: list[str] = []
        missing: list[str] = []
        source_keys: dict[str, str] = {}

        for logical_name, (target_name, aliases) in channel_specs.items():
            values = None
            source_key = None
            for key in aliases:
                candidate = test_data.get(key)
                if candidate is not None:
                    values = candidate
                    source_key = key
                    break

            if values is None:
                series[target_name] = [math.nan] * n
                missing.append(logical_name)
                continue

            if not isinstance(values, list) or len(values) != n:
                raise ValueError(
                    f"{info.path.name}: TestData.{source_key} has invalid length"
                )

            series[target_name] = _numeric_list_fast(
                values, f"{info.path.name}: TestData.{source_key}"
            )
            available.append(logical_name)
            source_keys[logical_name] = str(source_key)

        nominal_capacity = _optional_float_value(
            battery_info.get("CN"),
            default=_optional_float_value(test_info.get("nominal_capacity_ah"), 1.0),
        )

        metadata = {
            "source_file": str(info.path),
            "source_format": "basytec_fast",
            "time_source": "dateStart + Time_s" if absolute_start_s is not None else "Time_s",
            "available_signals": tuple(available),
            "missing_signals": tuple(missing),
            "source_signal_keys": source_keys,
            "dateStart": test_info.get("dateStart"),
            "dateEnd": test_info.get("dateEnd"),
            "testName": test_info.get("testName"),
            "testNumber": test_info.get("testNumber"),
        }

        return absolute_start_s, AgingSegment(
            battery_id=battery_id,
            start_time_s=0.0,
            local_time_s=local_time_s,
            voltage_v=series["voltage_v"],
            current_a=series["current_a"],
            temperature_c=series["temperature_c"],
            humidity_pct=series["humidity_pct"],
            acceleration_g=series["acceleration_g"],
            nominal_capacity_ah=float(nominal_capacity),
            current_positive_is_charge=True,
            metadata=metadata,
        )

    def _parse_normalized_aging(
        self,
        battery_id: str,
        info: DataFileInfo,
        raw: dict[str, Any],
    ) -> tuple[float | None, AgingSegment | None]:
        signals = _mapping(raw.get("signals"), f"{info.path.name}: signals")
        metadata = _optional_mapping(raw.get("metadata"))

        time_s = _float_list(signals.get("time_s"), f"{info.path.name}: signals.time_s")
        n = len(time_s)
        if n == 0:
            return None, None

        available: list[str] = []
        missing: list[str] = []
        series: dict[str, list[float]] = {}
        mapping = {
            "voltage": "voltage_v",
            "current": "current_a",
            "temperature": "temperature_c",
            "humidity": "humidity_pct",
            "acceleration": "acceleration_g",
        }
        for logical_name, key in mapping.items():
            values = _optional_numeric_series(signals.get(key), n, f"{info.path.name}: signals.{key}")
            if values is None:
                series[key] = [math.nan] * n
                missing.append(logical_name)
            else:
                series[key] = values
                available.append(logical_name)

        normalized_metadata = {
            **metadata,
            "source_file": str(info.path),
            "source_format": "normalized",
            "time_source": "time_s",
            "available_signals": tuple(available),
            "missing_signals": tuple(missing),
        }

        return None, AgingSegment(
            battery_id=battery_id,
            start_time_s=float(raw.get("start_time_s", 0.0)),
            local_time_s=time_s,
            voltage_v=series["voltage_v"],
            current_a=series["current_a"],
            temperature_c=series["temperature_c"],
            humidity_pct=series["humidity_pct"],
            acceleration_g=series["acceleration_g"],
            nominal_capacity_ah=float(metadata.get("nominal_capacity_ah", 1.0)),
            current_positive_is_charge=bool(metadata.get("current_positive_is_charge", True)),
            metadata=normalized_metadata,
        )

    def load_cu_data(self, battery_id: str) -> list[CUData]:
        """Load CU files for one battery.

        Supported CU layouts:

        1. Normalized dummy format with ``signals``.
        2. Full platform/BasyTec format with ``TestInfo`` + ``TestData``.
        3. Pre-evaluated CU files with top-level ``MetaData`` only.  These
           files already contain capacity and ``R_DC`` results and therefore
           do not need raw time-series channels for the first RPT stage.

        Empty/missing metadata values are preserved as missing values rather
        than converted to zero.
        """
        self._ensure_scanned()
        result: list[CUData] = []

        for info in self._files(battery_id, "CU"):
            raw = self._load_json_or_none(info.path)
            if raw is None:
                continue

            try:
                if isinstance(raw.get("TestData"), dict):
                    item = self._parse_basytec_cu(battery_id, info, raw)
                elif isinstance(raw.get("MetaData"), dict):
                    item = self._parse_precomputed_cu_metadata(battery_id, info, raw)
                else:
                    item = self._parse_normalized_cu(battery_id, info, raw)
            except (TypeError, ValueError, KeyError):
                if self.skip_invalid_json:
                    continue
                raise

            if item is not None:
                result.append(item)

        return sorted(
            result,
            key=lambda item: (
                _optional_float_value(item.metadata.get("date_start_epoch_s"), float("inf")),
                item.rpt_index,
                item.source_file,
            ),
        )

    def _parse_precomputed_cu_metadata(
        self,
        battery_id: str,
        info: DataFileInfo,
        raw: dict[str, Any],
    ) -> CUData:
        """Parse metadata-only CU exports.

        Pre-evaluated capacity/R_DC values are retained together with the
        pOCV charge/discharge curves required by the optional pyDMA analysis.
        The pOCV arrays are referenced directly from the decoded JSON and are
        not duplicated a second time.
        """
        meta = _mapping(raw.get("MetaData"), f"{info.path.name}: MetaData")

        date_start = _date_value_to_iso(raw.get("dateStart"))
        date_end = _date_value_to_iso(raw.get("dateEnd"))
        start_epoch = None
        if date_start:
            try:
                start_epoch = _parse_iso_timestamp(date_start)
            except (TypeError, ValueError):
                start_epoch = None

        capacity_candidates = {
            "Capacity_CCCV_Ah": _finite_float_or_none(meta.get("Capacity_CCCV_Ah")),
            "Capacity_Ah": _finite_float_or_none(meta.get("Capacity_Ah")),
        }

        r_dc = meta.get("R_DC")
        if not isinstance(r_dc, dict):
            r_dc = {}

        pocv_charge = _extract_pocv_metadata(meta.get("pOCV_charge"))
        pocv_discharge = _extract_pocv_metadata(meta.get("pOCV_discharge"))

        rpt_index = _infer_cu_index(raw, raw, info.path)
        efc = _first_float(raw.get("efc"), raw.get("EFC"), default=float("nan"))

        cell_id = _first_text(
            raw.get("cell_id"),
            raw.get("batterySerial"),
            raw.get("battery"),
            info.cell_id,
            default="pack",
        )

        metadata = {
            "source_format": "precomputed_cu_metadata",
            "source_file": str(info.path),
            "dateStart": date_start,
            "dateEnd": date_end,
            "date_start_epoch_s": start_epoch,
            "testName": raw.get("testName"),
            "testNumber": raw.get("testNumber"),
            "testPlan": raw.get("testPlan"),
            "capacity_candidates": capacity_candidates,
            "charge_throughput_ah": _finite_float_or_none(meta.get("charge_throughput_Ah")),
            "r_dc": r_dc,
            "pocv_charge": pocv_charge,
            "pocv_discharge": pocv_discharge,
        }

        # Raw measurement arrays are intentionally empty: this file type is an
        # already-evaluated CU export.  The CU analysis consumes the metadata
        # directly and only falls back to time-series calculations when raw
        # channels actually exist.
        return CUData(
            battery_id=battery_id,
            cell_id=cell_id,
            rpt_index=rpt_index,
            efc=efc,
            time_s=[],
            voltage_v=[],
            current_a=[],
            nominal_capacity_ah=1.0,
            current_positive_is_charge=True,
            metadata=metadata,
            source_file=str(info.path),
        )

    def _parse_basytec_cu(
        self,
        battery_id: str,
        info: DataFileInfo,
        raw: dict[str, Any],
    ) -> CUData | None:
        test_data = _mapping(raw.get("TestData"), f"{info.path.name}: TestData")
        test_info = _optional_mapping(raw.get("TestInfo"))
        battery_info = _optional_mapping(test_info.get("BatteryInfo"))

        time_raw = test_data.get("Time_s")
        current_raw = test_data.get("I_A")
        voltage_raw = test_data.get("U_V")
        if not isinstance(time_raw, list) or not isinstance(current_raw, list) or not isinstance(voltage_raw, list):
            raise ValueError(
                f"{info.path.name}: TestData must contain Time_s, I_A and U_V"
            )

        time_s = _numeric_list_fast(time_raw, f"{info.path.name}: TestData.Time_s")
        n = len(time_s)
        if n == 0:
            return None
        if len(current_raw) != n or len(voltage_raw) != n:
            raise ValueError(f"{info.path.name}: Time_s, I_A and U_V must have equal length")

        current_a = _numeric_list_fast(current_raw, f"{info.path.name}: TestData.I_A")
        voltage_v = _numeric_list_fast(voltage_raw, f"{info.path.name}: TestData.U_V")

        # Keep only auxiliary CU channels required for capacity / pulse-R analysis.
        aux_specs = {
            "line": "Line",
            "ah": "Ah",
            "r_ohm": "R_Ohm",
            "delta_i_pulse_a": "dIpulse_A",
            "dt_pulse_s": "dtpulse_s",
            "t_step_s": "t_Step_s",
        }
        aux: dict[str, list[float] | None] = {}
        for target, source in aux_specs.items():
            value = test_data.get(source)
            if value is None:
                aux[target] = None
                continue
            if not isinstance(value, list) or len(value) != n:
                raise ValueError(
                    f"{info.path.name}: TestData.{source} has invalid length"
                )
            aux[target] = _numeric_list_fast(value, f"{info.path.name}: TestData.{source}")

        start_iso = _date_value_to_iso(test_info.get("dateStart"))
        start_epoch = None
        if start_iso:
            try:
                start_epoch = _parse_iso_timestamp(start_iso)
            except (TypeError, ValueError):
                start_epoch = None

        rpt_index = _infer_cu_index(raw, test_info, info.path)
        efc = _first_float(
            raw.get("efc"),
            test_info.get("efc"),
            test_info.get("EFC"),
            default=float("nan"),
        )

        cell_id = _first_text(
            raw.get("cell_id"),
            test_info.get("cell_id"),
            battery_info.get("cell_id"),
            battery_info.get("identifier"),
            battery_info.get("batterySerial"),
            default="pack",
        )

        nominal_capacity = _first_float(
            battery_info.get("CN"),
            test_info.get("nominal_capacity_ah"),
            default=1.0,
        )

        test_table = test_info.get("testTable")
        if not isinstance(test_table, list):
            test_table = []

        top_meta = _optional_mapping(raw.get("MetaData"))
        pocv_charge = _extract_pocv_metadata(top_meta.get("pOCV_charge"))
        pocv_discharge = _extract_pocv_metadata(top_meta.get("pOCV_discharge"))

        metadata = {
            "source_format": "basytec_cu",
            "source_file": str(info.path),
            "dateStart": start_iso,
            "dateEnd": _date_value_to_iso(test_info.get("dateEnd")),
            "date_start_epoch_s": start_epoch,
            "testName": test_info.get("testName"),
            "testNumber": test_info.get("testNumber"),
            "testPlan": test_info.get("testPlan"),
            "test_table": test_table,
            "pocv_charge": pocv_charge,
            "pocv_discharge": pocv_discharge,
            **aux,
        }

        return CUData(
            battery_id=battery_id,
            cell_id=cell_id,
            rpt_index=rpt_index,
            efc=efc,
            time_s=time_s,
            voltage_v=voltage_v,
            current_a=current_a,
            nominal_capacity_ah=float(nominal_capacity),
            current_positive_is_charge=True,
            metadata=metadata,
            source_file=str(info.path),
        )

    def _parse_normalized_cu(
        self,
        battery_id: str,
        info: DataFileInfo,
        raw: dict[str, Any],
    ) -> CUData | None:
        signals = _mapping(raw.get("signals"), f"{info.path.name}: signals")
        metadata = _optional_mapping(raw.get("metadata"))

        time_s = _float_list(signals.get("time_s"), f"{info.path.name}: signals.time_s")
        n = len(time_s)
        if n == 0:
            return None

        return CUData(
            battery_id=battery_id,
            cell_id=str(raw.get("cell_id", info.cell_id or "cell_1")),
            rpt_index=int(raw.get("rpt_index", _infer_cu_index(raw, metadata, info.path))),
            efc=float(raw.get("efc", metadata.get("efc", float("nan")))),
            time_s=time_s,
            voltage_v=_required_series(signals, "voltage_v", n, info.path.name),
            current_a=_required_series(signals, "current_a", n, info.path.name),
            nominal_capacity_ah=float(metadata.get("nominal_capacity_ah", 1.0)),
            current_positive_is_charge=bool(metadata.get("current_positive_is_charge", True)),
            metadata={**metadata, "source_format": "normalized"},
            source_file=str(info.path),
        )

    def load_geis_data(self, battery_id: str) -> list[GEISData]:
        """Fully load only the selected battery's GEIS JSON files.

        Supported formats:
        1) normalized dummy data with ``eis`` block
        2) platform raw format with ``TestInfo`` / ``TestData``
        3) summary-only checkup format with characteristic GEIS points
        """
        self._ensure_scanned()
        result: list[GEISData] = []

        for info in self._files(battery_id, "GEIS"):
            raw = self._load_json_or_none(info.path)
            if raw is None:
                continue

            metadata: dict[str, Any] = {}
            cell_id = str(info.cell_id or "cell_1")
            rpt_index = 0
            efc = 0.0
            frequency_hz: list[float] = []
            z_real_ohm: list[float] = []
            z_imag_ohm: list[float] = []

            if isinstance(raw.get("eis"), dict):
                eis = _mapping(raw.get("eis"), f"{info.path.name}: eis")
                frequency_hz = _float_list(eis.get("frequency_hz"), f"{info.path.name}: eis.frequency_hz")
                n = len(frequency_hz)
                if n == 0:
                    continue
                z_real_ohm = _float_list(eis.get("z_real_ohm"), f"{info.path.name}: eis.z_real_ohm")
                z_imag_ohm = _float_list(eis.get("z_imag_ohm"), f"{info.path.name}: eis.z_imag_ohm")
                if len(z_real_ohm) != n or len(z_imag_ohm) != n:
                    raise ValueError(f"{info.path.name}: EIS arrays must have equal length")

                metadata = _optional_mapping(raw.get("metadata"))
                metadata["source_format"] = "normalized"
                cell_id = str(raw.get("cell_id", info.cell_id or "cell_1"))
                rpt_index = int(raw.get("rpt_index", _infer_geis_index(raw, metadata, info.path)))
                efc = float(raw.get("efc", metadata.get("efc", 0.0)))

            elif _has_geis_root_arrays(raw):
                frequency_hz, z_real_ohm, z_imag_ohm, format_meta = _extract_geis_root_arrays(
                    raw, info.path.name
                )
                if not frequency_hz:
                    continue

                metadata = {**format_meta, "source_format": "root_arrays"}
                cell_id = _first_text(raw.get("cell_id"), info.cell_id, default="cell_1")
                rpt_index = _infer_geis_index(raw, raw, info.path)
                efc = _first_float(raw.get("efc"), default=0.0)

            elif isinstance(raw.get("TestData"), dict) or isinstance(raw.get("TestInfo"), dict):
                test_info = _optional_mapping(raw.get("TestInfo"))
                test_data = _mapping(raw.get("TestData"), f"{info.path.name}: TestData")

                frequency_hz, z_real_ohm, z_imag_ohm, format_meta = _extract_geis_testdata_arrays(test_data, info.path.name)
                if not frequency_hz:
                    continue

                metadata = {**test_info, **format_meta, "source_format": "testdata_raw"}
                cell_id = _first_text(raw.get("cell_id"), test_info.get("cell_id"), info.cell_id, default="cell_1")
                rpt_index = _infer_geis_index(raw, test_info, info.path)
                efc = _first_float(raw.get("efc"), test_info.get("efc"), default=0.0)

            else:
                frequency_hz, z_real_ohm, z_imag_ohm, summary_meta = _extract_geis_summary_arrays(raw)
                if not frequency_hz:
                    continue
                metadata = {**summary_meta, "source_format": "summary_only"}
                cell_id = _first_text(raw.get("cell_id"), info.cell_id, default="cell_1")
                rpt_index = _infer_geis_index(raw, raw, info.path)
                efc = _first_float(raw.get("efc"), default=0.0)

            result.append(
                GEISData(
                    battery_id=battery_id,
                    cell_id=cell_id,
                    rpt_index=int(rpt_index),
                    efc=float(efc),
                    frequency_hz=frequency_hz,
                    z_real_ohm=z_real_ohm,
                    z_imag_ohm=z_imag_ohm,
                    metadata=metadata,
                    source_file=str(info.path),
                )
            )

        return sorted(result, key=lambda item: ((item.rpt_index if item.rpt_index > 0 else 10**9), item.efc, item.cell_id, item.source_file))

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------

    def get_file_tree(self) -> dict[str, dict[str, list[Path]]]:
        """Return the discovered file tree without loading measurement data."""
        self._ensure_scanned()
        tree: dict[str, dict[str, list[Path]]] = {}

        for battery_id, files in sorted(self._index.items()):
            tree[battery_id] = {
                "AGING": sorted(
                    (info.path for info in files if info.file_type == "AGING"),
                    key=lambda path: str(path).lower(),
                ),
                "CU": sorted(
                    (info.path for info in files if info.file_type == "CU"),
                    key=lambda path: str(path).lower(),
                ),
                "GEIS": sorted(
                    (info.path for info in files if info.file_type == "GEIS"),
                    key=lambda path: str(path).lower(),
                ),
            }

        return tree

    @classmethod
    def clear_scan_cache(cls) -> None:
        with cls._CACHE_LOCK:
            cls._SCAN_CACHE.clear()

    # ------------------------------------------------------------------
    # FAST INDEXING -- NO json.load() BELOW THIS LINE
    # ------------------------------------------------------------------

    def _build_path_index(
        self,
    ) -> tuple[dict[str, list[DataFileInfo]], list[Path]]:
        """Build a path-only index.

        This method performs directory listing only.  If ``discover_cell_ids``
        is enabled, it additionally reads a small text prefix of CU/GEIS files.
        It never calls ``json.load()``.
        """
        index: dict[str, list[DataFileInfo]] = defaultdict(list)
        watched_dirs: list[Path] = [self.root]

        with os.scandir(self.root) as entries:
            battery_entries = sorted(
                (
                    entry
                    for entry in entries
                    if entry.is_dir(follow_symlinks=False)
                ),
                key=lambda entry: entry.name.lower(),
            )

        for battery_entry in battery_entries:
            battery_id = battery_entry.name
            battery_dir = Path(battery_entry.path)
            watched_dirs.append(battery_dir)

            category_dirs = self._find_category_dirs(battery_dir)

            for folder_key, category_dir in category_dirs.items():
                file_type = self._FOLDER_TYPES[folder_key]
                watched_dirs.append(category_dir)

                paths, nested_dirs = self._list_json_paths(category_dir)
                watched_dirs.extend(nested_dirs)

                for path in paths:
                    # Absolutely no _safe_load / _require_load / json.load here.
                    cell_id: str | None = None
                    if self.discover_cell_ids and file_type in {"CU", "GEIS"}:
                        cell_id = self._probe_cell_id(path)

                    index[battery_id].append(
                        DataFileInfo(
                            path=path,
                            battery_id=battery_id,
                            file_type=file_type,
                            cell_id=cell_id,
                        )
                    )

            # Keep batteries visible even if one/all category folders are empty.
            index.setdefault(battery_id, [])

        return index, _deduplicate_paths(watched_dirs)

    def _find_category_dirs(self, battery_dir: Path) -> dict[str, Path]:
        """Find Aging/Ageing/CU/GEIS case-insensitively."""
        found: dict[str, Path] = {}
        try:
            with os.scandir(battery_dir) as entries:
                for entry in entries:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    key = entry.name.lower()
                    if key in self._FOLDER_TYPES:
                        found[key] = Path(entry.path)
        except FileNotFoundError:
            return {}
        return found

    def _list_json_paths(self, category_dir: Path) -> tuple[list[Path], list[Path]]:
        if not self.recursive:
            try:
                with os.scandir(category_dir) as entries:
                    paths = [
                        Path(entry.path)
                        for entry in entries
                        if entry.is_file(follow_symlinks=False)
                        and entry.name.lower().endswith(".json")
                    ]
            except FileNotFoundError:
                return [], []

            paths.sort(key=lambda path: path.name.lower())
            return paths, []

        paths: list[Path] = []
        nested_dirs: list[Path] = []
        for current_root, dirs, files in os.walk(category_dir):
            current = Path(current_root)
            if current != category_dir:
                nested_dirs.append(current)
            dirs.sort(key=str.lower)
            for name in sorted(files, key=str.lower):
                if name.lower().endswith(".json"):
                    paths.append(current / name)

        return paths, nested_dirs

    def _build_battery_info(
        self,
        index: dict[str, list[DataFileInfo]],
    ) -> list[BatteryInfo]:
        result: list[BatteryInfo] = []

        for battery_id in sorted(index, key=str.lower):
            files = index[battery_id]
            cell_ids = sorted({info.cell_id for info in files if info.cell_id})

            result.append(
                BatteryInfo(
                    battery_id=battery_id,
                    aging_files=sum(info.file_type == "AGING" for info in files),
                    cu_files=sum(info.file_type == "CU" for info in files),
                    geis_files=sum(info.file_type == "GEIS" for info in files),
                    cell_ids=tuple(cell_ids),
                )
            )

        return result

    def _probe_cell_id(self, path: Path) -> str | None:
        """Extract ``cell_id`` from a small text prefix; does not parse JSON."""
        if self.metadata_probe_bytes <= 0:
            return None

        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read(self.metadata_probe_bytes)
        except OSError:
            return None

        match = self._CELL_ID_RE.search(text)
        if match is None:
            return None

        try:
            return str(json.loads(f'"{match.group(1)}"'))
        except (json.JSONDecodeError, TypeError):
            return match.group(1)

    # ------------------------------------------------------------------
    # Scan cache
    # ------------------------------------------------------------------

    def _cache_key(self) -> tuple[str, bool, bool, int]:
        try:
            root_key = str(self.root.resolve())
        except OSError:
            root_key = str(self.root.absolute())

        return (
            os.path.normcase(root_key),
            self.recursive,
            self.discover_cell_ids,
            self.metadata_probe_bytes,
        )

    def _cache_is_current(self, entry: _ScanCacheEntry) -> bool:
        try:
            return self._directory_signature(entry.watched_dirs) == entry.signature
        except OSError:
            return False

    @staticmethod
    def _directory_signature(
        directories: Iterable[Path],
    ) -> tuple[tuple[str, int], ...]:
        signature: list[tuple[str, int]] = []

        for directory in directories:
            try:
                mtime_ns = directory.stat().st_mtime_ns
            except OSError:
                mtime_ns = -1

            signature.append((os.path.normcase(str(directory)), mtime_ns))

        return tuple(signature)

    def _restore_cache(self, entry: _ScanCacheEntry) -> None:
        self._index = defaultdict(
            list,
            {
                battery_id: list(files)
                for battery_id, files in entry.index.items()
            },
        )
        self._batteries = list(entry.batteries)

    # ------------------------------------------------------------------
    # Full JSON loading -- ONLY called by load_* methods
    # ------------------------------------------------------------------

    def _files(self, battery_id: str, file_type: str) -> list[DataFileInfo]:
        if battery_id not in self._index:
            available = ", ".join(sorted(self._index)) or "<none>"
            raise KeyError(
                f"Unknown battery_id {battery_id!r}. "
                f"Available batteries: {available}"
            )

        return [
            info
            for info in self._index[battery_id]
            if info.file_type == file_type
        ]

    def _ensure_scanned(self) -> None:
        if not self._has_scanned:
            self.scan()

    def _validate_root(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Data root is not a directory: {self.root}")

    def _load_json_or_none(self, path: Path) -> dict[str, Any] | None:
        try:
            return self._require_load(path)
        except (OSError, json.JSONDecodeError, ValueError):
            if self.skip_invalid_json:
                return None
            raise

    @staticmethod
    def _require_load(path: Path) -> dict[str, Any]:
        """Decode one selected JSON file, preferring ``orjson`` when installed.

        ``orjson`` is optional; the class remains functional with the standard
        library parser. On large platform JSONs, orjson is typically several
        times faster.
        """
        if orjson is not None:
            with path.open("rb") as handle:
                raw = orjson.loads(handle.read())
        else:
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)

        if not isinstance(raw, dict):
            raise ValueError(f"{path.name}: JSON root must be an object")

        return raw


# ----------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------


def _extract_pocv_metadata(value: Any) -> dict[str, list[Any]] | None:
    """Return the minimal pOCV payload required by pyDMA.

    Optional/malformed pOCV blocks do not invalidate the whole CU file. Only
    ``AhStep_Ah`` and ``U_V`` are retained because this is the input pair used
    by the existing pyDMA notebook workflow.
    """
    if not isinstance(value, dict):
        return None
    capacity = value.get("AhStep_Ah")
    voltage = value.get("U_V")
    if not isinstance(capacity, list) or not isinstance(voltage, list):
        return None
    if not capacity or len(capacity) != len(voltage):
        return None
    return {"capacity_ah": capacity, "voltage_v": voltage}


def _date_value_to_iso(value: Any) -> str | None:
    """Return an ISO timestamp from plain or Mongo Extended JSON dates."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        inner = value.get("$date")
        if isinstance(inner, str) and inner.strip():
            return inner.strip()
    return None


def _finite_float_or_none(value: Any) -> float | None:
    """Convert a scalar to float while treating null/empty/NaN as missing."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_text(*values: Any, default: str = "") -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _first_float(*values: Any, default: float = 0.0) -> float:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def _infer_cu_index(raw: dict[str, Any], info: dict[str, Any], path: Path) -> int:
    # Prefer the CU number encoded in the file/test name. BasyTec ``testNumber``
    # is a test-system counter and is not the CU/RPT index.
    candidates = [path.stem, str(info.get("testName", ""))]
    patterns = (r"(?i)(?:^|[_\-])CU[_\-]?(\d+)(?:$|[_\-])", r"(?i)CU(\d+)")
    for text in candidates:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))

    for value in (raw.get("rpt_index"), info.get("rpt_index")):
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    return 0

def _infer_geis_index(raw: dict[str, Any], info: dict[str, Any], path: Path) -> int:
    candidates: list[str] = [path.stem, str(info.get("testName", ""))]
    filename_value = raw.get("Filename")
    if isinstance(filename_value, list):
        candidates.extend(str(item) for item in filename_value if item is not None)
    elif filename_value is not None:
        candidates.append(str(filename_value))

    patterns = (r"(?i)(?:^|[_\-])GEIS[_\-]?(\d+)(?:$|[_\-])", r"(?i)GEIS(\d+)")
    for text in candidates:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))

    for value in (raw.get("rpt_index"), info.get("rpt_index"), info.get("testNumber")):
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    return 0



def _infer_series_length(*values: Any) -> int:
    for value in values:
        if isinstance(value, list) and value:
            return len(value)
    return 0


def _parse_iso_timestamp(value: str) -> float:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _timestamp_list_to_epoch_seconds(value: Any, path: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    try:
        return [_parse_iso_timestamp(str(item)) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} contains invalid timestamps") from exc


def _numeric_list_fast(value: list[Any], path: str) -> list[float]:
    """Convert a platform numeric array with minimal copying.

    JSON decoders already return numbers as int/float. If the array contains no
    null values, the original list is reused directly. This avoids another
    full-size Python list allocation for each signal.
    """
    has_none = False
    for item in value:
        if item is None:
            has_none = True
        elif not isinstance(item, (int, float)):
            raise ValueError(f"{path} contains non-numeric values")
    if not has_none:
        return value  # type: ignore[return-value]
    return [math.nan if item is None else float(item) for item in value]


def _optional_numeric_series(value: Any, n: int, path: str) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list when present")
    if len(value) != n:
        raise ValueError(f"{path} has {len(value)} values, expected {n}")
    result: list[float] = []
    for item in value:
        if item is None:
            result.append(math.nan)
        else:
            try:
                result.append(float(item))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path} contains non-numeric values") from exc
    return result


def _numeric_series_or_none(value: Any, n: int, path: str) -> list[float] | None:
    return _optional_numeric_series(value, n, path)


def _first_numeric_series(
    container: dict[str, Any],
    aliases: tuple[str, ...],
    n: int,
    file_name: str,
) -> tuple[list[float] | None, str | None]:
    for key in aliases:
        if key in container and container.get(key) is not None:
            return (
                _optional_numeric_series(
                    container.get(key), n, f"{file_name}: TestData.{key}"
                ),
                key,
            )
    return None, None


def _has_geis_root_arrays(raw: dict[str, Any]) -> bool:
    return (
        _find_present_key(raw, ("f_Hz", "Freq_Hz", "frequency_hz", "Frequency_Hz")) is not None
        and _find_present_key(raw, ("ReZ_Ohm", "z_real_ohm", "Re_Z_Ohm")) is not None
        and _find_present_key(raw, ("ImZ_Ohm", "z_imag_ohm", "Im_Z_Ohm", "-ImZ_Ohm")) is not None
    )


def _extract_geis_root_arrays(
    raw: dict[str, Any],
    file_name: str,
) -> tuple[list[float], list[float], list[float], dict[str, Any]]:
    frequency_key = _find_present_key(
        raw,
        ("f_Hz", "Freq_Hz", "frequency_hz", "Frequency_Hz"),
    )
    z_real_key = _find_present_key(
        raw,
        ("ReZ_Ohm", "z_real_ohm", "Re_Z_Ohm"),
    )
    z_imag_key = _find_present_key(
        raw,
        ("ImZ_Ohm", "z_imag_ohm", "Im_Z_Ohm", "-ImZ_Ohm"),
    )
    if frequency_key is None or z_real_key is None or z_imag_key is None:
        return [], [], [], {}

    frequency_hz = _float_list(raw.get(frequency_key), f"{file_name}: {frequency_key}")
    z_real_ohm = _float_list(raw.get(z_real_key), f"{file_name}: {z_real_key}")
    z_imag_ohm = _float_list(raw.get(z_imag_key), f"{file_name}: {z_imag_key}")

    n = len(frequency_hz)
    if n == 0:
        return [], [], [], {}
    if len(z_real_ohm) != n or len(z_imag_ohm) != n:
        raise ValueError(f"{file_name}: GEIS arrays must have equal length")

    # If the source already stores -Im(Z), convert it back to Im(Z).
    if z_imag_key.casefold() == "-imz_ohm":
        z_imag_ohm = [-value for value in z_imag_ohm]

    return frequency_hz, z_real_ohm, z_imag_ohm, {
        "frequency_key": frequency_key,
        "z_real_key": z_real_key,
        "z_imag_key": z_imag_key,
    }


def _extract_geis_testdata_arrays(
    test_data: dict[str, Any],
    file_name: str,
) -> tuple[list[float], list[float], list[float], dict[str, Any]]:
    frequency_key = _find_present_key(
        test_data,
        ("frequency_hz", "Frequency_Hz", "Freq_Hz", "f_Hz", "f_Hz_vec"),
    )
    if frequency_key is None:
        return [], [], [], {}

    frequency_hz = _float_list(test_data.get(frequency_key), f"{file_name}: TestData.{frequency_key}")
    n = len(frequency_hz)
    if n == 0:
        return [], [], [], {}

    z_real_key = _find_present_key(
        test_data,
        ("z_real_ohm", "ReZ_Ohm", "Re_Z_Ohm", "realZ_Ohm", "Real_Z_Ohm"),
    )
    if z_real_key is None:
        return [], [], [], {}
    z_real_ohm = _float_list(test_data.get(z_real_key), f"{file_name}: TestData.{z_real_key}")

    z_imag_key = _find_present_key(
        test_data,
        ("z_imag_ohm", "ImZ_Ohm", "Im_Z_Ohm", "imagZ_Ohm", "Imag_Z_Ohm", "-ImZ_Ohm", "MinusImZ_Ohm"),
    )
    if z_imag_key is None:
        return [], [], [], {}
    z_imag_ohm = _float_list(test_data.get(z_imag_key), f"{file_name}: TestData.{z_imag_key}")
    if z_imag_key.casefold() in {"-imz_ohm", "minusimz_ohm"}:
        z_imag_ohm = [-value for value in z_imag_ohm]

    if len(z_real_ohm) != n or len(z_imag_ohm) != n:
        raise ValueError(f"{file_name}: GEIS arrays must have equal length")

    return frequency_hz, z_real_ohm, z_imag_ohm, {
        "frequency_key": frequency_key,
        "z_real_key": z_real_key,
        "z_imag_key": z_imag_key,
    }


def _extract_geis_summary_arrays(raw: dict[str, Any]) -> tuple[list[float], list[float], list[float], dict[str, Any]]:
    point_specs = [
        ("ReZmin", "f_ReZmin_Hz", "ReZ_ReZmin_Ohm", "ImZ_ReZmin_Ohm"),
        ("1kHz", "f_1kHz_Hz", "ReZ_1kHz_Ohm", "ImZ_1kHz_Ohm"),
        ("zero_crossing", "f_zc_Hz", "ReZ_zc_Ohm", "ImZ_zc_Ohm"),
        ("localmax", "f_localmax_Hz", "ReZ_localmax_Ohm", "ImZ_localmax_Ohm"),
        ("localmin", "f_localmin_Hz", "ReZ_localmin_Ohm", "ImZ_localmin_Ohm"),
    ]

    points: list[tuple[float, float, float, str]] = []
    for label, f_key, re_key, im_key in point_specs:
        f_val = _finite_float_or_none(raw.get(f_key))
        re_val = _finite_float_or_none(raw.get(re_key))
        im_val = _finite_float_or_none(raw.get(im_key))
        if f_val is None or re_val is None or im_val is None:
            continue
        points.append((f_val, re_val, im_val, label))

    if not points:
        return [], [], [], {}

    points.sort(key=lambda item: item[0], reverse=True)
    return (
        [item[0] for item in points],
        [item[1] for item in points],
        [item[2] for item in points],
        {
            "summary_point_labels": [item[3] for item in points],
            "summary_filename": raw.get("Filename"),
        },
    )


def _find_present_key(container: dict[str, Any], aliases: tuple[str, ...]) -> str | None:
    for key in aliases:
        if key in container and container.get(key) is not None:
            return key
    return None


def _optional_float_value(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _required_series(
    signals: dict[str, Any],
    key: str,
    n: int,
    file_name: str,
) -> list[float]:
    values = _float_list(signals.get(key), f"{file_name}: signals.{key}")
    if len(values) != n:
        raise ValueError(
            f"{file_name}: signals.{key} has {len(values)} values, expected {n}"
        )
    return values


def _float_list(value: Any, path: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")

    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} contains non-numeric values") from exc


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _optional_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _deduplicate_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []

    for path in paths:
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            result.append(path)

    return result
