from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import AgingSegment, BatteryInfo, CUData, GEISData


class BatteryDataSource(ABC):
    @abstractmethod
    def scan(self) -> list[BatteryInfo]:
        """Find all batteries and classify their files."""

    @abstractmethod
    def load_aging_segments(self, battery_id: str) -> list[AgingSegment]:
        """Load all operating/Aging segments for one battery."""

    @abstractmethod
    def load_cu_data(self, battery_id: str) -> list[CUData]:
        """Load all CU RPTs for one battery."""

    @abstractmethod
    def load_geis_data(self, battery_id: str) -> list[GEISData]:
        """Load all GEIS RPTs for one battery."""
