from pathlib import Path

from src.analysis.config import load_json_config
from src.analysis.cu import analyze_cu_record
from src.analysis.drt import calculate_drt
from src.analysis.prediction import analyze_fischer_prediction
from src.datasource.local_folder_tree import LocalFolderTreeDataSource


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
CONFIG = load_json_config(ROOT / "config" / "rpt_analysis.json")


def _source() -> LocalFolderTreeDataSource:
    source = LocalFolderTreeDataSource(DATA_ROOT, discover_cell_ids=False)
    source.scan()
    return source


def test_example_battery_is_discovered() -> None:
    source = _source()
    batteries = {item.battery_id: item for item in source.scan()}
    assert "BAT_EXAMPLE" in batteries
    info = batteries["BAT_EXAMPLE"]
    assert info.aging_files == 3
    assert info.cu_files == 5
    assert info.geis_files == 5


def test_example_data_can_be_loaded() -> None:
    source = _source()
    aging = source.load_aging_segments("BAT_EXAMPLE")
    cu = source.load_cu_data("BAT_EXAMPLE")
    geis = source.load_geis_data("BAT_EXAMPLE")

    assert len(aging) == 3
    assert len(cu) == 5
    assert len(geis) == 5
    assert all(len(item.frequency_hz) >= 20 for item in geis)


def test_example_rpt_drt_and_prediction_work() -> None:
    source = _source()
    cu = source.load_cu_data("BAT_EXAMPLE")
    geis = source.load_geis_data("BAT_EXAMPLE")

    cu_points = [analyze_cu_record(item, CONFIG["cu"]) for item in cu]
    assert all(point.discharge_capacity_ah is not None for point in cu_points)

    drt = calculate_drt(geis[0], CONFIG["geis"])
    assert drt.tau_s
    assert drt.gamma_ohm

    prediction = analyze_fischer_prediction(cu, cu_points, CONFIG["prediction"])
    assert prediction.ready
    assert prediction.beta1 is not None
    assert prediction.beta2 is not None
    assert prediction.estimated_eol_usage is not None
