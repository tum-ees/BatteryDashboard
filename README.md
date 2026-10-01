# Battery Health Dashboard

Battery health monitoring and lifetime prediction dashboard developed in the context of the **HealthBatt** research project at the Chair of Electrical Energy Storage Technology, Technical University of Munich. The Streamlit application combines an operational **traffic-light monitoring system** with analysis of battery operating and check-up data. The repository provides a file-based data source that can later be replaced by another backend without changing the analysis and UI layers.

The dashboard currently supports:

- operating-data assessment from `Aging` measurements using a configurable traffic-light status system, including historical status evaluation and time-series plots,
- CU/check-up analysis of capacity and DC resistance,
- GEIS Nyquist plots and DRT analysis,
- degradation mode analysis (DMA) with [PyDMA](https://github.com/tum-ees/pydma),
- capacity-SOH / lifetime estimation based on the implemented resistance–SOH power-law workflow.

A small synthetic example battery is included in `data/BAT_EXAMPLE` so the dashboard can be started directly after installation.

## 1. Requirements

Recommended environment:

- Python **3.12 or 3.13**
- Windows, Linux or macOS
- a recent `pip`

Python 3.12+ is recommended because the current PyDMA release used by the project requires a modern Python environment.

## 2. Installation

Clone the repository and enter the project directory:

```bash
git clone <repository-url>
cd <repository-folder>
```

Create a virtual environment.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the dashboard with:

```bash
streamlit run app.py
```

Streamlit will print the local URL in the terminal, normally `http://localhost:8501`.

For development and tests, install the additional development dependencies:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## 3. Measurement-data directory

The dashboard reads data from the repository-local `data/` directory. The **first directory level is the battery identifier**. Below each battery identifier, measurement files are separated by measurement type:

```text
data/
└── <BATTERY_ID>/
    ├── Aging/
    │   ├── *.json
    │   └── ...
    ├── CU/
    │   ├── *.json
    │   └── ...
    └── GEIS/
        ├── *.json
        └── ...
```

`Ageing/` is accepted as an alternative spelling of `Aging/`.

The folder structure is important:

- the battery ID is taken from `<BATTERY_ID>`,
- the measurement type is determined from the `Aging`, `CU` or `GEIS` folder,
- JSON files should be placed directly in the corresponding measurement-type folder,
- not every battery needs to contain all three measurement types.

The startup scan is path-based and does not load all measurement arrays. JSON files are parsed only when the corresponding analysis is selected.

### Aging data

Two forms are supported by the current data source:

1. normalized dashboard JSON containing a `signals` object, e.g. `time_s`, `voltage_v`, `current_a`, `temperature_c` and optionally humidity / acceleration;
2. BasyTec/platform-style JSON containing `TestInfo` and `TestData` with the required measurement channels.

Missing optional sensors are shown as unavailable and do not influence the overall status.

### CU / check-up data

The CU loader supports:

- normalized time-series JSON with voltage/current signals,
- BasyTec-style raw CU files,
- already evaluated metadata/check-up files containing capacity, `R_DC` and optional pOCV data.

The RPT/CU analysis can use pre-evaluated `R_DC` metadata when available and otherwise falls back to the supported raw-signal evaluation paths.

For the Fischer-oriented prediction workflow, comparable capacity and DC-resistance values must be available across several check-ups. Native EFC values are used when available; otherwise the implementation can derive an approximate usage axis from charge throughput when the required metadata are present.

### GEIS data

Full GEIS spectra are supported in several forms, including:

- normalized `eis` objects with frequency, real and imaginary impedance arrays,
- root-level arrays such as `f_Hz`, `ReZ_Ohm` and `ImZ_Ohm`,
- supported BasyTec/platform exports.

A full spectrum is required for DRT analysis. Summary-only GEIS metadata can be loaded for characteristic points but cannot replace a complete frequency spectrum for DRT inversion.

## 4. Example data

The repository includes one synthetic battery:

```text
data/BAT_EXAMPLE/
├── Aging/   # 3 operating-data segments
├── CU/      # 5 check-ups
└── GEIS/    # 5 impedance spectra
```

The example is intentionally small and is only intended to demonstrate the dashboard workflow. It contains operating data, capacity evolution, structured DC-resistance metadata and GEIS spectra suitable for Nyquist and DRT visualization.

The example can be regenerated at any time with:

```bash
python scripts/generate_example_data.py
```

Only `data/BAT_EXAMPLE` is replaced by this script. Other battery folders in `data/` are left untouched.

The `.gitignore` keeps `BAT_EXAMPLE` under version control but ignores additional local battery folders by default. This reduces the risk of accidentally committing large or confidential measurement data.

## 5. PyDMA / half-cell curves

DMA is implemented using PyDMA. The default half-cell curve locations are configured in `config/rpt_analysis.json`:

```text
resources/halfcell/anodeOCP.mat
resources/halfcell/cathodeOCP.mat
```

These paths can be changed in the configuration or in the dashboard input fields. For meaningful DMA results, the half-cell curves must match the investigated cell chemistry and the CU data must contain suitable pOCV information.

## 6. Configuration

The main configuration files are:

```text
config/
├── thresholds.json       # operating-data status limits
├── thresholds_modul.json # alternative/module thresholds
└── rpt_analysis.json     # CU, DRT, DMA and prediction settings
```

Typical settings that can be changed without modifying the analysis code include:

- operating-data thresholds,
- reference `R_DC` condition,
- DRT regularization/grid settings,
- PyDMA fitting settings and half-cell paths,
- prediction feature, EOL-SOH and forecast settings.

## 7. Repository structure

```text
.
├── app.py
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── config/
├── data/
│   └── BAT_EXAMPLE/
│       ├── Aging/
│       ├── CU/
│       └── GEIS/
├── resources/
│   └── halfcell/
├── scripts/
│   └── generate_example_data.py
├── src/
│   ├── analysis/
│   ├── datasource/
│   ├── ui/
│   ├── models.py
│   └── status.py
└── tests/
```

## 8. Data-source abstraction

The UI does not directly read individual measurement files. The current implementation uses `LocalFolderTreeDataSource`, which implements the common `BatteryDataSource` interface.

A future REST, database or platform connector can replace the local data source by implementing the same methods:

```text
scan()
load_aging_segments(battery_id)
load_cu_data(battery_id)
load_geis_data(battery_id)
```

The analysis and visualization layers can then remain unchanged.

## 9. License and citation

The source code in this repository is released under the **BSD 3-Clause License**. See [`LICENSE`](LICENSE).

The included demonstration data in `data/BAT_EXAMPLE/` and the experimentally measured half-cell electrode curves in `resources/halfcell/` are released under **CC0 1.0 Universal** and may be freely reused, modified and redistributed. See [`DATA_LICENSE.md`](DATA_LICENSE.md).

Third-party Python packages remain subject to their respective licenses. An overview of the direct dependencies is provided in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

For scientific use, citation metadata for this repository are provided in [`CITATION.cff`](CITATION.cff). The DMA functionality uses PyDMA; please also follow the citation guidance of the PyDMA project when using results obtained from this part of the dashboard.

User-provided measurement data are **not** covered by the repository's data dedication. Users remain responsible for ensuring that they have the necessary rights to process, publish or redistribute their own data.
