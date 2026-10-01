# Third-Party Software

This repository depends on third-party open-source software. These packages are
**not relicensed** under the BSD 3-Clause License of this repository; they remain
subject to their respective upstream licenses.

## Direct runtime dependencies

| Package | License | Upstream project |
|---|---|---|
| Streamlit | Apache-2.0 | https://github.com/streamlit/streamlit |
| Plotly.py | MIT | https://github.com/plotly/plotly.py |
| NumPy | BSD-3-Clause | https://github.com/numpy/numpy |
| SciPy | BSD-3-Clause | https://github.com/scipy/scipy |
| pandas | BSD-3-Clause | https://github.com/pandas-dev/pandas |
| orjson | MPL-2.0 AND (Apache-2.0 OR MIT) | https://github.com/ijl/orjson |
| PyDMA | BSD-3-Clause | https://github.com/tum-ees/PyDMA |

## Development dependency

| Package | License | Upstream project |
|---|---|---|
| pytest | MIT | https://github.com/pytest-dev/pytest |

The authoritative license text and copyright notices are provided by each
upstream project and by the installed package distributions.

This file lists the project's **direct** dependencies from `requirements.txt`
and `requirements-dev.txt`. Transitive dependencies installed by `pip` may add
further license obligations and are not exhaustively reproduced here. For a
redistributed binary/application bundle, generate and review a license report
for the exact resolved environment used for that release.

## PyDMA citation

The degradation mode analysis functionality uses PyDMA. For scientific use,
please also follow the citation information provided by the PyDMA project:
https://github.com/tum-ees/PyDMA/blob/main/CITATION.cff
