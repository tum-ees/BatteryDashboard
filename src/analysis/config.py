from __future__ import annotations

import json
from pathlib import Path


def load_json_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
