from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class MondrianConfig:
    k: int
    qi_columns: Sequence[str]
    categorical_qi: Sequence[str]
    numeric_qi: Sequence[str]


@dataclass
class Partition:
    idx: List[int]
    bounds: Dict[str, Any]
