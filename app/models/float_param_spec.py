from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FloatParamSpec:
    key: str
    label: str
    minimum: float
    maximum: float
    step: float
    default: float
