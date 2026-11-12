"""ממשק InpaintModel (ARCHITECTURE §11). ה-Pipeline מכיר רק אותו, לא backend ספציפי."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..errors import CancelledError


class CancelToken:
    """אסימון ביטול. ביטול נבדק בין שלבים; הרצת Inference שכבר התחילה אינה ניתנת להפסקה."""

    def __init__(self) -> None:
        self._ev = threading.Event()

    def cancel(self) -> None:
        self._ev.set()

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()

    def raise_if_cancelled(self) -> None:
        if self._ev.is_set():
            raise CancelledError("הפעולה בוטלה")


@dataclass(frozen=True)
class ModelInfo:
    name: str
    multiple: int = 8          # H ו-W חייבים להיות כפולה של ערך זה (MODEL_CARD §3)
    min_size: int = 64         # טווח README; לא אומת מול הקובץ (PROGRESS)
    max_size: int = 4096


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    threads: int = 3                       # ברירת מחדל מוכרעת ב-M1.5 (משאירה ליבה ל-UI)
    sha256: str | None = None              # אם הוגדר: אי-התאמה = ModelLoadError


@runtime_checkable
class InpaintModel(Protocol):
    """run: image uint8 [H,W,3] RGB, mask uint8 [H,W] (0/1, 1 = למילוי) -> uint8 [H,W,3].

    Pre/Post של הטנזורים (חלקי 255, ריפוד לכפולה של 8, חיתוך, truncation) הם באחריות ה-backend,
    דרך הפונקציות המשותפות ב-pipeline.prepost."""

    info: ModelInfo

    def load(self, cfg: ModelConfig) -> None: ...

    def run(self, image: np.ndarray, mask: np.ndarray, cancel: CancelToken | None = None) -> np.ndarray: ...

    def unload(self) -> None: ...
