"""בניית מנוע ההסרה לממשק (M3.4): טעינת המודל ובניית `JobScheduler`, לריצה ב-`BackgroundRunner`.

`default_engine_factory()` מחזיר פונקציה בלי ארגומנטים שמחזירה `JobScheduler` **שטרם הופעל** (`start` נקרא ב-thread של
ה-UI אחרי שהטעינה הצליחה). הטעינה כבדה (`read_model` כ-1.2 שניות, עוד compile ב-Warm-up), ולכן אינה רצה בבנאי החלון:
`MainWindow(engine_factory=...)` מקבל אותה רק מ-`ui.app.main`, ובדיקות מזריקות מנוע מדומה."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..cli import DEFAULT_MODELS
from ..engine import JobScheduler
from ..models import ModelManager
from ..pipeline import Pipeline, PipelineConfig

EngineFactory = Callable[[], JobScheduler]

DEFAULT_MODEL = "big-lama"
DEFAULT_THREADS = 3          # M1.5: 3 threads מותיר ליבה ל-UI (ADR-011)


def default_engine_factory(models_dir: str | Path | None = None, model: str = DEFAULT_MODEL,
                           threads: int = DEFAULT_THREADS) -> EngineFactory:
    def make() -> JobScheduler:
        loaded = ModelManager(models_dir or DEFAULT_MODELS).load(model, threads=threads)
        return JobScheduler(Pipeline(loaded, PipelineConfig()))
    return make
