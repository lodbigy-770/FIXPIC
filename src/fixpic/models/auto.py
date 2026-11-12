"""בחירת backend לפי ADR-011: OpenVINO ראשי, ONNX Runtime fallback. כשל (טעינה, compile או הרצה)
גורם למעבר קבוע ל-ORT בלי חריגה כלפי חוץ; ה-fallback נרשם ב-`fallback_reason`."""
from __future__ import annotations

import logging

import numpy as np

from ..errors import InferenceError, ModelLoadError, OutOfMemoryError
from .base import CancelToken, ModelConfig, ModelInfo
from .ort_model import OrtModel, check_model_file

log = logging.getLogger("fixpic.models")


class AutoModel:
    def __init__(self, info: ModelInfo | None = None, primary=None, fallback=None) -> None:
        self.info = info or ModelInfo(name="big-lama")
        self._primary = primary if primary is not None else _default_primary(self.info)
        self._fallback = fallback if fallback is not None else OrtModel(self.info)
        self._cfg: ModelConfig | None = None
        self._active = None
        self.backend = ""                  # "openvino" | "onnxruntime" | ""
        self.fallback_reason: str | None = None

    def load(self, cfg: ModelConfig) -> None:
        check_model_file(cfg)              # מודל חסר/LFS/hash: שגיאה ישירה, ללא fallback
        self._cfg = cfg
        try:
            if self._primary is None:
                raise ModelLoadError("OpenVINO לא מותקן")
            self._primary.load(cfg)
            self._active, self.backend = self._primary, "openvino"
        except ModelLoadError as e:
            self._switch(str(e))

    def _switch(self, reason: str) -> None:
        log.warning("מעבר ל-ONNX Runtime: %s", reason)
        self.fallback_reason = reason
        if self._primary is not None:
            self._primary.unload()
        self._fallback.load(self._cfg)
        self._active, self.backend = self._fallback, "onnxruntime"

    def warmup(self, sizes, cancel: CancelToken | None = None) -> None:
        if self._active is self._primary and hasattr(self._primary, "warmup"):
            try:
                self._primary.warmup(sizes, cancel)
            except OutOfMemoryError:
                raise
            except (InferenceError, ModelLoadError) as e:   # CancelledError עולה כלפי חוץ
                self._switch(str(e))

    def run(self, image: np.ndarray, mask: np.ndarray, cancel: CancelToken | None = None) -> np.ndarray:
        if self._active is None:
            raise InferenceError("המודל לא נטען")
        if self._active is self._fallback:
            return self._fallback.run(image, mask, cancel)
        try:
            return self._primary.run(image, mask, cancel)
        except OutOfMemoryError:                          # אינו כשל backend: ללא מעבר ל-ORT
            raise
        except (InferenceError, ModelLoadError) as e:     # compile או הרצה נכשלו: ללא קריסה
            self._switch(str(e))
            return self._fallback.run(image, mask, cancel)

    def unload(self) -> None:
        for m in (self._primary, self._fallback):
            if m is not None:
                m.unload()
        self._active, self.backend = None, ""


def _default_primary(info: ModelInfo):
    try:
        from .ov_model import OvModel
        import openvino  # noqa: F401
    except ImportError:
        return None
    return OvModel(info)
