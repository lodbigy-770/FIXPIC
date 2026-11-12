"""מימוש InpaintModel מעל ONNX Runtime CPU (ה-fallback לפי ADR-011).
שמות Tensors נקראים מהמודל (ADR-005), לא מקודדים."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..errors import InferenceError, ModelLoadError, OutOfMemoryError, is_oom
from ..pipeline.prepost import postprocess, preprocess
from .base import CancelToken, ModelConfig, ModelInfo

_LFS_MAGIC = b"version https://git-lfs"


def check_model_file(cfg: ModelConfig) -> None:
    """בדיקות משותפות לכל backend: קיום, LFS pointer, hash."""
    p = Path(cfg.path)
    if not p.is_file():
        raise ModelLoadError(f"קובץ המודל חסר: {p}")
    with open(p, "rb") as f:
        head = f.read(64)
    if head.startswith(_LFS_MAGIC):
        raise ModelLoadError(f"הקובץ הוא LFS pointer ולא המודל עצמו: {p}")
    if cfg.sha256:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest().lower() != cfg.sha256.lower():
            raise ModelLoadError("hash של קובץ המודל לא תואם")


class OrtModel:
    def __init__(self, info: ModelInfo | None = None) -> None:
        self.info = info or ModelInfo(name="big-lama")
        self._sess = None
        self._img_name = self._mask_name = self._out_name = ""

    def load(self, cfg: ModelConfig) -> None:
        check_model_file(cfg)
        try:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.intra_op_num_threads = cfg.threads
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._sess = ort.InferenceSession(str(cfg.path), so, providers=["CPUExecutionProvider"])
        except Exception as e:  # noqa: BLE001 - מומר לשגיאה מוגדרת
            raise ModelLoadError(f"יצירת Session נכשלה: {e}") from e
        ins = {i.name: i.shape for i in self._sess.get_inputs()}
        # זיהוי תפקיד לפי מספר הערוצים (3 = תמונה, 1 = מסכה), לא לפי שם קבוע
        try:
            self._img_name = next(n for n, s in ins.items() if len(s) == 4 and s[1] == 3)
            self._mask_name = next(n for n, s in ins.items() if len(s) == 4 and s[1] == 1)
        except StopIteration as e:
            raise ModelLoadError(f"מבנה קלט לא צפוי: {ins}") from e
        self._out_name = self._sess.get_outputs()[0].name

    def run(self, image: np.ndarray, mask: np.ndarray, cancel: CancelToken | None = None) -> np.ndarray:
        if self._sess is None:
            raise InferenceError("המודל לא נטען")
        if cancel:
            cancel.raise_if_cancelled()
        rgb, m, hw = preprocess(image, mask, self.info.multiple)
        try:
            out = self._sess.run([self._out_name], {self._img_name: rgb, self._mask_name: m})[0]
        except Exception as e:  # noqa: BLE001
            if is_oom(e):
                raise OutOfMemoryError(f"נגמר הזיכרון בהרצת המודל: {e}") from e
            raise InferenceError(f"הרצת המודל נכשלה: {e}") from e
        return postprocess(out, hw)

    def unload(self) -> None:
        self._sess = None
