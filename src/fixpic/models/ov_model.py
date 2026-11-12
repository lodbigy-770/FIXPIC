"""backend OpenVINO CPU (ראשי לפי ADR-011): f32, LATENCY, מספר threads מוגדר, compile לכל צורה קבועה."""
from __future__ import annotations

import threading

import numpy as np

from ..errors import InferenceError, ModelLoadError
from ..pipeline.prepost import postprocess, preprocess
from .base import CancelToken, ModelConfig, ModelInfo
from .ort_model import check_model_file


class OvModel:
    def __init__(self, info: ModelInfo | None = None) -> None:
        self.info = info or ModelInfo(name="big-lama")
        self._core = None
        self._model = None
        self._cfg: ModelConfig | None = None
        self._compiled: dict[tuple[int, int], object] = {}
        self._lock = threading.Lock()
        self._img_name = self._mask_name = self._out_name = ""

    def load(self, cfg: ModelConfig) -> None:
        check_model_file(cfg)
        try:
            import openvino as ov

            self._ov = ov
            self._core = ov.Core()
            self._model = self._core.read_model(str(cfg.path))
            ins = {i.get_any_name(): list(i.get_partial_shape()) for i in self._model.inputs}
            self._img_name = next(n for n, s in ins.items() if len(s) == 4 and str(s[1]) == "3")
            self._mask_name = next(n for n, s in ins.items() if len(s) == 4 and str(s[1]) == "1")
            self._out_name = self._model.outputs[0].get_any_name()
        except Exception as e:  # noqa: BLE001 - ImportError, StopIteration, כשל קריאה
            raise ModelLoadError(f"טעינת OpenVINO נכשלה: {e}") from e
        self._cfg = cfg
        self._compiled.clear()

    def _get(self, h: int, w: int):
        key = (h, w)
        with self._lock:
            c = self._compiled.get(key)
            if c is None:
                try:
                    ov = self._ov
                    m = self._model.clone()
                    m.reshape({self._img_name: [1, 3, h, w], self._mask_name: [1, 1, h, w]})
                    c = self._core.compile_model(m, "CPU", {
                        "PERFORMANCE_HINT": "LATENCY",
                        "INFERENCE_PRECISION_HINT": ov.Type.f32,
                        "INFERENCE_NUM_THREADS": self._cfg.threads,
                        "NUM_STREAMS": 1,
                    })
                except Exception as e:  # noqa: BLE001
                    raise ModelLoadError(f"compile נכשל לצורה {h}x{w}: {e}") from e
                self._compiled[key] = c
            return c

    def warmup(self, sizes, cancel: CancelToken | None = None) -> None:
        """compile (וריצה אחת ריקה) לכל bucket; נקרא ברקע, הגודל הרגיל קודם."""
        for s in sizes:
            if cancel:
                cancel.raise_if_cancelled()
            c = self._get(s, s)
            try:
                c({self._img_name: np.zeros((1, 3, s, s), np.float32),
                   self._mask_name: np.zeros((1, 1, s, s), np.float32)})
            except Exception as e:  # noqa: BLE001
                raise InferenceError(f"Warm-up נכשל: {e}") from e

    def run(self, image: np.ndarray, mask: np.ndarray, cancel: CancelToken | None = None) -> np.ndarray:
        if self._model is None:
            raise InferenceError("המודל לא נטען")
        if cancel:
            cancel.raise_if_cancelled()
        rgb, m, hw = preprocess(image, mask, self.info.multiple)
        c = self._get(rgb.shape[2], rgb.shape[3])      # ModelLoadError אם compile נכשל
        try:
            res = c({self._img_name: rgb, self._mask_name: m})
            out = res[c.outputs[0]]
        except Exception as e:  # noqa: BLE001
            raise InferenceError(f"הרצת המודל נכשלה: {e}") from e
        return postprocess(np.asarray(out), hw)

    def unload(self) -> None:
        self._compiled.clear()
        self._model = self._core = None
