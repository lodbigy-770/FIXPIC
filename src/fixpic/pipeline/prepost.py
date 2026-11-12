"""Pre/Post של טנזורי LaMa, כמו ב-predict המקורי (MODEL_CARD §4, אומת ב-M1)."""
from __future__ import annotations

import cv2
import numpy as np


def preprocess(image: np.ndarray, mask: np.ndarray, multiple: int = 8):
    """uint8 [H,W,3] + uint8 [H,W] -> (rgb f32 [1,3,H',W'], mask f32 [1,1,H',W'], (h, w)).

    ריפוד ימין/תחתון בהשתקפות symmetric (BORDER_REFLECT) לכפולה של `multiple`."""
    h, w = image.shape[:2]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        image = cv2.copyMakeBorder(image, 0, ph, 0, pw, cv2.BORDER_REFLECT)
        mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    rgb = np.ascontiguousarray((image.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
    m = np.ascontiguousarray((mask > 0).astype(np.float32)[None, None])
    return rgb, m, (h, w)


def postprocess(out: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """f32 [1,3,H',W'] בטווח 0..1 -> uint8 [h,w,3]: חיתוך ריפוד, clip, truncation (לא עיגול)."""
    h, w = hw
    img = out[0].transpose(1, 2, 0)[:h, :w]
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)
