"""Feathering פנימי: alpha ∈ [0,1] שאינו חורג מהמסכה (מחוץ למסכה alpha = 0, כלומר זהות לבייט).
alpha עולה לינארית עם המרחק מקצה המסכה עד `feather` פיקסלים. רכיב דק (קו, שריטה) מנורמל לעובי שלו,
כדי שמרכזו יוחלף במלואו ולא ישאר "רפאים" של האובייקט."""
from __future__ import annotations

import cv2
import numpy as np


def feather_alpha(mask: np.ndarray, feather: int) -> np.ndarray:
    """mask uint8 0/1 -> float32 alpha באותו גודל. feather <= 0: alpha = mask."""
    if feather <= 0:
        return mask.astype(np.float32)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    maxd = np.zeros(n, np.float32)
    np.maximum.at(maxd, labels.ravel(), dist.ravel())
    f = np.minimum(np.float32(feather), np.maximum(maxd[labels], 1.0))
    return np.clip(dist / f, 0.0, 1.0).astype(np.float32) * (mask > 0)
