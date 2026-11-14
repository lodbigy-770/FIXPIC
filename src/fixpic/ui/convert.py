"""המרות בין Qt ל-numpy (הדבקה מהלוח)."""
from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage

from ..errors import InvalidInputError


def qimage_to_array(img: QImage) -> np.ndarray:
    """QImage -> uint8 [H,W,4] RGBA (עותק עצמאי, בלי ריפוד שורות). QImage ריקה = InvalidInputError."""
    if img.isNull() or img.width() == 0 or img.height() == 0:
        raise InvalidInputError("אין תמונה")
    rgba = img.convertToFormat(QImage.Format_RGBA8888)
    w, h = rgba.width(), rgba.height()
    buf = np.frombuffer(rgba.constBits(), np.uint8, count=rgba.sizeInBytes())
    return buf.reshape(h, rgba.bytesPerLine())[:, :w * 4].reshape(h, w, 4).copy()
