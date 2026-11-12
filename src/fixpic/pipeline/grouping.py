"""קיבוץ רכיבי מסכה לקבוצות עיבוד (מסכות מרובות). רכיבים קרובים (הקשר חופף, או ששניהם נכנסים
בחלון אחד בגודל S) מעובדים יחד; רכיבים רחוקים מעובדים בנפרד, כך שחלון אחד לא נמתח על כל התמונה."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Group:
    box: tuple[int, int, int, int]   # x0, y0, x1, y1 (בלעדי)
    mask: np.ndarray                 # uint8 0/1 בגודל ה-box: רק הפיקסלים של קבוצה זו


def _margin(box, ratio, min_margin):
    return max(min_margin, math.ceil(ratio * max(box[2] - box[0], box[3] - box[1])))


def _union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _should_merge(a, b, size, ratio, min_margin):
    ma, mb = _margin(a, ratio, min_margin), _margin(b, ratio, min_margin)
    if not (a[2] + ma <= b[0] - mb or b[2] + mb <= a[0] - ma or a[3] + ma <= b[1] - mb or b[3] + mb <= a[1] - ma):
        return True                                   # חלונות ההקשר חופפים
    u = _union(a, b)
    return max(u[2] - u[0], u[3] - u[1]) + 2 * _margin(u, ratio, min_margin) <= size


def group_mask(m: np.ndarray, size: int, margin_ratio: float = 0.25, min_margin: int = 16,
               max_components: int = 256) -> list[Group]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return []
    boxes = [(int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 0] + stats[i, 2]), int(stats[i, 1] + stats[i, 3]))
             for i in range(1, n)]
    ids = [[i] for i in range(1, n)]
    if len(boxes) > max_components:                   # הגנה מ-O(n²): קבוצה אחת לכל המסכה
        boxes, ids = [_union_all(boxes)], [list(range(1, n))]
    else:
        merged = True
        while merged and len(boxes) > 1:
            merged = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if _should_merge(boxes[i], boxes[j], size, margin_ratio, min_margin):
                        boxes[i] = _union(boxes[i], boxes[j]); ids[i] += ids[j]
                        del boxes[j], ids[j]
                        merged = True
                        break
                if merged:
                    break
    order = sorted(range(len(boxes)), key=lambda k: (boxes[k][1], boxes[k][0]))
    out = []
    for k in order:
        x0, y0, x1, y1 = boxes[k]
        crop = labels[y0:y1, x0:x1]
        gm = (crop > 0) if len(ids[k]) == n - 1 else np.isin(crop, ids[k])
        out.append(Group((x0, y0, x1, y1), gm.astype(np.uint8)))
    return out


def _union_all(boxes):
    u = boxes[0]
    for b in boxes[1:]:
        u = _union(u, b)
    return u
