#!/usr/bin/env python3
"""M1.5 follow-up (session A): UI Spike - smooth Zoom/Pan on a 24MP image in PySide6.

Design under test (what the editor would use): a custom QWidget that keeps the
image as an in-memory PYRAMID (level k = 1/2^k, cv2.INTER_AREA) and in paintEvent
draws only the VISIBLE source rectangle of the best level with QPainter.drawImage.
Optional --mask draws a semi-transparent mask overlay (same pyramid) on top, as the
brush layer would.

Two modes:
  benchmark (default)  scripted zoom/pan sequence, one synchronous repaint per step;
                       prints paint time p50/p95/max per phase. Measures our paint
                       code only (not the display compositor), so also try:
  --interactive        opens the window: wheel = zoom at cursor, left-drag = pan,
                       F = fit, 1 = 100%, M = toggle mask. Title shows paint ms.
                       Judge smoothness by eye and tell Claude how it felt.

  python tools\\m1\\ui_spike.py                      synthetic 6000x4000 (24MP), benchmark
  python tools\\m1\\ui_spike.py --mask
  python tools\\m1\\ui_spike.py --interactive --mask
  python tools\\m1\\ui_spike.py --image photo.jpg    your own photo (any size)

Writes ui_spike.json (--out) in benchmark mode. Do not use the computer meanwhile.
"""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


def crop_resize(img, w, h, interp):
    """Center-crop to the target aspect, then scale to w x h (no stretching). Used for image AND mask."""
    ih, iw = img.shape[:2]
    cw, ch = (iw, int(round(iw * h / w))) if iw * h <= ih * w else (int(round(ih * w / h)), ih)
    y0, x0 = (ih - ch) // 2, (iw - cw) // 2
    return cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=interp)


def synthetic(w, h, seed_img):
    """Detailed RGB test image of w x h: the sample photo, tiled and resized, plus fine noise."""
    base = cv2.imdecode(np.fromfile(str(seed_img), dtype=np.uint8), cv2.IMREAD_COLOR) if seed_img else None
    if base is None:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        return np.stack([(np.sin(xx / 37) + 1) * 127, (np.cos(yy / 53) + 1) * 127,
                         ((xx + yy) % 255)], axis=2).astype(np.uint8)
    big = crop_resize(base, w, h, cv2.INTER_CUBIC)
    noise = np.random.default_rng(0).integers(-12, 12, big.shape, dtype=np.int16)
    return np.clip(big.astype(np.int16) + noise, 0, 255).astype(np.uint8)


class Pyramid:
    def __init__(self, bgr, alpha=False):
        self.levels, self.arrays = [], []
        fmt = QImage.Format_ARGB32_Premultiplied if alpha else QImage.Format_RGB888
        cur = bgr
        while True:
            a = np.ascontiguousarray(cur)
            self.arrays.append(a)
            self.levels.append(QImage(a.data, a.shape[1], a.shape[0], a.strides[0], fmt))
            if min(cur.shape[:2]) <= 512:
                break
            cur = cv2.resize(cur, (cur.shape[1] // 2, cur.shape[0] // 2), interpolation=cv2.INTER_AREA)
        self.w, self.h = self.levels[0].width(), self.levels[0].height()


class Viewer(QWidget):
    def __init__(self, rgb, mask_rgba=None):
        super().__init__()
        t = time.perf_counter()
        self.pyr = Pyramid(rgb)
        self.mpyr = Pyramid(mask_rgba, alpha=True) if mask_rgba is not None else None
        self.build_s = time.perf_counter() - t
        self.scale, self.off = 1.0, QPointF(0, 0)   # view = (image_point * scale) + off
        self.show_mask = self.mpyr is not None
        self.last_paint_ms, self._drag = 0.0, None
        self.setMinimumSize(300, 200)

    # --- view math
    def fit(self):
        self.scale = min(self.width() / self.pyr.w, self.height() / self.pyr.h)
        self.off = QPointF((self.width() - self.pyr.w * self.scale) / 2, (self.height() - self.pyr.h * self.scale) / 2)

    def zoom_at(self, pos, factor):
        new = min(max(self.scale * factor, 0.02), 32.0)
        k = new / self.scale
        self.off = QPointF(pos.x() - (pos.x() - self.off.x()) * k, pos.y() - (pos.y() - self.off.y()) * k)
        self.scale = new

    # --- painting: only the visible source rect of the best pyramid level
    def draw(self, p, pyr, opacity=1.0):
        lvl = 0 if self.scale >= 1 else min(int(math.floor(math.log2(1 / self.scale))), len(pyr.levels) - 1)
        f = 1 / (2 ** lvl)
        vis = QRectF(-self.off.x() / self.scale, -self.off.y() / self.scale, self.width() / self.scale,
                     self.height() / self.scale).intersected(QRectF(0, 0, pyr.w, pyr.h))
        if vis.isEmpty():
            return
        src = QRectF(vis.x() * f, vis.y() * f, vis.width() * f, vis.height() * f)
        dst = QRectF(vis.x() * self.scale + self.off.x(), vis.y() * self.scale + self.off.y(),
                     vis.width() * self.scale, vis.height() * self.scale)
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.scale < 2)
        p.setOpacity(opacity)
        p.drawImage(dst, pyr.levels[lvl], src)

    def paintEvent(self, _):
        t = time.perf_counter()
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.darkGray)
        self.draw(p, self.pyr)
        if self.show_mask and self.mpyr:
            self.draw(p, self.mpyr, 0.5)
        p.end()
        self.last_paint_ms = (time.perf_counter() - t) * 1e3
        if self._drag is not None or self.isVisible():
            self.setWindowTitle(f"UI spike - scale {self.scale:.3f}  paint {self.last_paint_ms:.1f} ms")

    # --- interaction
    def wheelEvent(self, e):
        self.zoom_at(e.position(), 1.25 if e.angleDelta().y() > 0 else 0.8)
        self.update()

    def mousePressEvent(self, e):
        self._drag = e.position()

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            self.off += e.position() - self._drag
            self._drag = e.position()
            self.update()

    def mouseReleaseEvent(self, _):
        self._drag = None

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_F:
            self.fit()
        elif e.key() == Qt.Key_1:
            self.zoom_at(QPointF(self.width() / 2, self.height() / 2), 1 / self.scale)
        elif e.key() == Qt.Key_M and self.mpyr:
            self.show_mask = not self.show_mask
        self.update()

    def resizeEvent(self, _):
        if not getattr(self, "_fitted", False):
            self.fit()
            self._fitted = True


def stats(v):
    v = sorted(v)
    return {"n": len(v), "p50_ms": round(statistics.median(v), 2), "p95_ms": round(v[int(len(v) * 0.95)], 2),
            "max_ms": round(v[-1], 2)}


def benchmark(w):
    c = QPointF(w.width() / 2, w.height() / 2)
    phases = {}

    def step(name, fn, n):
        ts = []
        for i in range(n):
            fn(i)
            t = time.perf_counter()
            w.repaint()
            ts.append((time.perf_counter() - t) * 1e3)
            QApplication.processEvents()
        phases[name] = stats(ts)

    w.fit()
    step("zoom fit -> 100% (40 steps)", lambda i: w.zoom_at(c, (1 / w.scale) ** (1 / 40) if i < 40 else 1), 40)
    w.zoom_at(c, 1 / w.scale)
    step("pan at 100% (150 moves)", lambda i: setattr(w, "off", w.off + QPointF(-35, -22)), 150)
    step("zoom 100% -> 400% (20 steps)", lambda i: w.zoom_at(c, 4 ** (1 / 20)), 20)
    step("pan at 400% (150 moves)", lambda i: setattr(w, "off", w.off + QPointF(30, 18)), 150)
    w.zoom_at(c, 0.5 / w.scale)
    step("pan at 50% (150 moves)", lambda i: setattr(w, "off", w.off + QPointF(-25, -14)), 150)
    w.fit()
    step("fast wheel zoom in/out (60 steps)", lambda i: w.zoom_at(c, 1.25 if (i // 6) % 2 == 0 else 0.8), 60)
    return phases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="optional photo; default is a synthetic 24MP image")
    ap.add_argument("--width", type=int, default=6000)
    ap.add_argument("--height", type=int, default=4000)
    ap.add_argument("--window", default="1600x900")
    ap.add_argument("--mask", action="store_true", help="draw a mask overlay pyramid on top")
    ap.add_argument("--mask-file", help="mask image (white = masked), e.g. samples\\person_mask.png; implies --mask")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--out", default="ui_spike.json")
    a = ap.parse_args()
    a.mask = a.mask or bool(a.mask_file)

    app = QApplication(sys.argv[:1])
    if a.image:
        bgr = cv2.imdecode(np.fromfile(a.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        bgr = synthetic(a.width, a.height, HERE.parent.parent / "samples" / "person.png")
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    mask = None
    mask_file = a.mask_file
    if a.mask and not mask_file and not a.image:       # default synthetic image = person.png -> its own mask
        cand = HERE.parent.parent / "samples" / "person_mask.png"
        mask_file = str(cand) if cand.exists() else None
    if mask_file:
        m = cv2.imdecode(np.fromfile(mask_file, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if not a.image:                               # same crop/scale as the synthetic image
            m = crop_resize(m, rgb.shape[1], rgb.shape[0], cv2.INTER_LINEAR)
        elif m.shape[:2] != rgb.shape[:2]:
            m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = np.zeros((*rgb.shape[:2], 4), np.uint8)
        mask[m > 127] = (0, 0, 160, 160)              # BGRA premultiplied-ish red, ~63% alpha
        print(f"mask from {mask_file}")
    elif a.mask:
        print("mask: built-in demo shapes (pass --mask-file to use your own mask)")
        mask = np.zeros((*rgb.shape[:2], 4), np.uint8)
        cv2.circle(mask, (rgb.shape[1] // 2, rgb.shape[0] // 2), rgb.shape[0] // 5, (0, 0, 160, 160), -1)
        cv2.rectangle(mask, (rgb.shape[1] // 8, rgb.shape[0] // 8), (rgb.shape[1] // 3, rgb.shape[0] // 3), (160, 0, 0, 160), -1)
    base_rss = rss_mb()
    w = Viewer(rgb, mask)
    ww, hh = (int(x) for x in a.window.lower().split("x"))
    w.resize(ww, hh)
    w.show()
    app.processEvents()
    print(f"image {rgb.shape[1]}x{rgb.shape[0]} ({rgb.shape[0] * rgb.shape[1] / 1e6:.1f} MP), "
          f"{len(w.pyr.levels)} pyramid levels, built in {w.build_s:.2f}s, window {ww}x{hh}, mask={bool(mask is not None)}")
    if a.interactive:
        w.fit()
        w.update()
        sys.exit(app.exec())
    res = benchmark(w)
    after = rss_mb()
    print(f"\n{'phase':<36} | p50 ms | p95 ms | max ms")
    for k, v in res.items():
        print(f"{k:<36} | {v['p50_ms']:>6} | {v['p95_ms']:>6} | {v['max_ms']:>6}")
    if after:
        print(f"\nprocess RSS: {after:.0f} MB (before pyramid {base_rss:.0f} MB)")
    print("(60 fps needs about 16 ms per frame or less)")
    Path(a.out).write_text(json.dumps({"image": [rgb.shape[1], rgb.shape[0]], "window": [ww, hh], "mask": bool(mask is not None),
                                       "pyramid_build_s": round(w.build_s, 3), "rss_mb": after, "phases": res},
                                      indent=1), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
