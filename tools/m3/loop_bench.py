"""M3.4: מדידת עלות הציור והעדכון בלולאת העריכה על תמונה גדולה (offscreen; לא i5-6500T).

מודד: (1) בניית שכבת המסכה לציור בזום "התאמה" ובזום 100% על 24MP, (2) מחזור משיכה מלא (ציור במסכה + ציור הקנבס) לכל
תזוזת עכבר, (3) החלפת התמונה בקנבס אחרי Commit (בניית פירמידה), עם Alpha וללא.
הרצה: `QT_QPA_PLATFORM=offscreen python tools/m3/loop_bench.py [רוחב גובה]`"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
from PySide6.QtCore import QPointF  # noqa: E402

from fixpic.session import MaskLayer  # noqa: E402
from fixpic.ui.app import create_app  # noqa: E402
from fixpic.ui.canvas import Canvas  # noqa: E402


def med_ms(fn, n=15):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(ts), 1), round(max(ts), 1)


def main() -> None:
    w, h = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (6000, 4000)
    create_app([])
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    alpha = np.full((h, w), 255, np.uint8)
    alpha[::7] = 128
    c = Canvas()
    c.resize(1100, 700)
    c.show()
    t0 = time.perf_counter()
    c.set_image(rgb)
    print(f"image {w}x{h} ({w * h / 1e6:.1f}MP): set_image first {1000 * (time.perf_counter() - t0):.0f}ms")
    layer = MaskLayer(h, w)
    c.set_mask_layer(layer)
    c.set_painting(True)
    d = round(min(w, h) * 0.03)
    layer.paint([(w * 0.3, h * 0.5), (w * 0.35, h * 0.52)], d)

    def overlay_cost():
        c._ov_cache = None
        c._overlay_items(c.visible_source_rect())

    print(f"fit scale={c.scale:.3f}: overlay render (median, max ms) = {med_ms(overlay_cost)}")
    c.scale, c.offset, c._fit_mode = 1.0, QPointF(-w * 0.3 + 500, -h * 0.5 + 300), False
    print(f"scale 1.0: overlay render = {med_ms(overlay_cost)}")
    c.fit()

    state = {"x": w * 0.4}

    def move():
        state["x"] += 3
        layer.paint([(state["x"] - 3, h * 0.5), (state["x"], h * 0.5)], d)
        c.grab()                                  # ציור מלא: תמונה + שכבת מסכה (מטמון נפסל כי הגרסה השתנתה)

    print(f"fit: one mouse-move cycle (mask.paint + full canvas repaint) = {med_ms(move, 30)}")

    def hover():
        c._hover = QPointF(300, 300)
        c.grab()
    print(f"fit: hover repaint (overlay cached) = {med_ms(hover, 30)}")

    print(f"commit refresh: canvas.set_image(keep_view) no alpha = {med_ms(lambda: c.set_image(rgb, keep_view=True), 8)}")
    print(f"commit refresh: canvas.set_image(keep_view) with alpha = {med_ms(lambda: c.set_image(rgb, alpha, keep_view=True), 8)}")


if __name__ == "__main__":
    main()
