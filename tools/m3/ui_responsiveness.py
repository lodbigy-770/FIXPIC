"""M3.6: תגובתיות הממשק בזמן עיבוד, על תמונה גדולה ודרך הנתיב האמיתי (ModelManager -> OpenVINO/ORT -> Scheduler).

המודל הוא ONNX קטן (אותו חוזה כמו Big LaMa) ולכן זמן ה-Inference עצמו אינו מייצג; מה שנמדד כאן הוא כל מה שקורה **ב-thread
של ה-UI** ומסביב: ציור משיכה, בניית ה-Patch וה-Commit (`Session.finish`), ורענון הקנבס. כל עיכוב כזה חוסם את לולאת
האירועים בלי קשר למהירות המודל.

שיטה: טיימר "פעימת לב" של 4ms; פער בין פעימות = זמן שהלולאה הייתה חסומה. בנוסף נמדדים בנפרד `_handle_done` (Commit)
ו-`_image_changed` (רענון קנבס). משיכות נשלחות כמו משתמש: גרירה עם תזוזות עכבר כל כמה מילישניות.

הרצה: `QT_QPA_PLATFORM=offscreen python tools/m3/ui_responsiveness.py [רוחב גובה] [--alpha]`
(ללא `--alpha` התמונה אטומה; `--no-engine` = בסיס השוואה בלי עיבוד). אינו מייצג i5-6500T או Windows, ולא את זמן ה-Inference של Big LaMa."""
from __future__ import annotations

import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import TensorProto as T, helper as h  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fixpic.imgio import image_from_array  # noqa: E402
from fixpic.ui.app import create_app  # noqa: E402
from fixpic.ui.edit_loop import EngineState  # noqa: E402
from fixpic.ui.engine import default_engine_factory  # noqa: E402
from fixpic.ui.main_window import MainWindow  # noqa: E402

BEAT_MS = 4


def make_tiny_model(dst: Path) -> None:
    rgb = h.make_tensor_value_info("rgb", T.FLOAT, [1, 3, "8*height", "8*width"])
    mask = h.make_tensor_value_info("mask", T.FLOAT, [1, 1, "8*height", "8*width"])
    out = h.make_tensor_value_info("inpainted", T.FLOAT, [1, 3, "8*height", "8*width"])
    one = h.make_tensor("one", T.FLOAT, [], [1.0])
    nodes = [h.make_node("Sub", ["one", "mask"], ["inv"]), h.make_node("Mul", ["rgb", "inv"], ["keep"]),
             h.make_node("Sub", ["one", "rgb"], ["flip"]), h.make_node("Mul", ["flip", "mask"], ["fill"]),
             h.make_node("Add", ["keep", "fill"], ["inpainted"])]
    m = h.make_model(h.make_graph(nodes, "t", [rgb, mask], [out], [one]), opset_imports=[h.make_opsetid("", 13)])
    m.ir_version = 8
    d = dst / "tiny"
    d.mkdir(parents=True)
    onnx.save(m, d / "w.onnx")
    (d / "model.json").write_text(json.dumps({"schema": 1, "name": "tiny", "file": "w.onnx"}), encoding="utf-8")


class Beat:
    def __init__(self) -> None:
        self.gaps: list[tuple[float, float]] = []     # (זמן, פער ב-ms)
        self.last = time.perf_counter()
        self.t = QTimer()
        self.t.setInterval(BEAT_MS)
        self.t.timeout.connect(self._tick)

    def _tick(self) -> None:
        now = time.perf_counter()
        self.gaps.append((now, (now - self.last) * 1000))
        self.last = now

    def start(self) -> None:
        self.gaps.clear()
        self.last = time.perf_counter()
        self.t.start()

    def stop(self) -> list[float]:
        self.t.stop()
        return [g for _, g in self.gaps]


def timed(obj, name, bucket):
    orig = getattr(obj, name)

    def wrapper(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            bucket.append((time.perf_counter() - t0) * 1000)
    setattr(obj, name, wrapper)


def pump(ms: float) -> None:
    end = time.perf_counter() + ms / 1000
    while time.perf_counter() < end:
        QApplication.processEvents()
        time.sleep(0.001)


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * q))], 1) if xs else 0.0


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    alpha_on = "--alpha" in sys.argv
    no_engine = "--no-engine" in sys.argv      # בסיס השוואה: ציור משיכות בלבד (המסכה נשארת), בלי עיבוד
    w, hh = (int(args[0]), int(args[1])) if len(args) == 2 else (6000, 4000)
    app = create_app([])
    tmp = Path(tempfile.mkdtemp())
    try:
        make_tiny_model(tmp / "models")
        win = MainWindow(engine_factory=None if no_engine else default_engine_factory(tmp / "models", "tiny", threads=2))
        win.show()
        t0 = time.perf_counter()
        while not no_engine and win.controller.state is not EngineState.READY and time.perf_counter() - t0 < 60:
            pump(20)
        print(f"engine {win.controller.state.value} after {time.perf_counter() - t0:.1f}s")
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 256, (hh, w, 3), dtype=np.uint8)
        if alpha_on:
            a = np.full((hh, w), 255, np.uint8)
            a[::5] = 200
            arr = np.dstack([arr, a])
        win._install(image_from_array(arr), "big.png")
        pump(300)
        cv = win.canvas
        done_ms: list[float] = []
        refresh_ms: list[float] = []
        timed(win.controller, "_handle_done", done_ms)
        timed(win, "_image_changed", refresh_ms)

        def drag(points, step_ms=8):
            QTest.mousePress(cv, Qt.LeftButton, Qt.NoModifier, points[0])
            for p in points[1:]:
                QTest.mouseMove(cv, p)
                pump(step_ms)
            QTest.mouseRelease(cv, Qt.LeftButton, Qt.NoModifier, points[-1])

        def stroke_points(cx, cy, n=50):
            return [QPoint(cx + i * 4, cy + (i % 5)) for i in range(n)]

        beat = Beat()
        s = win.session
        d = win.brush_panel.diameter
        print(f"image {w}x{hh} alpha={alpha_on}; brush {d}px; fit scale {cv.scale:.3f}")

        # א. משיכות ברצף בזום "התאמה": 6 משיכות, בלי להמתין לסיום ביניהן
        beat.start()
        t_start = time.perf_counter()
        for i in range(6):
            drag(stroke_points(120 + (i % 3) * 330, 90 + (i // 3) * 200))
        t_input = time.perf_counter() - t_start
        busy_after_input = s.busy
        while s.busy or win.controller.waiting is not None:
            pump(2)
        gaps = beat.stop()
        print(f"[A] 6 strokes fit-zoom: input {t_input:.2f}s, busy after last release={busy_after_input}, "
              f"drained after {time.perf_counter() - t_start:.2f}s")
        print(f"    event-loop gap ms: median={pct(gaps, .5)} p99={pct(gaps, .99)} max={round(max(gaps), 1)} "
              f"(>50ms: {sum(g > 50 for g in gaps)}, >100ms: {sum(g > 100 for g in gaps)}) over {len(gaps)} beats")
        print(f"    _handle_done (Commit on UI thread) ms: {[round(x, 1) for x in done_ms]}")
        print(f"    _image_changed (canvas refresh) ms:    {[round(x, 1) for x in refresh_ms]}")
        print(f"    commits={s.history.undo_count}, modified={s.modified}")

        # ב. זום 100% ומשיכות מהירות (משתמש מדייק)
        done_ms.clear()
        refresh_ms.clear()
        cv.zoom_at(QPointF(cv.width() / 2, cv.height() / 2), 1 / cv.scale)
        pump(50)
        beat.start()
        for i in range(3):
            drag([QPoint(150 + j * 5, 150 + i * 60) for j in range(40)], step_ms=4)
        while s.busy or win.controller.waiting is not None:
            pump(2)
        gaps = beat.stop()
        print(f"[B] 3 strokes at 100% zoom: gap ms median={pct(gaps, .5)} p99={pct(gaps, .99)} max={round(max(gaps), 1)} "
              f"(>50ms: {sum(g > 50 for g in gaps)})")
        print(f"    _handle_done ms: {[round(x, 1) for x in done_ms]}; _image_changed ms: "
              f"{[round(x, 1) for x in refresh_ms]}")

        # ג. Undo/Redo ברצף (כל צעד מרענן את הקנבס)
        refresh_ms.clear()
        beat.start()
        t0 = time.perf_counter()
        n_undo = s.history.undo_count
        for _ in range(n_undo):
            win.act_undo.trigger()
            pump(1)
        for _ in range(n_undo):
            win.act_redo.trigger()
            pump(1)
        gaps = beat.stop()
        print(f"[C] {n_undo} undo + {n_undo} redo: total {time.perf_counter() - t0:.2f}s, "
              f"refresh ms median={pct(refresh_ms, .5)} max={round(max(refresh_ms), 1) if refresh_ms else 0}; "
              f"loop gap max={round(max(gaps), 1)}")
        win.controller.shutdown()
        win.session = None
        win.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
