"""M5 – בדיקת תיקיית האריזה (PyInstaller onedir) שנבנתה מ-`fixpic-app.spec`.

שימוש:  python tools/m5/check_dist.py dist/FixPic [--smoke-seconds 10] [--json out.json]

בודק: (1) `THIRD_PARTY_NOTICES.txt` נארז; (2) רכיבים שהוחרגו (טלמטריה של OpenVINO, `openvino/tools`, WebEngine, Pdf, Qml, Quick, 3D)
אינם בתיקייה; (3) קובץ המודל נארז והוא אמיתי (לא מצביע LFS); (4) ספריות Qt הן קבצים נפרדים ולא מוטמעות ב-exe (קישור דינמי, LGPL);
(5) גודל כולל והקבצים הגדולים ביותר; (6) אופציונלי: הרצת ה-exe ב-`QT_QPA_PLATFORM=offscreen` והוכחה שהוא חי כמה שניות בלי להיסגר בשגיאה.
קוד יציאה 1 אם יש בדיקה שנכשלה (`fail`); `warn` אינו מכשיל.

מגבלות: אין בדיקה שהאפליקציה *עובדת* (עיבוד תמונה) אלא רק שהיא עולה; אין בדיקת חתימה/Defender; אין מחשב נקי בלי Python."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

NOTICES = "THIRD_PARTY_NOTICES.txt"
MODEL = "inpainter.onnx"
MIN_MODEL_BYTES = 1_000_000
# תת-מחרוזות (באותיות קטנות, עם "/" כמפריד) שאסור שיופיעו בנתיב יחסי של קובץ או תיקייה באריזה
FORBIDDEN = ("openvino_telemetry", "openvino/tools", "qtwebengine", "qt6webengine", "qt6pdf", "pyside6/qtpdf", "qt6qml", "qt6quick",
             "pyside6/qtqml", "pyside6/qtquick", "qt63d", "pyside6/qt3d")
QT_CORE = ("qt6core", "qtcore")


def walk(root: Path) -> list[tuple[str, int]]:
    out = []
    for p in root.rglob("*"):
        if p.is_file():
            out.append((p.relative_to(root).as_posix(), p.stat().st_size))
    return out


def find_exe(root: Path) -> Path | None:
    for n in ("FixPic.exe", "FixPic"):
        p = root / n
        if p.is_file():
            return p
    return None


def inspect_dist(root: Path) -> dict:
    files = walk(root)
    checks: list[dict] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    notices = [(r, s) for r, s in files if r.rsplit("/", 1)[-1] == NOTICES]
    if notices and notices[0][1] > 10_000:
        add("THIRD_PARTY_NOTICES", "ok", f"{notices[0][0]} ({notices[0][1] // 1024}KB)")
    elif notices:
        add("THIRD_PARTY_NOTICES", "fail", f"הקובץ קטן מדי ({notices[0][1]} בייט): {notices[0][0]}")
    else:
        add("THIRD_PARTY_NOTICES", "fail", "הקובץ לא נארז")

    bad = sorted({r for r, _ in files if any(f in r.lower() for f in FORBIDDEN)})
    add("רכיבים שהוחרגו", "fail" if bad else "ok",
        f"{len(bad)} קבצים אסורים, למשל: {', '.join(bad[:5])}" if bad else "לא נמצא אף רכיב אסור")

    models = [(r, s) for r, s in files if r.rsplit("/", 1)[-1] == MODEL]
    if not models:
        add("מודל", "fail", f"{MODEL} לא נארז")
    elif models[0][1] < MIN_MODEL_BYTES:
        add("מודל", "fail", f"{models[0][0]}: {models[0][1]} בייט (נראה כמצביע LFS ולא מודל אמיתי)")
    else:
        add("מודל", "ok", f"{models[0][0]} ({models[0][1] // 2**20}MB)")

    mj = [r for r, _ in files if r.endswith("models/big-lama/model.json")]
    add("model.json", "ok" if mj else "fail", mj[0] if mj else "model.json לא נארז")

    exe = find_exe(root)
    qt_libs = [r for r, _ in files if any(q in r.lower().rsplit("/", 1)[-1] for q in QT_CORE)]
    if exe is None:
        add("קישור דינמי (Qt)", "fail", "לא נמצא קובץ ההרצה FixPic")
    elif not qt_libs:
        add("קישור דינמי (Qt)", "fail", "לא נמצאו ספריות QtCore כקבצים נפרדים")
    else:
        add("קישור דינמי (Qt)", "ok" if exe.stat().st_size < 50 * 2**20 else "warn",
            f"{len(qt_libs)} קבצי QtCore נפרדים; גודל ה-exe {exe.stat().st_size // 2**20}MB")

    total = sum(s for _, s in files)
    top = sorted(files, key=lambda x: -x[1])[:10]
    add("גודל", "ok", f"{total / 2**20:.0f}MB ב-{len(files)} קבצים; הגדולים: " +
        ", ".join(f"{r.rsplit('/', 1)[-1]} {s // 2**20}MB" for r, s in top[:5]))
    return {"root": str(root), "total_mb": round(total / 2**20), "files": len(files), "checks": checks,
            "largest": [{"path": r, "mb": round(s / 2**20, 1)} for r, s in top]}


def smoke(root: Path, seconds: float) -> dict:
    """מריץ את ה-exe ב-offscreen. עובר אם התהליך עדיין חי אחרי `seconds` (נסגר בכוח), נכשל אם נסגר בעצמו."""
    exe = find_exe(root)
    if exe is None:
        return {"name": "הרצה (smoke)", "status": "fail", "detail": "לא נמצא קובץ הרצה"}
    import tempfile
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    t0 = time.perf_counter()
    with tempfile.TemporaryFile() as log:                      # קובץ ולא pipe: תהליך-בן שנשאר חי לא חוסם את ההמתנה
        p = subprocess.Popen([str(exe)], env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(root))
        try:
            p.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            return {"name": "הרצה (smoke)", "status": "ok", "detail": f"האפליקציה חיה {seconds:.0f}s ב-offscreen (נסגרה בכוח). לא נבדק עיבוד תמונה."}
        log.seek(0)
        tail = log.read()[-400:].decode("utf-8", "replace")
        return {"name": "הרצה (smoke)", "status": "fail",
                "detail": f"התהליך נסגר מעצמו אחרי {time.perf_counter() - t0:.1f}s עם קוד {p.returncode}: {tail}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dist", type=Path)
    ap.add_argument("--smoke-seconds", type=float, default=0, help="0 = בלי הרצה")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    if not a.dist.is_dir():
        print(f"התיקייה לא קיימת: {a.dist}", file=sys.stderr)
        return 2
    rep = inspect_dist(a.dist)
    if a.smoke_seconds > 0:
        rep["checks"].append(smoke(a.dist, a.smoke_seconds))
    for c in rep["checks"]:
        print(f"[{c['status']:4}] {c['name']}: {c['detail']}")
    if a.json:
        a.json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if any(c["status"] == "fail" for c in rep["checks"]) else 0


if __name__ == "__main__":
    sys.exit(main())
