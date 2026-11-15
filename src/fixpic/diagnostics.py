"""M5: יומן מקומי וטיפול בקריסות (בלי רשת: שום דבר לא נשלח החוצה; הכול קבצים בתיקייה מקומית).

* `setup_logging()`: קובץ יומן מתגלגל (`fixpic.log`, עד 5 גיבויים) תחת תיקיית הנתונים של המשתמש, ל-logger `fixpic`.
  כשל ביצירת היומן (תיקייה לקריאה בלבד, דיסק מלא) לעולם אינו מפיל את היישום.
* `install_crash_handlers()`: חריגה שלא נתפסה (ב-thread הראשי, ב-threads אחרים או ב-"unraisable") נרשמת ביומן,
  נכתב דוח `crash-<זמן>.txt` (traceback, גרסאות, סוף היומן), ומופעל `on_crash(נתיב)` (הממשק מציג הודעה). `faulthandler`
  כותב לקובץ `native-crash.log` קריסות native (segfault, נפילה ב-OpenVINO/ORT/Qt) שאין להן traceback של Python.
  קריסה שהורגת את התהליך בלי חריגה (OOM של מערכת ההפעלה) אינה ניתנת לתפיסה מבפנים; השורה האחרונה ביומן תעיד על הפעולה האחרונה.

מיקום (`log_dir()`): `FIXPIC_LOG_DIR` אם הוגדר; ב-Windows `%LOCALAPPDATA%\\FIXPIC\\logs`; אחרת `$XDG_STATE_HOME/fixpic/logs`
(ברירת מחדל `~/.local/state/fixpic/logs`).

פרטיות: היומן והדוחות עלולים להכיל שמות קבצים ופרטי שגיאה, אך לא פיקסלים. הם נשארים במחשב; המשתמש מחליט אם לשלוח."""
from __future__ import annotations

import datetime as _dt
import faulthandler
import logging
import logging.handlers
import os
import platform
import sys
import threading
import traceback
from importlib import metadata
from pathlib import Path
from typing import Callable

LOGGER_NAME = "fixpic"
LOG_NAME = "fixpic.log"
NATIVE_NAME = "native-crash.log"
KEEP_REPORTS = 10
_FORMAT = "%(asctime)s %(levelname)-8s %(name)s [%(threadName)s] %(message)s"
_PACKAGES = ("numpy", "opencv-python-headless", "pillow", "onnxruntime", "openvino", "PySide6", "PySide6-Essentials")

_state: dict = {}


def log_dir() -> Path:
    env = os.environ.get("FIXPIC_LOG_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "FIXPIC" / "logs"
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "fixpic" / "logs"


def environment_info() -> list[str]:
    """שורות סביבה לראש היומן ולדוחות קריסה (גרסאות בלבד, בלי ייבוא כבד)."""
    out = [f"python {sys.version.split()[0]} {platform.platform()} {platform.machine()}"]
    try:
        out.append(f"fixpic {metadata.version('fixpic')}")
    except metadata.PackageNotFoundError:
        out.append("fixpic (לא מותקן כחבילה)")
    seen = set()
    for name in _PACKAGES:
        try:
            out.append(f"{name} {metadata.version(name)}")
            seen.add(name)
        except metadata.PackageNotFoundError:
            pass
    return out


def setup_logging(directory: str | Path | None = None, level: int = logging.INFO, max_bytes: int = 1_000_000,
                  backups: int = 5) -> Path | None:
    """מגדיר יומן מתגלגל ל-logger `fixpic`. מחזיר את נתיב הקובץ, או None אם אי אפשר ליצור אותו. בטוח לקריאה חוזרת."""
    logger = logging.getLogger(LOGGER_NAME)
    old = _state.pop("handler", None)
    if old is not None:
        logger.removeHandler(old)
        old.close()
    try:
        d = Path(directory) if directory is not None else log_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / LOG_NAME
        handler = logging.handlers.RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FORMAT))
        handler.setLevel(level)
    except Exception:  # noqa: BLE001 - יומן שאי אפשר ליצור אינו סיבה להפיל את היישום
        return None
    logger.addHandler(handler)
    if logger.level == logging.NOTSET or logger.level > level:
        logger.setLevel(level)
    _state["handler"], _state["dir"] = handler, d
    logger.info("=== התחלה ===")
    for line in environment_info():
        logger.info(line)
    return path


def _tail(path: Path, lines: int = 60) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-lines:])
    except OSError:
        return ""


def purge_old_reports(directory: Path, keep: int = KEEP_REPORTS) -> None:
    try:
        files = sorted(directory.glob("crash-*.txt"), key=lambda p: p.name, reverse=True)
        for p in files[keep:]:
            p.unlink(missing_ok=True)
    except OSError:
        pass


def write_crash_report(exc_type, exc, tb, *, kind: str = "uncaught", thread: str | None = None) -> Path | None:
    """כותב דוח קריסה ומתעד ביומן. מחזיר את נתיב הדוח, או None. לעולם אינו זורק."""
    try:
        logging.getLogger(LOGGER_NAME).critical("חריגה לא נתפסת (%s)%s", kind, f" ב-{thread}" if thread else "",
                                                exc_info=(exc_type, exc, tb))
    except Exception:  # noqa: BLE001
        pass
    try:
        d = _state.get("dir") or log_dir()
        d.mkdir(parents=True, exist_ok=True)
        now = _dt.datetime.now()
        path = d / f"crash-{now:%Y%m%d-%H%M%S}-{now.microsecond // 1000:03d}.txt"
        body = [f"FIXPIC crash report {now.isoformat(timespec='seconds')} ({kind}{', ' + thread if thread else ''})", "",
                *environment_info(), "", "".join(traceback.format_exception(exc_type, exc, tb)),
                "--- סוף היומן ---", _tail(d / LOG_NAME)]
        path.write_text("\n".join(body), encoding="utf-8")
        purge_old_reports(d)
        return path
    except Exception:  # noqa: BLE001
        return None


def install_crash_handlers(on_crash: Callable[[Path | None], None] | None = None, *, native: bool = True) -> None:
    """מחליף את `sys.excepthook`, `threading.excepthook` ו-`sys.unraisablehook`; `on_crash(נתיב_הדוח)` נקרא אחרי הכתיבה
    (כל חריגה בתוכו נבלעת). `KeyboardInterrupt` ממשיך להתנהג כרגיל. קריאה חוזרת מחליפה את ה-callback."""
    if "orig" not in _state:
        _state["orig"] = (sys.excepthook, threading.excepthook, sys.unraisablehook)
    orig_sys, orig_thr, orig_unr = _state["orig"]
    _state["on_crash"] = on_crash

    def notify(path):
        cb = _state.get("on_crash")
        if cb is not None:
            try:
                cb(path)
            except Exception:  # noqa: BLE001
                logging.getLogger(LOGGER_NAME).exception("on_crash נכשל")

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            return orig_sys(exc_type, exc, tb)
        notify(write_crash_report(exc_type, exc, tb, thread=threading.current_thread().name))
        return None

    def thread_hook(args):
        if args.exc_type is SystemExit:
            return
        notify(write_crash_report(args.exc_type, args.exc_value, args.exc_traceback, kind="thread",
                                  thread=getattr(args.thread, "name", None)))

    def unraisable(u):
        if u.exc_type is not None:
            write_crash_report(u.exc_type, u.exc_value, u.exc_traceback, kind="unraisable")

    sys.excepthook, threading.excepthook, sys.unraisablehook = hook, thread_hook, unraisable
    if native and "native_file" not in _state:
        try:
            d = _state.get("dir") or log_dir()
            d.mkdir(parents=True, exist_ok=True)
            f = open(d / NATIVE_NAME, "a", encoding="utf-8")
            _state["was_enabled"] = faulthandler.is_enabled()
            faulthandler.enable(f, all_threads=True)
            _state["native_file"] = f
        except Exception:  # noqa: BLE001
            pass


def uninstall_crash_handlers() -> None:
    """מחזיר את ה-hooks המקוריים (לבדיקות ולכיבוי מסודר)."""
    orig = _state.pop("orig", None)
    if orig:
        sys.excepthook, threading.excepthook, sys.unraisablehook = orig
    _state.pop("on_crash", None)
    f = _state.pop("native_file", None)
    if f is not None:
        faulthandler.disable()
        if _state.pop("was_enabled", False):
            faulthandler.enable()
        f.close()
