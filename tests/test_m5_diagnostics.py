"""M5: יומן מקומי וטיפול בקריסות (`fixpic.diagnostics`, `ui/crash_dialog.py`)."""
from __future__ import annotations

import faulthandler
import logging
import os
import sys
import threading
from pathlib import Path

import pytest

from fixpic import diagnostics as dg


@pytest.fixture(autouse=True)
def clean():
    yield
    dg.uninstall_crash_handlers()
    lg = logging.getLogger(dg.LOGGER_NAME)
    h = dg._state.pop("handler", None)
    if h is not None:
        lg.removeHandler(h)
        h.close()
    dg._state.pop("dir", None)


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------- יומן
def test_setup_logging_writes_file_with_env_header(tmp_path):
    path = dg.setup_logging(tmp_path)
    assert path == tmp_path / "fixpic.log"
    logging.getLogger("fixpic.pipeline").warning("שלום עולם")
    text = read(path)
    assert "שלום עולם" in text and "=== התחלה ===" in text and "python " in text and "numpy" in text
    assert "WARNING" in text and "fixpic.pipeline" in text


def test_setup_logging_is_idempotent_no_duplicate_lines(tmp_path):
    dg.setup_logging(tmp_path)
    dg.setup_logging(tmp_path)
    logging.getLogger("fixpic.x").error("פעם אחת")
    assert read(tmp_path / "fixpic.log").count("פעם אחת") == 1
    assert len([h for h in logging.getLogger("fixpic").handlers if isinstance(h, logging.handlers.RotatingFileHandler)]) == 1


def test_log_rotates(tmp_path):
    dg.setup_logging(tmp_path, max_bytes=2000, backups=2)
    for i in range(300):
        logging.getLogger("fixpic.x").error("שורה %d " + "x" * 40, i)
    names = sorted(p.name for p in tmp_path.glob("fixpic.log*"))
    assert names == ["fixpic.log", "fixpic.log.1", "fixpic.log.2"]


def test_setup_logging_unwritable_returns_none_and_does_not_raise(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert dg.setup_logging(blocker / "sub") is None       # "תיקייה" שאבא שלה קובץ
    logging.getLogger("fixpic.x").error("עדיין עובד")      # אין חריגה


def test_log_dir_env_and_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("FIXPIC_LOG_DIR", str(tmp_path / "L"))
    assert dg.log_dir() == tmp_path / "L"
    monkeypatch.delenv("FIXPIC_LOG_DIR")
    if sys.platform != "win32":
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "S"))
        assert dg.log_dir() == tmp_path / "S" / "fixpic" / "logs"


# ---------------------------------------------------------------- קריסות
def boom():
    raise ValueError("קריסה מכוונת")


def test_excepthook_writes_report_and_calls_back(tmp_path):
    dg.setup_logging(tmp_path)
    got = []
    dg.install_crash_handlers(on_crash=got.append, native=False)
    logging.getLogger("fixpic.x").info("פעולה אחרונה לפני הקריסה")
    try:
        boom()
    except ValueError:
        sys.excepthook(*sys.exc_info())
    assert len(got) == 1 and got[0].parent == tmp_path and got[0].name.startswith("crash-")
    rep = read(got[0])
    assert "ValueError: קריסה מכוונת" in rep and "boom" in rep and "פעולה אחרונה לפני הקריסה" in rep and "python " in rep
    assert "CRITICAL" in read(tmp_path / "fixpic.log")


def test_callback_failure_is_swallowed(tmp_path):
    dg.setup_logging(tmp_path)

    def bad(_):
        raise RuntimeError("callback שבור")

    dg.install_crash_handlers(on_crash=bad, native=False)
    try:
        boom()
    except ValueError:
        sys.excepthook(*sys.exc_info())          # לא זורק
    assert "on_crash נכשל" in read(tmp_path / "fixpic.log")


def test_keyboard_interrupt_goes_to_original_hook(tmp_path, monkeypatch):
    dg.setup_logging(tmp_path)
    seen = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: seen.append(a[0]))
    dg.install_crash_handlers(on_crash=lambda p: seen.append("crash"), native=False)
    sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    assert seen == [KeyboardInterrupt]
    assert not list(tmp_path.glob("crash-*.txt"))


def test_thread_exception_reported(tmp_path):
    dg.setup_logging(tmp_path)
    got = []
    dg.install_crash_handlers(on_crash=got.append, native=False)
    th = threading.Thread(target=boom, name="worker-1")
    th.start()
    th.join()
    rep = read(got[0])
    assert len(got) == 1 and "worker-1" in rep and "(thread, worker-1)" in rep


def test_systemexit_in_thread_is_not_a_crash(tmp_path):
    dg.setup_logging(tmp_path)
    got = []
    dg.install_crash_handlers(on_crash=got.append, native=False)
    th = threading.Thread(target=lambda: sys.exit(0))
    th.start()
    th.join()
    assert got == []


def test_unraisable_logged(tmp_path):
    dg.setup_logging(tmp_path)
    dg.install_crash_handlers(native=False)

    class Bad:
        def __del__(self):
            raise RuntimeError("בהשמדה")

    Bad()
    import gc
    gc.collect()
    assert list(tmp_path.glob("crash-*.txt"))


def test_old_reports_purged(tmp_path):
    dg.setup_logging(tmp_path)
    for i in range(15):
        (tmp_path / f"crash-20260101-0000{i:02d}-000.txt").write_text("x")
    dg.purge_old_reports(tmp_path, keep=10)
    assert len(list(tmp_path.glob("crash-*.txt"))) == 10
    assert (tmp_path / "crash-20260101-000014-000.txt").exists() and not (tmp_path / "crash-20260101-000000-000.txt").exists()


def test_report_without_log_dir_never_raises(tmp_path):
    blocker = tmp_path / "f"
    blocker.write_text("x")
    dg._state["dir"] = blocker / "no"
    try:
        raise ValueError("x")
    except ValueError:
        assert dg.write_crash_report(*sys.exc_info()) is None


def test_native_crash_file_and_restore(tmp_path):
    dg.setup_logging(tmp_path)
    before = (sys.excepthook, threading.excepthook)
    dg.install_crash_handlers(native=True)
    assert faulthandler.is_enabled() and (tmp_path / dg.NATIVE_NAME).exists()
    dg.uninstall_crash_handlers()
    assert (sys.excepthook, threading.excepthook) == before


# ---------------------------------------------------------------- ממשק
def test_crash_dialog_from_worker_thread(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QMessageBox

    from fixpic.ui.app import create_app
    from fixpic.ui.crash_dialog import CrashNotifier

    app = create_app([])
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda parent, title, text: shown.append((title, text))))
    n = CrashNotifier()
    dg.setup_logging(tmp_path)
    dg.install_crash_handlers(on_crash=n.notify, native=False)
    th = threading.Thread(target=boom)
    th.start()
    th.join()
    app.processEvents()
    assert n.shown == 1 and "שגיאה חמורה" in shown[0][0] and str(tmp_path) in shown[0][1] and "לא נשלח" in shown[0][1]
    n.notify(None)
    app.processEvents()
    assert "לא ניתן היה לשמור" in shown[1][1]


def test_crash_dialog_not_stacked_while_open(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QMessageBox

    from fixpic.ui.app import create_app
    from fixpic.ui.crash_dialog import CrashNotifier

    app = create_app([])
    n = CrashNotifier()
    calls = []

    def fake(parent, title, text):
        calls.append(1)
        n.notify(None)                    # שגיאה נוספת בזמן שההודעה פתוחה
        app.processEvents()

    monkeypatch.setattr(QMessageBox, "critical", staticmethod(fake))
    n.notify(None)
    app.processEvents()
    assert calls == [1]


def test_exception_in_qt_slot_reaches_hook(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QObject, Signal

    from fixpic.ui.app import create_app

    app = create_app([])
    dg.setup_logging(tmp_path)
    got = []
    dg.install_crash_handlers(on_crash=got.append, native=False)

    class S(QObject):
        sig = Signal()

    s = S()
    s.sig.connect(boom)
    s.sig.emit()
    app.processEvents()
    assert got and "ValueError" in read(got[0])
