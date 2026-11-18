"""M5: בדיקת Offline. חוסמת בתוך התהליך כל ניסיון חיבור רשת, ומריצה את המסלולים העיקריים (CLI, Pipeline, Session,
טעינת מנוע דרך הממשק, חלון ראשי). כל ניסיון נרשם ומכשיל את הבדיקה.

מה שהבדיקה מוכיחה: בקוד של FIXPIC ובספריות שנטענות במסלולים האלה אין פנייה ל-socket ברמת Python (חיבור, DNS, שליחה).
מה שלא נבדק: חסימת Firewall אמיתית ב-Windows; פנייה מקוד נייטיב (Qt, OpenVINO, ORT) שאינה עוברת דרך `socket`
של Python; מסלולים שלא מורצים כאן (שמירה אמיתית מהדיאלוג, חבילה ארוזה). לכן הבדיקה משלימה בדיקת Firewall ולא מחליפה אותה."""
from __future__ import annotations

import json
import os
import shutil
import socket
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

from fixpic import cli
from fixpic.engine import JobScheduler
from fixpic.imgio import image_from_array
from fixpic.pipeline import Pipeline, PipelineConfig
from fixpic.session import Session
from test_backends import tiny  # noqa: F401  (fixture)
from test_engine import FakeModel, rnd

FORBIDDEN_MODULES = ("requests", "urllib3", "httpx", "aiohttp", "PySide6.QtNetwork", "webbrowser")

_BLOCKED_METHODS = ("connect", "connect_ex", "sendto", "sendmsg")
_BLOCKED_FUNCS = ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "create_connection")


class NetworkAttempt(AssertionError):
    pass


@pytest.fixture
def netguard(monkeypatch):
    """חוסם פניות רשת ורושם אותן. חיבור ל-AF_UNIX מקומי מותר (אינו רשת)."""
    attempts: list[str] = []

    def wrap_method(name, orig):
        def guarded(self, *a, **k):
            if getattr(self, "family", None) == getattr(socket, "AF_UNIX", None):
                return orig(self, *a, **k)
            attempts.append(f"socket.{name}{a[:1]}")
            raise NetworkAttempt(f"ניסיון רשת: socket.{name} {a[:1]}")
        return guarded

    def wrap_func(name):
        def guarded(*a, **k):
            attempts.append(f"socket.{name}{a[:1]}")
            raise NetworkAttempt(f"ניסיון רשת: socket.{name} {a[:1]}")
        return guarded

    for n in _BLOCKED_METHODS:
        if not hasattr(socket.socket, n):                                      # sendmsg לא קיים ב-Windows
            continue
        monkeypatch.setattr(socket.socket, n, wrap_method(n, getattr(socket.socket, n)))
    for n in _BLOCKED_FUNCS:
        monkeypatch.setattr(socket, n, wrap_func(n))
    yield attempts


def _assert_no_network(attempts):
    assert attempts == [], f"נרשמו ניסיונות רשת: {attempts}"
    loaded = [m for m in FORBIDDEN_MODULES if m in sys.modules]
    assert not loaded, f"נטענו מודולי רשת: {loaded}"


@pytest.fixture
def models_dir(tmp_path, tiny):  # noqa: F811
    d = tmp_path / "models" / "tiny"
    d.mkdir(parents=True)
    shutil.copy(tiny, d / "w.onnx")
    (d / "model.json").write_text(json.dumps({"schema": 1, "name": "tiny", "file": "w.onnx"}), encoding="utf-8")
    return tmp_path / "models"


def _mask(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


# ------------------------------------------------------------------ בדיקת תקינות המגן עצמו
def test_guard_blocks_connect_dns_and_records(netguard):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkAttempt):
            s.connect(("127.0.0.1", 9))
        with pytest.raises(NetworkAttempt):
            socket.getaddrinfo("example.com", 80)
        with pytest.raises(NetworkAttempt):
            socket.create_connection(("example.com", 80))
    finally:
        s.close()
    assert len(netguard) == 3
    netguard.clear()                                             # שהבדיקה עצמה לא תיחשב ככשל


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="אין AF_UNIX")
def test_guard_allows_local_unix_socketpair(netguard):
    a, b = socket.socketpair()
    try:
        a.sendall(b"x")
        assert b.recv(1) == b"x"
    finally:
        a.close()
        b.close()
    assert netguard == []


# ------------------------------------------------------------------ המסלולים עצמם
def test_cli_end_to_end_without_network(netguard, tmp_path, models_dir):
    d = tmp_path / "תיקייה"
    d.mkdir()
    img = rnd(300, 400, 5)
    cv2.imencode(".png", img[..., ::-1])[1].tofile(str(d / "תמונה.png"))
    cv2.imencode(".png", _mask(300, 400, 100, 140, 150, 230))[1].tofile(str(d / "מסכה.png"))
    out = d / "פלט.png"
    code = cli.main([str(d / "תמונה.png"), str(d / "מסכה.png"), "--model", "tiny", "--models-dir", str(models_dir),
                     "-o", str(out), "--size", "256", "--dilate", "0", "--feather", "0"])
    assert code == 0 and out.exists()
    _assert_no_network(netguard)


def test_pipeline_and_session_without_network(netguard):
    img = rnd(200, 200, 1)
    with JobScheduler(Pipeline(FakeModel(), PipelineConfig())).start(warmup=False) as sched:
        s = Session(image_from_array(img), sched)
        m = _mask(200, 200, 60, 100, 60, 100)
        job = sched.submit(img, m, 256)
        assert job.wait(30)
        assert s is not None
    _assert_no_network(netguard)


def test_ui_main_window_and_engine_without_network(netguard, models_dir):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from fixpic.ui.engine import default_engine_factory
    from fixpic.ui.main_window import MainWindow

    import time

    from fixpic.ui.edit_loop import EngineState

    app = QApplication.instance() or QApplication([])
    win = MainWindow(engine_factory=default_engine_factory(models_dir, "tiny", threads=2))
    try:
        win.show()
        end = time.monotonic() + 15
        while win.controller.state is not EngineState.READY and time.monotonic() < end:
            app.processEvents()
            time.sleep(0.01)
        assert win.controller.state is EngineState.READY       # המודל נטען מהדיסק בלבד
    finally:
        win._runner.wait_idle()
        win.controller.shutdown()
        win.session = None
        win.close()
        app.processEvents()
    _assert_no_network(netguard)
