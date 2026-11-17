"""M5: אייקון היישום (קבצים, טעינה, חיבור ל-QApplication, נתיב מארוז)."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402

from fixpic.ui import icon as ic  # noqa: E402
from fixpic.ui.app import create_app  # noqa: E402
from test_m33 import app  # noqa: E402,F401


def test_assets_exist_and_valid():
    d = ic.assets_dir()
    png = Image.open(d / "fixpic.png")
    assert png.size == (256, 256) and png.mode == "RGBA"
    assert png.getpixel((0, 0))[3] == 0            # פינות שקופות (ריבוע מעוגל)
    assert png.getpixel((128, 20))[3] == 255       # הגוף אטום
    ico = Image.open(d / "fixpic.ico")
    assert {(16, 16), (32, 32), (48, 48), (256, 256)} <= set(ico.info["sizes"])


def test_app_icon_loads_and_is_set(app):
    assert not ic.app_icon().isNull()
    a = create_app([])
    assert not a.windowIcon().isNull()


def test_missing_assets_gives_empty_icon_not_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(ic, "assets_dir", lambda: tmp_path)
    assert ic.app_icon().isNull()


def test_frozen_path_uses_meipass(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert ic.assets_dir() == tmp_path / "fixpic" / "ui" / "assets"


def test_make_icon_is_deterministic(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "tools" / "m5" / "make_icon.py"
    spec = importlib.util.spec_from_file_location("make_icon", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    monkeypatch.setattr(m, "OUT", tmp_path)
    assert m.main() == 0
    assert (tmp_path / "fixpic.ico").stat().st_size > 5000
    assert (tmp_path / "fixpic.png").read_bytes() == (ic.assets_dir() / "fixpic.png").read_bytes()
