"""M5: מסך "אודות" והצגת THIRD_PARTY_NOTICES (offscreen)."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDialog  # noqa: E402

from fixpic import strings  # noqa: E402
from fixpic.ui import about_dialog as ad  # noqa: E402
from fixpic.ui.main_window import MainWindow  # noqa: E402
from test_m33 import app  # noqa: E402,F401


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("FIXPIC_NOTICES", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)


def test_candidates_order_env_then_meipass_then_exe_then_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("FIXPIC_NOTICES", str(tmp_path / "env.txt"))
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "mei"), raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    c = ad.notices_candidates()
    assert c[0] == tmp_path / "env.txt" and c[1] == tmp_path / "mei" / ad.NOTICES_NAME
    assert c[2].name == ad.NOTICES_NAME and c[-1].parts[-2:] == ("build", ad.NOTICES_NAME)


def test_find_prefers_first_existing(monkeypatch, tmp_path):
    f = tmp_path / "n.txt"
    f.write_text("hello", encoding="utf-8")
    monkeypatch.setenv("FIXPIC_NOTICES", str(f))
    assert ad.find_notices_path() == f and ad.read_notices()[0] == "hello"


def test_missing_returns_none_and_lists_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("FIXPIC_NOTICES", str(tmp_path / "nope.txt"))
    monkeypatch.setattr(ad, "notices_candidates", lambda: [tmp_path / "nope.txt"])
    text, cands = ad.read_notices()
    assert text is None and cands == [tmp_path / "nope.txt"]


def test_dialog_shows_notices_text(app, monkeypatch, tmp_path):
    f = tmp_path / "n.txt"
    f.write_text("THIRD-PARTY NOTICES\nnumpy BSD", encoding="utf-8")
    monkeypatch.setenv("FIXPIC_NOTICES", str(f))
    d = ad.AboutDialog()
    assert d.notices_found and "numpy BSD" in d.notices.toPlainText() and d.notices.isReadOnly()
    assert "Big LaMa" in d.body.text() and ad.app_version() in d.body.text()
    assert d.windowTitle() == strings.t("dialog.about_title")


def test_dialog_missing_file_shows_message_with_paths(app, monkeypatch, tmp_path):
    p = tmp_path / "gone.txt"
    monkeypatch.setattr(ad, "notices_candidates", lambda: [p])
    d = ad.AboutDialog()
    assert not d.notices_found and str(p) in d.notices.toPlainText() and "THIRD_PARTY_NOTICES.txt" in d.notices.toPlainText()


def test_close_button_accepts(app):
    d = ad.AboutDialog()
    d.show()
    d.btn_close.click()
    assert d.result() == QDialog.Accepted


def test_help_menu_has_about_and_action_opens_dialog(app, monkeypatch):
    w = MainWindow()
    titles = [a.text() for a in w.menuBar().actions()]
    assert strings.t("menu.help") in titles
    help_menu = next(a.menu() for a in w.menuBar().actions() if a.text() == strings.t("menu.help"))
    assert w.act_about in help_menu.actions()
    opened = []
    monkeypatch.setattr(ad.AboutDialog, "exec", lambda self: opened.append(self) or 0)
    w.act_about.trigger()
    assert len(opened) == 1 and isinstance(opened[0], ad.AboutDialog)
