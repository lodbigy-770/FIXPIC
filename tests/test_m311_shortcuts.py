"""M3.11: קיצורי מקלדת לזום והתאמה אישית של קיצורים."""
from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fixpic.ui.shortcuts import (ShortcutsDialog, ShortcutStore, find_conflicts, normalize,  # noqa: E402
                                 with_hebrew)
from test_m33 import app  # noqa: E402,F401
from test_m4_select_ui import SWin  # noqa: E402


def texts(action):
    return [s.toString(QKeySequence.PortableText) for s in action.shortcuts()]


def test_normalize_and_hebrew_twins():
    assert normalize("ctrl+z") == "Ctrl+Z" and normalize("") is None and normalize("Ctrl+A, B") is None
    assert [s.toString() for s in with_hebrew(["G"])] == ["G", "ע"]
    assert [s.toString() for s in with_hebrew(["Ctrl+G"])] == ["Ctrl+G"]          # רק אות בודדת
    assert [s.toString() for s in with_hebrew(["+"])] == ["+"]


def test_find_conflicts():
    assert find_conflicts({"a": ["G"], "b": ["Ctrl+G"], "c": []}) == {}
    assert find_conflicts({"a": ["g"], "b": ["G"]}) == {"G": ["a", "b"]}


def test_store_roundtrip_and_bad_files(tmp_path):
    st = ShortcutStore(tmp_path / "s.json")
    assert st.load() == {}                                                   # אין קובץ
    st.save({"action.zoom_in": ["Z"], "action.undo": []})
    assert ShortcutStore(tmp_path / "s.json").load() == {"action.zoom_in": ["Z"], "action.undo": []}
    (tmp_path / "bad.json").write_text("{לא json", encoding="utf-8")
    assert ShortcutStore(tmp_path / "bad.json").load() == {}
    (tmp_path / "odd.json").write_text(json.dumps({"shortcuts": {"a": ["Ctrl+A, B"], "b": [3], "c": ["X"]}}))
    assert ShortcutStore(tmp_path / "odd.json").load() == {"c": ["X"]}        # ערכים לא תקינים נזרקים


def test_zoom_shortcuts_default_and_work(app):
    sw = SWin()
    try:
        w = sw.w
        assert {"Ctrl++", "+", "Ctrl+="} <= set(texts(w.act_zoom_in))
        assert {"Ctrl+-", "-"} <= set(texts(w.act_zoom_out))
        assert "0" in texts(w.act_fit) and "1" in texts(w.act_actual)
        w.act_actual.trigger()
        s0 = w.canvas.scale
        w.act_zoom_in.trigger()
        assert w.canvas.scale > s0
        w.act_zoom_out.trigger()
        assert abs(w.canvas.scale - s0) < 1e-6
    finally:
        sw.close()


def test_apply_overrides_and_reset(app, tmp_path):
    st = ShortcutStore(tmp_path / "s.json")
    st.save({"action.zoom_in": ["Z"], "action.undo": []})
    sw = SWin()
    try:
        w = sw.w
        w.shortcut_store = st
        w.apply_shortcuts()                                                  # טעינה מהקובץ
        assert texts(w.act_zoom_in) == ["Z", "ז"] and texts(w.act_undo) == []
        assert "Ctrl+Z" not in texts(w.act_undo)
        w.apply_shortcuts({})
        assert "Ctrl++" in texts(w.act_zoom_in) and "Ctrl+Z" in texts(w.act_undo)
    finally:
        sw.close()


def test_dialog_overrides_conflict_and_defaults(app):
    sw = SWin()
    try:
        w = sw.w
        dlg = ShortcutsDialog(w.shortcut_entries(), w)
        assert dlg.result_overrides() == {}                                  # בלי שינוי: אין החלפות
        keys = {e[0] for e in w.shortcut_entries()}
        assert {"action.zoom_in", "action.zoom_out", "action.remove_bg", "action.undo"} <= keys
        dlg._set("action.zoom_in", ["Y"])
        assert dlg.result_overrides() == {"action.zoom_in": ["Y"]}
        dlg._set("action.zoom_out", ["Y"])
        assert find_conflicts(dlg.assignment()) == {"Y": ["action.zoom_in", "action.zoom_out"]}
        dlg.reset_all()
        assert dlg.result_overrides() == {}
    finally:
        sw.close()


def test_dialog_keeps_existing_override_and_default_resets_it(app, tmp_path):
    sw = SWin()
    try:
        w = sw.w
        w.shortcut_store.save({"action.zoom_in": ["Y"]})
        dlg = ShortcutsDialog(w.shortcut_entries(), w)
        assert dlg.result_overrides() == {"action.zoom_in": ["Y"]}           # לא שונה: ההחלפה נשמרת
        dlg._set("action.zoom_in", ["Ctrl++"])                               # חזרה לברירת המחדל
        assert dlg.result_overrides() == {}
    finally:
        sw.close()


def test_edit_shortcuts_saves_and_applies(app, monkeypatch):
    sw = SWin()
    try:
        w = sw.w

        def fake_exec(self):
            self._set("action.zoom_in", ["Y"])
            return ShortcutsDialog.Accepted

        monkeypatch.setattr(ShortcutsDialog, "exec", fake_exec)
        w.edit_shortcuts()
        assert texts(w.act_zoom_in) == ["Y", "ט"]
        assert json.loads(w.shortcut_store.path.read_text(encoding="utf-8"))["shortcuts"] == {"action.zoom_in": ["Y"]}
        w2 = type(w)()                                                       # חלון חדש טוען את ההחלפה
        try:
            assert texts(w2.act_zoom_in) == ["Y", "ט"]
        finally:
            w2.close()
    finally:
        sw.close()
