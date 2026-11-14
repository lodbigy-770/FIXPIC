"""M4 (ממשק): לוח כלי המסכה `MaskToolsPanel` וחיבורו לחלון הראשי. Qt על פלטפורמת offscreen."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from fixpic import strings  # noqa: E402
from fixpic.imgio import image_from_array  # noqa: E402
from fixpic.session import MaskLayer  # noqa: E402
from fixpic.ui.app import create_app  # noqa: E402
from fixpic.ui.main_window import MainWindow  # noqa: E402
from fixpic.ui.mask_panel import MaskToolsPanel  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return create_app([])


def painted(h=100, w=120):
    layer = MaskLayer(h, w)
    layer.paint([(30, 30), (60, 30)], 10)
    return layer


def panel_for(layer):
    p = MaskToolsPanel()
    p.bind(layer)
    p.changes = []
    p.maskChanged.connect(p.changes.append)
    p.errors = []
    p.failed.connect(p.errors.append)
    return p


def test_unbound_is_inert(app):
    p = MaskToolsPanel()
    assert not p.active
    assert not any(b.isEnabled() for b in (p.btn_grow, p.btn_shrink, p.btn_small, p.btn_clear, p.btn_undo, p.btn_redo))
    assert p.grow() is None and p.undo() is None and p.clear() is None


def test_empty_mask_disables_edit_buttons(app):
    p = panel_for(MaskLayer(50, 50))
    assert p.active
    assert not p.btn_grow.isEnabled() and not p.btn_clear.isEnabled()
    assert p.spin_px.isEnabled() and not p.btn_undo.isEnabled()


def test_grow_and_shrink_emit_rect_and_undo(app):
    layer = painted()
    before = layer.snapshot()
    n0 = layer.pixel_count
    p = panel_for(layer)
    rect = p.grow(4)
    assert rect is not None and layer.pixel_count > n0
    assert p.changes == [rect] and p.btn_undo.isEnabled()
    grown = layer.snapshot()
    p.shrink(2)
    assert layer.pixel_count < grown.sum() and len(p.changes) == 2
    p.undo()
    assert np.array_equal(layer.snapshot(), grown)
    p.undo()
    assert np.array_equal(layer.snapshot(), before)
    assert not p.btn_undo.isEnabled() and p.btn_redo.isEnabled()
    p.redo()
    assert np.array_equal(layer.snapshot(), grown)


def test_buttons_use_spinbox_values(app):
    layer = painted()
    p = panel_for(layer)
    p.spin_px.setValue(5)
    n0 = layer.pixel_count
    p.btn_grow.click()
    assert layer.pixel_count > n0
    n1 = layer.pixel_count
    p.btn_shrink.click()
    assert layer.pixel_count < n1


def test_noop_does_not_emit_or_record(app):
    layer = MaskLayer(50, 50)
    p = panel_for(layer)
    assert p.grow(3) is None and p.remove_small(10) is None and p.clear() is None
    assert p.changes == [] and not p.btn_undo.isEnabled()


def test_remove_small_keeps_big_component(app):
    layer = painted()
    layer.paint([(100, 80)], 2)            # כתם קטן
    total = layer.pixel_count
    p = panel_for(layer)
    rect = p.remove_small(30)
    assert isinstance(rect, tuple) and len(rect) == 4 and 0 < layer.pixel_count < total
    p.undo()
    assert layer.pixel_count == total
    assert p.remove_small(1) is None


def test_remove_component_at(app):
    layer = painted()
    layer.paint([(100, 80)], 6)
    p = panel_for(layer)
    assert p.remove_component_at(0, 0) is None            # לא על רכיב
    assert p.remove_component_at(100, 80) is not None
    assert len(layer.components()) == 1
    p.undo()
    assert len(layer.components()) == 2


def test_clear_is_one_undoable_step(app):
    layer = painted()
    before = layer.snapshot()
    p = panel_for(layer)
    rect = p.clear()
    assert isinstance(rect, tuple) and len(rect) == 4 and layer.empty and not p.btn_clear.isEnabled()
    p.undo()
    assert np.array_equal(layer.snapshot(), before)


def test_work_disabled_blocks_actions(app):
    layer = painted()
    n0 = layer.pixel_count
    p = panel_for(layer)
    p.set_enabled_for_work(False)
    assert not p.btn_grow.isEnabled() and p.grow(3) is None and p.clear() is None
    assert layer.pixel_count == n0
    p.set_enabled_for_work(True)
    assert p.btn_grow.isEnabled()


def test_external_change_makes_history_stale(app):
    layer = painted()
    p = panel_for(layer)
    p.grow(2)
    assert p.btn_undo.isEnabled()
    layer.paint([(10, 10)], 4)             # שינוי מבחוץ (למשל submit_stroke)
    p.refresh()
    assert not p.btn_undo.isEnabled()


def test_bind_new_layer_resets_history(app):
    p = panel_for(painted())
    p.grow(2)
    p.bind(painted())
    assert not p.btn_undo.isEnabled()
    p.bind(None)
    assert not p.active


def test_strings_exist(app):
    for k in ("dock.mask", "mask.amount", "mask.grow", "mask.shrink", "mask.min_area",
              "mask.remove_small", "mask.clear", "mask.undo", "mask.redo"):
        assert strings.t(k).strip()


# ------------------------------------------------------------------------------ חיבור לחלון הראשי
def make_window():
    w = MainWindow()
    w._error = lambda key, **f: None
    w.show()
    return w


def test_window_has_dock_and_binds_session(app):
    w = make_window()
    assert w.mask_dock.widget() is w.mask_panel and not w.mask_panel.active
    img = image_from_array(np.random.default_rng(0).integers(0, 256, (60, 80, 3), dtype=np.uint8))
    w._install(img, "x")
    assert w.mask_panel.layer is w.session.mask and w.mask_panel.active
    w.session.mask.paint([(20, 20), (40, 20)], 8)
    w.mask_panel.refresh()
    assert w.mask_panel.btn_grow.isEnabled()
    w.mask_panel.grow(3)
    assert w.mask_panel.btn_undo.isEnabled()
    w.close()


def test_window_busy_disables_panel(app):
    w = make_window()
    img = image_from_array(np.random.default_rng(1).integers(0, 256, (60, 80, 3), dtype=np.uint8))
    w._install(img, "x")
    w._set_busy(True, "…")
    assert not w.mask_panel.active
    w._set_busy(False)
    assert w.mask_panel.active
    w.close()
    QApplication.processEvents()
