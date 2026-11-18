"""לוח הבחירה החכמה (M4, AUTO-SELECT.md §4 ו-§7 שלבים 4א/4ב): בחירת כלי (מברשת / שרביט / מלבן / לאסו), אישור וביטול הצעה, ורמז.

חוזים:
- הלוח אינו מכיר את הקנבס ואת הבקר: הוא משדר `toolChanged(str)` (`"brush"` | `"wand"` | `"box"` | `"lasso"` | `"mask_fill"` | `"mask_erase"` | `"mask_component"`; שלושת האחרונים עורכים את המסכה ישירות), `acceptRequested`,
  `cancelRequested` ו-`toleranceChanged(int)` (רק משינוי של המשתמש בסליידר/ספינר, שפעילים בשרביט בלבד; `set_tolerance` מבחוץ
  אינו משדר), והחלון הראשי מחבר. `set_state(has_proposal, has_input)` מעדכן כפתורים ורמז.
- כשהחלון עסוק (`set_enabled_for_work(False)`) כל הלוח מושבת.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QHBoxLayout, QLabel, QPushButton, QRadioButton, QSlider, QSpinBox,
                               QVBoxLayout, QWidget)

from ..strings import t
from .wand_tool import MAX_UI_TOLERANCE

TOOLS = ("brush", "wand", "box", "lasso", "mask_fill", "mask_erase", "mask_component")


class SelectPanel(QWidget):
    toolChanged = Signal(str)
    acceptRequested = Signal()
    cancelRequested = Signal()
    toleranceChanged = Signal(int)
    removeBackgroundRequested = Signal()
    makeTransparentRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tool = "brush"
        self._work_ok = True
        self._has_proposal = False
        self._has_input = False
        self.radios: dict[str, QRadioButton] = {
            "brush": QRadioButton(t("select.tool_brush")),
            "wand": QRadioButton(t("select.tool_wand")),
            "box": QRadioButton(t("select.tool_box")),
            "lasso": QRadioButton(t("select.tool_lasso")),
            "mask_fill": QRadioButton(t("select.tool_mask_fill")),
            "mask_erase": QRadioButton(t("select.tool_mask_erase")),
            "mask_component": QRadioButton(t("select.tool_mask_component")),
        }
        self._group = QButtonGroup(self)
        lay = QVBoxLayout(self)
        for name, r in self.radios.items():
            self._group.addButton(r)
            lay.addWidget(r)
            r.toggled.connect(lambda on, n=name: self._radio_toggled(n, on))
        self.radios["brush"].setChecked(True)
        self.slider = QSlider(Qt.Horizontal)                  # סבילות השרביט
        self.slider.setRange(0, MAX_UI_TOLERANCE)
        self.spin = QSpinBox()
        self.spin.setRange(0, MAX_UI_TOLERANCE)
        row = QHBoxLayout()
        row.addWidget(QLabel(t("select.tolerance")))
        row.addWidget(self.spin)
        lay.addLayout(row)
        lay.addWidget(self.slider)
        self.slider.valueChanged.connect(self._tolerance_from_user)
        self.spin.valueChanged.connect(self._tolerance_from_user)
        self.lbl_hint = QLabel()
        self.lbl_hint.setWordWrap(True)
        lay.addWidget(self.lbl_hint)
        self.btn_accept = QPushButton(t("select.accept"))
        self.btn_cancel = QPushButton(t("select.cancel"))
        self.btn_bg = QPushButton(t("select.remove_bg"))
        lay.addWidget(self.btn_accept)
        self.btn_tr = QPushButton(t("select.make_transparent"))
        lay.addWidget(self.btn_bg)
        lay.addWidget(self.btn_tr)
        lay.addWidget(self.btn_cancel)
        lay.addStretch(1)
        self.btn_accept.clicked.connect(lambda _c=False: self.acceptRequested.emit())
        self.btn_cancel.clicked.connect(lambda _c=False: self.cancelRequested.emit())
        self.btn_bg.clicked.connect(lambda _c=False: self.removeBackgroundRequested.emit())
        self.btn_tr.clicked.connect(lambda _c=False: self.makeTransparentRequested.emit())
        self.set_mask_tools_visible(False)
        self.set_state(False, False)

    # ------------------------------------------------------------------ מצב
    @property
    def tolerance(self) -> int:
        return self.slider.value()

    def set_tolerance(self, v: int) -> None:
        """קובע את הסבילות המוצגת בלי לשדר `toleranceChanged`."""
        for w in (self.slider, self.spin):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)

    def _tolerance_from_user(self, v: int) -> None:
        self.set_tolerance(v)
        self.toleranceChanged.emit(v)

    @property
    def tool(self) -> str:
        return self._tool

    def set_tool(self, tool: str) -> None:
        if tool not in TOOLS:
            raise ValueError(f"כלי לא מוכר: {tool}")
        self.radios[tool].setChecked(True)          # משדר toolChanged דרך _radio_toggled אם השתנה

    def _radio_toggled(self, name: str, on: bool) -> None:
        if on and name != self._tool:
            self._tool = name
            self.set_state(self._has_proposal, self._has_input)
            self.toolChanged.emit(name)

    def set_mask_tools_visible(self, on: bool) -> None:
        """שלושת כלי עריכת המסכה (לאסו מילוי/מחיקה, מחיקת רכיב) מוצגים רק כשיש מסכה נצברת (החלה ידנית)."""
        for n in ("mask_fill", "mask_erase", "mask_component"):
            self.radios[n].setVisible(on)
        if not on and self._tool.startswith("mask_"):
            self.set_tool("brush")

    def set_enabled_for_work(self, ok: bool) -> None:
        self._work_ok = ok
        self.set_state(self._has_proposal, self._has_input)

    def set_state(self, has_proposal: bool, has_input: bool) -> None:
        self._has_proposal, self._has_input = has_proposal, has_input
        self.setEnabled(self._work_ok)
        self.btn_accept.setEnabled(has_proposal)
        self.btn_bg.setEnabled(has_proposal)
        self.btn_tr.setEnabled(has_proposal)
        self.btn_cancel.setEnabled(has_input or has_proposal)
        self.slider.setEnabled(self._tool == "wand")
        self.spin.setEnabled(self._tool == "wand")
        if self._tool == "brush":
            hint = t("select.hint_brush")
        elif self._tool == "wand":
            hint = t("select.hint_wand")
        elif has_input:
            hint = t("select.hint_scribble")
        else:
            hint = t({"box": "select.hint_box", "lasso": "select.hint_lasso", "mask_fill": "select.hint_mask_fill",
                      "mask_erase": "select.hint_mask_erase", "mask_component": "select.hint_mask_component"}[self._tool])
        self.lbl_hint.setText(hint)
