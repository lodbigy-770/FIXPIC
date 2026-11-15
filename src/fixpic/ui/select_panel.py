"""לוח הבחירה החכמה (M4, AUTO-SELECT.md §4 ו-§7 שלב 4ב): בחירת כלי (מברשת / מלבן / לאסו), אישור וביטול הצעה, ורמז.

חוזים:
- הלוח אינו מכיר את הקנבס ואת הבקר: הוא משדר `toolChanged(str)` (`"brush"` | `"box"` | `"lasso"`), `acceptRequested` ו-
  `cancelRequested`, והחלון הראשי מחבר. `set_state(has_proposal, has_input)` מעדכן כפתורים ורמז.
- כשהחלון עסוק (`set_enabled_for_work(False)`) כל הלוח מושבת.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QLabel, QPushButton, QRadioButton, QVBoxLayout, QWidget

from ..strings import t

TOOLS = ("brush", "box", "lasso")


class SelectPanel(QWidget):
    toolChanged = Signal(str)
    acceptRequested = Signal()
    cancelRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tool = "brush"
        self._work_ok = True
        self._has_proposal = False
        self._has_input = False
        self.radios: dict[str, QRadioButton] = {
            "brush": QRadioButton(t("select.tool_brush")),
            "box": QRadioButton(t("select.tool_box")),
            "lasso": QRadioButton(t("select.tool_lasso")),
        }
        self._group = QButtonGroup(self)
        lay = QVBoxLayout(self)
        for name, r in self.radios.items():
            self._group.addButton(r)
            lay.addWidget(r)
            r.toggled.connect(lambda on, n=name: self._radio_toggled(n, on))
        self.radios["brush"].setChecked(True)
        self.lbl_hint = QLabel()
        self.lbl_hint.setWordWrap(True)
        lay.addWidget(self.lbl_hint)
        self.btn_accept = QPushButton(t("select.accept"))
        self.btn_cancel = QPushButton(t("select.cancel"))
        lay.addWidget(self.btn_accept)
        lay.addWidget(self.btn_cancel)
        lay.addStretch(1)
        self.btn_accept.clicked.connect(lambda _c=False: self.acceptRequested.emit())
        self.btn_cancel.clicked.connect(lambda _c=False: self.cancelRequested.emit())
        self.set_state(False, False)

    # ------------------------------------------------------------------ מצב
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

    def set_enabled_for_work(self, ok: bool) -> None:
        self._work_ok = ok
        self.set_state(self._has_proposal, self._has_input)

    def set_state(self, has_proposal: bool, has_input: bool) -> None:
        self._has_proposal, self._has_input = has_proposal, has_input
        self.setEnabled(self._work_ok)
        self.btn_accept.setEnabled(has_proposal)
        self.btn_cancel.setEnabled(has_input or has_proposal)
        if self._tool == "brush":
            hint = t("select.hint_brush")
        elif has_input:
            hint = t("select.hint_scribble")
        else:
            hint = t("select.hint_box") if self._tool == "box" else t("select.hint_lasso")
        self.lbl_hint.setText(hint)
