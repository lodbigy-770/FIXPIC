"""לוח המברשת (M3.4): גודל (פיקסלי תמונה) ומצב מחיקה.

הלוח אינו מכיר את ה-Canvas או ה-Session: הוא משדר `diameterChanged(int)` ו-`eraseChanged(bool)`.
הגודל הוא בפיקסלי **תמונה** (כמו `brush_px` של ה-Job, ולכן הוא קובע גם את ההרחבה האוטומטית של המסכה). ברירת המחדל
והטווח נקבעים לפי גודל התמונה ב-`configure_for_image`, כך שעל תמונת 24MP המברשת אינה זעירה במסך.

Feathering (M4): תיבת סימון "לפי ברירת מחדל" (מסומנת = `feather_px` הוא None, כלומר לפי ה-cfg של המנוע) וסליידר/ספינר
0..`MAX_FEATHER` בפיקסלי תמונה (0 = מיזוג בינארי). הלוח משדר `featherChanged(object)` (None או מספר שלם).
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QGridLayout, QLabel, QSlider, QSpinBox, QWidget

from ..strings import t

DEFAULT_FRACTION = 0.03        # ברירת מחדל: 3% מהצלע הקצרה של התמונה
MIN_DEFAULT_PX = 8
MIN_RANGE_MAX = 32
MAX_RANGE_MAX = 2000
MAX_FEATHER = 64               # כמו `pipeline.pipeline.MAX_FEATHER`
DEFAULT_FEATHER_SHOWN = 3      # הערך שמוצג בסליידר כשעוברים מ"ברירת מחדל" לידני (כמו `PipelineConfig.feather_px`)


def brush_range_for(width: int, height: int) -> tuple[int, int]:
    """(ברירת מחדל, מקסימום) לפי גודל תמונה."""
    short = max(1, min(width, height))
    top = min(MAX_RANGE_MAX, max(MIN_RANGE_MAX, short // 4))
    default = min(top, max(MIN_DEFAULT_PX, round(short * DEFAULT_FRACTION)))
    return default, top


class BrushPanel(QWidget):
    diameterChanged = Signal(int)
    eraseChanged = Signal(bool)
    featherChanged = Signal(object)         # None (ברירת מחדל) | int

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.slider = QSlider(Qt.Horizontal)
        self.spin = QSpinBox()
        self.spin.setSuffix(" px")
        self.chk_erase = QCheckBox(t("brush.erase"))
        self.chk_feather_auto = QCheckBox(t("brush.feather_auto"))
        self.chk_feather_auto.setChecked(True)
        self.feather_slider = QSlider(Qt.Horizontal)
        self.feather_spin = QSpinBox()
        self.feather_spin.setSuffix(" px")
        for w in (self.feather_slider, self.feather_spin):
            w.setRange(0, MAX_FEATHER)
            w.setValue(DEFAULT_FEATHER_SHOWN)
            w.setEnabled(False)
        self._set_range(1, 200)
        self.slider.setValue(20)
        self.spin.setValue(20)
        g = QGridLayout(self)
        g.addWidget(QLabel(t("brush.size")), 0, 0)
        g.addWidget(self.spin, 0, 1)
        g.addWidget(self.slider, 1, 0, 1, 2)
        g.addWidget(self.chk_erase, 2, 0, 1, 2)
        g.addWidget(QLabel(t("brush.feather")), 3, 0)
        g.addWidget(self.feather_spin, 3, 1)
        g.addWidget(self.feather_slider, 4, 0, 1, 2)
        g.addWidget(self.chk_feather_auto, 5, 0, 1, 2)
        g.setRowStretch(6, 1)
        self.feather_slider.valueChanged.connect(self.set_feather)
        self.feather_spin.valueChanged.connect(self.set_feather)
        self.chk_feather_auto.toggled.connect(self._feather_auto_toggled)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self.chk_erase.toggled.connect(self.eraseChanged)

    # ----------------------------------------------------------------- מצב
    @property
    def diameter(self) -> int:
        return self.spin.value()

    @property
    def erase(self) -> bool:
        return self.chk_erase.isChecked()

    @property
    def feather_px(self) -> int | None:
        """None = לפי ברירת המחדל של המנוע; אחרת רוחב ה-Feathering שנבחר (0..MAX_FEATHER)."""
        return None if self.chk_feather_auto.isChecked() else self.feather_spin.value()

    def set_feather(self, px: int | None) -> None:
        """None = חזרה לברירת מחדל; מספר = ידני (נחתך לטווח) ומשדר."""
        if px is None:
            self.chk_feather_auto.setChecked(True)
            return
        px = max(0, min(MAX_FEATHER, int(round(px))))
        for w in (self.feather_slider, self.feather_spin):
            w.blockSignals(True)
            w.setValue(px)
            w.blockSignals(False)
        if self.chk_feather_auto.isChecked():
            self.chk_feather_auto.setChecked(False)       # משדר דרך _feather_auto_toggled
        else:
            self.featherChanged.emit(px)

    def _feather_auto_toggled(self, auto: bool) -> None:
        for w in (self.feather_slider, self.feather_spin):
            w.setEnabled(not auto)
        self.featherChanged.emit(self.feather_px)

    def _set_range(self, lo: int, hi: int) -> None:
        for w in (self.slider, self.spin):
            w.blockSignals(True)
            w.setRange(lo, hi)
            w.blockSignals(False)

    def configure_for_image(self, width: int, height: int) -> int:
        """קובע טווח וברירת מחדל לתמונה חדשה ומשדר את הגודל. מחזיר את הגודל."""
        default, top = brush_range_for(width, height)
        self._set_range(1, top)
        self.set_diameter(default)
        return default

    def set_diameter(self, d: int) -> None:
        d = max(self.spin.minimum(), min(self.spin.maximum(), int(round(d))))
        for w in (self.slider, self.spin):
            w.blockSignals(True)
            w.setValue(d)
            w.blockSignals(False)
        self.diameterChanged.emit(d)

    def scale_diameter(self, factor: float) -> None:
        """הגדלה/הקטנה יחסית (קיצורי `]` ו-`[`), לפחות פיקסל אחד של שינוי."""
        cur = self.spin.value()
        new = round(cur * factor)
        if new == cur:
            new = cur + (1 if factor > 1 else -1)
        self.set_diameter(new)

    def set_erase(self, on: bool) -> None:
        self.chk_erase.setChecked(on)

    # --------------------------------------------------------------- פנימי
    def _from_slider(self, v: int) -> None:
        self.set_diameter(v)

    def _from_spin(self, v: int) -> None:
        self.set_diameter(v)
