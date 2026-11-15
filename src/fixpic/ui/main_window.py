"""החלון הראשי (M3.3): פתיחה (דיאלוג/גרירה/הדבקה), Canvas עם Zoom/Pan, שמירה/ייצוא ברקע עם מחוון.

חוזים:
- `Session` חי רק ב-thread של ה-UI. טעינה ושמירה רצות ב-`BackgroundRunner` ומקבלות נתונים בלתי-משתנים בלבד.
- פעולה אחת ברקע בכל זמן (`busy`): ניסיון לפתוח/לשמור בזמן פעולה אחרת מציג הודעה ואינו מתחיל.
- **שמירה (Ctrl+S) אינה דורסת את קובץ המקור**: היא שומרת לנתיב האחרון שנבחר ב"שמירה בשם", ואם אין כזה
  מתנהגת כמו "שמירה בשם" (כך אי אפשר לדרוס צילום מקורי בטעות, ובמיוחד לא לדחוס JPEG מחדש).
- שינויים שלא נשמרו: בפתיחת תמונה אחרת, בהדבקה ובסגירה, שואלים שמירה/התעלמות/ביטול.
- חלונות דיאלוג מבודדים במתודות (`_error`, `_ask_discard`, `_pick_open`, `_pick_save`) כדי שבדיקות יחליפו אותן.
- **לולאת העריכה (M3.4):** ה-Canvas במצב ציור כשיש תמונה ואין טעינה/שמירה ברקע. משיכות מגיעות ל-`EditController`
  (`edit_loop.py`), שמצייר ב-`Session.mask`, מגיש Job בשחרור ומבצע Commit בסיומו; החלון רק מגיב לאותות שלו
  (ציור מחדש, מחוון עיבוד אחרי 200ms, הודעות, דיאלוג שגיאה אחד בכל פעם).
- **Undo/Redo והשוואה (M3.5):** Ctrl+Z, Ctrl+Y ו-Ctrl+Shift+Z פועלים על `Session` דרך `EditController` (לא על היסטוריית
  המסכה שבלוח הכלים). השוואה לפני/אחרי: מתג (`C`) וגם החזקה (מקש `B`, או לחיצה ממושכת על כפתור בסרגל); בזמן השוואה אין
  ציור, והקנבס מציג את `Session.original_*`. מצב ההשוואה מתאפס בתמונה חדשה.
- **בחירה חכמה (M4, שלב 4ב):** לוח `SelectPanel` (dock) בוחר כלי: מברשת / מלבן (S) / לאסו (L); P חוזר למברשת. הקנבס מצייר
  מלבן/לאסו/משיכה מתקנת ומשדר אותם ל-`SmartSelectController` (GrabCut ברקע), וההצעה מצוירת בירוק. Enter מאשר (מוסיף למסכה
  בצעד Undo אחד של לוח כלי המסכה, בלי להפעיל Inpaint), Esc מבטל קודם מחוות פעילה, אחר כך הצעה/קלט, ורק אחרי זה עיבוד Inpaint.
  תמונה חדשה, Commit, Undo ו-Redo של התמונה זורקים את ההצעה. מחוון "הבחירה לוקחת זמן" בשורת המצב אחרי 200ms.
- **ייצוא:** ברזולוציה מלאה תמיד, בלי מגבלה. ב-JPG/WebP "שמירה בשם" שואלת איכות (1-100, ברירת מחדל 92, מבודד ב-`_ask_quality`);
  Ctrl+S חוזר על האיכות והנתיב האחרונים.
- **מנוע ההסרה** נטען ברקע רק אם ניתן `engine_factory` (`ui.app.main` נותן; בדיקות מזריקות מנוע מדומה). בלי מנוע
  משיכות רק מציירות מסכה.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (QDockWidget, QFileDialog, QInputDialog, QLabel, QMainWindow, QMessageBox, QProgressBar,
                               QToolBar, QToolButton)

from ..casefile import save_case
from ..errors import FixPicError, InvalidInputError
from ..imgio import LoadedImage, SaveResult, image_from_array, load_image, save_image
from ..session import Session
from ..strings import t
from .brush_panel import BrushPanel
from .canvas import Canvas
from .convert import qimage_to_array
from .edit_loop import EditController
from .mask_panel import MaskToolsPanel
from .select_panel import SelectPanel
from .select_tool import SmartSelectController
from .workers import BackgroundRunner

log = logging.getLogger("fixpic.ui")

_EXT_BY_FILTER = {"filter.png": ".png", "filter.jpg": ".jpg", "filter.webp": ".webp"}
_FILTER_BY_FORMAT = {"JPEG": "filter.jpg", "WEBP": "filter.webp"}
_LOSSY = {".jpg", ".jpeg", ".webp"}
DEFAULT_QUALITY = 92
_HEBREW_KEY = {"C": "ב", "E": "ק", "P": "פ", "S": "ד", "L": "ך", "M": "צ", "D": "ג"}     # קיצורי אות במקלדת עברית


class MainWindow(QMainWindow):
    def __init__(self, engine_factory: Callable | None = None) -> None:
        """`engine_factory`: פונקציה בלי ארגומנטים שמחזירה `JobScheduler` (שטרם הופעל); רצה ב-thread ברקע."""
        super().__init__()
        self.session: Session | None = None
        self.display_name: str = ""
        self.save_path: Path | None = None
        self.busy = False
        self._error_open = False
        self._cmp_toggle = False
        self._cmp_hold = False
        self.save_quality = DEFAULT_QUALITY       # איכות JPG/WebP; נשאלת ב"שמירה בשם" ונשמרת ל-Ctrl+S
        self._runner = BackgroundRunner(self)
        self.canvas = Canvas(self)
        self.controller = EditController(self)
        self.select_ctl = SmartSelectController(self, history_source=lambda: self.mask_panel.history,
                                                busy=self._inpaint_busy)
        self.setCentralWidget(self.canvas)
        self.setAcceptDrops(True)
        self.resize(1100, 760)
        self._build_actions()
        self._build_status()
        self._build_brush_dock()
        self._build_mask_dock()
        self._build_select_dock()
        self.canvas.scaleChanged.connect(self._update_status)
        self._connect_editing()
        self._connect_select()
        self._refresh()
        if engine_factory is not None:
            self._start_engine(engine_factory)

    # ------------------------------------------------------------------ בנייה
    def _action(self, key: str, slot: Callable, shortcut=None) -> QAction:
        a = QAction(t(key), self)
        if shortcut is not None:
            he = _HEBREW_KEY.get(shortcut) if isinstance(shortcut, str) else None
            if he:      # אות לטינית + האות העברית שעל אותו מקש (פריסה עברית סטנדרטית)
                a.setShortcuts([QKeySequence(shortcut), QKeySequence(he)])
            else:
                a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(lambda _checked=False: slot())
        self.addAction(a)
        return a

    def _build_actions(self) -> None:
        self.act_open = self._action("action.open", self.open_dialog, QKeySequence.Open)
        self.act_save = self._action("action.save", self.save, QKeySequence.Save)
        self.act_save_as = self._action("action.save_as", self.save_as, QKeySequence("Ctrl+Shift+S"))
        self.act_paste = self._action("action.paste", self.paste, QKeySequence.Paste)
        self.act_quit = self._action("action.quit", self.close, QKeySequence("Ctrl+Q"))
        self.act_fit = self._action("action.fit", self.canvas.fit, "Ctrl+0")
        self.act_actual = self._action("action.actual_size", self.canvas.actual_size, "Ctrl+1")
        self.act_zoom_in = self._action("action.zoom_in", lambda: self._zoom(1.25), QKeySequence.ZoomIn)
        self.act_zoom_out = self._action("action.zoom_out", lambda: self._zoom(0.8), QKeySequence.ZoomOut)
        self.act_cancel = self._action("action.cancel_jobs", self._escape, "Esc")
        self.act_undo = self._action("action.undo", self._undo, QKeySequence.Undo)
        self.act_redo = self._action("action.redo", self._redo, QKeySequence.Redo)
        self.act_redo.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        self.act_compare = self._action("action.compare_toggle", lambda: None, "C")
        self.act_compare.setCheckable(True)
        self.act_compare.toggled.connect(self._compare_toggled)
        self.act_erase = self._action("action.erase_mode", lambda: self.brush_panel.set_erase(not self.brush_panel.erase),
                                      "E")
        self.act_brush_bigger = self._action("action.brush_bigger", lambda: self.brush_panel.scale_diameter(1.2), "]")
        self.act_brush_smaller = self._action("action.brush_smaller", lambda: self.brush_panel.scale_diameter(1 / 1.2),
                                              "[")
        self.act_tool_brush = self._action("action.select_brush", lambda: self.select_panel.set_tool("brush"), "P")
        self.act_tool_box = self._action("action.select_box", lambda: self.select_panel.set_tool("box"), "S")
        self.act_tool_lasso = self._action("action.select_lasso", lambda: self.select_panel.set_tool("lasso"), "L")
        self.act_tool_mask_fill = self._action("action.mask_lasso_fill", lambda: self.select_panel.set_tool("mask_fill"), "M")
        self.act_tool_mask_erase = self._action("action.mask_lasso_erase",
                                                lambda: self.select_panel.set_tool("mask_erase"), "D")
        self.act_select_accept = self._action("action.select_accept", self._select_accept, "Return")
        self.act_select_accept.setShortcuts([QKeySequence("Return"), QKeySequence("Enter")])
        self.act_save_case = self._action("action.save_case", self.save_case)
        mb = self.menuBar()
        m = mb.addMenu(t("menu.file"))
        for a in (self.act_open, self.act_save, self.act_save_as, self.act_paste):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.act_save_case)
        m.addSeparator()
        m.addAction(self.act_quit)
        ed = mb.addMenu(t("menu.edit"))
        for a in (self.act_undo, self.act_redo, self.act_cancel, self.act_erase, self.act_brush_bigger,
                  self.act_brush_smaller, self.act_compare):
            ed.addAction(a)
        ed.addSeparator()
        for a in (self.act_tool_brush, self.act_tool_box, self.act_tool_lasso, self.act_tool_mask_fill,
                  self.act_tool_mask_erase, self.act_select_accept):
            ed.addAction(a)
        v = mb.addMenu(t("menu.view"))
        for a in (self.act_zoom_in, self.act_zoom_out, self.act_fit, self.act_actual):
            v.addAction(a)
        tb = QToolBar(t("toolbar.main"), self)
        tb.setMovable(False)
        for a in (self.act_open, self.act_save, self.act_paste, self.act_undo, self.act_redo, self.act_fit,
                  self.act_actual, self.act_cancel, self.act_compare):
            tb.addAction(a)
        self.btn_hold = QToolButton()                         # החזקה בעכבר: לחיצה מציגה את המקור, שחרור מחזיר
        self.btn_hold.setText(t("action.compare_hold"))
        self.btn_hold.pressed.connect(lambda: self._compare_hold(True))
        self.btn_hold.released.connect(lambda: self._compare_hold(False))
        tb.addWidget(self.btn_hold)
        self.addToolBar(tb)

    def _build_brush_dock(self) -> None:
        self.brush_panel = BrushPanel(self)
        self.brush_dock = QDockWidget(t("dock.brush"), self)
        self.brush_dock.setWidget(self.brush_panel)
        self.brush_dock.setFeatures(QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.brush_dock)

    def _build_mask_dock(self) -> None:
        """לוח כלי המסכה (M4). הלוח משנה את `Session.mask`; ציור שכבת המסכה על ה-Canvas הוא של M3.4."""
        self.mask_panel = MaskToolsPanel(self)
        self.mask_dock = QDockWidget(t("dock.mask"), self)
        self.mask_dock.setWidget(self.mask_panel)
        self.mask_dock.setFeatures(QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.mask_dock)
        self.mask_panel.maskChanged.connect(lambda _rect: self.canvas.update())
        self.mask_panel.failed.connect(lambda detail: self._error("error.unexpected", detail=detail))

    def _build_select_dock(self) -> None:
        self.select_panel = SelectPanel(self)
        self.select_dock = QDockWidget(t("dock.select"), self)
        self.select_dock.setWidget(self.select_panel)
        self.select_dock.setFeatures(QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.select_dock)

    def _build_status(self) -> None:
        sb = self.statusBar()
        self.lbl_size, self.lbl_zoom = QLabel(), QLabel()
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)               # מחוון "עסוק" (אין התקדמות מדידה בטעינה/קידוד)
        self.progress.setFixedWidth(140)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.lbl_cmp = QLabel()
        self.lbl_cmp.hide()
        sb.addPermanentWidget(self.lbl_cmp)
        self.lbl_sel = QLabel(t("select.working"))      # מחוון הבחירה החכמה (אחרי 200ms)
        self.lbl_sel.hide()
        sb.addPermanentWidget(self.lbl_sel)
        self.lbl_proc = QLabel()
        self.proc_progress = QProgressBar()        # עיבוד Inpaint; נפרד ממחוון הטעינה/שמירה
        self.proc_progress.setRange(0, 0)
        self.proc_progress.setFixedWidth(140)
        self.proc_progress.setTextVisible(False)
        self.lbl_proc.hide()
        self.proc_progress.hide()
        sb.addPermanentWidget(self.lbl_proc)
        sb.addPermanentWidget(self.proc_progress)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.lbl_size)
        sb.addPermanentWidget(self.lbl_zoom)
        sb.showMessage(t("status.ready"))

    # ------------------------------------------------------------------- מצב
    def _zoom(self, factor: float) -> None:
        self.canvas.zoom_at(self.canvas.rect().center().toPointF(), factor)

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.busy = busy
        self.progress.setVisible(busy)
        if message:
            self.statusBar().showMessage(message)
        self._refresh()

    def _refresh(self) -> None:
        has = self.session is not None
        self.act_save.setEnabled(has and not self.busy)
        self.act_save_as.setEnabled(has and not self.busy)
        self.act_open.setEnabled(not self.busy)
        self.mask_panel.set_enabled_for_work(not self.busy)
        self.act_paste.setEnabled(not self.busy)
        for a in (self.act_fit, self.act_actual, self.act_zoom_in, self.act_zoom_out):
            a.setEnabled(has)
        working = has and (self.session.busy or self.controller.waiting is not None)
        sel = self.select_ctl
        self.act_cancel.setEnabled(working or sel.active or self.canvas.gesture_active)
        can_select = has and not self.busy and not self._comparing
        self.select_panel.set_enabled_for_work(can_select)
        self.select_panel.set_state(sel.has_proposal, sel.has_input)
        self.act_select_accept.setEnabled(can_select and sel.has_proposal)
        for a in (self.act_tool_brush, self.act_tool_box, self.act_tool_lasso, self.act_tool_mask_fill,
                  self.act_tool_mask_erase):
            a.setEnabled(can_select)
        c = self.controller
        self.mask_panel.refresh()
        self.act_undo.setEnabled(has and not self.busy and (c.can_undo or self._mask_can(False)))
        self.act_redo.setEnabled(has and not self.busy and (c.can_redo or self._mask_can(True)))
        self.act_compare.setEnabled(has)
        self.btn_hold.setEnabled(has)
        painting = has and not self.busy and not self._comparing
        if painting != self.canvas.painting:
            self.canvas.set_painting(painting)
        for a in (self.act_erase, self.act_brush_bigger, self.act_brush_smaller):
            a.setEnabled(has)
        self.brush_panel.setEnabled(has)
        mod = t("title.modified") if has and self.session.modified else ""
        name = self.display_name or t("title.untitled")
        self.setWindowTitle(f"{name}{mod} - {t('app.name')}" if has else t("app.name"))
        self._update_status()

    def _update_status(self, *_) -> None:
        if self.session is None:
            self.lbl_size.setText("")
            self.lbl_zoom.setText("")
            return
        w, h = self.session.size
        self.lbl_size.setText(t("status.size", w=w, h=h))
        self.lbl_zoom.setText(t("status.zoom", percent=round(self.canvas.scale * 100)))

    # -------------------------------------------------------------- דיאלוגים
    def _error(self, key: str, **fmt) -> None:
        QMessageBox.warning(self, t("error.title"), t(key, **fmt))

    def _error_once(self, key: str, **fmt) -> None:
        """כמו `_error`, אך בזמן שדיאלוג שגיאה פתוח (המשך כשלי עיבוד ברצף) רק מעדכן את שורת המצב."""
        if self._error_open:
            self.statusBar().showMessage(t(key, **fmt).splitlines()[0])
            return
        self._error_open = True
        try:
            self._error(key, **fmt)
        finally:
            self._error_open = False

    def _ask_discard(self) -> str:
        """"save" / "discard" / "cancel"."""
        box = QMessageBox(QMessageBox.Question, t("msg.discard_title"), t("msg.discard_text"), parent=self)
        b_save = box.addButton(t("msg.save"), QMessageBox.AcceptRole)
        b_discard = box.addButton(t("msg.discard"), QMessageBox.DestructiveRole)
        box.addButton(t("msg.cancel"), QMessageBox.RejectRole)
        box.exec()
        c = box.clickedButton()
        return "save" if c is b_save else "discard" if c is b_discard else "cancel"

    def _pick_open(self) -> str:
        return QFileDialog.getOpenFileName(self, t("dialog.open_title"), "", t("filter.images"))[0]

    def _pick_save(self, suggested: str, selected_filter: str) -> tuple[str, str]:
        filters = ";;".join(t(k) for k in _EXT_BY_FILTER)
        return QFileDialog.getSaveFileName(self, t("dialog.save_title"), suggested, filters, selected_filter)

    def _pick_dir(self) -> str:
        return QFileDialog.getExistingDirectory(self, t("dialog.case_title"))

    def save_case(self) -> None:
        """שומר את הקלט של המשיכה האחרונה (תמונה לפני, מסכה, פרמטרים) והתוצאה הנוכחית בתיקייה, כמקרה בדיקה איכות
        (`fixpic.casefile`). לבדיקות איכות על המודל האמיתי, בלי לצייר מחדש."""
        s = self.session
        if s is None or self.busy:
            return
        sub = s.last_submission
        if sub is None:
            self._error("msg.no_case")
            return
        folder = self._pick_dir()
        if not folder:
            return
        d = Path(folder) / f"case_{datetime.now():%Y%m%d_%H%M%S}"
        res, params = s.rgb, {"brush_px": sub.brush_px, "size": sub.size, "feather_px": sub.feather_px}
        self._set_busy(True, t("status.saving"))

        def work() -> Path:
            return save_case(d, mask=sub.mask, original_rgb=sub.rgb, result_rgb=res, params=params,
                             note="נשמר מהממשק (שמירת מקרה בדיקה): result = התמונה הנוכחית")

        def done(p: Path) -> None:
            self._set_busy(False, t("status.case_saved", name=p.name))

        def failed(e: BaseException) -> None:
            self._set_busy(False, t("status.ready"))
            self._report("error.save", e)

        self._runner.submit(work, done, failed)

    def _ask_quality(self, fmt: str, current: int) -> int | None:
        """איכות ייצוא (1-100) ל-JPG/WebP. None = ביטול."""
        v, ok = QInputDialog.getInt(self, t("dialog.quality_title"), t("dialog.quality_label", fmt=fmt), current, 1, 100, 1)
        return int(v) if ok else None

    def _report(self, key: str, e: BaseException) -> None:
        if isinstance(e, FixPicError):
            self._error(key, detail=str(e))
        else:
            log.error("שגיאה לא צפויה", exc_info=e)
            self._error("error.unexpected", detail=f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------- לולאת עריכה
    def _connect_editing(self) -> None:
        c, cv = self.controller, self.canvas
        cv.strokeStarted.connect(lambda pt, erase: c.begin_stroke(pt, erase))
        cv.strokeMoved.connect(c.extend_stroke)
        cv.strokeEnded.connect(c.end_stroke)
        self.brush_panel.diameterChanged.connect(self._brush_changed)
        self.brush_panel.eraseChanged.connect(self._erase_changed)
        c.changed.connect(self._edit_changed)
        c.imageChanged.connect(self._image_changed)
        cv.compareHoldChanged.connect(self._compare_hold)
        c.message.connect(self.statusBar().showMessage)
        c.failed.connect(lambda detail: self._error_once("error.inpaint", detail=detail))
        c.processingChanged.connect(self._processing_changed)
        c.progress.connect(self._processing_progress)
        self._brush_changed(self.brush_panel.diameter)

    def _brush_changed(self, d: int) -> None:
        self.controller.diameter = float(d)
        self.canvas.set_brush_diameter(d)

    def _erase_changed(self, on: bool) -> None:
        self.controller.erase_mode = on
        self.canvas.erase_mode = on
        self.canvas.update()

    def _edit_changed(self) -> None:
        self.canvas.update()
        self.mask_panel.refresh()
        self._refresh()
        if self.controller.indicator_on:
            self._show_processing_text()

    def _image_changed(self, box=None) -> None:
        """התמונה העובדת השתנתה (Commit, Undo, Redo): מרענן את הקנבס באזור `box` בלבד, בלי לאבד זום והזזה."""
        s = self.session
        if s is not None:
            self.canvas.update_image_region(s.rgb, s.alpha, box)
        self._refresh()

    # ------------------------------------------------------------ בחירה חכמה
    def _inpaint_busy(self) -> bool:
        """נקרא מה-Worker של הבחירה (קריאה בלבד): Inpaint פעיל או ממתין, ו-GrabCut ממתין לו."""
        s = self.session
        return s is not None and (s.busy or self.controller.waiting is not None)

    def _connect_select(self) -> None:
        cv, ctl, panel = self.canvas, self.select_ctl, self.select_panel
        cv.selectBoxDrawn.connect(ctl.request_box)
        cv.selectLassoDrawn.connect(ctl.request_lasso)
        cv.maskLassoDrawn.connect(self._mask_lasso)
        cv.selectScribbleDrawn.connect(lambda path, positive: ctl.request_scribble(
            path, positive, self.brush_panel.diameter / 2))
        panel.toolChanged.connect(self._tool_changed)
        panel.acceptRequested.connect(self._select_accept)
        panel.cancelRequested.connect(self.select_ctl.cancel)
        ctl.changed.connect(self._select_changed)
        ctl.message.connect(self.statusBar().showMessage)
        ctl.processingChanged.connect(self.lbl_sel.setVisible)
        ctl.accepted.connect(self._selection_accepted)
        self.controller.imageChanged.connect(ctl.on_image_changed)

    def _tool_changed(self, tool: str) -> None:
        self.select_ctl.cancel()
        self.canvas.set_select_mode(None if tool == "brush" else tool)
        self._select_changed()

    def _select_changed(self) -> None:
        ctl = self.select_ctl
        self.canvas.set_proposal(ctl.proposal)
        self.canvas.set_select_scribble(ctl.has_input)
        self._refresh()

    def _mask_lasso(self, points, erase: bool) -> None:
        """לאסו ישיר על המסכה (M4): מילוי = כמו שחרור מברשת (המסכה מוגשת ל-Inpaint); מחיקה = רק עדכון המסכה והתצוגה.
        שניהם צעד Undo אחד בלוח כלי המסכה."""
        if self.session is None or self.busy or self._comparing:
            return
        rect = self.mask_panel.fill_polygon(points, erase=erase)
        self.canvas.invalidate_overlay()
        self._refresh()
        if rect is not None and not erase:
            self.controller.submit()

    def _selection_accepted(self, _rect) -> None:
        """אישור הצעת בחירה = הסרה: המסכה שנוספה מוגשת ל-Inpaint בדיוק כמו אחרי שחרור מברשת (בדיקה ידנית ב-Windows:
        בחירה בלי הסרה הייתה מבלבלת). בלי מנוע המסכה רק נשארת מצוירת."""
        self.mask_panel.refresh()
        self.canvas.invalidate_overlay()
        self.controller.submit()

    def _select_accept(self) -> None:
        if self.session is not None and not self.busy:
            self.select_ctl.accept()

    def _escape(self) -> None:
        """Esc: קודם מחוות פעילה, אחר כך הצעה/קלט בחירה, ורק אחרי זה ביטול עיבוד Inpaint."""
        if self.canvas.cancel_gesture():
            self._refresh()
        elif not self.select_ctl.cancel():
            self.controller.cancel()

    # ----------------------------------------------------- Undo/Redo והשוואה
    def _mask_can(self, redo: bool) -> bool:
        """יש עריכת מסכה ממתינה (עדיין לא הוגשה ל-Inpaint) שאפשר לבטל/לשחזר."""
        h = self.mask_panel.history
        return self.mask_panel.active and h is not None and (h.can_redo if redo else h.can_undo)

    def _undo(self) -> None:
        """Ctrl+Z: קודם עריכת מסכה ממתינה (לאסו, הרחבה, בחירה מאושרת שלא הוגשה); אחרי הגשה ההיסטוריה מיושנת
        (המסכה רוקנה) ולכן Ctrl+Z מבטל את שינוי התמונה האחרון."""
        if self.session is None or self.busy:
            return
        if self._mask_can(False):
            self.select_ctl.cancel()
            self.mask_panel.undo()
            self.canvas.invalidate_overlay()
            self._refresh()
        else:
            self.controller.undo()

    def _redo(self) -> None:
        if self.session is None or self.busy:
            return
        if self._mask_can(True):
            self.select_ctl.cancel()
            self.mask_panel.redo()
            self.canvas.invalidate_overlay()
            self._refresh()
        else:
            self.controller.redo()

    @property
    def _comparing(self) -> bool:
        return self._cmp_toggle or self._cmp_hold

    def _compare_toggled(self, on: bool) -> None:
        self._cmp_toggle = on
        self._apply_compare()

    def _compare_hold(self, on: bool) -> None:
        self._cmp_hold = on and self.session is not None
        self._apply_compare()

    def _reset_compare(self) -> None:
        self._cmp_hold = False
        self._cmp_toggle = False
        if self.act_compare.isChecked():
            self.act_compare.blockSignals(True)
            self.act_compare.setChecked(False)
            self.act_compare.blockSignals(False)
        self.canvas.clear_compare_cache()
        self.lbl_cmp.hide()

    def _apply_compare(self) -> None:
        s, on = self.session, self._comparing
        if s is None:
            on = False
        self.canvas.set_compare(on, s.original_rgb if s else None, s.original_alpha if s else None,
                                t("compare.badge"))
        on = self.canvas.comparing
        self.lbl_cmp.setText(t("status.comparing"))
        self.lbl_cmp.setVisible(on)
        self._refresh()

    def _show_processing_text(self) -> None:
        n = len(self.session.pending_masks) if self.session is not None else 0
        self.lbl_proc.setText(t("status.processing_n", count=n) if n > 1 else t("status.processing"))

    def _processing_changed(self, on: bool) -> None:
        self.lbl_proc.setVisible(on)
        self.proc_progress.setVisible(on)
        if on:
            self.proc_progress.setRange(0, 0)
            self._show_processing_text()
        else:
            self.lbl_proc.setText("")

    def _processing_progress(self, done: int, total: int) -> None:
        if self.controller.indicator_on and total > 1:
            self.proc_progress.setRange(0, total)
            self.proc_progress.setValue(done)

    # ------------------------------------------------------------------ מנוע
    def _start_engine(self, factory: Callable) -> None:
        self.controller.engine_loading()
        self.statusBar().showMessage(t("status.engine_loading"))
        self._runner.submit(factory, self._engine_loaded, self._engine_failed)

    def _engine_loaded(self, scheduler) -> None:
        scheduler.start()                       # Warm-up ברקע (compile לכל bucket)
        self.controller.engine_ready(scheduler)
        self.statusBar().showMessage(t("status.engine_ready"))

    def _engine_failed(self, e: BaseException) -> None:
        detail = str(e) if isinstance(e, FixPicError) else f"{type(e).__name__}: {e}"
        if not isinstance(e, FixPicError):
            log.error("טעינת המנוע נכשלה", exc_info=e)
        self.controller.engine_failed(detail)
        self.statusBar().showMessage(t("status.ready"))
        self._error_once("error.engine", detail=detail)

    # ------------------------------------------------------- אישור שינויים
    def confirm_discard(self, then: Callable[[], None]) -> None:
        """מריץ `then` אחרי טיפול בשינויים שלא נשמרו (אם יש). שמירה אסינכרונית: `then` רץ בסיומה."""
        if self.session is None or not self.session.modified:
            then()
            return
        choice = self._ask_discard()
        if choice == "discard":
            then()
        elif choice == "save":
            self.save(then=then)

    def _busy_guard(self) -> bool:
        if self.busy:
            self._error("msg.busy")
        return self.busy

    # ----------------------------------------------------------------- פתיחה
    def open_dialog(self) -> None:
        if self._busy_guard():
            return
        path = self._pick_open()
        if path:
            self.open_path(path)

    def open_path(self, path: str | Path) -> None:
        if self._busy_guard():
            return
        p = Path(path)

        def start() -> None:
            self._set_busy(True, t("status.loading"))
            self._runner.submit(lambda: load_image(p), lambda img: self._loaded(img, p.name, p), self._load_failed)
        self.confirm_discard(start)

    def _load_failed(self, e: BaseException) -> None:
        self._set_busy(False, t("status.ready"))
        self._report("error.load", e)

    def _loaded(self, img: LoadedImage, name: str, path: Path | None) -> None:
        self._install(img, name)
        self.statusBar().showMessage(t("status.loaded", name=name))

    def _install(self, img: LoadedImage, name: str) -> None:
        try:
            session = Session(img)
        except FixPicError as e:
            self._set_busy(False, t("status.ready"))
            self._report("error.load", e)
            return
        self.session, self.display_name, self.save_path = session, name, None
        self._reset_compare()
        self.controller.bind(session)
        self.select_ctl.bind(session)
        self.canvas.set_image(session.rgb, session.alpha)
        self.canvas.set_mask_layer(session.mask)
        self.canvas.set_pending_source(lambda: session.pending_masks)
        self.mask_panel.bind(session.mask)
        self.brush_panel.configure_for_image(*session.size)
        self.canvas.setFocus()
        self._set_busy(False)

    # ---------------------------------------------------------------- הדבקה
    def paste(self) -> None:
        if self._busy_guard():
            return
        qimg = QGuiApplication.clipboard().image()
        try:
            img = image_from_array(qimage_to_array(qimg))
        except InvalidInputError:
            self._error("error.clipboard_empty")
            return
        except FixPicError as e:
            self._report("error.load", e)
            return
        self.confirm_discard(lambda: self._install(img, ""))

    # --------------------------------------------------------------- גרירה
    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        urls: list[QUrl] = [u for u in e.mimeData().urls() if u.isLocalFile()]
        if not urls:
            self._error("error.not_image_drop")
            return
        e.acceptProposedAction()
        self.open_path(urls[0].toLocalFile())

    # ----------------------------------------------------------------- שמירה
    def save(self, then: Callable[[], None] | None = None) -> None:
        if self.session is None or self._busy_guard():
            return
        if self.save_path is None:
            self.save_as(then=then)
        else:
            self._start_save(self.save_path, then)

    def save_as(self, then: Callable[[], None] | None = None) -> None:
        if self.session is None or self._busy_guard():
            return
        s = self.session
        stem = Path(self.display_name).stem if self.display_name else t("title.untitled")
        base = s.source_path.parent if s.source_path else Path("")
        sel = t(_FILTER_BY_FORMAT.get((s.source_format or "").upper(), "filter.png"))
        ext = _EXT_BY_FILTER[next((k for k in _EXT_BY_FILTER if t(k) == sel), "filter.png")]
        path, chosen = self._pick_save(str(base / f"{stem}{ext}"), sel)
        if not path:
            return
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(next((e for k, e in _EXT_BY_FILTER.items() if t(k) == chosen), ".png"))
        if p.suffix.lower() in _LOSSY:
            q = self._ask_quality(p.suffix.lstrip(".").upper().replace("JPEG", "JPG"), self.save_quality)
            if q is None:
                return
            self.save_quality = q
        self._start_save(p, then)

    def _start_save(self, path: Path, then: Callable[[], None] | None) -> None:
        s = self.session
        rgb, alpha, icc, exif = s.rgb, s.alpha, s.icc_profile, s.exif   # בלתי-משתנים (קריאה-בלבד)
        state_id = s.history.current_id                                  # מצב שנשמר בפועל
        quality = self.save_quality
        self._set_busy(True, t("status.saving"))

        def work() -> SaveResult:
            return save_image(path, rgb, alpha, icc_profile=icc, exif=exif, quality=quality)

        def done(res: SaveResult) -> None:
            self.save_path = path
            s.mark_saved(state_id)
            self.display_name = path.name
            self._set_busy(False, t("status.saved", name=path.name))
            if res.alpha_flattened:
                self.statusBar().showMessage(t("msg.alpha_flattened"))
            if then is not None:
                then()

        def failed(e: BaseException) -> None:
            self._set_busy(False, t("status.ready"))
            self._report("error.save", e)

        self._runner.submit(work, done, failed)

    # ---------------------------------------------------------------- סגירה
    def closeEvent(self, e) -> None:
        if self.busy:
            self._error("msg.busy")
            e.ignore()
            return
        if self.session is not None and self.session.modified:
            e.ignore()
            self.confirm_discard(self._close_now)
            return
        self.select_ctl.shutdown()
        self.controller.shutdown()
        e.accept()

    def _close_now(self) -> None:
        if self.session is not None:
            self.session.mark_saved()      # אחרת closeEvent היה שואל שוב
        self.close()
