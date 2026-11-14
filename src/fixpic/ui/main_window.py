"""החלון הראשי (M3.3): פתיחה (דיאלוג/גרירה/הדבקה), Canvas עם Zoom/Pan, שמירה/ייצוא ברקע עם מחוון.

חוזים:
- `Session` חי רק ב-thread של ה-UI. טעינה ושמירה רצות ב-`BackgroundRunner` ומקבלות נתונים בלתי-משתנים בלבד.
- פעולה אחת ברקע בכל זמן (`busy`): ניסיון לפתוח/לשמור בזמן פעולה אחרת מציג הודעה ואינו מתחיל.
- **שמירה (Ctrl+S) אינה דורסת את קובץ המקור**: היא שומרת לנתיב האחרון שנבחר ב"שמירה בשם", ואם אין כזה
  מתנהגת כמו "שמירה בשם" (כך אי אפשר לדרוס צילום מקורי בטעות, ובמיוחד לא לדחוס JPEG מחדש).
- שינויים שלא נשמרו: בפתיחת תמונה אחרת, בהדבקה ובסגירה, שואלים שמירה/התעלמות/ביטול.
- חלונות דיאלוג מבודדים במתודות (`_error`, `_ask_discard`, `_pick_open`, `_pick_save`) כדי שבדיקות יחליפו אותן.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (QFileDialog, QLabel, QMainWindow, QMessageBox, QProgressBar, QToolBar)

from ..errors import FixPicError, InvalidInputError
from ..imgio import LoadedImage, SaveResult, image_from_array, load_image, save_image
from ..session import Session
from ..strings import t
from .canvas import Canvas
from .convert import qimage_to_array
from .workers import BackgroundRunner

log = logging.getLogger("fixpic.ui")

_EXT_BY_FILTER = {"filter.png": ".png", "filter.jpg": ".jpg", "filter.webp": ".webp"}
_FILTER_BY_FORMAT = {"JPEG": "filter.jpg", "WEBP": "filter.webp"}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.session: Session | None = None
        self.display_name: str = ""
        self.save_path: Path | None = None
        self.busy = False
        self._runner = BackgroundRunner(self)
        self.canvas = Canvas(self)
        self.setCentralWidget(self.canvas)
        self.setAcceptDrops(True)
        self.resize(1100, 760)
        self._build_actions()
        self._build_status()
        self.canvas.scaleChanged.connect(self._update_status)
        self._refresh()

    # ------------------------------------------------------------------ בנייה
    def _action(self, key: str, slot: Callable, shortcut=None) -> QAction:
        a = QAction(t(key), self)
        if shortcut is not None:
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
        mb = self.menuBar()
        m = mb.addMenu(t("menu.file"))
        for a in (self.act_open, self.act_save, self.act_save_as, self.act_paste):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.act_quit)
        v = mb.addMenu(t("menu.view"))
        for a in (self.act_zoom_in, self.act_zoom_out, self.act_fit, self.act_actual):
            v.addAction(a)
        tb = QToolBar(t("toolbar.main"), self)
        tb.setMovable(False)
        for a in (self.act_open, self.act_save, self.act_paste, self.act_fit, self.act_actual):
            tb.addAction(a)
        self.addToolBar(tb)

    def _build_status(self) -> None:
        sb = self.statusBar()
        self.lbl_size, self.lbl_zoom = QLabel(), QLabel()
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)               # מחוון "עסוק" (אין התקדמות מדידה בטעינה/קידוד)
        self.progress.setFixedWidth(140)
        self.progress.setTextVisible(False)
        self.progress.hide()
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
        self.act_paste.setEnabled(not self.busy)
        for a in (self.act_fit, self.act_actual, self.act_zoom_in, self.act_zoom_out):
            a.setEnabled(has)
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

    def _report(self, key: str, e: BaseException) -> None:
        if isinstance(e, FixPicError):
            self._error(key, detail=str(e))
        else:
            log.error("שגיאה לא צפויה", exc_info=e)
            self._error("error.unexpected", detail=f"{type(e).__name__}: {e}")

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
        self.canvas.set_image(session.rgb, session.alpha)
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
        self._start_save(p, then)

    def _start_save(self, path: Path, then: Callable[[], None] | None) -> None:
        s = self.session
        rgb, alpha, icc, exif = s.rgb, s.alpha, s.icc_profile, s.exif   # בלתי-משתנים (קריאה-בלבד)
        state_id = s.history.current_id                                  # מצב שנשמר בפועל
        self._set_busy(True, t("status.saving"))

        def work() -> SaveResult:
            return save_image(path, rgb, alpha, icc_profile=icc, exif=exif)

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
        e.accept()

    def _close_now(self) -> None:
        if self.session is not None:
            self.session.mark_saved()      # אחרת closeEvent היה שואל שוב
        self.close()
