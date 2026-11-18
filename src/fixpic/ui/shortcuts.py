"""התאמה אישית של קיצורי מקלדת: מאגר (קובץ JSON בתיקיית ההגדרות), בדיקת התנגשויות ודיאלוג עריכה.

- נשמרות רק החלפות (`overrides`) לפי מפתח הפעולה (למשל `action.zoom_in`); פעולה בלי החלפה משתמשת בברירת המחדל.
  רשימה ריקה = בלי קיצור. קובץ פגום או ערכים לא תקינים מתעלמים מהם (ברירת מחדל), בלי לקרוס.
- אות לטינית בודדת (בלי Ctrl/Shift/Alt) מקבלת אוטומטית גם את האות העברית שעל אותו מקש (`with_hebrew`).
- מיקום: `FIXPIC_CONFIG_DIR`; ב-Windows `%LOCALAPPDATA%\\FIXPIC`; אחרת `$XDG_CONFIG_HOME/fixpic`.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QKeySequenceEdit, QLabel, QMessageBox,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..strings import t

log = logging.getLogger("fixpic.ui")

# האות העברית שעל אותו מקש (פריסה עברית סטנדרטית); q ו-w הם סימני פיסוק ולכן אינם כאן
HEBREW_KEY = {"E": "ק", "R": "ר", "T": "א", "Y": "ט", "U": "ו", "I": "ן", "O": "ם", "P": "פ", "A": "ש", "S": "ד",
              "D": "ג", "F": "כ", "G": "ע", "H": "י", "J": "ח", "K": "ל", "L": "ך", "Z": "ז", "X": "ס", "C": "ב",
              "V": "ה", "B": "נ", "N": "מ", "M": "צ"}


def config_dir() -> Path:
    env = os.environ.get("FIXPIC_CONFIG_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "FIXPIC"
    return Path(os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")) / "fixpic"


def parse_sequence(text: str) -> QKeySequence | None:
    """מקש אחד (לא רצף `A, B`); None אם ריק או לא תקין."""
    seq = QKeySequence(text, QKeySequence.PortableText)
    return seq if seq.count() == 1 else None


def normalize(text: str) -> str | None:
    seq = parse_sequence(text)
    return seq.toString(QKeySequence.PortableText) if seq is not None else None


def with_hebrew(texts: list[str]) -> list[QKeySequence]:
    """רצפי הקיצור בפועל: כל קיצור, ולאות לטינית בודדת גם האות העברית שעל אותו מקש."""
    out: list[QKeySequence] = []
    for s in texts:
        seq = parse_sequence(s)
        if seq is None:
            continue
        out.append(seq)
        he = HEBREW_KEY.get(s.upper()) if len(s) == 1 else None
        if he:
            out.append(QKeySequence(he))
    return out


def find_conflicts(assignment: dict[str, list[str]]) -> dict[str, list[str]]:
    """קיצור (מנורמל) -> מפתחות הפעולות שמחזיקות בו, רק כשיש שתי פעולות או יותר."""
    owners: dict[str, list[str]] = {}
    for key, seqs in assignment.items():
        for s in dict.fromkeys(n for n in (normalize(x) for x in seqs) if n):
            owners.setdefault(s, []).append(key)
    return {s: ks for s, ks in owners.items() if len(ks) > 1}


class ShortcutStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else config_dir() / "keyboard.json"
        self.overrides: dict[str, list[str]] = {}

    def load(self) -> dict[str, list[str]]:
        self.overrides = {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            raw = data["shortcuts"]
            if not isinstance(raw, dict):
                raise ValueError("shortcuts")
        except FileNotFoundError:
            return self.overrides
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.warning("קובץ קיצורי המקלדת לא נקרא (%s); ברירות מחדל", e)
            return self.overrides
        for key, seqs in raw.items():
            if isinstance(key, str) and isinstance(seqs, list) and all(isinstance(s, str) for s in seqs):
                norm = [n for n in (normalize(s) for s in seqs) if n]
                if len(norm) == len([s for s in seqs if s.strip()]):
                    self.overrides[key] = norm
        return self.overrides

    def save(self, overrides: dict[str, list[str]]) -> None:
        """כתיבה אטומית (קובץ זמני והחלפה); `OSError` עולה למי שקרא."""
        self.overrides = {k: list(v) for k, v in overrides.items()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": 1, "shortcuts": self.overrides}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, self.path)


class ShortcutsDialog(QDialog):
    """טבלת פעולות וקיצורים. `entries`: [(key, תווית, ברירת מחדל [טקסט], נוכחי [טקסט])].
    `result_overrides()` מחזיר רק את מה שנבדל מברירת המחדל."""

    def __init__(self, entries: list[tuple[str, str, list[str], list[str]]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("shortcuts.title"))
        self._entries = entries
        self._edits: dict[str, QKeySequenceEdit] = {}
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(t("shortcuts.hint")))
        self.table = QTableWidget(len(entries), 3, self)
        self.table.setHorizontalHeaderLabels([t("shortcuts.col_action"), t("shortcuts.col_key"), ""])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setColumnWidth(0, 230)
        self.table.setColumnWidth(1, 170)
        for row, (key, label, default, current) in enumerate(entries):
            item = QTableWidgetItem(label)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, item)
            ed = QKeySequenceEdit(QKeySequence(current[0]) if current else QKeySequence(), self.table)
            ed.setMaximumSequenceLength(1)
            self._edits[key] = ed
            self.table.setCellWidget(row, 1, ed)
            btn = QPushButton(t("shortcuts.default"), self.table)
            btn.clicked.connect(lambda _c=False, k=key, d=default: self._set(k, d))
            self.table.setCellWidget(row, 2, btn)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.btn_reset_all = QPushButton(t("shortcuts.reset_all"))
        self.btn_reset_all.clicked.connect(self.reset_all)
        row.addWidget(self.btn_reset_all)
        row.addStretch(1)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        row.addWidget(self.buttons)
        lay.addLayout(row)
        self.resize(560, 520)

    def _set(self, key: str, seqs: list[str]) -> None:
        ed = self._edits[key]
        ed.setKeySequence(QKeySequence(seqs[0]) if seqs else QKeySequence())

    def reset_all(self) -> None:
        for key, _l, default, _c in self._entries:
            self._set(key, default)

    def assignment(self) -> dict[str, list[str]]:
        out = {}
        for key, ed in self._edits.items():
            s = ed.keySequence().toString(QKeySequence.PortableText)
            out[key] = [s] if s else []
        return out

    def result_overrides(self) -> dict[str, list[str]]:
        """החלפות לשמירה. שדה שלא שונה משאיר את מצבו (ברירת מחדל כולל קיצורים נוספים, או החלפה קיימת);
        שדה ששונה לברירת המחדל הראשונה מבטל את ההחלפה; כל ערך אחר הוא החלפה."""
        cur = self.assignment()
        out: dict[str, list[str]] = {}
        for key, _l, default, current in self._entries:
            dflt = [n for n in (normalize(x) for x in default) if n]
            cur_list = [n for n in (normalize(x) for x in current) if n]
            if cur[key] == cur_list[:1]:                      # לא שונה בדיאלוג
                if cur_list != dflt:
                    out[key] = cur_list
            elif cur[key] != dflt[:1]:
                out[key] = cur[key]
        return out

    def _label(self, key: str) -> str:
        return next(l for k, l, _d, _c in self._entries if k == key)

    def _accept(self) -> None:
        conflicts = find_conflicts(self.assignment())
        if conflicts:
            lines = [f"{s}: " + ", ".join(self._label(k) for k in ks) for s, ks in conflicts.items()]
            QMessageBox.warning(self, t("shortcuts.title"), t("shortcuts.conflict") + "\n" + "\n".join(lines))
            return
        self.accept()
