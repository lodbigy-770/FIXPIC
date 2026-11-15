# test_model_card_updater.py
"""
Unit tests for model_card_updater (task 12.3).

The tests run on a synthetic MODEL_CARD that mirrors the structure of the card *before* stage-0 validation
(legend with a bold [pointer], section 1 with [pointer] markers, section 3 table, section 8 checklist), so they do
not depend on the real, hand-edited docs/architecture/cleanup-ai/MODEL_CARD.md (which is never touched here).

Run from the repo root:  python -m pytest stage0_validation/test_model_card_updater.py -q
"""

import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from .config import MODEL_CARD_PATH
from .model_card_updater import (
    add_verification_note,
    backup_model_card,
    parse_model_card,
    update_section_1,
    update_section_3,
    update_task_list,
    write_model_card,
)

CARD = """# MODEL_CARD – Big LaMa (ONNX)
> עודכן: 2026-11-10.
> מקרא: **[README]** = מוצהר | **[pointer]** = מתוך ה-LFS pointer | **[אומת]** = נבדק בפועל | [ידע כללי] | [לא ידוע].

## 1. מקור ורישיון

| פריט | ערך | מקור |
|------|-----|------|
| SHA256 | `ef85edd8` [pointer] | LFS |
| גודל | 208,851,664 בתים [pointer] | LFS |
| רישיון | Apache-2.0 [README] | HF |
| Revision | לא ידוע [לא ידוע] | - |
| מחבר | Carve [ידע כללי] | - |

## 2. ארכיטקטורה

- גרף ONNX יחיד. [pointer]
- [ ] SHA256 במקום לא נכון (לא בסעיף 8)

## 3. מבנה קלט ופלט [README – לאמת ב-`probe`]

| | ערך |
|---|---|
| Input תמונה | `rgb`, `[1, 3, H, W]`, float, ערכים 0–1 |
| Input מסכה | `mask`, `[1, 1, H, W]`; **1 = אזור למילוי/הסרה, 0 = לשמירה** |
| Output | `inpainted`, `[1, 3, H, W]`, 0–1 |

## 8. משימות להשלמת הכרטיס

- [ ] להשיג את תוכן הקובץ ולאמת SHA256 מול ה-pointer.
- [ ] `probe`: אימות שמות, טיפוסים ומידות דינמיות (סעיף 3).
- [~] אימות הריפוד על תמונה אמיתית (סעיף 4).
- [x] משימה שכבר הושלמה, כוללת probe.
- [ ] Revision ב-HF ותנאי נתוני האימון (סעיף 1).

## 9. נספח

- [ ] SHA256 אחרי סעיף 8 (לא אמור להשתנות)
"""

META = {
    "image": SimpleNamespace(name="image"),
    "mask": SimpleNamespace(name="mask_input"),
    "output": SimpleNamespace(name="output"),
}
FILE_INFO = {"sha256": "ef85edd8", "size": 208851664}


def headings(text):
    return [l for l in text.split("\n") if l.startswith("#")]


# ---------------------------------------------------------------------------------------------- parse / backup / write
class TestParseBackupWrite:
    def test_parse_reads_utf8_hebrew(self, tmp_path):
        p = tmp_path / "MODEL_CARD.md"
        p.write_text(CARD, encoding="utf-8")
        assert parse_model_card(p) == CARD
        assert "## 1. מקור ורישיון" in parse_model_card(p)

    def test_parse_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_model_card(tmp_path / "nope.md")

    def test_backup_name_content_and_original_untouched(self, tmp_path):
        p = tmp_path / "MODEL_CARD.md"
        p.write_text(CARD, encoding="utf-8")
        b = backup_model_card(p)
        assert b.name == "MODEL_CARD.md.bak" and b.parent == p.parent
        assert b.read_text(encoding="utf-8") == CARD and p.read_text(encoding="utf-8") == CARD

    def test_backup_overwrites_previous_backup(self, tmp_path):
        p = tmp_path / "MODEL_CARD.md"
        p.write_text("v1", encoding="utf-8")
        backup_model_card(p)
        p.write_text("v2", encoding="utf-8")
        assert backup_model_card(p).read_text(encoding="utf-8") == "v2"

    def test_backup_keeps_mtime(self, tmp_path):
        p = tmp_path / "MODEL_CARD.md"
        p.write_text(CARD, encoding="utf-8")
        import os
        os.utime(p, (1_600_000_000, 1_600_000_000))
        assert abs(backup_model_card(p).stat().st_mtime - 1_600_000_000) < 2

    def test_backup_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            backup_model_card(tmp_path / "nope.md")

    def test_write_roundtrip_hebrew_no_bom(self, tmp_path):
        p = tmp_path / "x.md"
        text = "# TEST\n> שורה בעברית\nבדיקה 123\n"
        write_model_card(p, text)
        assert p.read_bytes() == text.encode("utf-8") and not p.read_bytes().startswith(b"\xef\xbb\xbf")

    def test_write_overwrites(self, tmp_path):
        p = tmp_path / "x.md"
        write_model_card(p, "long old content" * 10)
        write_model_card(p, "new")
        assert p.read_text(encoding="utf-8") == "new"

    def test_write_into_missing_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            write_model_card(tmp_path / "no_dir" / "x.md", "a")


# ---------------------------------------------------------------------------------------------- section 1
class TestSection1:
    def test_plain_pointer_replaced_bold_legend_kept(self):
        out = update_section_1(CARD, FILE_INFO)
        plain = out.count("[pointer]") - out.count("**[pointer]**")
        assert plain == 0 and "**[pointer]**" in out
        assert "| SHA256 | `ef85edd8` [אומת] |" in out and "208,851,664 בתים [אומת]" in out

    def test_replaces_every_plain_occurrence_including_other_sections(self):
        out = update_section_1(CARD, FILE_INFO)
        assert "- גרף ONNX יחיד. [אומת]" in out                       # גם בסעיף 2 (ההחלפה גלובלית, לא רק בסעיף 1)
        assert out.count("[אומת]") == CARD.count("[pointer]") - CARD.count("**[pointer]**") + CARD.count("[אומת]")

    def test_other_markers_untouched(self):
        out = update_section_1(CARD, FILE_INFO)
        for marker in ("[README]", "[לא ידוע]", "[ידע כללי]"):
            assert out.count(marker) == CARD.count(marker)

    def test_structure_preserved(self):
        out = update_section_1(CARD, FILE_INFO)
        assert headings(out) == headings(CARD) and len(out.split("\n")) == len(CARD.split("\n"))

    def test_idempotent(self):
        once = update_section_1(CARD, FILE_INFO)
        assert update_section_1(once, FILE_INFO) == once

    def test_file_info_is_not_used(self):
        assert update_section_1(CARD, {}) == update_section_1(CARD, FILE_INFO) == update_section_1(CARD, None)

    def test_no_markers_is_noop_and_no_placeholder_leak(self):
        assert update_section_1("# T\nאין כאן סימון\n", FILE_INFO) == "# T\nאין כאן סימון\n"
        assert "\x00" not in update_section_1(CARD, FILE_INFO)

    def test_several_on_one_line(self):
        assert update_section_1("a [pointer] b [pointer] **[pointer]**", FILE_INFO) == "a [אומת] b [אומת] **[pointer]**"


# ---------------------------------------------------------------------------------------------- section 3
class TestSection3:
    def test_header_marker_replaced(self):
        out = update_section_3(CARD, META)
        assert "[README – לאמת ב-`probe`]" not in out
        assert "## 3. מבנה קלט ופלט [אומת]" in out

    def test_tensor_names_replaced_in_rows(self):
        out = update_section_3(CARD, META)
        assert "| Input תמונה | `image`, `[1, 3, H, W]`, float, ערכים 0–1 |" in out
        assert "| Input מסכה | `mask_input`, `[1, 1, H, W]`;" in out
        assert "| Output | `output`, `[1, 3, H, W]`, 0–1 |" in out

    def test_rest_of_row_text_kept(self):
        out = update_section_3(CARD, META)
        assert "**1 = אזור למילוי/הסרה, 0 = לשמירה**" in out

    def test_same_names_gives_same_card_except_header(self):
        same = {"image": SimpleNamespace(name="rgb"), "mask": SimpleNamespace(name="mask"),
                "output": SimpleNamespace(name="inpainted")}
        out = update_section_3(CARD, same)
        assert out == CARD.replace("[README – לאמת ב-`probe`]", "[אומת]")

    def test_other_sections_untouched(self):
        out = update_section_3(CARD, META)
        assert out.split("## 3.")[0] == CARD.split("## 3.")[0]
        assert out.split("## 8.")[1] == CARD.split("## 8.")[1]

    def test_structure_preserved(self):
        out = update_section_3(CARD, META)
        assert len(out.split("\n")) == len(CARD.split("\n"))
        assert [h.split(" [")[0] for h in headings(out)] == [h.split(" [")[0] for h in headings(CARD)]

    def test_idempotent(self):
        once = update_section_3(CARD, META)
        assert update_section_3(once, META) == once

    def test_missing_rows_are_a_noop(self):
        text = "# T\n## 3. מבנה\nאין כאן טבלה\n"
        assert update_section_3(text, META) == text

    def test_metadata_missing_key_raises(self):
        with pytest.raises(KeyError):
            update_section_3(CARD, {"image": META["image"]})

    def test_tensor_without_name_raises(self):
        with pytest.raises(AttributeError):
            update_section_3(CARD, {"image": object(), "mask": META["mask"], "output": META["output"]})


# ---------------------------------------------------------------------------------------------- section 8
class TestTaskList:
    def lines(self, text):
        return text.split("\n")

    def test_marks_matching_tasks_only_in_section_8(self):
        out = update_task_list(CARD, ["SHA256", "probe"])
        assert "- [x] להשיג את תוכן הקובץ ולאמת SHA256 מול ה-pointer." in out
        assert "- [x] `probe`: אימות שמות, טיפוסים ומידות דינמיות (סעיף 3)." in out

    def test_lines_outside_section_8_untouched(self):
        out = update_task_list(CARD, ["SHA256"])
        assert "- [ ] SHA256 במקום לא נכון (לא בסעיף 8)" in out          # סעיף 2, לפני 8
        assert "- [ ] SHA256 אחרי סעיף 8 (לא אמור להשתנות)" in out        # סעיף 9, אחרי 8

    def test_non_matching_tasks_stay_open(self):
        out = update_task_list(CARD, ["SHA256", "probe"])
        assert "- [ ] Revision ב-HF ותנאי נתוני האימון (סעיף 1)." in out

    def test_partial_and_done_markers_untouched(self):
        out = update_task_list(CARD, ["הריפוד", "הושלמה", "probe"])
        assert "- [~] אימות הריפוד על תמונה אמיתית (סעיף 4)." in out
        assert "- [x] משימה שכבר הושלמה, כוללת probe." in out

    def test_empty_list_is_noop(self):
        assert update_task_list(CARD, []) == CARD

    def test_unknown_task_is_noop(self):
        assert update_task_list(CARD, ["אין כזו"]) == CARD

    def test_only_checkbox_changes(self):
        out = update_task_list(CARD, ["SHA256", "probe"])
        diff = [(a, b) for a, b in zip(self.lines(CARD), self.lines(out)) if a != b]
        assert len(diff) == 2 and all(a.replace("- [ ]", "- [x]", 1) == b for a, b in diff)

    def test_one_replacement_per_line(self):
        text = "## 8. משימות\n- [ ] משימה A - [ ] טקסט\n"
        assert update_task_list(text, ["A"]) == "## 8. משימות\n- [x] משימה A - [ ] טקסט\n"

    def test_several_matches_for_one_line_use_one_replacement(self):
        text = "## 8. משימות\n- [ ] SHA256 probe\n"
        assert update_task_list(text, ["SHA256", "probe"]) == "## 8. משימות\n- [x] SHA256 probe\n"

    def test_idempotent(self):
        once = update_task_list(CARD, ["SHA256", "probe"])
        assert update_task_list(once, ["SHA256", "probe"]) == once

    def test_no_section_8_is_noop(self):
        text = "# T\n## 1. א\n- [ ] SHA256\n"
        assert update_task_list(text, ["SHA256"]) == text

    def test_indented_task_line(self):
        text = "## 8. משימות\n  - [ ] SHA256\n"
        assert update_task_list(text, ["SHA256"]) == "## 8. משימות\n  - [x] SHA256\n"


# ---------------------------------------------------------------------------------------------- verification note
class TestVerificationNote:
    NOTE = "> אומת בשלב 0 ב-2026-11-11 14:19"

    def test_exact_format(self):
        assert self.NOTE in add_verification_note(CARD, "2026-11-11 14:19").split("\n")

    def test_inserted_after_existing_blockquotes_before_content(self):
        lines = add_verification_note(CARD, "2026-11-11 14:19").split("\n")
        i = lines.index(self.NOTE)
        assert lines[0].startswith("# MODEL_CARD") and lines[i - 1].startswith(">") and lines[i + 1] == ""

    def test_exactly_one_line_added_nothing_else_changes(self):
        out = add_verification_note(CARD, "2026-11-11 14:19").split("\n")
        rest = [l for l in out if l != self.NOTE]
        assert len(out) == len(CARD.split("\n")) + 1 and rest == CARD.split("\n")

    def test_heading_without_blockquote_gets_note_right_after_heading(self):
        out = add_verification_note("# MODEL_CARD – X\n\nגוף\n", "T").split("\n")
        assert out[:3] == ["# MODEL_CARD – X", "> אומת בשלב 0 ב-T", ""]

    def test_auto_timestamp_format(self):
        out = add_verification_note(CARD)
        assert re.search(r"^> אומת בשלב 0 ב-\d{4}-\d{2}-\d{2} \d{2}:\d{2}$", out, re.M)

    def test_no_heading_is_noop(self):
        text = "אין כותרת\n> שורה\n"
        assert add_verification_note(text, "T") == text

    @pytest.mark.xfail(strict=True, reason="ידוע: כל הרצה של validate_stage0.py מוסיפה הערה נוספת (אין בדיקת כפילות)")
    def test_idempotent_for_same_timestamp(self):
        once = add_verification_note(CARD, "2026-11-11 14:19")
        assert add_verification_note(once, "2026-11-11 14:19") == once


# ---------------------------------------------------------------------------------------------- whole flow
class TestPipeline:
    def run(self, text):
        text = update_section_1(text, FILE_INFO)
        text = update_section_3(text, META)
        text = update_task_list(text, ["SHA256", "probe"])
        return add_verification_note(text, "2026-11-11 14:19")

    def test_orchestrator_order_end_to_end_via_files(self, tmp_path):
        p = tmp_path / "MODEL_CARD.md"
        p.write_text(CARD, encoding="utf-8")
        backup_model_card(p)
        write_model_card(p, self.run(parse_model_card(p)))
        out = p.read_text(encoding="utf-8")
        assert (tmp_path / "MODEL_CARD.md.bak").read_text(encoding="utf-8") == CARD
        assert "> אומת בשלב 0 ב-2026-11-11 14:19" in out and "[README – לאמת ב-`probe`]" not in out
        assert out.count("- [x]") == CARD.count("- [x]") + 2

    def test_headings_and_line_count_preserved(self):
        out = self.run(CARD)
        assert [h.split(" [")[0] for h in headings(out)] == [h.split(" [")[0] for h in headings(CARD)]
        assert len(out.split("\n")) == len(CARD.split("\n")) + 1

    def test_rerun_of_the_three_updaters_is_a_noop(self):
        """הרצה שנייה של שלושת העדכונים (סעיף 1, סעיף 3, משימות) על הכרטיס המעודכן אינה משנה דבר."""
        first = self.run(CARD)
        second = update_task_list(update_section_3(update_section_1(first, FILE_INFO), META), ["SHA256", "probe"])
        assert second == first


# ---------------------------------------------------------------------------------------------- real card (read only)
def test_real_model_card_still_has_the_structure_the_updater_relies_on():
    """הכרטיס האמיתי נקרא בלבד (לא נכתב): שמונת הסעיפים, המקרא וסעיף המשימות קיימים, אחרת העדכון האוטומטי ישתתק."""
    if not Path(MODEL_CARD_PATH).exists():
        pytest.skip("MODEL_CARD.md לא נמצא")
    text = parse_model_card(MODEL_CARD_PATH)
    for n in range(1, 9):
        assert re.search(rf"^## {n}\. ", text, re.M), f"חסר סעיף {n}"
    assert "**[pointer]**" in text and "- [" in text.split("## 8.")[1]
