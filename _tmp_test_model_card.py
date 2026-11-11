"""
Inline checkpoint tests for ModelCardUpdater.
Works entirely in memory — MODEL_CARD.md is never modified.
"""

import sys
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

# Make sure the package is importable from the workspace root
sys.path.insert(0, str(Path(__file__).parent))

from stage0_validation.model_card_updater import (
    parse_model_card,
    backup_model_card,
    update_section_1,
    update_section_3,
    update_task_list,
    add_verification_note,
    write_model_card,
)
from stage0_validation.config import MODEL_CARD_PATH

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f"\n         ↳ {detail}" if not condition else ""))
    results.append((name, condition))


# ---------------------------------------------------------------------------
# 1. parse_model_card
# ---------------------------------------------------------------------------
print("\n--- 1. parse_model_card ---")
content = parse_model_card(MODEL_CARD_PATH)
check("Returns non-empty string", isinstance(content, str) and len(content) > 0)
check("Contains Hebrew characters", "מקור ורישיון" in content)
check("Contains section header #1", "## 1. מקור ורישיון" in content)
check("Contains [pointer] marker", "[pointer]" in content)
# Section 3 marker may already have been updated to [אומת] by a prior run
check("Contains section 3 (header or already updated)",
      "[README – לאמת ב-`probe`]" in content or "## 3. מבנה קלט ופלט" in content)
check("Contains section 8 task list", "## 8. משימות" in content)


# ---------------------------------------------------------------------------
# 2. backup_model_card
# ---------------------------------------------------------------------------
print("\n--- 2. backup_model_card ---")
with tempfile.TemporaryDirectory() as tmp:
    tmp_original = Path(tmp) / "MODEL_CARD.md"
    shutil.copy2(MODEL_CARD_PATH, tmp_original)

    backup_path = backup_model_card(tmp_original)
    check("Backup has .md.bak suffix", backup_path.suffix == ".bak")
    check("Backup file name", backup_path.name == "MODEL_CARD.md.bak")
    check("Backup exists", backup_path.exists())
    check("Backup content identical",
          backup_path.read_text(encoding="utf-8") == tmp_original.read_text(encoding="utf-8"))
    check("Original untouched", tmp_original.exists())


# ---------------------------------------------------------------------------
# 3. update_section_1
# ---------------------------------------------------------------------------
print("\n--- 3. update_section_1 ---")
# Feed it the real file content
orig = content
file_info = {"sha256": "ef85edd865769c03c753b5faa066be713a46a9c80c8f77954c67c0bb3a4ba2ff",
             "size": 208851664}
updated1 = update_section_1(orig, file_info)

# Only plain (non-bold) [pointer] should be gone; **[pointer]** in the legend is preserved.
plain_pointer_count = updated1.count("[pointer]") - updated1.count("**[pointer]**")
check("[pointer] replaced by [אומת] (plain only)", plain_pointer_count == 0,
      f"Remaining plain [pointer] occurrences: {plain_pointer_count}")
check("**[pointer]** in legend preserved", "**[pointer]**" in updated1,
      "Bold legend was incorrectly replaced")
check("[אומת] appears in updated content", "[אומת]" in updated1)
check("Other markers intact – [README]", "[README]" in updated1)
check("Other markers intact – [לא ידוע]", "[לא ידוע]" in updated1)
check("Other markers intact – [ידע כללי]", "[ידע כללי]" in updated1)


# ---------------------------------------------------------------------------
# 4. update_section_3
# ---------------------------------------------------------------------------
print("\n--- 4. update_section_3 ---")

class FakeTensor:
    def __init__(self, name):
        self.name = name

metadata = {
    "image":  FakeTensor("image"),
    "mask":   FakeTensor("mask_input"),
    "output": FakeTensor("output"),
}
updated3 = update_section_3(orig, metadata)

check("Section-3 header marker replaced",
      "[README – לאמת ב-`probe`]" not in updated3,
      "Header marker not replaced")
check("Section-3 now shows [אומת]", "[אומת]" in updated3)
check("Image tensor name updated",
      "| Input תמונה | `image`" in updated3,
      "rgb → image replacement failed")
check("Mask tensor name updated",
      "| Input מסכה | `mask_input`" in updated3,
      "mask → mask_input replacement failed")
check("Output tensor name updated",
      "| Output | `output`" in updated3,
      "inpainted → output replacement failed")


# ---------------------------------------------------------------------------
# 5. update_task_list
# ---------------------------------------------------------------------------
print("\n--- 5. update_task_list ---")

completed = ["SHA256", "probe"]
updated_tasks = update_task_list(orig, completed)

lines = updated_tasks.split("\n")
sha_lines    = [l for l in lines if "SHA256" in l or "sha256" in l.lower()]
probe_lines  = [l for l in lines if "probe" in l and "- [" in l]
other_unchecked = [l for l in lines
                   if l.strip().startswith("- [ ]")
                   and "SHA256" not in l and "sha256" not in l.lower()
                   and "probe" not in l]

check("SHA256 task marked [x]",
      all("- [x]" in l for l in sha_lines) and len(sha_lines) > 0,
      f"Lines: {sha_lines}")
check("probe task marked [x]",
      any("- [x]" in l for l in probe_lines),
      f"Lines: {probe_lines}")
check("Other tasks remain unchecked",
      len(other_unchecked) > 0,
      "No unchecked tasks remaining – possible over-replacement")


# ---------------------------------------------------------------------------
# 6. add_verification_note
# ---------------------------------------------------------------------------
print("\n--- 6. add_verification_note ---")

updated_note = add_verification_note(orig, timestamp="2024-01-15 10:30")
note_line = "> אומת בשלב 0 ב-2024-01-15 10:30"

check("Verification note present", note_line in updated_note,
      f"Expected: {note_line!r}")

# Check it appears after existing blockquotes, before non-blockquote content
lines_note = updated_note.split("\n")
note_idx = next((i for i, l in enumerate(lines_note) if l == note_line), -1)
check("Note found in content", note_idx >= 0, "Note not found in lines")

if note_idx > 0:
    before = lines_note[note_idx - 1]
    check("Inserted after an existing blockquote (> line)",
          before.startswith(">"),
          f"Line before note: {before!r}")

# Calling without timestamp should use current time and not crash
updated_auto = add_verification_note(orig)
check("Auto-timestamp note contains prefix", "> אומת בשלב 0 ב-" in updated_auto)


# ---------------------------------------------------------------------------
# 7. write_model_card
# ---------------------------------------------------------------------------
print("\n--- 7. write_model_card ---")

with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode='w', encoding='utf-8') as tf:
    tmp_write_path = Path(tf.name)

test_content = "# TEST\n> שורה בעברית\nבדיקה 123\n"
write_model_card(tmp_write_path, test_content)
read_back = tmp_write_path.read_text(encoding="utf-8")
tmp_write_path.unlink()

check("Write-read round-trip identical", read_back == test_content,
      f"Expected {test_content!r}, got {read_back!r}")
check("Hebrew preserved after write", "שורה בעברית" in read_back)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
total  = len(results)
print(f"Results: {passed}/{total} passed, {failed} failed")

if failed:
    print("\nFailed tests:")
    for name, ok in results:
        if not ok:
            print(f"  ✗ {name}")

sys.exit(0 if failed == 0 else 1)
