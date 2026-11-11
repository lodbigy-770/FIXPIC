# model_card_updater.py
"""
MODEL_CARD.md updater for Stage 0 Model Validation.

Provides functions to read, back up, and update the MODEL_CARD.md file
with verified data from the model inspection steps.

MODEL_CARD structure (docs/architecture/cleanup-ai/MODEL_CARD.md):
  - Section 1: מקור ורישיון  – SHA256 / file-size status markers ([pointer] → [אומת])
  - Section 3: מבנה קלט ופלט – tensor names + "[README – לאמת ב-`probe`]" header marker
  - Section 8: משימות להשלמת הכרטיס – task checklist (- [ ] / - [x])
  Top-level blockquotes carry the "אומת בשלב 0" timestamp note.

Actual MODEL_CARD section-3 table rows (as they exist on disk):
  | Input תמונה | `rgb`, `[1, 3, H, W]`, float, ערכים 0–1 |
  | Input מסכה | `mask`, `[1, 1, H, W]`; **1 = אזור למילוי/הסרה, 0 = לשמירה** |
  | Output | `inpainted`, `[1, 3, H, W]`, 0–1 |

The replace calls use the `` `name` `` substring (which always appears right after
the pipe separator) so they work regardless of what follows on the same row.
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# 6.1  parse_model_card
# ---------------------------------------------------------------------------

def parse_model_card(path: Path) -> str:
    """
    Read MODEL_CARD.md into memory as a string.

    Reads the entire file with explicit UTF-8 encoding to preserve Hebrew
    characters and returns the full content ready for in-memory editing.

    Args:
        path: Path to the MODEL_CARD.md file.

    Returns:
        Full file content as a Unicode string.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError:   If the file cannot be read.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ---------------------------------------------------------------------------
# 6.2  backup_model_card
# ---------------------------------------------------------------------------

def backup_model_card(path: Path) -> Path:
    """
    Create a backup of MODEL_CARD.md before any modifications.

    Copies *path* to a sibling file with the ``.md.bak`` extension using
    ``shutil.copy2`` so that file metadata (timestamps) is also preserved.

    Args:
        path: Path to the MODEL_CARD.md file to back up.

    Returns:
        Path to the created backup file (same directory, ``.md.bak`` suffix).

    Raises:
        FileNotFoundError: If the source file does not exist.
        PermissionError:   If the file or its directory cannot be written.
    """
    backup_path = path.with_suffix('.md.bak')
    shutil.copy2(path, backup_path)
    return backup_path


# ---------------------------------------------------------------------------
# 6.3  update_section_1
# ---------------------------------------------------------------------------

def update_section_1(content: str, file_info: dict) -> str:
    """
    Update section 1 status markers for SHA256 and file size.

    Replaces every occurrence of the ``[pointer]`` status marker with
    ``[אומת]``.  All other markers ([README], [לא ידוע], [ידע כללי]) are
    left untouched.

    The ``file_info`` dict is accepted for interface consistency with the
    orchestrator (it carries ``sha256`` and ``size`` values), but the
    replacement target is the literal marker string ``[pointer]`` which
    appears in the MODEL_CARD rows for both SHA256 and file-size entries.

    Args:
        content:   Full MODEL_CARD content as a string.
        file_info: Dict with keys ``"sha256"`` (str) and ``"size"`` (int)
                   describing the verified file.  Used for documentation
                   purposes; the actual replacement key is ``[pointer]``.

    Returns:
        Updated content string with all plain ``[pointer]`` markers replaced
        by ``[אומת]``.  Bold occurrences (``**[pointer]**``) in the legend
        line are intentionally left untouched so the glossary remains correct.
    """
    # Replace only plain [pointer], not **[pointer]** in the legend.
    # We do this by replacing the bold variant with a placeholder, performing
    # the plain replacement, then restoring the bold variant.
    PLACEHOLDER = "\x00BOLD_POINTER\x00"
    content = content.replace("**[pointer]**", PLACEHOLDER)
    content = content.replace("[pointer]", "[אומת]")
    content = content.replace(PLACEHOLDER, "**[pointer]**")
    return content


# ---------------------------------------------------------------------------
# 6.4  update_section_3
# ---------------------------------------------------------------------------

def update_section_3(content: str, metadata: dict) -> str:
    """
    Update section 3 (מבנה קלט ופלט) with verified tensor names.

    Performs two categories of replacement:

    1. **Section header marker** – replaces the string
       ``[README – לאמת ב-`probe`]`` in the section-3 heading with
       ``[אומת]``.

    2. **Table row tensor names** – replaces the backtick-quoted placeholder
       names (``rgb``, ``mask``, ``inpainted``) in the input/output rows
       with the actual names read from the ONNX model:
         - ``| Input תמונה | `rgb``` → ``| Input תמונה | `{image_name}```
         - ``| Input מסכה | `mask``` → ``| Input מסכה | `{mask_name}```
         - ``| Output | `inpainted``` → ``| Output | `{output_name}```

    All other markers ([README], [לא ידוע], [ידע כללי]) are left untouched.

    Args:
        content:  Full MODEL_CARD content as a string.
        metadata: Dict with keys ``"image"``, ``"mask"``, ``"output"`` whose
                  values are ``TensorInfo`` objects (each having a ``.name``
                  attribute).

    Returns:
        Updated content string.
    """
    image_name = metadata["image"].name
    mask_name = metadata["mask"].name
    output_name = metadata["output"].name

    # Replace "[README – לאמת ב-`probe`]" in the section 3 header
    content = content.replace(
        "[README – לאמת ב-`probe`]",
        "[אומת]",
    )

    # Update the image-input tensor name in the table row.
    # Actual row: | Input תמונה | `rgb`, `[1, 3, H, W]`, float, ערכים 0–1 |
    content = content.replace(
        "| Input תמונה | `rgb`",
        f"| Input תמונה | `{image_name}`",
    )

    # Update the mask-input tensor name in the table row.
    # Actual row: | Input מסכה | `mask`, `[1, 1, H, W]`; ...
    content = content.replace(
        "| Input מסכה | `mask`",
        f"| Input מסכה | `{mask_name}`",
    )

    # Update the output tensor name in the table row.
    # Actual row: | Output | `inpainted`, `[1, 3, H, W]`, 0–1 |
    content = content.replace(
        "| Output | `inpainted`",
        f"| Output | `{output_name}`",
    )

    return content


# ---------------------------------------------------------------------------
# 6.5  update_task_list
# ---------------------------------------------------------------------------

def update_task_list(content: str, completed_tasks: List[str]) -> str:
    """
    Mark completed tasks in section 8 of MODEL_CARD.

    Iterates over lines inside section 8 (משימות להשלמת הכרטיס) and
    changes ``- [ ]`` to ``- [x]`` for every task whose text contains at
    least one entry from *completed_tasks*.  Lines outside section 8 and
    already-completed tasks are left unchanged.

    Args:
        content:         Full MODEL_CARD content as a string.
        completed_tasks: List of substrings to match against task lines.

    Returns:
        Updated content string.
    """
    lines = content.split('\n')
    in_section_8 = False

    for i, line in enumerate(lines):
        if line.startswith('## 8.'):
            in_section_8 = True
        elif in_section_8 and line.startswith('##'):
            # Entered a new section — stop scanning
            break
        elif in_section_8 and line.strip().startswith('- [ ]'):
            for task in completed_tasks:
                if task in line:
                    lines[i] = line.replace('- [ ]', '- [x]', 1)
                    break  # Only replace once per line

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 6.6  add_verification_note
# ---------------------------------------------------------------------------

def add_verification_note(content: str, timestamp: str | None = None) -> str:
    """
    Insert a verification note into the MODEL_CARD header blockquote.

    Adds a blockquote line of the form
    ``> אומת בשלב 0 ב-{timestamp}``
    immediately after any existing blockquote lines that follow the main
    ``# MODEL_CARD`` heading, so it sits inside the same visual block.

    Args:
        content:   Full MODEL_CARD content as a string.
        timestamp: Optional ISO-like timestamp string (``YYYY-MM-DD HH:MM``).
                   If not provided, the current local time is used.

    Returns:
        Updated content string.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    note_line = f"> אומת בשלב 0 ב-{timestamp}"

    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('# MODEL_CARD'):
            # Find the last consecutive blockquote line after the heading
            insert_pos = i + 1
            while insert_pos < len(lines) and lines[insert_pos].startswith('>'):
                insert_pos += 1
            lines.insert(insert_pos, note_line)
            break

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 6.7  write_model_card
# ---------------------------------------------------------------------------

def write_model_card(path: Path, content: str) -> None:
    """
    Write updated MODEL_CARD content back to disk.

    Uses explicit UTF-8 encoding to ensure Hebrew characters are preserved
    without corruption on all platforms.

    Args:
        path:    Destination path (normally the original MODEL_CARD.md).
        content: Full updated content to write.

    Raises:
        PermissionError: If the file cannot be written.
    """
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
