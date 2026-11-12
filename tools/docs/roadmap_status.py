#!/usr/bin/env python3
"""Regenerates the status table at the top of docs/architecture/cleanup-ai/ROADMAP.md.

Counts the checkboxes under every '## ' section: [x] done, [~] partial, [ ] open.
The table sits between the markers <!-- STATUS:BEGIN --> and <!-- STATUS:END -->.
Hand-written remarks per section live in NOTES below (keyed by the heading prefix).

  python tools/docs/roadmap_status.py          rewrite the table in ROADMAP.md
  python tools/docs/roadmap_status.py --check  exit 1 if the table is out of date
"""
import re
import sys
from pathlib import Path

ROADMAP = Path(__file__).resolve().parents[2] / "docs" / "architecture" / "cleanup-ai" / "ROADMAP.md"
BEGIN, END = "<!-- STATUS:BEGIN -->", "<!-- STATUS:END -->"
NOTES = {
    "שלב 0": "נשארו: Revision ב-HF ותיעוד מלא ב-README של המודל (דורש גישה ל-Hugging Face)",
    "Milestone 1.5": "נסגר; אימות אריזה על Python 3.14 הועבר ל-M5",
    "Milestone 2": "בעבודה (שיחה B): M2.1–M2.5 בוצעו; נשארו M2.6–M2.9",
}


def sections(text):
    cur, out = None, []
    for line in text.splitlines():
        if line.startswith("## "):
            cur = {"title": line[3:].strip(), "x": 0, "p": 0, "o": 0}
            out.append(cur)
        elif cur and re.match(r"- \[( |x|~)\]", line):
            cur[{"x": "x", "~": "p", " ": "o"}[line[3]]] += 1
    return [s for s in out if s["x"] + s["p"] + s["o"]]


def status(s):
    if s["p"] == 0 and s["o"] == 0:
        return "בוצע"
    if s["x"] == 0 and s["p"] == 0:
        return "טרם התחיל"
    if s["o"] == 0:
        return "בוצע בעיקרו"
    return "בעבודה"


def table(text):
    rows = ["| שלב | מצב | בוצע | חלקי | פתוח | הערה |", "|-----|-----|------|------|------|------|"]
    for s in sections(text):
        note = next((v for k, v in NOTES.items() if s["title"].startswith(k)), "")
        rows.append(f"| {s['title']} | {status(s)} | {s['x']} | {s['p']} | {s['o']} | {note} |")
    return "\n".join(rows)


def main():
    text = ROADMAP.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        sys.exit("markers missing in ROADMAP.md")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{BEGIN}\n{table(text)}\n{END}{tail}"
    if "--check" in sys.argv:
        sys.exit(0 if new == text else 1)
    ROADMAP.write_text(new, encoding="utf-8")
    print(table(new))


if __name__ == "__main__":
    main()
