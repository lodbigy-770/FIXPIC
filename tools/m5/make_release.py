"""M5 – שחרור גרסה: יוצר תג git מתויג (`vX.Y.Z`), דוחף אותו, ויוצר Release ב-GitHub עם קובץ המתקין
(`FixPicSetup-X.Y.Z.exe`, תוצר `packaging/fixpic.iss`) כנספח. רץ אצל המשתמש, אחרי שהמתקין נבנה ב-Windows
(האריזה עצמה ו-Inno Setup לא רצים כאן). רק ספריית התקן של Python (`subprocess`, `urllib.request`) - בלי תלויות נוספות.

שימוש:
  python tools/m5/make_release.py 0.1.0 --installer dist/installer/FixPicSetup-0.1.0.exe --notes "ראשון עם מתקין"
  python tools/m5/make_release.py 0.1.0 --dry-run                     # בלי לגעת ב-git או ברשת

דורש משתנה סביבה עם טוקן גישה ל-GitHub (ברירת מחדל GITHUB_TOKEN; ראו --token-env), בהרשאת `repo` לכתיבת Releases.
הטוקן עצמו אף פעם לא נכתב לקובץ או ללוג.

מגבלות: לא בודק שעץ העבודה נקי או מעודכן מול המקור (יש להריץ `git pull --rebase` לפני, כרגיל); לא בונה את המתקין בעצמו;
לא מוחק/מתקן Release או תג קיימים – נכשל אם התג כבר קיים.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = "lodbigy-770/FIXPIC"
VERSION_RX = re.compile(r"^\d+\.\d+\.\d+(?:[-.][0-9A-Za-z]+)*$")  # semver בסיסי, כולל סיומת כמו -alpha1


class ReleaseError(Exception):
    pass


def tag_name(version: str) -> str:
    if not VERSION_RX.match(version):
        raise ReleaseError(f"מספר גרסה לא תקין: {version!r} (צפוי כמו 0.1.0 או 0.1.0-alpha1)")
    return f"v{version}"


def run_git(args: list[str], *, cwd: Path = REPO_ROOT) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise ReleaseError(f"git {' '.join(args)} נכשל:\n{r.stderr.strip()}")
    return r.stdout.strip()


def tag_exists_locally(tag: str) -> bool:
    return bool(run_git(["tag", "-l", tag]))


def create_and_push_tag(tag: str, message: str, remote: str) -> None:
    run_git(["tag", "-a", tag, "-m", message])
    run_git(["push", remote, tag])


def github_request(url: str, token: str, *, method: str = "GET", data: bytes | None = None,
                    content_type: str = "application/json") -> dict:
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "fixpic-make-release")
    if data is not None:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise ReleaseError(f"בקשת GitHub ל-{url} נכשלה ({e.code}): {body}") from e


def upload_url_for(release: dict, asset_name: str) -> str:
    """`upload_url` מגיע בתבנית `...{?name,label}`; מחליף בפרמטר `name` האמיתי."""
    base = release["upload_url"].split("{", 1)[0]
    return f"{base}?name={asset_name}"


def create_release(repo: str, token: str, tag: str, name: str, body: str) -> dict:
    url = f"https://api.github.com/repos/{repo}/releases"
    payload = json.dumps({"tag_name": tag, "name": name, "body": body, "draft": False, "prerelease": False}).encode()
    return github_request(url, token, method="POST", data=payload)


def upload_asset(release: dict, token: str, installer: Path) -> dict:
    url = upload_url_for(release, installer.name)
    data = installer.read_bytes()
    return github_request(url, token, method="POST", data=data, content_type="application/octet-stream")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version", help="לדוגמה 0.1.0 (בלי 'v' מוביל; גם pyproject.toml[project.version] אמור להתעדכן בנפרד)")
    ap.add_argument("--installer", type=Path, help="קובץ FixPicSetup-X.Y.Z.exe להעלאה כנספח ל-Release; אם לא ניתן, הReleaseנוצר בלי נספח")
    ap.add_argument("--notes", default="", help="טקסט תיאור ה-Release")
    ap.add_argument("--notes-file", type=Path, help="קובץ טקסט לתיאור ה-Release, במקום --notes")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--token-env", default="GITHUB_TOKEN")
    ap.add_argument("--dry-run", action="store_true", help="מדפיס מה היה קורה, בלי git push או קריאות רשת")
    a = ap.parse_args(argv)

    try:
        tag = tag_name(a.version)
    except ReleaseError as e:
        print(str(e), file=sys.stderr)
        return 2

    if a.installer is not None and not a.installer.is_file():
        print(f"קובץ המתקין לא נמצא: {a.installer}", file=sys.stderr)
        return 2

    notes = a.notes
    if a.notes_file is not None:
        notes = a.notes_file.read_text(encoding="utf-8")
    message = f"FixPic {a.version}"

    if a.dry_run:
        print(f"[dry-run] היה נוצר תג {tag} ונדחף ל-{a.remote}")
        print(f"[dry-run] היה נוצר Release ב-{a.repo}: tag={tag}, name={message}, body={notes!r}")
        if a.installer is not None:
            print(f"[dry-run] היה מועלה נספח: {a.installer} ({a.installer.stat().st_size // 2**20}MB)")
        return 0

    if tag_exists_locally(tag):
        print(f"התג {tag} כבר קיים מקומית; הרץ גרסה אחרת או מחק אותו תחילה (git tag -d {tag})", file=sys.stderr)
        return 2

    token = os.environ.get(a.token_env)
    if not token:
        print(f"חסר טוקן: הגדר את משתנה הסביבה {a.token_env}", file=sys.stderr)
        return 2

    try:
        create_and_push_tag(tag, message, a.remote)
        print(f"נוצר ונדחף תג {tag}")
        release = create_release(a.repo, token, tag, message, notes)
        print(f"נוצר Release: {release.get('html_url', '?')}")
        if a.installer is not None:
            asset = upload_asset(release, token, a.installer)
            print(f"הועלה נספח: {asset.get('browser_download_url', asset.get('name', '?'))}")
    except ReleaseError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
