"""M5: בדיקות ל-`tools/m5/make_release.py` — לוגיקה טהורה (אימות גרסה, upload_url, payload) ו-CLI במצב --dry-run
בלבד (בלי git push או קריאות רשת אמיתיות)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_release", REPO / "tools" / "m5" / "make_release.py")
mr = importlib.util.module_from_spec(spec)
sys.modules["make_release"] = mr
spec.loader.exec_module(mr)


def test_tag_name_valid():
    assert mr.tag_name("0.1.0") == "v0.1.0"
    assert mr.tag_name("1.2.3-alpha1") == "v1.2.3-alpha1"


def test_tag_name_rejects_bad_version():
    for bad in ("v0.1.0", "0.1", "latest", ""):
        try:
            mr.tag_name(bad)
        except mr.ReleaseError:
            continue
        raise AssertionError(f"{bad!r} אמור היה להידחות")


def test_upload_url_for_strips_template_and_adds_name():
    release = {"upload_url": "https://uploads.github.com/repos/o/r/releases/1/assets{?name,label}"}
    url = mr.upload_url_for(release, "FixPicSetup-0.1.0.exe")
    assert url == "https://uploads.github.com/repos/o/r/releases/1/assets?name=FixPicSetup-0.1.0.exe"


def test_create_release_payload(monkeypatch):
    captured = {}

    def fake_request(url, token, *, method="GET", data=None, content_type="application/json"):
        captured["url"] = url
        captured["token"] = token
        captured["method"] = method
        captured["data"] = json.loads(data.decode())
        captured["content_type"] = content_type
        return {"html_url": "https://github.com/o/r/releases/tag/v0.1.0", "upload_url": "https://u/{?name}"}

    monkeypatch.setattr(mr, "github_request", fake_request)
    out = mr.create_release("o/r", "tok", "v0.1.0", "FixPic 0.1.0", "הערות")
    assert captured["url"] == "https://api.github.com/repos/o/r/releases"
    assert captured["method"] == "POST"
    assert captured["data"] == {"tag_name": "v0.1.0", "name": "FixPic 0.1.0", "body": "הערות",
                                 "draft": False, "prerelease": False}
    assert out["html_url"].endswith("v0.1.0")


def test_upload_asset_sends_bytes_with_octet_stream(monkeypatch, tmp_path):
    installer = tmp_path / "FixPicSetup-0.1.0.exe"
    installer.write_bytes(b"\x00\x01binary-ish")
    captured = {}

    def fake_request(url, token, *, method="GET", data=None, content_type="application/json"):
        captured.update(url=url, method=method, data=data, content_type=content_type)
        return {"browser_download_url": "https://github.com/o/r/releases/download/v0.1.0/" + installer.name}

    monkeypatch.setattr(mr, "github_request", fake_request)
    release = {"upload_url": "https://uploads.github.com/repos/o/r/releases/1/assets{?name,label}"}
    out = mr.upload_asset(release, "tok", installer)
    assert captured["url"].endswith(f"?name={installer.name}")
    assert captured["method"] == "POST"
    assert captured["data"] == installer.read_bytes()
    assert captured["content_type"] == "application/octet-stream"
    assert out["browser_download_url"].endswith(installer.name)


def test_dry_run_touches_neither_git_nor_network(monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("אסור לקרוא לזה במצב --dry-run")

    monkeypatch.setattr(mr, "run_git", boom)
    monkeypatch.setattr(mr, "tag_exists_locally", boom)
    monkeypatch.setattr(mr, "github_request", boom)
    rc = mr.main(["0.1.0", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "v0.1.0" in out and "dry-run" in out


def test_dry_run_with_installer_reports_size(monkeypatch, tmp_path, capsys):
    installer = tmp_path / "FixPicSetup-0.1.0.exe"
    installer.write_bytes(b"x" * 2_000_000)
    monkeypatch.setattr(mr, "run_git", lambda *a, **k: (_ for _ in ()).throw(AssertionError("git")))
    rc = mr.main(["0.1.0", "--installer", str(installer), "--dry-run"])
    assert rc == 0
    assert "FixPicSetup-0.1.0.exe" in capsys.readouterr().out


def test_main_rejects_bad_version():
    assert mr.main(["not-a-version", "--dry-run"]) == 2


def test_main_rejects_missing_installer_file(tmp_path):
    missing = tmp_path / "nope.exe"
    assert mr.main(["0.1.0", "--installer", str(missing), "--dry-run"]) == 2


def test_main_requires_token_when_not_dry_run(monkeypatch, tmp_path):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(mr, "tag_exists_locally", lambda tag: False)
    rc = mr.main(["0.1.0"])
    assert rc == 2


def test_main_fails_if_tag_already_exists(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setattr(mr, "tag_exists_locally", lambda tag: True)
    rc = mr.main(["0.1.0"])
    assert rc == 2
