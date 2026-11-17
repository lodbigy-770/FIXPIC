"""M5: תיקיית המודלים בהרצה מארוזה (PyInstaller: `sys._MEIPASS`)."""
from __future__ import annotations

import sys
from pathlib import Path

from fixpic import cli


def test_dev_path_is_repo_models(monkeypatch):
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    d = cli.default_models_dir()
    assert d.name == "models" and (d / "big-lama" / "model.json").is_file()


def test_frozen_path_uses_meipass(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert cli.default_models_dir() == Path(tmp_path) / "models"
