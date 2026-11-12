import hashlib
import json

import pytest

from fixpic.errors import ModelLoadError
from fixpic.models import ModelManager

from test_backends import tiny  # noqa: F401  (fixture)


def make(root, folder="m", **over):
    name = folder
    d = root / folder
    d.mkdir(parents=True)
    d_json = {"schema": 1, "name": name, "version": "1", "license": "Apache-2.0", "file": "w.onnx",
              "input": {"multiple": 8, "min_size": 64, "max_size": 4096}}
    d_json.update(over)
    (d / "model.json").write_text(json.dumps(d_json), encoding="utf-8")
    return d


def test_list_skips_broken_and_reads_info(tmp_path):
    make(tmp_path, "good")
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "model.json").write_text("{nope", encoding="utf-8")
    (tmp_path / "nojson").mkdir()
    ms = ModelManager(tmp_path).list()
    assert [m.name for m in ms] == ["good"] and ms[0].info.multiple == 8 and ms[0].license == "Apache-2.0"


@pytest.mark.parametrize("over", [{"schema": 2}, {"file": "../x.onnx"}, {"file": "/abs.onnx"},
                                  {"sha256": "zz"}, {"input": {"multiple": 0}}, {"name": None}])
def test_invalid_manifest(tmp_path, over):
    make(tmp_path, "m", **over)
    with pytest.raises(ModelLoadError):
        ModelManager(tmp_path).manifest("m")


def test_missing_manifest_and_file(tmp_path):
    with pytest.raises(ModelLoadError):
        ModelManager(tmp_path).manifest("nope")
    make(tmp_path, "m")
    with pytest.raises(ModelLoadError, match="חסר"):
        ModelManager(tmp_path).load("m")


def test_load_with_hash_ok_and_mismatch(tmp_path, tiny):  # noqa: F811
    d = make(tmp_path, "t", sha256=hashlib.sha256(tiny.read_bytes()).hexdigest())
    (d / "w.onnx").write_bytes(tiny.read_bytes())
    m = ModelManager(tmp_path).load("t")
    assert m.backend in ("openvino", "onnxruntime")
    d2 = make(tmp_path, "t2", sha256="0" * 64)
    (d2 / "w.onnx").write_bytes(tiny.read_bytes())
    with pytest.raises(ModelLoadError, match="hash"):
        ModelManager(tmp_path).load("t2")


def test_size_mismatch(tmp_path, tiny):  # noqa: F811
    d = make(tmp_path, "s", size_bytes=1234567)
    (d / "w.onnx").write_bytes(tiny.read_bytes())
    with pytest.raises(ModelLoadError, match="גודל"):
        ModelManager(tmp_path).load("s")


def test_repo_big_lama_manifest_is_valid():
    from pathlib import Path
    mf = ModelManager(Path(__file__).resolve().parents[1] / "models").manifest("big-lama")
    assert mf.sha256 and mf.info.multiple == 8 and mf.path.name == "inpainter.onnx"
