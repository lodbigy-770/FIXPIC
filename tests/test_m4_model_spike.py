"""M4 / AUTO-SELECT שלב 5: `tools/m4/model_spike.py` (סקריפט מדידת מועמדי מודל לחיצה-חכמה).

הבדיקות רצות מול זוג מודלי ONNX **סינתטי** בחוזה SAM (מקודד AveragePool, מפענח שמחזיר מסכות קבועות), ולכן בודקות
את הלוגיקה של הסקריפט (קדם-עיבוד, מיפוי קואורדינטות, בניית קלט המפענח, שחזור מסכה, שער, JSON) ולא את איכות
או חוזה של מודל אמיתי: זה נמדד אצל המשתמש."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import pytest
from onnx import TensorProto as T, helper as h

SPEC = importlib.util.spec_from_file_location("model_spike", Path(__file__).resolve().parents[1] / "tools" / "m4" / "model_spike.py")
ms = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ms)


def save(nodes, inputs, outputs, inits, path):
    g = h.make_graph(nodes, "t", inputs, outputs, inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)
    return path


def make_encoder(path, side=64, out_name="image_embeddings"):
    x = h.make_tensor_value_info("input_image", T.FLOAT, [1, 3, side, side])
    y = h.make_tensor_value_info(out_name, T.FLOAT, [1, 3, side // 8, side // 8])
    node = h.make_node("AveragePool", ["input_image"], [out_name], kernel_shape=[8, 8], strides=[8, 8])
    return save([node], [x], [y], [], path)


def make_decoder(path, extra_input=None, square_output=False):
    """masks: (K=2): [0] כל התמונה חיובית, [1] רק הרבע השמאלי-עליון. iou = [0.3, 0.9] => הטוב הוא הרבע.
    ברירת מחדל: Resize לפי orig_im_size (כמו ייצוא SAM). square_output: פלט קבוע 16x16 (מרופד/נמוך)."""
    low = np.full((1, 2, 16, 16), 5.0, np.float32)
    low[0, 1] = -5.0
    low[0, 1, :8, :8] = 5.0
    ins = [h.make_tensor_value_info("image_embeddings", T.FLOAT, [1, 3, 8, 8]),
           h.make_tensor_value_info("point_coords", T.FLOAT, [1, "n", 2]),
           h.make_tensor_value_info("point_labels", T.FLOAT, [1, "n"]),
           h.make_tensor_value_info("mask_input", T.FLOAT, [1, 1, 16, 16]),
           h.make_tensor_value_info("has_mask_input", T.FLOAT, [1]),
           h.make_tensor_value_info("orig_im_size", T.FLOAT, [2])]
    if extra_input:
        ins.append(h.make_tensor_value_info(extra_input, T.FLOAT, [1]))
    inits = [h.make_tensor("low", T.FLOAT, low.shape, low.ravel().tolist()),
             h.make_tensor("lead", T.INT64, [2], [1, 2]),
             h.make_tensor("iou", T.FLOAT, [1, 2], [0.3, 0.9])]
    if square_output:
        nodes = [h.make_node("Identity", ["low"], ["masks"]), h.make_node("Identity", ["iou"], ["iou_predictions"])]
        outs = [h.make_tensor_value_info("masks", T.FLOAT, [1, 2, 16, 16]),
                h.make_tensor_value_info("iou_predictions", T.FLOAT, [1, 2])]
    else:
        nodes = [h.make_node("Cast", ["orig_im_size"], ["sz"], to=T.INT64),
                 h.make_node("Concat", ["lead", "sz"], ["sizes"], axis=0),
                 h.make_node("Resize", ["low", "", "", "sizes"], ["masks"], mode="nearest"),
                 h.make_node("Identity", ["iou"], ["iou_predictions"])]
        outs = [h.make_tensor_value_info("masks", T.FLOAT, [1, 2, "h", "w"]),
                h.make_tensor_value_info("iou_predictions", T.FLOAT, [1, 2])]
    return save(nodes, ins, outs, inits, path)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("spike")
    return make_encoder(d / "enc.onnx"), make_decoder(d / "dec.onnx"), d


def scene(h_=96, w_=128):
    img = np.full((h_, w_, 3), 80, np.uint8)
    cv2.rectangle(img, (30, 20), (90, 70), (200, 40, 40), -1)
    return img


# ------------------------------------------------------------------ פונקציות טהורות
def test_preprocess_shape_padding_and_scale():
    x, scale, newh, neww = ms.preprocess(scene(96, 128), 64, "none", "area")
    assert x.shape == (1, 3, 64, 64) and x.dtype == np.float32
    assert scale == 0.5 and (newh, neww) == (48, 64)
    assert x[0, :, 48:, :].max() == 0           # ריפוד באפסים מימין/מטה בלבד
    assert x[0, :, :48, :].min() >= 0 and x[0, 0, 5, 5] == pytest.approx(80, abs=1)


def test_preprocess_sam_norm_and_upscale_small_image():
    img = np.full((10, 20, 3), 123, np.uint8)
    x, scale, newh, neww = ms.preprocess(img, 64, "sam", "area")
    assert scale == pytest.approx(3.2) and (newh, neww) == (32, 64)       # תמונה קטנה מוגדלת (כמו SAM)
    exp = (123 - ms.SAM_MEAN) / ms.SAM_STD
    assert np.allclose(x[0, :, 0, 0], exp, atol=1e-4)
    with pytest.raises(ms.SpikeError):
        ms.preprocess(img, 64, "bogus", "area")


def test_encode_prompts_click_adds_padding_point():
    c, l = ms.encode_prompts([(100, 50, True), (10, 10, False)], None, 0.5)
    assert c.shape == (1, 3, 2) and l.shape == (1, 3)
    assert c[0].tolist() == [[50, 25], [5, 5], [0, 0]] and l[0].tolist() == [1, 0, -1]


def test_encode_prompts_box_labels_and_no_padding():
    c, l = ms.encode_prompts([], (20, 40, 60, 80), 0.5)
    assert c[0].tolist() == [[10, 20], [30, 40]] and l[0].tolist() == [2, 3]
    c, l = ms.encode_prompts([(1, 2, True)], (20, 40, 60, 80), 1.0)
    assert l[0].tolist() == [1, 2, 3] and c.dtype == np.float32


@pytest.mark.parametrize("name,role", [("image_embeddings", "embeddings"), ("point_coords", "coords"),
                                       ("point_labels", "labels"), ("mask_input", "mask_input"),
                                       ("has_mask_input", "has_mask"), ("orig_im_size", "orig_size"),
                                       ("something_else", None)])
def test_role_of(name, role):
    assert ms.role_of(name) == role


def test_to_original_resized_and_padded_and_probabilities():
    lg = np.full((48, 64), -3.0, np.float32)
    lg[:24, :32] = 3.0
    m = ms.to_original(lg, 96, 128, 48, 64, 64)                           # פלט בגודל התמונה המוקטנת -> מקורית
    assert m.shape == (96, 128) and m[:40, :60].all() and not m[60:, 70:].any()
    sq = np.full((16, 16), -3.0, np.float32)                              # מרובע ומרופד: התמונה תופסת 12 שורות מ-16
    sq[:6, :8] = 3.0
    m = ms.to_original(sq, 96, 128, 48, 64, 64)
    assert m.shape == (96, 128) and m[:40, :60].all() and not m[60:, 70:].any()
    p = np.zeros((48, 64), np.float32)
    p[:10] = 0.8                                                          # הסתברויות 0..1: סף 0.5, לא 0
    assert ms.to_original(p, 48, 64, 48, 64, 64)[:10].all() and not ms.to_original(p, 48, 64, 48, 64, 64)[10:].any()
    with pytest.raises(ms.SpikeError):
        ms.to_original(np.zeros((10, 20), np.float32), 96, 128, 48, 64, 64)


def test_check_model_file(tmp_path):
    with pytest.raises(ms.SpikeError, match="not found"):
        ms.check_model_file(tmp_path / "missing.onnx")
    p = tmp_path / "x.onnx"
    p.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n")
    with pytest.raises(ms.SpikeError, match="LFS"):
        ms.check_model_file(p)


def test_license_info_lists_files_and_is_unverified(tmp_path):
    (tmp_path / "LICENSE").write_text("Apache License\nVersion 2.0", "utf-8")
    (tmp_path / "README.md").write_text("x", "utf-8")
    (tmp_path / "model.onnx").write_bytes(b"0")
    info = ms.license_info(tmp_path)
    assert info["files_next_to_model"] == ["LICENSE", "README.md"] and info["verified"] is False
    assert info["license_excerpt"]["LICENSE"].startswith("Apache License Version 2.0")


def test_evaluate_gate_numbers_and_verdicts():
    def run(threads, cond, prep_ms, prop_ms, extra=500):
        return {"threads": threads, "condition": cond, "prepare_person": {"warm_ms": prep_ms / 4},
                "prepare_24mp": {"warm_ms": prep_ms}, "propose": [{"total_ms": prop_ms}, {"total_ms": prop_ms / 2}],
                "memory": {"extra_peak_mb": extra}}
    g = ms.evaluate_gate([run(4, "idle", 3000, 100), run(1, "idle", 9000, 300, 700)])
    assert g["prepare_s_best"] == 3.0 and g["propose_s_worst_at_fewest_threads"] == 0.3
    assert g["prepare_ok"] and g["propose_ok"] and g["extra_peak_mb"] == 500
    g = ms.evaluate_gate([run(4, "idle", 6000, 100), run(1, "contention", 9000, 800)])
    assert not g["prepare_ok"] and g["propose_s_worst_during_inpaint"] == 0.8 and not g["propose_ok"]


# ------------------------------------------------------------------ מועמד מול מודלים סינתטיים
class Spy:
    """עוטף session של המפענח ורושם את ה-feed האחרון."""

    def __init__(self, inner):
        self.inner, self.feed = inner, None

    def get_inputs(self):
        return self.inner.get_inputs()

    def run(self, names, feed):
        self.feed = feed
        return self.inner.run(names, feed)


def test_candidate_prepare_and_propose_end_to_end(models):
    enc, dec, _ = models
    c = ms.Candidate("syn", enc, dec, 1, "none", 1024, "area")
    assert c.size == 64                                                   # נקרא מצורת הקלט של המקודד
    img = scene()
    ctx = c.prepare(img)
    assert ctx["emb"]["image_embeddings"].shape == (1, 3, 8, 8) and ctx["scale"] == 0.5
    spy = c.dec = Spy(c.dec)
    res = c.propose(ctx, [(60, 40, True)], None)
    f = spy.feed
    assert f["orig_im_size"].tolist() == [48, 64]                         # גודל התמונה המוקטנת, לא המקורית
    assert f["point_coords"].tolist() == [[[30, 20], [0, 0]]] and f["point_labels"].tolist() == [[1, -1]]
    assert f["mask_input"].shape == (1, 1, 16, 16) and not f["mask_input"].any() and f["has_mask_input"].tolist() == [0]
    assert res["best"] == 1 and res["scores"] == [0.3, 0.9] and len(res["masks"]) == 2
    best = res["masks"][res["best"]]
    assert best.shape == (96, 128) and best[:40, :60].all() and not best[60:, 70:].any()
    assert res["masks"][0].all()
    c.propose(ctx, [], (10, 20, 90, 70))
    assert spy.feed["point_labels"].tolist() == [[2, 3]]


def test_candidate_square_output_decoder(tmp_path):
    enc, dec = make_encoder(tmp_path / "e.onnx"), make_decoder(tmp_path / "d.onnx", square_output=True)
    c = ms.Candidate("sq", enc, dec, 1, "none", 1024, "area")
    res = c.propose(c.prepare(scene()), [(60, 40, True)], None)
    m = res["masks"][res["best"]]
    assert m.shape == (96, 128) and m[:40, :60].all() and not m[60:, 70:].any()


def test_candidate_unknown_decoder_input_reports_names(tmp_path):
    enc, dec = make_encoder(tmp_path / "e.onnx"), make_decoder(tmp_path / "d.onnx", extra_input="weird_tensor")
    with pytest.raises(ms.SpikeError) as e:
        ms.Candidate("bad", enc, dec, 1, "none", 1024, "area")
    assert "weird_tensor" in str(e.value) and "decoder inputs" in str(e.value)


def test_candidate_renamed_encoder_output_still_feeds_embeddings_input(tmp_path):
    enc = make_encoder(tmp_path / "e.onnx", out_name="feats")             # שם פלט שונה מזה שהמפענח מצפה לו
    dec = make_decoder(tmp_path / "d.onnx")
    c = ms.Candidate("x", enc, dec, 1, "none", 1024, "area")
    res = c.propose(c.prepare(scene()), [(10, 10, True)], None)
    assert res["best"] == 1 and res["masks"][1].shape == (96, 128)


def test_candidate_run_failure_is_spike_error_with_contract(tmp_path):
    enc = make_encoder(tmp_path / "e.onnx", side=128)                     # embedding 16x16, המפענח מצפה ל-8x8
    dec = make_decoder(tmp_path / "d.onnx")
    c = ms.Candidate("mismatch", enc, dec, 1, "none", 1024, "area")
    ctx = c.prepare(scene())
    with pytest.raises(ms.SpikeError, match="decoder run failed") as e:
        c.propose(ctx, [(10, 10, True)], None)
    assert "decoder inputs" in str(e.value)


# ------------------------------------------------------------------ מקצה לקצה דרך main
def lama_like(path):
    rgb = h.make_tensor_value_info("rgb", T.FLOAT, [1, 3, "8*height", "8*width"])
    mask = h.make_tensor_value_info("mask", T.FLOAT, [1, 1, "8*height", "8*width"])
    out = h.make_tensor_value_info("inpainted", T.FLOAT, [1, 3, "8*height", "8*width"])
    node = h.make_node("Mul", ["rgb", "mask"], ["inpainted"])
    return save([node], [rgb, mask], [out], [], path)


def test_main_end_to_end_json_images_and_contention(models, tmp_path):
    enc, dec, d = models
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    cv2.imencode(".png", cv2.cvtColor(scene(), cv2.COLOR_RGB2BGR))[1].tofile(str(imgs / "syn.png"))
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"name": "obj", "image": "syn", "box": [30, 20, 90, 70]},
                                 {"name": "obj2", "image": "syn", "box": [30, 20, 90, 70], "point": [60, 45]}]))
    lama = lama_like(tmp_path / "lama.onnx")
    out = tmp_path / "out"
    rc = ms.main(["--candidate", "syn", str(enc), str(dec), "--threads", "1", "--cases", str(cases),
                  "--images-dir", str(imgs), "--prepare-repeats", "1", "--propose-repeats", "2",
                  "--contention", str(lama), "--out", str(out)])
    assert rc == 0
    data = json.loads((out / "model_spike_results.json").read_text("utf-8"))
    cand = data["candidates"][0]
    assert cand["name"] == "syn" and set(cand["sha256"]) == {"encoder", "decoder"} and "gate" in cand
    conds = [r["condition"] for r in cand["runs"]]
    assert conds == ["idle", "contention"]
    idle = cand["runs"][0]
    assert idle["prepare_24mp"]["warm_ms"] > 0 and idle["fullres_mask_24mp_ms"] >= 0 and idle["enc_size"] == 64
    kinds = {(r["case"], r["kind"]) for r in idle["propose"]}
    assert kinds == {("obj", "click"), ("obj", "box"), ("obj2", "click"), ("obj2", "box")}
    assert all(r["best"] == 1 and 0.0 < r["area_frac"] < 1.0 for r in idle["propose"])
    assert all("iou_click_vs_box" in r for r in idle["propose"])
    assert (out / "syn" / "t1" / "obj_click.png").is_file() and (out / "syn" / "t1" / "obj2_box.png").is_file()
    assert "propose_s_worst_during_inpaint" in cand["gate"]


def test_main_reports_lfs_pointer_with_exit_2(tmp_path, capsys):
    p = tmp_path / "enc.onnx"
    p.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 5\n")
    rc = ms.main(["--candidate", "x", str(p), str(p), "--out", str(tmp_path / "o")])
    assert rc == 2 and "LFS" in capsys.readouterr().err
