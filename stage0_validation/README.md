# Stage 0 — Model Validation

## Purpose

Stage 0 is a one-time pre-flight check that must pass before the M1 inference
pipeline is usable. It verifies that `models/big-lama/inpainter.onnx` is the
correct, fully-downloaded binary and extracts its tensor metadata into
`docs/architecture/cleanup-ai/MODEL_CARD.md`.

**What it does:**
1. Verifies SHA256 hash and file size of `inpainter.onnx`
2. Confirms the file is not a Git LFS pointer stub
3. Loads the model with `onnxruntime` and reads input/output tensor metadata
4. Updates `MODEL_CARD.md` with verified tensor names, shapes, and status markers

**What it does NOT do:** run any inference on images.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9+ |
| onnxruntime | ≥ 1.30.0 |

Install onnxruntime if needed:

```bash
pip install "onnxruntime>=1.30.0"
```

The script uses only the Python standard library plus onnxruntime — no other
third-party packages are required.

---

## How to Run

Run from the **workspace root** (the directory that contains `validate_stage0.py`):

```bash
python validate_stage0.py
```

Print usage/help:

```bash
python validate_stage0.py --help
```

---

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | All steps succeeded |
| `1` | SHA256 hash or file size mismatch |
| `2` | File is a Git LFS pointer (model not fully downloaded) |
| `3` | `onnxruntime.InferenceSession` failed to load the model |
| `4` | Input or output tensor metadata extraction failed |
| `5` | Tensor role identification failed (image/mask pattern not found) |
| `6` | Writing updated `MODEL_CARD.md` to disk failed |

---

## Expected Console Output

Each validation step prints a `[STEP]` line followed by an optional details line:

```
[FILE VALIDATION] started
[SHA256 COMPUTATION] success
  Details: ef85edd865769c03c753b5faa066be713a46a9c80c8f77954c67c0bb3a4ba2ff
[FILE SIZE CHECK] success
  Details: 208851664 bytes
[LFS POINTER CHECK] success
  Details: Not an LFS pointer
[FILE VALIDATION] success
[MODEL SESSION LOADING] started
[MODEL SESSION LOADING] success
[METADATA EXTRACTION] started
[INPUT METADATA EXTRACTION] success
  Details: 2 inputs
[OUTPUT METADATA EXTRACTION] success
  Details: 1 outputs
[TENSOR ROLE IDENTIFICATION] success
  Details: image=<name>, mask=<name>
[DIMENSION CONSTRAINTS] success
[METADATA EXTRACTION] success
[MODEL_CARD UPDATE] started
...
[MODEL_CARD UPDATE] success
[VALIDATION COMPLETE] success
```

Followed by a summary block:

```
============================================================
Stage 0 Validation Summary
============================================================

✓ sha256_computation: success
  → ef85edd865769c03c753b5faa066be713a46a9c80c8f77954c67c0bb3a4ba2ff

✓ file_size_check: success
  → 208851664

✓ lfs_pointer_check: success
  → Not an LFS pointer

... (one line per step)

Overall: SUCCESS
============================================================
```

If any step fails the symbol changes to `✗`, the step details show the error,
and `Overall: FAILED`.

---

## What Gets Updated in MODEL_CARD.md

After a successful run the following changes are written to
`docs/architecture/cleanup-ai/MODEL_CARD.md`:

- **Section 1** — SHA256 and file-size status markers change from `[pointer]`
  to `[אומת]`.
- **Section 3** — Input/output tensor names and shapes are filled in; markers
  change from `[README – לאמת ב-\`probe\`]` to `[אומת]`.
- **Section 8** (task list) — Completed tasks are checked off.
- A timestamped verification note is prepended to the document.

Markers that were not covered by this stage (`[README]`, `[לא ידוע]`,
`[ידע כללי]`) are left unchanged.

---

## Backup File

Before writing any changes, the script copies the original MODEL_CARD to:

```
docs/architecture/cleanup-ai/MODEL_CARD.md.bak
```

The `.bak` file is overwritten on every run. If you need to keep multiple
snapshots, copy it manually before re-running.
