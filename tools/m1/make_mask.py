#!/usr/bin/env python3
"""Draw a removal mask on an image and save an M1 sample pair.

Usage
  python tools\\m1\\make_mask.py                      (opens a file dialog)
  python tools\\m1\\make_mask.py photo.jpg
  python tools\\m1\\make_mask.py photo.jpg --name person --out samples --grow 4

Output (in --out, default "samples"), ready for  m1_check.py run --dir samples
  NAME.png        the image (decoded, EXIF rotation applied, re-saved lossless)
  NAME_mask.png   same size, black background, white = area to remove

Controls
  left button drag   paint            right button drag (or key E)   erase
  mouse wheel / [ ]  brush size       Ctrl+Z   undo (last 10 strokes)
  Ctrl+S             save             Esc      quit without saving

Needs only opencv + numpy (already in tools/m1/requirements.txt) and tkinter,
which ships with the standard Windows Python installer. Hebrew paths work.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

UNDO_LIMIT = 10
MAX_ZOOM = 2.0  # small images are enlarged on screen up to this factor


# ------------------------------------------------------------ core (no GUI)
def read_image(path):
    """Decode with np.fromfile so Unicode paths work; EXIF rotation is applied."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode image: {path}")
    return img


def write_png(path, arr):
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError(f"cannot encode PNG: {path}")
    buf.tofile(str(path))


def paint(mask, p0, p1, radius, value):
    """Round brush stroke from p0 to p1 on a uint8 mask (value 255 paint, 0 erase)."""
    r = max(1, int(round(radius)))
    p0 = (int(round(p0[0])), int(round(p0[1])))
    p1 = (int(round(p1[0])), int(round(p1[1])))
    cv2.line(mask, p0, p1, int(value), thickness=2 * r, lineType=cv2.LINE_8)
    cv2.circle(mask, p0, r, int(value), -1)
    cv2.circle(mask, p1, r, int(value), -1)


def grow_mask(mask, px):
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, k)


def to_ppm(bgr):
    """Raw P6 bytes that tkinter.PhotoImage(data=...) can load without PNG decoding."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return b"P6 %d %d 255\n" % (w, h) + rgb.tobytes()


def save_pair(img, mask, out_dir, name, grow=0):
    """Write NAME.png and NAME_mask.png; returns (image_path, mask_path, mask_pixels)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = grow_mask(((mask > 0).astype(np.uint8)) * 255, grow)
    ip, mp = out_dir / f"{name}.png", out_dir / f"{name}_mask.png"
    write_png(ip, img)
    write_png(mp, m)
    return ip, mp, int((m > 0).sum())


# ------------------------------------------------------------------- GUI
def run_gui(img_path, out_dir, name, grow):
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("make_mask")
    if img_path is None:
        sel = filedialog.askopenfilename(
            title="בחר תמונה",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All", "*.*")])
        if not sel:
            root.destroy()
            return 1
        img_path = Path(sel)
    img = read_image(img_path)
    H, W = img.shape[:2]
    name = name or img_path.stem

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    scale = min((sw - 60) / W, (sh - 220) / H, MAX_ZOOM)
    dw, dh = max(1, int(W * scale)), max(1, int(H * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    base = cv2.resize(img, (dw, dh), interpolation=interp)
    tint = (base.astype(np.float32) * 0.45 + np.array([40, 40, 255], np.float32) * 0.55
            ).astype(np.uint8)  # BGR, red

    mask = np.zeros((H, W), np.uint8)          # full resolution, saved
    mask_d = np.zeros((dh, dw), np.uint8)      # display resolution, shown
    undo = []
    st = {"last": None, "erase": False, "dirty": False, "sched": False, "photo": None}

    top = tk.Frame(root)
    top.pack(side="top", fill="x")
    canvas = tk.Canvas(root, width=dw, height=dh, highlightthickness=0, cursor="none")
    canvas.pack(side="top")
    status = tk.Label(root, anchor="w", text="")
    status.pack(side="bottom", fill="x")
    canvas_img = canvas.create_image(0, 0, anchor="nw")
    ring = canvas.create_oval(0, 0, 0, 0, outline="yellow", width=1)

    brush = tk.IntVar(value=max(6, int(min(dw, dh) * 0.03)))  # radius, screen px
    erase_var = tk.BooleanVar(value=False)

    def msg(text):
        status.config(text=text)

    def refresh():
        st["sched"] = False
        view = base.copy()
        np.copyto(view, tint, where=(mask_d > 0)[:, :, None])
        st["photo"] = tk.PhotoImage(data=to_ppm(view), format="PPM")
        canvas.itemconfig(canvas_img, image=st["photo"])
        st["dirty"] = False

    def request_refresh():
        st["dirty"] = True
        if not st["sched"]:
            st["sched"] = True
            root.after(15, refresh)

    def snapshot():
        undo.append(mask.copy())
        if len(undo) > UNDO_LIMIT:
            undo.pop(0)

    def resync_display():
        mask_d[:] = cv2.resize(mask, (dw, dh), interpolation=cv2.INTER_NEAREST)

    def stroke(p0, p1, erase):
        v = 0 if erase else 255
        r = brush.get()
        paint(mask_d, p0, p1, r, v)
        paint(mask, (p0[0] / scale, p0[1] / scale), (p1[0] / scale, p1[1] / scale),
              r / scale, v)
        request_refresh()

    def inside(e):
        return (min(max(e.x, 0), dw - 1), min(max(e.y, 0), dh - 1))

    def on_down(e, erase_btn=False):
        snapshot()
        er = erase_btn or erase_var.get()
        st["erase"] = er
        st["last"] = inside(e)
        stroke(st["last"], st["last"], er)

    def on_move(e):
        r = brush.get()
        canvas.coords(ring, e.x - r, e.y - r, e.x + r, e.y + r)
        if st["last"] is not None:
            p = inside(e)
            stroke(st["last"], p, st["erase"])
            st["last"] = p

    def on_up(_e):
        st["last"] = None
        n = int((mask > 0).sum())
        msg(f"{name}: {W}x{H}, mask pixels {n} ({100.0 * n / (W * H):.2f}%)")

    def do_undo(_e=None):
        if undo:
            mask[:] = undo.pop()
            resync_display()
            request_refresh()

    def do_clear():
        snapshot()
        mask[:] = 0
        mask_d[:] = 0
        request_refresh()

    def do_save(_e=None):
        if not (mask > 0).any():
            messagebox.showwarning("make_mask", "המסכה ריקה: צייר קודם על האזור להסרה.")
            return
        ip, mp, n = save_pair(img, mask, out_dir, name, grow)
        msg(f"נשמר: {ip}  |  {mp}  ({n} px)")
        messagebox.showinfo("make_mask", f"נשמר:\n{ip}\n{mp}")

    def wheel(e):
        d = 1 if (e.delta > 0 or getattr(e, "num", 0) == 4) else -1
        brush.set(min(200, max(2, brush.get() + 2 * d)))

    def toggle_erase(_e=None):
        erase_var.set(not erase_var.get())

    tk.Label(top, text="גודל מברשת").pack(side="right", padx=4)
    tk.Scale(top, from_=2, to=200, orient="horizontal", variable=brush, length=180
             ).pack(side="right")
    tk.Checkbutton(top, text="מחק (E)", variable=erase_var).pack(side="right", padx=8)
    tk.Button(top, text="בטל (Ctrl+Z)", command=do_undo).pack(side="left", padx=2)
    tk.Button(top, text="נקה", command=do_clear).pack(side="left", padx=2)
    tk.Button(top, text="שמור (Ctrl+S)", command=do_save).pack(side="left", padx=2)

    canvas.bind("<Button-1>", on_down)
    canvas.bind("<B1-Motion>", on_move)
    canvas.bind("<ButtonRelease-1>", on_up)
    canvas.bind("<Button-3>", lambda e: on_down(e, True))
    canvas.bind("<B3-Motion>", on_move)
    canvas.bind("<ButtonRelease-3>", on_up)
    canvas.bind("<Motion>", on_move)
    canvas.bind("<MouseWheel>", wheel)
    canvas.bind("<Button-4>", wheel)
    canvas.bind("<Button-5>", wheel)
    root.bind("<Control-z>", do_undo)
    root.bind("<Control-s>", do_save)
    root.bind("e", toggle_erase)
    root.bind("[", lambda _e: brush.set(max(2, brush.get() - 2)))
    root.bind("]", lambda _e: brush.set(min(200, brush.get() + 2)))
    root.bind("<Escape>", lambda _e: root.destroy())

    refresh()
    msg(f"{img_path.name}: {W}x{H}. צייר על האזור להסרה (מומלץ מעט מעבר לגבולות)")
    root.mainloop()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", help="image path (omit to pick with a dialog)")
    ap.add_argument("--out", default="samples", help="output directory (default: samples)")
    ap.add_argument("--name", help="sample name (default: image file name without extension)")
    ap.add_argument("--grow", type=int, default=0,
                    help="dilate the mask by N pixels when saving (default 0)")
    a = ap.parse_args()
    img_path = Path(a.image) if a.image else None
    if img_path is not None and not img_path.is_file():
        print(f"file not found: {img_path}", file=sys.stderr)
        return 2
    return run_gui(img_path, a.out, a.name, a.grow)


if __name__ == "__main__":
    sys.exit(main())
