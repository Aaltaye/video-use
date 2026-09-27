"""Fit a look (colour LUT + optional frame mask) from a before/after still pair.

Use it to copy a filter/preset from a phone app (PREQUEL, VSCO, CapCut...) that has
no export: screenshot the same frame without and with the preset, then run

    python helpers/fit_look.py before.png after.png --name prequel --glow 0.75 --frame

It writes, into static/looks/:
  <name>.cube        33^3 3D LUT for ffmpeg's lut3d filter (colour only)
  <name>_mask.png    (with --frame) the window traced from the after image:
                     white = picture, black = frame. Same size as the after image.
  <name>.json        alignment scale + glow settings used during the fit

How it works:
  1. Align: search the zoom/offset that maps the before image onto the after
     image (apps often shrink the picture inside a frame).
  2. Undo glow: the look = LUT then a screen-blended blur (`--glow` opacity,
     `--sigma` as fraction of frame height). We invert that blend so the LUT
     only has to explain colour, then refit twice.
  3. Fit LUT: average after-colour per RGB bin, fill empty bins from nearest
     neighbours, light smoothing so gradients don't band.

Needs numpy, scipy, pillow (pip install numpy scipy pillow).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

N = 33  # LUT size
OUT_DIR = Path(__file__).resolve().parent.parent / "static" / "looks"


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def trace_mask(after: np.ndarray, thresh: float = 22 / 255, run: int = 6) -> np.ndarray:
    """Trace the bright window in the after image, row by row, symmetric about the centre."""
    lum = after.max(2)
    h, w = lum.shape
    halfw = np.full(h, np.nan)
    for y in range(h):
        row = lum[y] > thresh
        # a run of bright pixels, so thin guide lines/dust don't count as picture
        runs = np.convolve(row, np.ones(run), "valid") >= run
        xs = np.flatnonzero(runs)
        if len(xs):
            left, right = xs[0], xs[-1] + run - 1
            halfw[y] = (right - left + 1) / 2
    rows = np.flatnonzero(~np.isnan(halfw))
    top, bottom = rows[0], rows[-1]
    hw = halfw[top:bottom + 1]
    hw = np.where(np.isnan(hw), np.nanmax(hw), hw)
    hw = ndimage.median_filter(hw, 5)
    hw = ndimage.uniform_filter1d(hw, 5)
    # anti-aliased coverage per pixel
    xs = np.arange(w) + 0.5
    cx = w / 2
    mask = np.zeros((h, w), np.float32)
    mask[top:bottom + 1] = np.clip(hw[:, None] - np.abs(xs[None, :] - cx) + 0.5, 0, 1)
    return mask


def align(before: np.ndarray, after: np.ndarray, region: tuple[slice, slice]) -> tuple[float, float, float]:
    """Find zoom s and offset (dy, dx) so after(y, x) ≈ before(c + (y - c)/s + d)."""
    def edges(img):
        g = ndimage.gaussian_filter(img.mean(2), 2)
        return np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))

    ea, eb = edges(after), edges(before)
    h, w = ea.shape
    ys, xs = region
    yy, xx = np.mgrid[ys.start:ys.stop:3, xs.start:xs.stop:3]
    target = ea[yy, xx].ravel()

    def score(s, dy, dx):
        p = ndimage.map_coordinates(eb, [h / 2 + (yy - h / 2) / s + dy, w / 2 + (xx - w / 2) / s + dx], order=1)
        return np.corrcoef(p.ravel(), target)[0, 1]

    best = max((score(s, dy, dx), s, dy, dx)
               for s in np.arange(0.86, 1.15, 0.02)
               for dy in range(-60, 61, 10) for dx in range(-60, 61, 10))
    _, s0, dy0, dx0 = best
    best = max((score(s, dy, dx), s, dy, dx)
               for s in np.arange(s0 - 0.02, s0 + 0.021, 0.004)
               for dy in range(dy0 - 10, dy0 + 11, 2) for dx in range(dx0 - 10, dx0 + 11, 2))
    print(f"  align: corr={best[0]:.3f} scale={best[1]:.3f} dy={best[2]} dx={best[3]}")
    return best[1], best[2], best[3]


def warp(before: np.ndarray, s: float, dy: float, dx: float) -> np.ndarray:
    h, w, _ = before.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    coords = [h / 2 + (yy - h / 2) / s + dy, w / 2 + (xx - w / 2) / s + dx]
    return np.stack([ndimage.map_coordinates(before[..., c], coords, order=1) for c in range(3)], -1)


def fit_lut(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """src, dst: (K, 3) in 0..1. Returns (N, N, N, 3) LUT indexed [r, g, b]."""
    idx = np.clip(np.rint(src * (N - 1)).astype(int), 0, N - 1)
    flat = (idx[:, 0] * N + idx[:, 1]) * N + idx[:, 2]
    cnt = np.bincount(flat, minlength=N ** 3).astype(np.float32)
    lut = np.stack([np.bincount(flat, dst[:, c], N ** 3) for c in range(3)], -1)
    filled = cnt > 0
    lut[filled] /= cnt[filled, None]
    lut = lut.reshape(N, N, N, 3)
    filled = filled.reshape(N, N, N)
    # fill empty bins from nearest filled bin, then smooth
    near = ndimage.distance_transform_edt(~filled, return_distances=False, return_indices=True)
    lut = lut[near[0], near[1], near[2]]
    lut = np.stack([ndimage.gaussian_filter(lut[..., c], 0.6) for c in range(3)], -1)
    return np.clip(lut, 0, 1)


def apply_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    coords = [img[..., c] * (N - 1) for c in range(3)]
    return np.stack([ndimage.map_coordinates(lut[..., k], coords, order=1) for k in range(3)], -1)


def glow_layer(img: np.ndarray, sigma_px: float) -> np.ndarray:
    return np.stack([ndimage.gaussian_filter(img[..., c], sigma_px) for c in range(3)], -1)


def write_cube(lut: np.ndarray, path: Path, title: str) -> None:
    with open(path, "w") as f:
        f.write(f'TITLE "{title}"\nLUT_3D_SIZE {N}\n')
        for b in range(N):          # .cube order: red fastest, then green, then blue
            for g in range(N):
                for r in range(N):
                    f.write("%.6f %.6f %.6f\n" % tuple(lut[r, g, b]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit a LUT (+ frame mask) from a before/after pair")
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--glow", type=float, default=0.0, help="Screen-blend glow opacity the look will add on top")
    ap.add_argument("--sigma", type=float, default=0.012, help="Glow blur, fraction of frame height")
    ap.add_argument("--frame", action="store_true", help="Trace a frame/window mask from the after image")
    args = ap.parse_args()

    before, after = load(args.before), load(args.after)
    if before.shape != after.shape:
        before = np.asarray(Image.fromarray((before * 255).astype(np.uint8)).resize(after.shape[1::-1], Image.LANCZOS)).astype(np.float32) / 255
    h, w, _ = after.shape
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mask = trace_mask(after) if args.frame else np.ones((h, w), np.float32)
    inner = ndimage.binary_erosion(mask > 0.99, iterations=12)
    rows, cols = np.flatnonzero(inner.any(1)), np.flatnonzero(inner.any(0))
    cy, cx = (rows[0] + rows[-1]) // 2, (cols[0] + cols[-1]) // 2
    ry, rx = (rows[-1] - rows[0]) // 4, (cols[-1] - cols[0]) // 3
    s, dy, dx = align(before, after, (slice(cy - ry, cy + ry), slice(cx - rx, cx + rx)))
    src_img = warp(before, s, dy, dx)

    # ignore letterbox bars in the source (pure black rows/cols) as well as the frame
    valid = inner & (src_img.max(2) > 2 / 255)
    src = src_img[valid]
    sigma_px = args.sigma * h

    target = after
    lut = fit_lut(src, after[valid])
    for it in range(3 if args.glow > 0 else 0):
        pre = apply_lut(src_img, lut)
        g = args.glow * glow_layer(pre * mask[..., None], sigma_px)
        # invert screen blend: after = 1 - (1 - x)(1 - g)  =>  x = 1 - (1 - after)/(1 - g)
        target = np.clip(1 - (1 - after) / np.maximum(1 - g, 1e-3), 0, 1)
        lut = fit_lut(src, target[valid])

    pre = apply_lut(src_img, lut)
    out = 1 - (1 - pre) * (1 - args.glow * glow_layer(pre * mask[..., None], sigma_px))
    err = np.abs(out[valid] - after[valid]).mean() * 255
    print(f"  mean abs error after fit: {err:.1f}/255")

    write_cube(lut, OUT_DIR / f"{args.name}.cube", args.name)
    if args.frame:
        Image.fromarray((mask * 255).round().astype(np.uint8), "L").save(OUT_DIR / f"{args.name}_mask.png")
    meta = {"scale": round(float(s), 4), "glow": args.glow, "sigma": args.sigma,
            "frame": args.frame, "fit_size": [w, h], "mean_abs_error": round(float(err), 2)}
    (OUT_DIR / f"{args.name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote {OUT_DIR}/{args.name}.*")


if __name__ == "__main__":
    main()
