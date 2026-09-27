"""Recreate the user's PREQUEL look: fitted colour LUT + haze/glow + rounded black frame.

Everything is fitted from a before/after pair by `fit_look.py` and lives in
static/looks/ (prequel.cube, prequel_mask.png, prequel.json):

  1. Shrink — PREQUEL zooms the picture out slightly (~4%) inside the frame.
  2. Colour — 3D LUT fitted pixel-for-pixel from the pair.
  3. Glow   — blurred copy screen-blended on top (bloom + haze).
  4. Frame  — the exact window traced from the after image; black outside.

`--dream` adds an old-TV-in-a-dream layer on top of the match: soft halation,
RGB fringing, fine scanlines and moving film grain.

Usage:
    python helpers/prequel_look.py <input> -o <output>            # exact match
    python helpers/prequel_look.py <input> -o <output> --dream    # + dreamy old-TV texture
    python helpers/prequel_look.py in.mp4 -o out.mp4 --glow 0.8 --no-frame

Works on videos and still images. The frame mask is stretched to the output size,
so it is exact for 9:16 vertical video (what it was traced from).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

LOOK_DIR = Path(__file__).resolve().parent.parent / "static" / "looks"
LUT = LOOK_DIR / "prequel.cube"
MASK = LOOK_DIR / "prequel_mask.png"
META = json.loads((LOOK_DIR / "prequel.json").read_text())

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def probe_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    for line in out.splitlines():
        if "Video:" in line:
            for tok in line.replace(",", " ").split():
                w, _, h = tok.partition("x")
                if w.isdigit() and h.isdigit():
                    return int(w), int(h)
    raise SystemExit(f"could not read video size of {path}")


def make_scanlines(w: int, h: int, out: Path) -> None:
    """Static scanline texture: soft dark line every ~1/540 of the height."""
    period = max(3, round(h / 540))
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color=white:s={w}x{h}:d=1",
         "-vf", f"format=gray,geq=lum='255-70*pow(sin(PI*Y/{period}),8)'",
         "-frames:v", "1", str(out)],
        check=True,
    )


def build_graph(w: int, h: int, glow: float, frame: bool, dream: bool) -> tuple[str, str]:
    s = META["scale"]
    sw, sh = round(w * s / 2) * 2, round(h * s / 2) * 2
    sigma = META["sigma"] * h
    lut = str(LUT).replace("\\", "/").replace(":", "\\:")
    g = (
        f"[0:v]scale={sw}:{sh},pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,"
        f"format=gbrp,lut3d=file='{lut}':interp=tetrahedral,split[base][hi];"
        f"[hi]gblur=sigma={sigma:.1f}[blur];"
        f"[base][blur]blend=all_mode=screen:all_opacity={glow}[graded]"
    )
    last = "[graded]"
    if dream:
        px = max(1, round(w / 540))  # fringe size scales with resolution
        g += (
            # halation: wide warm bloom
            f";{last}split[d0][d1];"
            f"[d1]gblur=sigma={h * 0.04:.1f},colorchannelmixer=rr=1:gg=0.75:bb=0.45[hal];"
            f"[d0][hal]blend=all_mode=screen:all_opacity=0.28,"
            # RGB fringing + grain
            f"rgbashift=rh=-{px}:bh={px},noise=alls=9:allf=t+u[tex];"
            # scanlines (input 2)
            f"[2:v]format=gbrp,scale={w}:{h}[scan];"
            f"[tex][scan]blend=all_mode=multiply:all_opacity=0.5[dreamy]"
        )
        last = "[dreamy]"
    if frame:
        g += (
            f";[1:v]format=gbrp,scale={w}:{h}:flags=bicubic[mask];"
            f"color=black:s={w}x{h},format=gbrp[black];"
            f"[black]{last}[mask]maskedmerge=planes=7[framed]"
        )
        last = "[framed]"
    return g, last


def apply(inp: Path, out: Path, glow: float, frame: bool, dream: bool) -> None:
    w, h = probe_size(inp)
    is_image = inp.suffix.lower() in IMAGE_EXTS
    loop = [] if is_image else ["-loop", "1"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(inp)]
        # input 1 is always the mask (unused when --no-frame), input 2 the scanlines
        cmd += loop + ["-i", str(MASK)]
        if dream:
            scan = Path(tmp) / "scan.png"
            make_scanlines(w, h, scan)
            cmd += loop + ["-i", str(scan)]
        graph, last = build_graph(w, h, glow, frame, dream)
        cmd += ["-filter_complex", graph, "-map", last]
        if is_image:
            cmd += ["-frames:v", "1", "-update", "1"]
        else:
            cmd += [
                "-map", "0:a?", "-shortest",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
            ]
        subprocess.run(cmd + [str(out)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="PREQUEL look: fitted LUT + glow + rounded black frame")
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--glow", type=float, default=META["glow"],
                    help=f"Haze/bloom strength 0..1 (default {META['glow']}, what the LUT was fitted with)")
    ap.add_argument("--dream", action="store_true", help="Add dreamy old-TV texture: halation, RGB fringe, scanlines, grain")
    ap.add_argument("--no-frame", action="store_true", help="Skip the rounded black frame")
    args = ap.parse_args()
    apply(args.input, args.output, args.glow, not args.no_frame, args.dream)
    print(f"done: {args.output}")


if __name__ == "__main__":
    main()
