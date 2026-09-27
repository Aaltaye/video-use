"""Recreate the user's PREQUEL look: warm orange vintage grade + haze/glow + rounded black frame.

Reverse-engineered from a before/after pair (vertical 9:16 clip). Three stages:

  1. Grade  — crushed blacks, amber/orange midtones, blue pulled from highlights
              (shadows keep some blue so night windows still read), +saturation.
  2. Glow   — blurred copy screen-blended over the grade: bloom on highlights
              plus a light haze that lifts the midtones.
  3. Frame  — everything outside a rounded "arch window" is solid black.
              The window spans ~2%–98% of width and ~22%–78% of height.

Works on videos and still images (use a stills first to dial it in).

Usage:
    python helpers/prequel_look.py <input> -o <output>
    python helpers/prequel_look.py in.mp4 -o out.mp4 --glow 0.4 --no-frame
    python helpers/prequel_look.py --print-grade          # grade chain only, for grade.py --filter
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Stage 1 — colour. eq adds the punch (runs in YUV, so it goes first), per-channel
# curves warm the whole image, then selectivecolor pushes whites/neutrals/skin to
# amber while keeping blues blue (night windows stay blue, like the original).
GRADE = (
    "eq=saturation=1.1:contrast=1.06,"
    "curves="
    "r='0/0 0.12/0.07 0.45/0.62 0.8/0.96 1/1':"
    "g='0/0 0.12/0.06 0.45/0.48 0.8/0.86 1/0.97':"
    "b='0/0.02 0.3/0.29 0.6/0.52 1/0.82',"
    "selectivecolor=correction_method=absolute:"
    "whites='0 0.03 0.3 0':"
    "neutrals='-0.05 0.04 0.3 0':"
    "yellows='-0.1 0.08 0.25 0':"
    "reds='0 0.08 0.2 0':"
    "blues='0.15 0 -0.2 0'"
)

# Stage 3 — frame geometry, as fractions of the output frame.
FRAME_LEFT, FRAME_RIGHT = 0.022, 0.978
FRAME_TOP, FRAME_BOTTOM = 0.222, 0.778
FRAME_RX, FRAME_RY = 0.34, 0.14  # elliptical corner radii, fraction of frame width / height


def probe_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path)],
        capture_output=True, text=True,
    ).stderr
    for line in out.splitlines():
        if "Video:" in line:
            for tok in line.replace(",", " ").split():
                w, _, h = tok.partition("x")
                if w.isdigit() and h.isdigit():
                    return int(w), int(h)
    raise SystemExit(f"could not read video size of {path}")


def make_mask(w: int, h: int, out: Path) -> None:
    """Write a grayscale PNG: white inside the rounded window, black outside, 2px feather."""
    cx = w / 2
    cy = h * (FRAME_TOP + FRAME_BOTTOM) / 2
    rx, ry = w * FRAME_RX, h * FRAME_RY
    a = w * (FRAME_RIGHT - FRAME_LEFT) / 2 - rx  # half-width of the straight part
    b = h * (FRAME_BOTTOM - FRAME_TOP) / 2 - ry  # half-height of the straight part
    # Normalised distance to the corner ellipse, rescaled to ~pixels for the feather.
    dist = f"(hypot(max(abs(X-{cx})-{a},0)/{rx},max(abs(Y-{cy})-{b},0)/{ry})-1)*{min(rx, ry)}"
    expr = f"255*clip(0.5-{dist}/2,0,1)"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color=black:s={w}x{h}:d=1",
         "-vf", f"format=gray,geq=lum='{expr}'", "-frames:v", "1", str(out)],
        check=True,
    )


def build_graph(w: int, h: int, glow: float, frame: bool) -> str:
    sigma = max(w, h) * 0.018
    graph = (
        f"[0:v]{GRADE},format=gbrp,split[base][hi];"
        f"[hi]gblur=sigma={sigma:.1f}[blur];"
        f"[base][blur]blend=all_mode=screen:all_opacity={glow}[graded]"
    )
    if frame:
        graph += (
            f";[1:v]format=gbrp,scale={w}:{h}[mask];"
            f"color=black:s={w}x{h},format=gbrp[black];"
            f"[black][graded][mask]maskedmerge=planes=7[framed]"
        )
    return graph


def apply(inp: Path, out: Path, glow: float, frame: bool) -> None:
    w, h = probe_size(inp)
    is_image = inp.suffix.lower() in IMAGE_EXTS
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(inp)]
        if frame:
            mask = Path(tmp) / "mask.png"
            make_mask(w, h, mask)
            cmd += ([] if is_image else ["-loop", "1"]) + ["-i", str(mask)]
        graph = build_graph(w, h, glow, frame)
        last = "[framed]" if frame else "[graded]"
        cmd += ["-filter_complex", graph, "-map", last]
        if is_image:
            cmd += ["-frames:v", "1"]
        else:
            cmd += [
                "-map", "0:a?", "-shortest",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
            ]
        subprocess.run(cmd + [str(out)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="PREQUEL-style warm glow + rounded black frame")
    ap.add_argument("input", type=Path, nargs="?")
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--glow", type=float, default=0.75, help="Haze/bloom strength 0..1 (default 0.75)")
    ap.add_argument("--no-frame", action="store_true", help="Skip the rounded black frame")
    ap.add_argument("--print-grade", action="store_true", help="Print the colour-only filter chain and exit")
    args = ap.parse_args()

    if args.print_grade:
        print(GRADE)
        return
    if not args.input or not args.output:
        ap.error("input and -o/--output are required")
    apply(args.input, args.output, args.glow, not args.no_frame)
    print(f"done: {args.output}")


if __name__ == "__main__":
    main()
