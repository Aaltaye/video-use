"""Recreate the user's PREQUEL / Life Mastery look on any clip.

Fitted frame-by-frame from a real before/after pair of the user's videos
(`static/looks/prequel.*`):

  1. Place  — output is always 1080x1920. The clip is scaled (never stretched) to
              cover a centred 1050x1400 box, exactly like the reference edit.
  2. Colour — 3D LUT fitted from ~60 matched frames (greys -> amber, whites -> cream,
              blacks -> dark brown).
  3. Frame  — rounded "arch window", traced from the reference; black outside.
  4. Glow   — blurred copy screen-blended on top AFTER the frame, so bright areas
              bloom and the glow spills softly over the frame edge.
  5. Text   — curved 1–2 line title along the top of the window + @lifemastri tag
              (see arc_title.py).

`--dream` optionally adds old-TV texture (RGB fringe, scanlines, grain).

Usage:
    python helpers/prequel_look.py clip.mp4 -o out.mp4 --title "The Loneliest Part" "of Waking Up"
    python helpers/prequel_look.py clip.mp4 -o out.mp4 --title "One Line Title" --brand ""
    python helpers/prequel_look.py raw_folder/ -o done_folder/
        # batch: a title for clip.mp4 is read from clip.txt (1–2 lines) if it exists
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Assets sit next to this script in the portable preset folder, or in static/looks/ in the repo.
_HERE = Path(__file__).resolve().parent
LOOK_DIR = _HERE if (_HERE / "prequel.cube").exists() else _HERE.parent / "static" / "looks"
LUT = LOOK_DIR / "prequel.cube"
MASK = LOOK_DIR / "prequel_mask.png"
META = json.loads((LOOK_DIR / "prequel.json").read_text())

sys.path.insert(0, str(_HERE))
import arc_title  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
DEFAULT_BRAND = "@lifemastri"


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
    period = max(3, round(h / 540))
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color=white:s={w}x{h}:d=1",
         "-vf", f"format=gray,geq=lum='255-70*pow(sin(PI*Y/{period}),8)'",
         "-frames:v", "1", str(out)],
        check=True,
    )


def build_graph(src_w: int, src_h: int, glow: float, frame: bool, dream: bool, has_text: bool) -> tuple[str, str]:
    W, H = META["canvas"]
    bw, bh = META["box"]
    s = max(bw / src_w, bh / src_h)                    # cover the box, keep aspect
    sw, sh = round(src_w * s / 2) * 2, round(src_h * s / 2) * 2
    lut = str(LUT).replace("\\", "/").replace(":", "\\:")
    g = (
        f"[0:v]scale={sw}:{sh},"
        f"pad=w='max(iw,{W})':h='max(ih,{H})':x='(ow-iw)/2':y='(oh-ih)/2':color=black,"
        f"crop={W}:{H},setsar=1,"
        f"format=gbrp,lut3d=file='{lut}':interp=tetrahedral[graded]"
    )
    last = "[graded]"
    if frame:
        g += (
            f";{last}split[gm0][gm1];[gm1]lutrgb=r=0:g=0:b=0[black];"   # black copy keeps the clip's timing
            f"[1:v]format=gbrp,scale={W}:{H}:flags=bicubic[mask];"
            f"[black][gm0][mask]maskedmerge=planes=7[framed]"
        )
        last = "[framed]"
    # glow after the frame so it spills over the edge like the reference; every layer
    # blurs the same framed image and is screen-blended on (the model the LUT was fitted with)
    layers = [l for l in META["glow"] if l["opacity"] * glow > 0]
    if layers:
        n = len(layers)
        g += f";{last}split={n + 1}[gb]" + "".join(f"[gs{i}]" for i in range(n))
        prev = "[gb]"
        for i, layer in enumerate(layers):
            op = min(layer["opacity"] * glow, 1)
            g += (
                f";[gs{i}]gblur=sigma={layer['sigma']:.1f}[gl{i}];"
                f"{prev}[gl{i}]blend=all_mode=screen:all_opacity={op:.3f}[glow{i}]"
            )
            prev = f"[glow{i}]"
        last = prev
    if dream:
        px = max(1, round(W / 540))
        g += (
            f";{last}rgbashift=rh=-{px}:bh={px},noise=alls=9:allf=t+u[tex];"
            f"[2:v]format=gbrp,scale={W}:{H}[scan];"
            f"[tex][scan]blend=all_mode=multiply:all_opacity=0.5[dreamy]"
        )
        last = "[dreamy]"
    if has_text:
        g += f";{last}format=rgba[base];[3:v]format=rgba[txt];[base][txt]overlay=0:0:format=auto[texted]"
        last = "[texted]"
    return g, last


def apply(inp: Path, out: Path, glow: float = 1.0, frame: bool = True, dream: bool = False,
          title: list[str] | None = None, brand: str | None = DEFAULT_BRAND) -> None:
    src_w, src_h = probe_size(inp)
    W, H = META["canvas"]
    is_image = inp.suffix.lower() in IMAGE_EXTS
    loop = [] if is_image else ["-loop", "1"]
    title = [t for t in (title or []) if t.strip()]
    has_text = bool(title or brand)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        scan, txt = Path(tmp) / "scan.png", Path(tmp) / "text.png"
        make_scanlines(W, H, scan) if dream else scan.write_bytes(MASK.read_bytes())
        if has_text:
            arc_title.render(title, brand, (W, H)).save(txt)
        else:
            txt.write_bytes(MASK.read_bytes())
        # inputs: 0 clip, 1 mask, 2 scanlines, 3 text overlay (unused ones are harmless)
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(inp)]
        for extra in (MASK, scan, txt):
            cmd += loop + ["-i", str(extra)]
        graph, last = build_graph(src_w, src_h, glow, frame, dream, has_text)
        cmd += ["-filter_complex", graph, "-map", last]
        if is_image:
            cmd += ["-frames:v", "1", "-update", "1"]
        else:
            cmd += [
                "-map", "0:a?", "-shortest",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            ]
        subprocess.run(cmd + [str(out)], check=True)


def title_for(path: Path) -> list[str]:
    side = path.with_suffix(".txt")
    return [l.strip() for l in side.read_text(encoding="utf-8").splitlines() if l.strip()][:2] if side.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser(description="PREQUEL / Life Mastery look: colour, glow, frame, curved title, brand")
    ap.add_argument("input", type=Path, help="Video/image file, or a folder of them")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output file, or folder when input is a folder")
    ap.add_argument("--title", nargs="+", metavar="LINE", help="Title: one or two lines (quote each line)")
    ap.add_argument("--brand", default=DEFAULT_BRAND, help=f"Brand tag (default {DEFAULT_BRAND}); '' to omit")
    ap.add_argument("--glow", type=float, default=1.0, help="Glow strength multiplier (1.0 = matched to reference)")
    ap.add_argument("--dream", action="store_true", help="Add old-TV texture: RGB fringe, scanlines, grain")
    ap.add_argument("--no-frame", action="store_true", help="Skip the rounded black frame")
    args = ap.parse_args()
    if args.title and len(args.title) > 2:
        ap.error("--title takes one or two lines")
    brand = args.brand or None
    opts = dict(glow=args.glow, frame=not args.no_frame, dream=args.dream, brand=brand)
    if args.input.is_dir():
        files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS)
        if not files:
            raise SystemExit(f"no videos or images found in {args.input}")
        for i, f in enumerate(files, 1):
            out = args.output / f"{f.stem}_prequel{f.suffix if f.suffix.lower() in IMAGE_EXTS else '.mp4'}"
            t = title_for(f) or args.title
            print(f"[{i}/{len(files)}] {f.name} -> {out.name}" + (f"  title: {' / '.join(t)}" if t else ""))
            apply(f, out, title=t, **opts)
        print(f"done: {len(files)} file(s) in {args.output}")
        return
    apply(args.input, args.output, title=args.title, **opts)
    print(f"done: {args.output}")


if __name__ == "__main__":
    main()
