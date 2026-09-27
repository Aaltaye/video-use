"""Render a curved (arched) title + small brand tag as a transparent PNG overlay.

The title's lines sit on concentric arcs that follow the top of the rounded frame,
like the user's Life Mastery videos: white serif (Playfair Display) with a soft dark
shadow. The brand tag (e.g. @lifemastri) is a small orange serif line near the bottom.

    python helpers/arc_title.py "The Loneliest Part" "of Waking Up" -o title.png
    python helpers/arc_title.py "One Line Title" --brand "@lifemastri" -o title.png

All geometry is in a 1080x1920 canvas (scaled for other sizes with --size).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Fonts sit in fonts/ next to this script in the portable preset folder, or static/fonts/ in the repo.
_HERE = Path(__file__).resolve().parent
FONT_DIR = _HERE / "fonts" if (_HERE / "fonts").is_dir() else _HERE.parent / "static" / "fonts"

# Measured from the user's reference video (1080x1920).
TITLE = dict(
    font="playfair-display.woff", size=89.6, tracking=-0.046,  # extra space per glyph, fraction of size
    line2_size=0.847,                                          # line 2 font size relative to line 1
    hscale=1.108,                                              # horizontal stretch of each glyph
    cx=541.3, cy=1098, radius=690, line_gap=88.2,              # baseline radius of line 1; line 2 sits line_gap inside
    max_span_deg=66,                                        # shrink font if a line would wrap further round the arc
    fill=(255, 255, 255), shadow=(0, 0, 0), shadow_blur=6, shadow_alpha=170, shadow_offset=(0, 4),
)
BRAND = dict(font="old-standard-tt.woff", size=48, y=1473, fill=(166, 92, 32), alpha=235)


def _font(name: str, size: float) -> ImageFont.FreeTypeFont:
    path = Path(name) if Path(name).exists() else FONT_DIR / name
    return ImageFont.truetype(str(path), max(1, round(size)))


def draw_arc_line(canvas: Image.Image, text: str, font: ImageFont.FreeTypeFont, cx: float, cy: float,
                  radius: float, tracking_px: float, fill, hscale: float = 1.0) -> None:
    """Draw `text` with its baseline on a circle (centre cx, cy), centred at the top."""
    import math
    # cumulative advances (includes kerning) + tracking
    xs = [font.getlength(text[:i]) * hscale + i * tracking_px for i in range(len(text) + 1)]
    total = xs[-1] - tracking_px
    ascent, descent = font.getmetrics()
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        adv = xs[i + 1] - xs[i] - tracking_px
        mid = xs[i] + adv / 2 - total / 2           # arc-length position of the glyph centre
        theta = mid / radius                          # radians, 0 = top of circle
        pad = int(font.size * 0.6)
        raw_adv = font.getlength(ch)
        g = Image.new("L", (int(raw_adv + 2 * pad), ascent + descent + 2 * pad), 0)
        ImageDraw.Draw(g).text((pad, pad), ch, font=font, fill=255)
        if hscale != 1.0:
            g = g.resize((max(1, round(g.width * hscale)), g.height), Image.BICUBIC)
        pad_x = pad * hscale
        # glyph image: baseline centre at (pad + adv/2, pad + ascent)
        bx, by = pad_x + raw_adv * hscale / 2, pad + ascent
        rot = g.rotate(-math.degrees(theta), resample=Image.BICUBIC, expand=True, center=(bx, by))
        # rotate() with expand keeps the rotation centre at the new image centre offset:
        # compute where (bx, by) lands in the expanded image
        w0, h0 = g.size
        w1, h1 = rot.size
        ox, oy = bx - w0 / 2, by - h0 / 2
        c, s = math.cos(theta), math.sin(theta)
        nx, ny = ox * c - oy * s, ox * s + oy * c
        px, py = w1 / 2 + nx, h1 / 2 + ny
        tx = cx + radius * math.sin(theta)
        ty = cy - radius * math.cos(theta)
        layer = Image.new("RGBA", rot.size, fill + (0,))
        layer.putalpha(rot)
        canvas.alpha_composite(layer, (round(tx - px), round(ty - py)))


def render(lines: list[str], brand: str | None, size=(1080, 1920), title_scale: float = 1.0) -> Image.Image:
    import math
    W, H = size
    k = W / 1080
    t = TITLE
    text_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    lines = [l for l in lines if l.strip()]
    for i, line in enumerate(lines):
        fsize = t["size"] * k * title_scale * (t["line2_size"] if i else 1.0)
        # one-line titles sit half a line lower so they're centred in the arch
        offset = i if len(lines) > 1 else 0.5
        radius = (t["radius"] - offset * t["line_gap"] * title_scale) * k
        while True:  # keep long lines inside the arch
            font = _font(t["font"], fsize)
            track = t["tracking"] * fsize
            span = (font.getlength(line) * t["hscale"] + track * (len(line) - 1)) / radius
            if math.degrees(span) <= t["max_span_deg"] or fsize < 30 * k:
                break
            fsize *= 0.96
        draw_arc_line(text_layer, line, font, t["cx"] * k, t["cy"] * k, radius, track, t["fill"], t["hscale"])

    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    # soft drop shadow from the text alpha
    a = text_layer.getchannel("A").filter(ImageFilter.GaussianBlur(t["shadow_blur"] * k))
    a = a.point(lambda v: v * t["shadow_alpha"] // 255)
    shadow = Image.new("RGBA", (W, H), t["shadow"] + (0,))
    shadow.putalpha(a)
    out.alpha_composite(shadow, (round(t["shadow_offset"][0] * k), round(t["shadow_offset"][1] * k)))
    out.alpha_composite(text_layer)

    if brand:
        b = BRAND
        font = _font(b["font"], b["size"] * k)
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.text((W / 2, b["y"] * k), brand, font=font, fill=b["fill"] + (b["alpha"],), anchor="mm")
        out.alpha_composite(layer)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Curved title + brand overlay PNG")
    ap.add_argument("lines", nargs="*", help="Title line(s): one or two")
    ap.add_argument("--brand", default="@lifemastri", help="Brand tag; pass '' to omit")
    ap.add_argument("--size", default="1080x1920")
    ap.add_argument("--title-scale", type=float, default=1.0)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()
    w, h = map(int, args.size.lower().split("x"))
    render(args.lines, args.brand or None, (w, h), args.title_scale).save(args.output)
    print(f"done: {args.output}")


if __name__ == "__main__":
    main()
