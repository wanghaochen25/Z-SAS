from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/whc/零样本风格迁移/ZSTAR_spatial020_minimal")
BASE = ROOT / "workdir/ours_strength_search_candy"
REF = ROOT / "workdir/multimodel_compare_candy_weighted_only_strong"
OUT = BASE / "summary"

ITEMS = [
    ("Content", REF / "inputs/content/demo_0.jpg"),
    ("Style", REF / "inputs/style/candy.jpg"),
    ("ZSTAR", REF / "outputs/ZSTAR_direct/demo_0_candy_reconstructed.png"),
    ("Prev 2.5/3.0", REF / "outputs/Ours_weighted_only/demo_0_candy_reconstructed.png"),
    ("W1 4/5", BASE / "W1_scale4_5/demo_0_candy_reconstructed.png"),
    ("W2 6/8", BASE / "W2_scale6_8/demo_0_candy_reconstructed.png"),
    ("W3 early 4/5", BASE / "W3_early_scale4_5/demo_0_candy_reconstructed.png"),
    ("W4 wide 4/5", BASE / "W4_wide_scale4_5/demo_0_candy_reconstructed.png"),
]


def font(size):
    candidate = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if candidate.exists():
        return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def square(path, size=260):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.LANCZOS)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [(name, path) for name, path in ITEMS if not path.exists()]
    if missing:
        raise FileNotFoundError("\n".join(f"{name}: {path}" for name, path in missing))

    cell = 260
    label_h = 56
    gap = 10
    margin = 16
    width = margin * 2 + len(ITEMS) * cell + (len(ITEMS) - 1) * gap
    height = margin * 2 + label_h + cell
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(18)
    sub_font = font(13)

    subtitles = {
        "Content": "input",
        "Style": "input",
        "ZSTAR": "direct 1.2/1.5",
        "Prev 2.5/3.0": "weighted only",
        "W1 4/5": "scale up",
        "W2 6/8": "max scale",
        "W3 early 4/5": "start step 0",
        "W4 wide 4/5": "layers 16-30",
    }

    for i, (name, path) in enumerate(ITEMS):
        x = margin + i * (cell + gap)
        y = margin
        draw.text((x, y), name, fill=(20, 20, 20), font=title_font)
        draw.text((x, y + 25), subtitles.get(name, ""), fill=(80, 80, 80), font=sub_font)
        canvas.paste(square(path, cell), (x, margin + label_h))
        draw.rectangle((x, margin + label_h, x + cell - 1, margin + label_h + cell - 1), outline=(210, 210, 210))

    out_path = OUT / "ours_candy_strength_search.png"
    canvas.save(out_path)
    report = OUT / "ours_candy_strength_search.md"
    report.write_text(
        "# Ours Candy Strength Search\n\n"
        "All variants disable internal face-part masks and use weighted-only spatial adaptive injection.\n\n"
        "- Prev 2.5/3.0: content_style=2.5, style_content=3.0, start=5, layers=20,22,24,26,28,30.\n"
        "- W1 4/5: content_style=4.0, style_content=5.0, start=5, same layers.\n"
        "- W2 6/8: content_style=6.0, style_content=8.0, start=5, same layers.\n"
        "- W3 early 4/5: content_style=4.0, style_content=5.0, start=0, same layers.\n"
        "- W4 wide 4/5: content_style=4.0, style_content=5.0, start=5, layers=16,18,20,22,24,26,28,30.\n",
        encoding="utf-8",
    )
    print(out_path)


if __name__ == "__main__":
    main()
