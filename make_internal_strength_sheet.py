from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/whc/零样本风格迁移/ZSTAR_spatial020_minimal")
WORK = ROOT / "workdir/ourmethod_internal_strength_test"
OUT = WORK / "summary"

STYLES = {
    "demo_1": {
        "stem": "demo_1",
        "base": ROOT / "workdir/multimodel_compare_demo_1_internal_attention",
    },
    "Starry": {
        "stem": "Starry",
        "base": ROOT / "workdir/multimodel_compare_Starry_internal_attention",
    },
}


def font(size):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def fit_square(path, size):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.Resampling.LANCZOS)


def make_style_sheet(style_name, info):
    stem = info["stem"]
    base = info["base"]
    methods = [
        (
            "ZSTAR",
            "direct",
            base / f"outputs/ZSTAR_direct/demo_0_{stem}_reconstructed.png",
        ),
        (
            "Ours-old",
            "internal + spatial 0.20",
            base / f"outputs/Ours_spatial020/demo_0_{stem}_reconstructed.png",
        ),
        (
            "A",
            "scale 3/4, protect strong",
            WORK / style_name / f"A_scale3_4/demo_0_{stem}_reconstructed.png",
        ),
        (
            "B",
            "scale 4/5, spatial light",
            WORK / style_name / f"B_scale4_5_light_spatial/demo_0_{stem}_reconstructed.png",
        ),
        (
            "C",
            "scale 6/8, masks softer",
            WORK / style_name / f"C_scale6_8_softmask_light_spatial/demo_0_{stem}_reconstructed.png",
        ),
    ]
    missing = [(name, path) for name, _, path in methods if not path.exists()]
    if missing:
        detail = "\n".join(f"{name}: {path}" for name, path in missing)
        raise FileNotFoundError(f"Missing images for {style_name}:\n{detail}")

    cell = 300
    label_h = 62
    margin = 18
    gap = 12
    width = margin * 2 + len(methods) * cell + (len(methods) - 1) * gap
    height = margin * 2 + label_h + cell
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(22)
    small_font = font(15)

    for idx, (name, subtitle, path) in enumerate(methods):
        x = margin + idx * (cell + gap)
        y = margin
        draw.text((x, y), name, fill=(20, 20, 20), font=title_font)
        draw.text((x, y + 29), subtitle, fill=(80, 80, 80), font=small_font)
        img = fit_square(path, cell)
        canvas.paste(img, (x, margin + label_h))
        draw.rectangle(
            (x, margin + label_h, x + cell - 1, margin + label_h + cell - 1),
            outline=(210, 210, 210),
        )

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"internal_strength_{stem}.png"
    canvas.save(out_path)
    return out_path


def make_mask_sheet(style_name, info):
    stem = info["stem"]
    methods = [
        (
            "A masks",
            WORK / style_name / "A_scale3_4/internal_attention_masks",
        ),
        (
            "B masks",
            WORK / style_name / "B_scale4_5_light_spatial/internal_attention_masks",
        ),
        (
            "C masks",
            WORK / style_name / "C_scale6_8_softmask_light_spatial/internal_attention_masks",
        ),
    ]
    concepts = ["eyes", "nose", "mouth", "lips"]
    paths = []
    for name, mask_dir in methods:
        for concept in concepts:
            paths.append((name, concept, mask_dir / f"demo_0_{concept}_internal_attn_mask.png"))
    missing = [(name, concept, path) for name, concept, path in paths if not path.exists()]
    if missing:
        detail = "\n".join(f"{name}/{concept}: {path}" for name, concept, path in missing)
        raise FileNotFoundError(f"Missing masks for {style_name}:\n{detail}")

    cell = 180
    label_h = 48
    margin = 16
    gap = 10
    cols = len(concepts)
    rows = len(methods)
    width = margin * 2 + cols * cell + (cols - 1) * gap
    height = margin * 2 + rows * (cell + label_h) + (rows - 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(18)
    small_font = font(14)

    for row, (name, mask_dir) in enumerate(methods):
        for col, concept in enumerate(concepts):
            x = margin + col * (cell + gap)
            y = margin + row * (cell + label_h + gap)
            draw.text((x, y), name, fill=(20, 20, 20), font=title_font)
            draw.text((x, y + 23), concept, fill=(80, 80, 80), font=small_font)
            path = mask_dir / f"demo_0_{concept}_internal_attn_mask.png"
            img = fit_square(path, cell)
            canvas.paste(img, (x, y + label_h))
            draw.rectangle((x, y + label_h, x + cell - 1, y + label_h + cell - 1), outline=(210, 210, 210))

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"internal_masks_{stem}.png"
    canvas.save(out_path)
    return out_path


def write_report(sheets, mask_sheets):
    lines = [
        "# Ours Internal-Attention Strength Test",
        "",
        "## Purpose",
        "",
        "The previous internal-attention version preserved facial regions but made the stylization too weak. This test isolates three possible suppressors: base ZSTAR scale, spatial-adaptive suppression, and semantic-mask protection strength.",
        "",
        "## Variants",
        "",
        "- A: content/style scales 3/4, spatial strength 0.20, strong internal masks.",
        "- B: content/style scales 4/5, spatial strength 0.05, strong internal masks.",
        "- C: content/style scales 6/8, spatial strength 0.02, softer internal masks.",
        "",
        "## Visual Outputs",
        "",
    ]
    for style, path in sheets.items():
        lines.append(f"- {style} style sheet: `{path}`")
    for style, path in mask_sheets.items():
        lines.append(f"- {style} internal masks: `{path}`")
    lines += [
        "",
        "## Reading",
        "",
        "If A remains weak, the original 0.20 spatial gate is suppressing style too aggressively. If B gains style while keeping the face stable, spatial suppression is the main problem. If C is much stronger but degrades facial parts, the semantic masks were useful but should not be removed entirely.",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "internal_strength_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    sheets = {}
    mask_sheets = {}
    for style_name, info in STYLES.items():
        sheets[style_name] = make_style_sheet(style_name, info)
        mask_sheets[style_name] = make_mask_sheet(style_name, info)
    write_report(sheets, mask_sheets)
    for path in [*sheets.values(), *mask_sheets.values(), OUT / "internal_strength_report.md"]:
        print(path)


if __name__ == "__main__":
    main()
