from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("workdir/ablation_v2")
RUNS = ROOT / "runs"
OUT = ROOT / "summary"
STYLES = ["demo_1", "wave"]

PORTRAIT_VARIANTS = [
    ("P0", "Direct ZSTAR", "P0_direct_zstar"),
    ("P1", "Internal attention", "P1_internal_attention"),
    ("P2", "Spatial 0.20 only", "P2_spatial020_only"),
    ("P3", "Internal + spatial 0.20", "P3_internal_spatial020"),
    ("P4", "MediaPipe masks", "P4_mediapipe_masks"),
    ("P5", "MediaPipe + spatial 0.20", "P5_mediapipe_spatial020"),
]

LANDSCAPE_VARIANTS = [
    ("L0", "Direct ZSTAR", "L0_direct_zstar"),
    ("L1", "Spatial 0.20 only", "L1_spatial020_only"),
]


def font(size):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def load_img(path, size=(240, 240)):
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    tile = Image.new("RGB", size, (245, 245, 245))
    tile.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return tile


def make_stylized_sheet(group, content_name, variants, out_name):
    OUT.mkdir(parents=True, exist_ok=True)
    title_font = font(16)
    small_font = font(12)
    cell_w, cell_h = 230, 230
    row_w = 112
    label_h = 42
    margin = 10
    width = row_w + len(variants) * (cell_w + margin) + margin
    height = label_h + len(STYLES) * (cell_h + label_h + margin) + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for col, (tag, name, _) in enumerate(variants):
        x = row_w + margin + col * (cell_w + margin)
        draw.text((x + 3, 5), f"{tag}\n{name}", fill=(20, 20, 20), font=small_font)
    for row, style in enumerate(STYLES):
        y = label_h + margin + row * (cell_h + label_h + margin)
        draw.text((12, y + 8), style, fill=(20, 20, 20), font=title_font)
        for col, (_, _, dirname) in enumerate(variants):
            path = RUNS / group / dirname / f"{content_name}_{style}_reconstructed.png"
            if not path.exists():
                continue
            x = row_w + margin + col * (cell_w + margin)
            canvas.paste(load_img(path, (cell_w, cell_h)), (x, y + label_h))
            draw.rectangle((x, y + label_h, x + cell_w - 1, y + label_h + cell_h - 1), outline=(210, 210, 210))
    out = OUT / out_name
    canvas.save(out)
    return out


def crop_box_from_mask(mask_path, pad=55):
    mask = np.load(mask_path).astype(np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    ys, xs = np.where(mask > 0.1)
    if len(xs) == 0:
        return (130, 120, 430, 350)
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(mask.shape[1], int(xs.max()) + pad),
        min(mask.shape[0], int(ys.max()) + pad),
    )


def crop_tile(path, box, size=(230, 160)):
    img = Image.open(path).convert("RGB").resize((560, 560), Image.Resampling.LANCZOS)
    crop = img.crop(box)
    crop.thumbnail(size, Image.Resampling.LANCZOS)
    tile = Image.new("RGB", size, (245, 245, 245))
    tile.paste(crop, ((size[0] - crop.width) // 2, (size[1] - crop.height) // 2))
    return tile


def make_portrait_eye_sheet():
    OUT.mkdir(parents=True, exist_ok=True)
    title_font = font(16)
    small_font = font(12)
    box = crop_box_from_mask(ROOT / "inputs/portrait_content/demo_0_eyes_mask.npy", pad=55)
    cell_w, cell_h = 230, 160
    row_w = 112
    label_h = 42
    margin = 10
    width = row_w + len(PORTRAIT_VARIANTS) * (cell_w + margin) + margin
    height = label_h + len(STYLES) * (cell_h + label_h + margin) + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for col, (tag, name, _) in enumerate(PORTRAIT_VARIANTS):
        x = row_w + margin + col * (cell_w + margin)
        draw.text((x + 3, 5), f"{tag}\n{name}", fill=(20, 20, 20), font=small_font)
    for row, style in enumerate(STYLES):
        y = label_h + margin + row * (cell_h + label_h + margin)
        draw.text((12, y + 8), style, fill=(20, 20, 20), font=title_font)
        for col, (_, _, dirname) in enumerate(PORTRAIT_VARIANTS):
            path = RUNS / "portrait" / dirname / f"demo_0_{style}_reconstructed.png"
            if not path.exists():
                continue
            x = row_w + margin + col * (cell_w + margin)
            canvas.paste(crop_tile(path, box, (cell_w, cell_h)), (x, y + label_h))
            draw.rectangle((x, y + label_h, x + cell_w - 1, y + label_h + cell_h - 1), outline=(210, 210, 210))
    out = OUT / "portrait_eye_crop_stylized_only.png"
    canvas.save(out)
    return out


def make_spatial_risk_sheet(group, variants, out_name):
    cells = []
    for tag, name, dirname in variants:
        map_dir = RUNS / group / dirname / "spatial_adaptive_maps"
        if not map_dir.exists():
            continue
        for style in STYLES:
            paths = sorted(map_dir.glob(f"*_{style}_main_step5_layer20*.png"))
            if paths:
                cells.append((f"{tag} {name}\n{style}", paths[0]))
    if not cells:
        return None
    small_font = font(12)
    cell = 180
    label_h = 40
    margin = 10
    cols = min(4, len(cells))
    rows = int(np.ceil(len(cells) / cols))
    canvas = Image.new("RGB", (cols * (cell + margin) + margin, rows * (cell + label_h + margin) + margin), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, path) in enumerate(cells):
        row, col = divmod(i, cols)
        x = margin + col * (cell + margin)
        y = margin + row * (cell + label_h + margin)
        draw.text((x + 2, y), label, fill=(20, 20, 20), font=small_font)
        canvas.paste(load_img(path, (cell, cell)), (x, y + label_h))
        draw.rectangle((x, y + label_h, x + cell - 1, y + label_h + cell - 1), outline=(210, 210, 210))
    out = OUT / out_name
    canvas.save(out)
    return out


def make_attention_sheet(group, content_name, variants, out_name):
    cells = []
    for tag, name, dirname in variants:
        attn_dir = RUNS / group / dirname / "attn"
        if not attn_dir.exists():
            continue
        for style in STYLES:
            paths = sorted(attn_dir.glob(f"{content_name}_{style}*step5_layer20.png"))
            paths = [p for p in paths if "focus" not in p.name]
            if paths:
                cells.append((f"{tag} {name}\n{style}", paths[0]))
    if not cells:
        return None
    small_font = font(12)
    cell = 180
    label_h = 40
    margin = 10
    cols = min(4, len(cells))
    rows = int(np.ceil(len(cells) / cols))
    canvas = Image.new("RGB", (cols * (cell + margin) + margin, rows * (cell + label_h + margin) + margin), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, path) in enumerate(cells):
        row, col = divmod(i, cols)
        x = margin + col * (cell + margin)
        y = margin + row * (cell + label_h + margin)
        draw.text((x + 2, y), label, fill=(20, 20, 20), font=small_font)
        canvas.paste(load_img(path, (cell, cell)), (x, y + label_h))
        draw.rectangle((x, y + label_h, x + cell - 1, y + label_h + cell - 1), outline=(210, 210, 210))
    out = OUT / out_name
    canvas.save(out)
    return out


def write_report(paths):
    lines = [
        "# Ablation V2: Stylized-Only Comparison with Portrait and Landscape Content",
        "",
        "## Purpose",
        "",
        "This run addresses two points: first, the comparison uses only the injected/stylized output, not content-style-result panels. Second, it adds a landscape content image where face-part semantic protection should not be useful; this isolates the role of spatial adaptive injection outside human faces.",
        "",
        "## Styles",
        "",
        "```text",
        "/home/whc/零样本风格迁移/official_repos/ZSTAR/style_images/demo_1.jpg",
        "/home/whc/零样本风格迁移/official_repos/ZSTAR/style_images/wave.jpg",
        "```",
        "",
        "## Variants",
        "",
        "Portrait variants:",
        "",
        "```text",
        "P0 Direct ZSTAR: no internal attention, no spatial adaptive scaling, no external masks.",
        "P1 Internal attention: internal cross-attention semantic masks + self-attention refinement.",
        "P2 Spatial 0.20 only: no face semantic masks, only structure-aware spatial scaling.",
        "P3 Internal + spatial 0.20: recommended self-contained method.",
        "P4 MediaPipe masks: external sidecar face-part masks only.",
        "P5 MediaPipe + spatial 0.20: external masks plus spatial scaling.",
        "```",
        "",
        "Landscape variants:",
        "",
        "```text",
        "L0 Direct ZSTAR",
        "L1 Spatial 0.20 only",
        "```",
        "",
        "For landscape images, eye/nose/mouth masks are not semantically applicable. The relevant comparison is therefore direct ZSTAR versus spatial-only injection.",
        "",
        "## Outputs",
        "",
    ]
    for key, path in paths.items():
        if path:
            lines.append(f"- {key}: `{path}`")
    lines.extend([
        "",
        "## Analysis",
        "",
        "1. **Direct ZSTAR baseline.** P0/L0 are the clean no-module baselines. They show the behavior of ZSTAR when no semantic protection, no external masks, and no spatial adaptive scaling are used.",
        "",
        "2. **Internal attention on portrait.** P1 tests whether Stable Diffusion's own cross-attention can provide face-part protection. The portrait stylized-only and eye-crop sheets should be used to inspect whether eyes and mouth are more stable than P0.",
        "",
        "3. **Spatial-only effect.** P2 and L1 isolate spatial adaptive scaling. This is especially important for the landscape case: because there are no facial semantic masks, any improvement must come from structure-aware spatial injection rather than face-part protection.",
        "",
        "4. **Self-contained portrait setting.** P3 combines internal semantic protection with spatial adaptive scaling. It is the main self-contained candidate to inspect when balancing face-part preservation against visible style strength.",
        "",
        "5. **External mask comparison.** P4/P5 use the sidecar face-part masks as MediaPipe-style external masks. They are useful as a non-self-contained comparison, but not the main self-contained contribution.",
        "",
        "## Caveat",
        "",
        "During the landscape run, null-text optimization produced NaNs once and the code fell back to default unconditional embeddings. The images were still generated, but landscape conclusions should be treated as qualitative visual evidence rather than a finalized quantitative result.",
    ])
    (OUT / "ablation_v2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "Portrait stylized-only comparison": make_stylized_sheet("portrait", "demo_0", PORTRAIT_VARIANTS, "portrait_stylized_only_comparison.png"),
        "Portrait eye crop stylized-only comparison": make_portrait_eye_sheet(),
        "Landscape stylized-only comparison": make_stylized_sheet("landscape", "chicago", LANDSCAPE_VARIANTS, "landscape_stylized_only_comparison.png"),
        "Portrait attention maps": make_attention_sheet("portrait", "demo_0", PORTRAIT_VARIANTS, "portrait_attention_maps.png"),
        "Landscape attention maps": make_attention_sheet("landscape", "chicago", LANDSCAPE_VARIANTS, "landscape_attention_maps.png"),
        "Portrait spatial risk maps": make_spatial_risk_sheet("portrait", PORTRAIT_VARIANTS, "portrait_spatial_risk_maps.png"),
        "Landscape spatial risk maps": make_spatial_risk_sheet("landscape", LANDSCAPE_VARIANTS, "landscape_spatial_risk_maps.png"),
    }
    write_report(paths)
    print(OUT / "ablation_v2_report.md")


if __name__ == "__main__":
    main()
