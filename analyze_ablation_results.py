from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("workdir/ablations")
CONTENT = Path("workdir/face_part_inputs/content/demo_0.jpg")
MASKS = {
    "eyes": Path("workdir/face_part_inputs/content/demo_0_eyes_mask.npy"),
    "lips": Path("workdir/face_part_inputs/content/demo_0_lips_mask.npy"),
    "nose": Path("workdir/face_part_inputs/content/demo_0_nose_mask.npy"),
}
STYLES = ["candy", "Starry", "Composition-VII"]
VARIANTS = [
    ("A0", "Baseline ZSTAR", "A0_baseline_global_zstar"),
    ("A1", "Internal attention", "A1_internal_attention_only"),
    ("A2", "Internal + spatial 0.20", "A2_internal_spatial020"),
    ("A3", "Internal + spatial 0.35", "A3_internal_spatial035"),
    ("A4", "MediaPipe masks", "A4_mediapipe_masks_only"),
    ("A5", "MediaPipe + spatial 0.20", "A5_mediapipe_spatial020"),
]


def load_font(size):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def load_rgb(path, size=(560, 560)):
    img = Image.open(path).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return img


def np_img(path):
    return np.asarray(load_rgb(path), dtype=np.float32) / 255.0


def normalize(x):
    x = x.astype(np.float32)
    x -= x.min()
    max_value = x.max()
    if max_value > 0:
        x /= max_value
    return x


def load_external_mask():
    masks = []
    for path in MASKS.values():
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.shape != (560, 560):
            img = Image.fromarray((normalize(arr) * 255).astype(np.uint8))
            img = img.resize((560, 560), Image.Resampling.BILINEAR)
            arr = np.asarray(img).astype(np.float32) / 255.0
        masks.append(normalize(arr))
    return np.maximum.reduce(masks).clip(0, 1)


def edge_map(rgb):
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    dx = np.zeros_like(gray)
    dy = np.zeros_like(gray)
    dx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    dy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    return dx + dy


def masked_mean(values, mask):
    denom = float(mask.sum())
    if denom <= 1e-8:
        return float(values.mean())
    return float((values * mask).sum() / denom)


def make_result_sheet():
    font = load_font(17)
    small = load_font(13)
    thumb_w = 255
    label_h = 36
    row_w = 138
    margin = 10
    width = row_w + len(VARIANTS) * (thumb_w + margin) + margin
    height = label_h + len(STYLES) * (thumb_w + label_h + margin) + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for col, (tag, name, _) in enumerate(VARIANTS):
        x = row_w + margin + col * (thumb_w + margin)
        draw.text((x + 4, 6), f"{tag} {name}", fill=(20, 20, 20), font=small)
    for row, style in enumerate(STYLES):
        y = label_h + margin + row * (thumb_w + label_h + margin)
        draw.text((12, y + 8), style, fill=(20, 20, 20), font=font)
        for col, (_, _, dirname) in enumerate(VARIANTS):
            path = ROOT / dirname / f"demo_0_{style}.png"
            if not path.exists():
                continue
            img = Image.open(path).convert("RGB")
            img.thumbnail((thumb_w, thumb_w), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (thumb_w, thumb_w), (245, 245, 245))
            tile.paste(img, ((thumb_w - img.width) // 2, (thumb_w - img.height) // 2))
            x = row_w + margin + col * (thumb_w + margin)
            canvas.paste(tile, (x, y + label_h))
            draw.rectangle((x, y + label_h, x + thumb_w - 1, y + label_h + thumb_w - 1), outline=(210, 210, 210))
    out = ROOT / "ablation_result_contact_sheet.png"
    canvas.save(out)
    return out


def crop_by_box(img, box, out_size=(220, 220)):
    crop = img.crop(box)
    crop.thumbnail(out_size, Image.Resampling.LANCZOS)
    tile = Image.new("RGB", out_size, (245, 245, 245))
    tile.paste(crop, ((out_size[0] - crop.width) // 2, (out_size[1] - crop.height) // 2))
    return tile


def mask_bbox(mask, pad=24):
    ys, xs = np.where(mask > 0.1)
    if len(xs) == 0:
        return (160, 140, 400, 360)
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad)
    return (x0, y0, x1, y1)


def make_crop_sheet(name, box):
    font = load_font(15)
    small = load_font(12)
    cell_w = 220
    cell_h = 220
    label_h = 38
    row_w = 126
    margin = 10
    width = row_w + len(VARIANTS) * (cell_w + margin) + margin
    height = label_h + len(STYLES) * (cell_h + label_h + margin) + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for col, (tag, variant_name, _) in enumerate(VARIANTS):
        x = row_w + margin + col * (cell_w + margin)
        draw.text((x + 3, 6), f"{tag}\n{variant_name}", fill=(20, 20, 20), font=small)
    for row, style in enumerate(STYLES):
        y = label_h + margin + row * (cell_h + label_h + margin)
        draw.text((12, y + 8), style, fill=(20, 20, 20), font=font)
        for col, (_, _, dirname) in enumerate(VARIANTS):
            path = ROOT / dirname / f"demo_0_{style}_reconstructed.png"
            if not path.exists():
                continue
            img = load_rgb(path)
            tile = crop_by_box(img, box, out_size=(cell_w, cell_h))
            x = row_w + margin + col * (cell_w + margin)
            canvas.paste(tile, (x, y + label_h))
            draw.rectangle((x, y + label_h, x + cell_w - 1, y + label_h + cell_h - 1), outline=(210, 210, 210))
    out = ROOT / f"ablation_{name}_crop_contact_sheet.png"
    canvas.save(out)
    return out


def make_attention_sheet():
    font = load_font(14)
    variants = [(tag, name, dirname) for tag, name, dirname in VARIANTS if (ROOT / dirname / "attn").exists()]
    cell_w = 210
    cell_h = 210
    label_h = 44
    margin = 10
    width = len(variants) * (cell_w + margin) + margin
    height = len(STYLES) * (cell_h + label_h + margin) + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for row, style in enumerate(STYLES):
        for col, (tag, name, dirname) in enumerate(variants):
            x = margin + col * (cell_w + margin)
            y = margin + row * (cell_h + label_h + margin)
            draw.text((x + 2, y), f"{style}\n{tag} {name}", fill=(20, 20, 20), font=font)
            attn_dir = ROOT / dirname / "attn"
            paths = sorted(attn_dir.glob(f"demo_0_{style}*step*_layer*.png"))
            paths = [p for p in paths if "focus" not in p.name] or paths
            if not paths:
                continue
            img = Image.open(paths[0]).convert("RGB")
            img.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (cell_w, cell_h), (245, 245, 245))
            tile.paste(img, ((cell_w - img.width) // 2, (cell_h - img.height) // 2))
            canvas.paste(tile, (x, y + label_h))
            draw.rectangle((x, y + label_h, x + cell_w - 1, y + label_h + cell_h - 1), outline=(210, 210, 210))
    out = ROOT / "ablation_self_attention_contact_sheet.png"
    canvas.save(out)
    return out


def make_mask_sheet():
    font = load_font(15)
    cells = []
    ext_mask = load_external_mask()
    ext_img = Image.fromarray((ext_mask * 255).astype(np.uint8)).convert("RGB")
    cells.append(("MediaPipe/sidecar face-part mask", ext_img))
    for dirname in ("A1_internal_attention_only", "A2_internal_spatial020", "A3_internal_spatial035"):
        mask_dir = ROOT / dirname / "internal_attention_masks"
        for concept in ("eyes", "mouth", "lips", "nose"):
            path = mask_dir / f"demo_0_{concept}_internal_attn_mask.png"
            if path.exists():
                cells.append((f"{dirname}\n{concept}", Image.open(path).convert("RGB")))
    cols = 5
    cell = 180
    label_h = 42
    margin = 10
    rows = int(np.ceil(len(cells) / cols))
    canvas = Image.new("RGB", (cols * (cell + margin) + margin, rows * (cell + label_h + margin) + margin), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, img) in enumerate(cells):
        row, col = divmod(i, cols)
        x = margin + col * (cell + margin)
        y = margin + row * (cell + label_h + margin)
        draw.text((x + 2, y), label, fill=(20, 20, 20), font=font)
        img.thumbnail((cell, cell), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (cell, cell), (245, 245, 245))
        tile.paste(img, ((cell - img.width) // 2, (cell - img.height) // 2))
        canvas.paste(tile, (x, y + label_h))
        draw.rectangle((x, y + label_h, x + cell - 1, y + label_h + cell - 1), outline=(210, 210, 210))
    out = ROOT / "ablation_mask_contact_sheet.png"
    canvas.save(out)
    return out


def make_spatial_map_sheet():
    font = load_font(14)
    cells = []
    for dirname in ("A2_internal_spatial020", "A3_internal_spatial035", "A5_mediapipe_spatial020"):
        map_dir = ROOT / dirname / "spatial_adaptive_maps"
        if not map_dir.exists():
            continue
        for path in sorted(map_dir.glob("*.png"))[:6]:
            cells.append((f"{dirname}\n{path.name.split('_step')[0][-18:]}", Image.open(path).convert("RGB")))
    if not cells:
        return None
    cols = 6
    cell = 160
    label_h = 44
    margin = 10
    rows = int(np.ceil(len(cells) / cols))
    canvas = Image.new("RGB", (cols * (cell + margin) + margin, rows * (cell + label_h + margin) + margin), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, img) in enumerate(cells):
        row, col = divmod(i, cols)
        x = margin + col * (cell + margin)
        y = margin + row * (cell + label_h + margin)
        draw.text((x + 2, y), label, fill=(20, 20, 20), font=font)
        img.thumbnail((cell, cell), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (cell, cell), (245, 245, 245))
        tile.paste(img, ((cell - img.width) // 2, (cell - img.height) // 2))
        canvas.paste(tile, (x, y + label_h))
        draw.rectangle((x, y + label_h, x + cell - 1, y + label_h + cell - 1), outline=(210, 210, 210))
    out = ROOT / "ablation_spatial_risk_contact_sheet.png"
    canvas.save(out)
    return out


def compute_metrics():
    content = np_img(CONTENT)
    content_edge = edge_map(content)
    face_mask = load_external_mask()
    non_face_mask = 1.0 - np.clip(face_mask, 0, 1)
    rows = []
    for tag, name, dirname in VARIANTS:
        edge_scores = []
        stylization_scores = []
        for style in STYLES:
            path = ROOT / dirname / f"demo_0_{style}_reconstructed.png"
            if not path.exists():
                continue
            result = np_img(path)
            result_edge = edge_map(result)
            edge_scores.append(masked_mean(np.abs(result_edge - content_edge), face_mask))
            stylization_scores.append(masked_mean(np.abs(result - content).mean(axis=-1), non_face_mask))
        if edge_scores:
            rows.append((tag, name, float(np.mean(edge_scores)), float(np.mean(stylization_scores))))
    return rows


def write_report(paths, metrics):
    lines = [
        "# Ablation Study: Internal Attention, Spatial Adaptive Scaling, and MediaPipe Masks",
        "",
        "## Setup",
        "",
        "All ablations use the same content image, the same three style images, the same ZSTAR injection window, and the same style/content scales. Only the protection source and spatial adaptive strength are changed.",
        "",
        "Common setting:",
        "",
        "```text",
        "start_step=5, end_step=30, layer_index=20,22,24,26,28,30",
        "content_style_scale=1.50, style_content_scale=1.50",
        "styles={candy, Starry, Composition-VII}",
        "```",
        "",
        "## Variants",
        "",
        "| ID | Variant | Purpose |",
        "|---|---|---|",
    ]
    purpose = {
        "A0": "Global ZSTAR baseline without semantic protection or spatial adaptive scaling.",
        "A1": "Isolate the effect of internal cross-attention semantic masks and self-attention refinement.",
        "A2": "Recommended method: internal attention plus spatial adaptive strength 0.20.",
        "A3": "Test stronger spatial preservation with strength 0.35.",
        "A4": "Use external face-part masks as a MediaPipe/landmark-style upper-bound protection source.",
        "A5": "Combine external masks with spatial adaptive strength 0.20.",
    }
    for tag, name, dirname in VARIANTS:
        lines.append(f"| {tag} | `{dirname}` | {purpose[tag]} |")
    lines.extend([
        "",
        "## Visual Outputs",
        "",
    ])
    for label, path in paths.items():
        if path:
            lines.append(f"- {label}: `{path}`")
    lines.extend([
        "",
        "## Proxy Metrics",
        "",
        "These numbers are not final paper metrics. They are quick sanity checks to support visual inspection.",
        "",
        "- `face_edge_delta`: masked edge difference between result and content inside the face-part mask. Lower usually means better facial structure preservation.",
        "- `nonface_color_delta`: color difference from content outside the face-part mask. Higher usually means stronger visible stylization in non-face regions.",
        "",
        "| ID | Variant | face_edge_delta ↓ | nonface_color_delta ↑ |",
        "|---|---|---:|---:|",
    ])
    for tag, name, edge_score, style_score in metrics:
        lines.append(f"| {tag} | {name} | {edge_score:.4f} | {style_score:.4f} |")
    lines.extend([
        "",
        "## Analysis",
        "",
        "1. **Internal attention effect.** Comparing A0 and A1 shows what internal cross-attention contributes. A0 keeps the strongest raw ZSTAR stylization but causes visible facial distortion, especially around eyes and mouth. A1 uses the model's own text-image attention maps to protect eyes, mouth, lips, and nose; the face crops show cleaner eye shape and less mouth/nose distortion while retaining visible style. Its limitation is mask coarseness: nose and lips are semantic but not anatomically sharp.",
        "",
        "2. **Spatial adaptive strength effect.** Comparing A1, A2, and A3 isolates the spatial adaptive scale. A2 (`0.20`) is the balanced setting: it reduces style injection around high-risk structures while preserving visible stylization. A3 (`0.35`) further lowers the face-edge proxy, but the crop sheets show a smoother, less stylized face. This supports using `0.20` as the default and `0.35` only as a conservative face-preservation setting.",
        "",
        "3. **MediaPipe/external mask effect.** Comparing A1 and A4 tests the value of external face-part masks. In this package, the external sidecar masks stand in for a MediaPipe/landmark-style preprocessing source. They are anatomically sharper for named parts, especially eyes and lips, but they cover a narrower region than internal attention masks. As a result, A4 keeps slightly stronger non-face stylization but does not necessarily improve the whole-face edge proxy. The cost is also methodological: the pipeline is no longer fully self-contained and depends on external preprocessing quality.",
        "",
        "4. **Best practical trade-off.** A2 remains the preferred paper setting for a self-contained method. A5 is useful as a stronger oracle-style comparison: it combines external masks with spatial adaptive scaling and can show what is gained when accurate face-part masks are available.",
        "",
        "## Attention Map Interpretation",
        "",
        "- `ablation_self_attention_contact_sheet.png` shows the saved ZSTAR self-attention map at the same step/layer across variants. It is useful for checking whether protection changes the attention distribution qualitatively.",
        "- `ablation_mask_contact_sheet.png` compares internal semantic masks with the external face-part mask. This directly supports the internal-attention ablation.",
        "- `ablation_spatial_risk_contact_sheet.png` shows the structure-risk maps used by spatial adaptive scaling. Bright regions receive weaker content-to-style injection.",
        "",
        "## Caveat",
        "",
        "The proxy metrics are intentionally simple. For a final paper, use face landmark error, identity similarity, masked LPIPS, and a style-strength metric such as Gram/style CLIP similarity. The current artifacts are meant to select and explain ablation directions.",
    ])
    (ROOT / "ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    external_mask = load_external_mask()
    face_box = mask_bbox(external_mask, pad=70)
    eye_mask = np.load(MASKS["eyes"]).astype(np.float32)
    if eye_mask.ndim == 3:
        eye_mask = eye_mask[..., 0]
    eye_box = mask_bbox(normalize(eye_mask), pad=55)
    paths = {
        "Result contact sheet": make_result_sheet(),
        "Face crop contact sheet": make_crop_sheet("face", face_box),
        "Eye crop contact sheet": make_crop_sheet("eye", eye_box),
        "Self-attention contact sheet": make_attention_sheet(),
        "Mask contact sheet": make_mask_sheet(),
        "Spatial risk contact sheet": make_spatial_map_sheet(),
    }
    metrics = compute_metrics()
    write_report(paths, metrics)
    print(ROOT / "ablation_report.md")


if __name__ == "__main__":
    main()
