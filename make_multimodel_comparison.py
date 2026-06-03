import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/whc/零样本风格迁移")
THIS = ROOT / "ZSTAR_spatial020_minimal"
WORK = Path(os.environ.get("MULTIMODEL_WORK", THIS / "workdir/multimodel_compare"))
OUT = WORK / "summary"

CONTENT = WORK / "inputs/content/demo_0.jpg"
STYLE_DIR = WORK / "inputs/style"
STYLE = sorted(
    p for p in STYLE_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
)[0]
STYLE_STEM = STYLE.stem
OURS_MODE = os.environ.get("OURS_MODE", "full")
if OURS_MODE == "weighted_only":
    OURS_NAME = "Ours-weighted"
    OURS_SUBTITLE = "weighted only, strong"
    OURS_OUTPUT_DIR = "Ours_weighted_only"
    OURS_DESCRIPTION = "Ours: weighted-only spatial adaptive injection with stronger ZSTAR scales"
else:
    OURS_NAME = "Ours"
    OURS_SUBTITLE = "internal + spatial 0.20"
    OURS_OUTPUT_DIR = "Ours_spatial020"
    OURS_DESCRIPTION = "Ours: internal attention protection + spatial adaptive scaling 0.20"

METHODS = [
    ("Content", CONTENT),
    ("Style", STYLE),
    ("AdaIN", WORK / f"outputs/AdaIN/demo_0_stylized_{STYLE_STEM}.jpg"),
    ("SANet", WORK / f"outputs/SANet/demo_0_stylized_{STYLE_STEM}.jpg"),
    ("StyTR2", WORK / f"outputs/StyTR2/demo_0_stylized_{STYLE_STEM}.jpg"),
    ("ZSTAR", WORK / f"outputs/ZSTAR_direct/demo_0_{STYLE_STEM}_reconstructed.png"),
    (OURS_NAME, WORK / f"outputs/{OURS_OUTPUT_DIR}/demo_0_{STYLE_STEM}_reconstructed.png"),
]


def font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def fit_square(path, size=320):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.Resampling.LANCZOS)


def make_sheet():
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [(name, path) for name, path in METHODS if not path.exists()]
    if missing:
        detail = "\n".join(f"{name}: {path}" for name, path in missing)
        raise FileNotFoundError(f"Missing comparison inputs:\n{detail}")

    cell = 320
    label_h = 58
    gap = 12
    margin = 18
    width = margin * 2 + len(METHODS) * cell + (len(METHODS) - 1) * gap
    height = margin * 2 + label_h + cell
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(23)
    small_font = font(16)

    for idx, (name, path) in enumerate(METHODS):
        x = margin + idx * (cell + gap)
        y = margin
        draw.text((x, y), name, fill=(20, 20, 20), font=title_font)
        if name in {"Content", "Style"}:
            subtitle = "input"
        elif name.startswith("Ours"):
            subtitle = OURS_SUBTITLE
        elif name == "ZSTAR":
            subtitle = "direct baseline"
        else:
            subtitle = "image-style baseline"
        draw.text((x, y + 28), subtitle, fill=(80, 80, 80), font=small_font)
        img = fit_square(path, cell)
        canvas.paste(img, (x, margin + label_h))
        draw.rectangle((x, margin + label_h, x + cell - 1, margin + label_h + cell - 1), outline=(210, 210, 210))

    out_path = OUT / f"multimodel_face_{STYLE_STEM}_comparison.png"
    canvas.save(out_path)
    return out_path


def write_report(sheet_path):
    lines = [
        f"# Multi-Model Face + {STYLE_STEM} Style Comparison",
        "",
        "## Setting",
        "",
        "All runnable image-style baselines are evaluated with the same portrait content image and the same reference style image.",
        "",
        "```text",
        f"content: {CONTENT}",
        f"style:   {STYLE}",
        "```",
        "",
        "## Compared Methods",
        "",
        "```text",
        "AdaIN",
        "SANet",
        "StyTR2",
        "ZSTAR direct baseline",
        OURS_DESCRIPTION,
        "```",
        "",
        "CLIPstyler is not included in this image-style sheet because the available implementation is text-condition/decoder-condition based rather than taking the same reference style image as input.",
        "",
        "## Output",
        "",
        f"- Comparison sheet: `{sheet_path}`",
        "",
        "## Qualitative Reading",
        "",
        "Use this sheet to compare three properties: visible style transfer strength, preservation of facial identity, and local stability of eyes, nose, and mouth. Feed-forward image-style baselines provide useful non-diffusion references; direct ZSTAR shows the original attention-rearrangement behavior. In weighted-only mode, ours disables semantic face-part masks and keeps only spatially weighted injection with stronger style scales.",
    ]
    (OUT / "multimodel_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    sheet = make_sheet()
    write_report(sheet)
    print(sheet)


if __name__ == "__main__":
    main()
