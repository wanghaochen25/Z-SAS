"""
CLIPSeg-based text-to-mask segmentation utilities.
Generates binary masks for localized style transfer in Z-STAR pipeline.

Usage:
    python segment_utils.py --image ./content_images/portrait.jpg --text "hair"
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image


def parse_region_text(text: str) -> list:
    """
    Parse comma-separated region text into a list of clean prompts.

    Example:
        "hair, face" -> ["hair", "face"]
        "hair ,  face" -> ["hair", "face"]  (handles stray spaces)
    """
    return [p.strip() for p in text.split(",") if p.strip()]


def clear_device_cache(device: torch.device):
    """
    Release GPU/MPS memory in a device-compatible way.

    Args:
        device: torch.device object (e.g. cuda:0, mps, cpu)
    """
    device_str = str(device)
    if "cuda" in device_str:
        torch.cuda.empty_cache()
    elif "mps" in device_str:
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass  # older PyTorch may not have mps.empty_cache


def load_clipseg_model(device: torch.device):
    """
    Load CLIPSeg model and processor from HuggingFace.

    Returns:
        (model, processor) tuple. Model is ~600 MB on first download.

    Note:
        Requires `transformers` package. Install with:
        pip install transformers>=4.30.0
    """
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

    model_id = "CIDAS/clipseg-rd64-refined"
    processor = CLIPSegProcessor.from_pretrained(model_id)
    model = CLIPSegForImageSegmentation.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, processor


def generate_mask_from_text(
    image_path: str,
    text_prompts: list,
    target_size: int = 560,
    threshold: float = 0.5,
    blur_sigma: float = 3.0,
    blur_kernel: int = 21,
    device: torch.device = None,
):
    """
    Generate a soft mask from a text description using CLIPSeg.

    The mask maps:
        +1.0 = style-transfer region (e.g. "hair")
        -1.0 = preserve-original region (everything else)

    Pipeline:
        PIL image -> resize to target_size -> CLIPSeg per prompt
        -> sigmoid probabilities -> max-pool across prompts
        -> remap to [-1, +1] -> Gaussian blur (soft edges)
        -> numpy array

    Args:
        image_path: Path to the content image.
        text_prompts: List of text descriptions, e.g. ["hair"] or ["hair", "face"].
        target_size: Output mask size (square), default 560 (matches Z-STAR).
        threshold: CLIPSeg confidence threshold for binarization (before blur).
        blur_sigma: Gaussian blur sigma. 0 = no blur (hard edges). Default 3.0.
        blur_kernel: Gaussian kernel size. Must be odd. Default 21.
        device: torch device. Auto-detected if None.

    Returns:
        np.ndarray of shape (target_size, target_size), dtype=float32,
        values in [-1.0, 1.0].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load and preprocess image ---
    image = Image.open(image_path).convert("RGB")
    image = image.resize((target_size, target_size), Image.BICUBIC)

    # --- CLIPSeg per prompt: collect sigmoid probabilities ---
    model, processor = load_clipseg_model(device)
    prob_maps = []  # list of torch.Tensor, each shape (target_size, target_size)

    for prompt in text_prompts:
        inputs = processor(
            text=[prompt],
            images=[image],
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # outputs.logits: may be (1, H, W) or (1, 1, H, W) depending on version
        logits = outputs.logits
        if logits.dim() == 4:
            logits = logits[0, 0]  # (H, W) from (1, 1, H, W)
        elif logits.dim() == 3:
            logits = logits[0]     # (H, W) from (1, H, W)
        # Upsample to target_size
        logits = logits.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        logits = F.interpolate(logits, size=(target_size, target_size), mode='bilinear', align_corners=False)
        logits = logits[0, 0]  # (target_size, target_size)
        prob = torch.sigmoid(logits)   # values in [0, 1]
        prob_maps.append(prob)

    # --- Multi-prompt union: take max probability across all prompts ---
    # This preserves edge probability info better than binary-OR.
    combined = torch.stack(prob_maps, dim=0).max(dim=0).values  # (H, W), in [0, 1]

    # --- Release CLIPSeg memory ---
    del model, processor, outputs, inputs
    clear_device_cache(device)

    # --- Remap to Z-STAR format: [+1 = style region, -1 = preserve region] ---
    mask_cpu = combined.cpu().numpy().astype(np.float32)
    mask_cpu = mask_cpu * 2.0 - 1.0  # [0,1] -> [-1, +1]

    # --- Gaussian blur for soft edges ---
    if blur_sigma > 0:
        ksize = max(3, blur_kernel)
        if ksize % 2 == 0:
            ksize += 1  # ensure odd kernel size
        mask_cpu = cv2.GaussianBlur(mask_cpu, (ksize, ksize), sigmaX=blur_sigma)

    return mask_cpu


def save_mask_npy(mask: np.ndarray, save_path: str):
    """
    Save mask array as .npy file for Z-STAR consumption.

    Z-STAR's ReweightCrossAttentionControl.get_batch_sim_with_mask()
    automatically loads `{content_image_name}_mask.npy`.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.save(save_path, mask.astype(np.float32))
    print(f"[segment_utils] Mask saved to: {save_path}")


def visualize_mask(mask: np.ndarray, save_path: str):
    """
    Save a visualization of the mask as a PNG for inspection.

    - Style regions (+1) shown in red
    - Preserved regions (-1) shown in blue
    - Transition regions shown as blend
    """
    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    # Remap [-1, +1] -> [0, 1]
    normalized = (mask + 1.0) / 2.0
    # Red channel = style intensity, Blue channel = preserve intensity
    vis[:, :, 0] = (normalized * 255).astype(np.uint8)        # Red
    vis[:, :, 2] = ((1.0 - normalized) * 255).astype(np.uint8)  # Blue
    Image.fromarray(vis).save(save_path)
    print(f"[segment_utils] Visualization saved to: {save_path}")


# ============================================================================
# Standalone test
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIPSeg mask generation test")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--text", type=str, default="hair",
                        help="Comma-separated region descriptions, e.g. 'hair, face'")
    parser.add_argument("--size", type=int, default=560,
                        help="Target mask size (default 560)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="CLIPSeg confidence threshold")
    parser.add_argument("--blur_sigma", type=float, default=3.0,
                        help="Gaussian blur sigma (0 = no blur)")
    parser.add_argument("--blur_kernel", type=int, default=21,
                        help="Gaussian kernel size (odd)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npy path (default: {image}_mask.npy)")
    parser.add_argument("--vis", type=str, default=None,
                        help="Save visualization PNG to this path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[segment_utils] Using device: {device}")

    prompts = parse_region_text(args.text)
    print(f"[segment_utils] Text prompts: {prompts}")

    mask = generate_mask_from_text(
        image_path=args.image,
        text_prompts=prompts,
        target_size=args.size,
        threshold=args.threshold,
        blur_sigma=args.blur_sigma,
        blur_kernel=args.blur_kernel,
        device=device,
    )

    out_path = args.output or args.image.replace(".jpg", "_mask.npy").replace(
        ".png", "_mask.npy").replace(".jpeg", "_mask.npy").replace(".bmp", "_mask.npy")
    save_mask_npy(mask, out_path)

    if args.vis or not args.output:
        vis_path = args.vis or out_path.replace(".npy", "_vis.png")
        visualize_mask(mask, vis_path)

    # Stats
    pos_ratio = (mask > 0).sum() / mask.size
    print(f"[segment_utils] Mask stats: shape={mask.shape}, "
          f"style_region_ratio={pos_ratio:.2%}, "
          f"range=[{mask.min():.3f}, {mask.max():.3f}]")