import argparse
from zstar.zstar import ReweightCrossAttentionControl
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from diffusers import DDIMScheduler
from zstar.diffuser_utils import ZstarPipeline
from zstar.zstar_utils import AttentionBase
from zstar.zstar_utils import CrossAttentionSemanticMaskExtractor
from zstar.zstar_utils import regiter_attention_editor_diffusers
from torchvision.utils import save_image
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from typing import Union
import torch.nn.functional as nnf
import numpy as np
import ptp_utils
import shutil
from torch.optim.adam import Adam
from PIL import Image
import pickle
import warnings
import segment_utils  # CLIPSeg text-to-mask for localized style transfer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)

MY_TOKEN = ""
TARGET_IMG_SIZE = 560
RESIZE_MODE = "square"
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
STABLE_DIFFUSION_MODEL_PATH = "./stable-diffusion-v1-5"
SEED = 9999
START_STEP = 5
END_STEP = 30
TOTAL_STEP = 30
NUM_DDIM_STEPS = TOTAL_STEP
LAYER_INDEX = [20, 22, 24, 26, 28, 30]
LAYER_INDEX_STRING = "_".join(str(x) for x in LAYER_INDEX)

torch.cuda.set_device(0)  # set the GPU device
seed_everything(SEED)

# Note that you may add your Hugging Face token to get access to the models
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
)
model = ZstarPipeline.from_pretrained(
    STABLE_DIFFUSION_MODEL_PATH, scheduler=scheduler, torch_dtype=torch.float16).to(device)


def open_image_rgb(image_path):
    image = Image.open(image_path)
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    return image


def prepare_image(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    if RESIZE_MODE == "pad":
        image.thumbnail((TARGET_IMG_SIZE, TARGET_IMG_SIZE), LANCZOS)
        canvas = Image.new("RGB", (TARGET_IMG_SIZE, TARGET_IMG_SIZE), (255, 255, 255))
        left = (TARGET_IMG_SIZE - image.width) // 2
        top = (TARGET_IMG_SIZE - image.height) // 2
        canvas.paste(image, (left, top))
        return canvas
    return image.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE), LANCZOS)


def load_img_to_numpy(image_path):
    if type(image_path) is str:
        image = open_image_rgb(image_path)
    else:
        image = Image.fromarray(image_path)
    return np.array(prepare_image(image))


def load_image(image_path, device, reverse=False):
    totensor = transforms.ToTensor()
    image = totensor(prepare_image(open_image_rgb(image_path)))
    image = image[:3].unsqueeze_(0).float() * 2.0 - 1.0
    if reverse:
        image = torch.flip(image, dims=[2])
    image = image.to(device).half()
    return image


def has_nan_embeddings(embeddings):
    if embeddings is None:
        return False
    return any(torch.isnan(t).any().item() for t in embeddings)


def get_image(data_dir):
    img_list = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if (
                file.endswith(".jpg")
                or file.endswith(".png")
                or file.endswith(".bmp")
                or file.endswith(".jpeg")
            ):
                img_list.append(os.path.join(root, file))
    assert len(img_list) > 0, "[ERROR] img_list is Empty!"
    return img_list


def sidecar_path(image_path, suffix):
    return (
        image_path.replace(".png", suffix)
        .replace(".jpg", suffix)
        .replace(".jpeg", suffix)
        .replace(".bmp", suffix)
    )


def first_existing_sidecar(image_path, suffixes):
    for suffix in suffixes:
        candidate = sidecar_path(image_path, suffix)
        if os.path.exists(candidate):
            return candidate
    return None


def clamp01(value):
    return float(max(0.0, min(1.0, value)))


def infer_nose_mask_from_parts(content, eyes_mask_path, mouth_mask_path):
    nose_mask_path = sidecar_path(content, "_nose_mask.npy")
    if os.path.exists(nose_mask_path):
        return nose_mask_path
    if not eyes_mask_path or not mouth_mask_path:
        return None
    if not os.path.exists(eyes_mask_path) or not os.path.exists(mouth_mask_path):
        return None

    eyes = np.load(eyes_mask_path).astype(np.float32)
    mouth = np.load(mouth_mask_path).astype(np.float32)
    if eyes.ndim == 3:
        eyes = eyes[..., 0]
    if mouth.ndim == 3:
        mouth = mouth[..., 0]
    if eyes.shape != mouth.shape:
        return None

    def bbox(mask):
        ys, xs = np.where(mask > 0.1)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return xs.min(), ys.min(), xs.max(), ys.max()

    eyes_box = bbox(eyes)
    mouth_box = bbox(mouth)
    if eyes_box is None or mouth_box is None:
        return None

    ex0, ey0, ex1, ey1 = eyes_box
    mx0, my0, mx1, my1 = mouth_box
    eye_cx = (ex0 + ex1) * 0.5
    eye_cy = (ey0 + ey1) * 0.5
    mouth_cx = (mx0 + mx1) * 0.5
    mouth_cy = (my0 + my1) * 0.5
    vertical_span = max(8.0, mouth_cy - eye_cy)
    cx = (eye_cx + mouth_cx) * 0.5
    cy = eye_cy + vertical_span * 0.52
    rx = max(4.0, (ex1 - ex0 + 1) * 0.12, (mx1 - mx0 + 1) * 0.18)
    ry = max(6.0, vertical_span * 0.22)

    yy, xx = np.mgrid[:eyes.shape[0], :eyes.shape[1]]
    nose = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0).astype(np.float32)
    np.save(nose_mask_path, nose)
    return nose_mask_path


def inversion_cache_path(image_path):
    base = sidecar_path(image_path, ".pkl")
    if TARGET_IMG_SIZE == 560 and RESIZE_MODE == "square" and NUM_DDIM_STEPS == 30:
        return base
    root, _ = os.path.splitext(base)
    return f"{root}_{RESIZE_MODE}_{TARGET_IMG_SIZE}_steps{NUM_DDIM_STEPS}.pkl"


def parse_int_list(value):
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_concept_strengths(value):
    concepts = []
    if not value:
        return concepts
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" in item:
            name, strength = item.split(":", 1)
            concepts.append((name.strip(), clamp01(float(strength))))
        else:
            concepts.append((item, 1.0))
    return concepts


def find_subsequence(sequence, pattern):
    if not pattern:
        return None
    for start in range(0, len(sequence) - len(pattern) + 1):
        if sequence[start:start + len(pattern)] == pattern:
            return list(range(start, start + len(pattern)))
    return None


def concept_token_indices(tokenizer, prompt, concepts):
    prompt_ids = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        return_tensors="pt",
    ).input_ids[0].tolist()
    output = {}
    for concept, _ in concepts:
        candidates = [concept, f" {concept}"]
        indices = None
        for candidate in candidates:
            concept_ids = tokenizer(
                candidate,
                add_special_tokens=False,
                return_tensors=None,
            ).input_ids
            indices = find_subsequence(prompt_ids, concept_ids)
            if indices:
                break
        if indices:
            output[concept] = indices
        else:
            print(f"[InternalAttn] Token not found in semantic prompt: {concept}")
    return output


def parse_protect_mask_specs(value, default_dilate, default_blur_sigma):
    if not value:
        return []
    masks = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) < 3:
            raise ValueError(
                "--protect_mask entries must use name:path:strength[:dilate[:blur_sigma]]"
            )
        masks.append({
            "name": parts[0],
            "path": parts[1],
            "suffixes": [],
            "strength": clamp01(float(parts[2])),
            "dilate": int(parts[3]) if len(parts) > 3 and parts[3] else default_dilate,
            "blur_sigma": float(parts[4]) if len(parts) > 4 and parts[4] else default_blur_sigma,
        })
    return masks


class NullInversion:

    def prev_step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
    ):
        prev_timestep = (
            timestep
            - self.scheduler.config.num_train_timesteps
            // self.scheduler.num_inference_steps
        )
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (
            sample - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = (
            alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        )
        return prev_sample

    def next_step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
    ):
        timestep, next_timestep = (
            min(
                timestep
                - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps,
                999,
            ),
            timestep,
        )
        alpha_prod_t = (
            self.scheduler.alphas_cumprod[timestep]
            if timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (
            sample - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = (
            alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
        )
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents.to(dtype=self.model.unet.dtype), t, encoder_hidden_states=context)[
            "sample"
        ]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        latents_input = latents_input.to(dtype=self.model.unet.dtype)
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)[
            "sample"
        ]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_prediction_text - noise_pred_uncond
        )
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type="np"):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)["sample"]
        if return_type == "np":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device).half()
                latents = self.model.vae.encode(image)["latent_dist"].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.model.text_encoder(
            uncond_input.input_ids.to(self.model.device)
        )[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(
            text_input.input_ids.to(self.model.device)
        )[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[
                len(self.model.scheduler.timesteps) - i - 1
            ]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        for i in tqdm(range(NUM_DDIM_STEPS)):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1.0 - i / 100.0))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(
                    latent_cur, t, cond_embeddings
                )
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(
                    latent_cur, t, uncond_embeddings
                )
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (
                    noise_pred_cond - noise_pred_uncond
                )
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                if loss_item < epsilon + i * 2e-5:
                    break
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        return uncond_embeddings_list

    def invert(
        self,
        image_path: str,
        prompt: str,
        num_inner_steps=10,
        early_stop_epsilon=1e-5,
        verbose=False
    ):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_img_to_numpy(image_path)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(
            ddim_latents, num_inner_steps, early_stop_epsilon
        )
        return (image_gt, image_rec), ddim_latents, ddim_latents[-1], uncond_embeddings

    def __init__(self, model):
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None


def parse_args():
    parser = argparse.ArgumentParser(description="Z-STAR style transfer with optional CLIPSeg localized masking")
    parser.add_argument('--sub_exp_name', type=str,
                        default="workdir/demo", help='sub exp name')
    parser.add_argument('--content_img_folder', type=str,
                        default="./content_images/", help='content image paths')
    parser.add_argument('--style_img_folder', type=str,
                        default="./style_images/", help='style image paths')
    parser.add_argument('--target_size', type=int, default=TARGET_IMG_SIZE,
                        help='Square working canvas size, preferably a multiple of 64 (default 560).')
    parser.add_argument('--resize_mode', choices=['square', 'pad'], default=RESIZE_MODE,
                        help='square stretches to target_size; pad preserves aspect ratio on a square canvas.')
    parser.add_argument('--start_step', type=int, default=START_STEP,
                        help='First denoising step for ZSTAR attention injection.')
    parser.add_argument('--end_step', type=int, default=END_STEP,
                        help='Exclusive final denoising step for ZSTAR attention injection.')
    parser.add_argument('--total_step', type=int, default=TOTAL_STEP,
                        help='Total DDIM sampling steps.')
    parser.add_argument('--layer_index', type=str, default=",".join(str(x) for x in LAYER_INDEX),
                        help='Comma-separated attention layer indices to inject, e.g. "20,22,24,26,28,30".')
    parser.add_argument('--content_style_scale', type=float, default=1.2,
                        help='Scale for content->style similarity logits (official default 1.2).')
    parser.add_argument('--style_content_scale', type=float, default=1.5,
                        help='Scale for style->content similarity logits (official default 1.5).')
    parser.add_argument('--eye_mask', type=str, default=None,
                        help='Optional .npy eye mask. If omitted, *_eyes_mask.npy next to the content image is used when present.')
    parser.add_argument('--eye_protect_strength', type=float, default=0.0,
                        help='Blend eye-region content-preserve attention into the stylized content branch, 0 disables, 1 fully protects.')
    parser.add_argument('--eye_mask_dilate', type=int, default=3,
                        help='Eye mask dilation radius before latent downsampling.')
    parser.add_argument('--eye_mask_blur_sigma', type=float, default=1.5,
                        help='Gaussian blur sigma for soft eye mask edges.')
    parser.add_argument('--face_protect', action='store_true',
                        help='Enable semantic protection for eyes, mouth/lips, and nose when masks are available.')
    parser.add_argument('--mouth_mask', type=str, default=None,
                        help='Optional .npy mouth/lips mask. If omitted, *_mouth_mask.npy or *_lips_mask.npy is used when present.')
    parser.add_argument('--nose_mask', type=str, default=None,
                        help='Optional .npy nose mask. If omitted, *_nose_mask.npy is used when present.')
    parser.add_argument('--face_eye_protect_strength', type=float, default=0.85,
                        help='Eye protection strength used by --face_protect when --eye_protect_strength is 0.')
    parser.add_argument('--mouth_protect_strength', type=float, default=0.75,
                        help='Mouth/lips protection strength used by --face_protect or --mouth_mask.')
    parser.add_argument('--nose_protect_strength', type=float, default=0.55,
                        help='Nose protection strength used by --face_protect or --nose_mask.')
    parser.add_argument('--protect_mask', type=str, default=None,
                        help='Extra protected masks as name:path:strength[:dilate[:blur_sigma]], comma-separated.')
    parser.add_argument('--protect_mask_dilate', type=int, default=None,
                        help='Shared dilation radius for face-part protection; defaults to --eye_mask_dilate.')
    parser.add_argument('--protect_mask_blur_sigma', type=float, default=None,
                        help='Shared Gaussian blur sigma for face-part protection; defaults to --eye_mask_blur_sigma.')
    parser.add_argument('--no_infer_nose_mask', action='store_true',
                        help='Disable approximate *_nose_mask.npy generation from eye and mouth masks.')
    parser.add_argument('--internal_attention_protect', action='store_true',
                        help='Derive protected face-part masks from Stable Diffusion cross-attention instead of external parsers.')
    parser.add_argument('--semantic_prompt', type=str,
                        default='a portrait photo with face eyes nose mouth lips hair background',
                        help='Prompt used to collect internal cross-attention semantic maps.')
    parser.add_argument('--semantic_protect_concepts', type=str,
                        default='eyes:0.85,mouth:0.75,lips:0.75,nose:0.55',
                        help='Concept strengths for internal attention protection, e.g. eyes:0.85,mouth:0.75,nose:0.55.')
    parser.add_argument('--semantic_collect_start', type=int, default=5,
                        help='First denoising step used for collecting internal semantic attention.')
    parser.add_argument('--semantic_collect_end', type=int, default=20,
                        help='Exclusive final denoising step used for collecting internal semantic attention.')
    parser.add_argument('--semantic_collect_layers', type=str, default='',
                        help='Optional comma-separated cross-attention layer ranks for semantic collection. Empty means all cross-attention layers.')
    parser.add_argument('--semantic_attn_max_res', type=int, default=64,
                        help='Max cross-attention grid resolution used for internal semantic masks.')
    parser.add_argument('--semantic_mask_quantile', type=float, default=0.80,
                        help='Drop low-response internal attention values below this quantile before using them as masks.')
    parser.add_argument('--semantic_mask_gamma', type=float, default=1.0,
                        help='Gamma applied to normalized internal attention masks after quantile sharpening.')
    parser.add_argument('--semantic_step_weight_mode', choices=['uniform', 'semantic'],
                        default='semantic',
                        help='How to weight denoising steps when averaging internal attention masks.')
    parser.add_argument('--semantic_layer_weight_mode', choices=['uniform', 'semantic'],
                        default='semantic',
                        help='How to weight attention layers when averaging internal attention masks.')
    parser.add_argument('--semantic_cross_layer_count', type=int, default=16,
                        help='Expected number of cross-attention layers for normalized semantic layer weighting.')
    parser.add_argument('--semantic_self_refine', action='store_true',
                        help='Refine internal cross-attention masks with self-attention token affinity.')
    parser.add_argument('--semantic_self_refine_weight', type=float, default=0.15,
                        help='Blend weight for self-attention-refined masks.')
    parser.add_argument('--semantic_self_refine_iters', type=int, default=2,
                        help='Number of self-attention propagation iterations.')
    parser.add_argument('--semantic_self_refine_max_res', type=int, default=35,
                        help='Max self-attention grid resolution used for mask refinement.')
    parser.add_argument('--semantic_self_refine_layers', type=str, default='',
                        help='Optional comma-separated self-attention layer ranks for mask refinement. Empty means all eligible self-attention layers.')
    parser.add_argument('--semantic_self_layer_count', type=int, default=16,
                        help='Expected number of self-attention layers for normalized semantic self-layer weighting.')
    parser.add_argument('--spatial_adaptive_scale', action='store_true',
                        help='Use self-attention/feature structure maps to reduce style injection on fragile spatial regions.')
    parser.add_argument('--spatial_adaptive_strength', type=float, default=0.35,
                        help='How strongly structure-risk regions reduce content->style injection scale.')
    parser.add_argument('--spatial_adaptive_min_scale', type=float, default=0.65,
                        help='Lower bound for spatial scale multiplier applied to content->style logits.')
    parser.add_argument('--spatial_adaptive_entropy_weight', type=float, default=0.35,
                        help='Weight for low-entropy self-attention focus in the structure-risk map.')
    parser.add_argument('--spatial_adaptive_edge_weight', type=float, default=0.55,
                        help='Weight for content feature edge strength in the structure-risk map.')
    parser.add_argument('--spatial_adaptive_local_weight', type=float, default=0.10,
                        help='Weight for local self-attention mass in the structure-risk map.')
    parser.add_argument('--spatial_adaptive_blur_sigma', type=float, default=1.0,
                        help='Gaussian blur sigma used to smooth the structure-risk map.')
    parser.add_argument('--spatial_adaptive_save_maps', action='store_true',
                        help='Save structure-risk maps for visual inspection.')
    parser.add_argument('--refresh_inversion_cache', action='store_true',
                        help='Ignore cached content inversion pkl and recompute it.')
    parser.add_argument('--diagnostic_triplet', action='store_true',
                        help='Save reconstruction-only, official full ZSTAR, and late/weak ZSTAR outputs for diagnosis.')
    parser.add_argument('--diagnostic_late_start', type=int, default=15,
                        help='Start step for the diagnostic late/weak stylization.')
    parser.add_argument('--diagnostic_late_end', type=int, default=30,
                        help='End step for the diagnostic late/weak stylization.')
    parser.add_argument('--diagnostic_content_style_scale', type=float, default=1.05,
                        help='Content->style scale for diagnostic late/weak stylization.')
    parser.add_argument('--diagnostic_style_content_scale', type=float, default=1.10,
                        help='Style->content scale for diagnostic late/weak stylization.')
    # --- CLIPSeg localized style transfer options ---
    parser.add_argument('--region_text', type=str, default=None,
                        help='Comma-separated region descriptions to stylize, '
                             'e.g. "hair" or "hair, face". '
                             'If not provided, global style transfer is applied.')
    parser.add_argument('--mask_threshold', type=float, default=0.5,
                        help='CLIPSeg confidence threshold (default 0.5)')
    parser.add_argument('--blur_sigma', type=float, default=3.0,
                        help='Gaussian blur sigma for mask edges (0 = no blur, default 3.0)')
    parser.add_argument('--blur_kernel', type=int, default=21,
                        help='Gaussian blur kernel size, must be odd (default 21)')
    parser.add_argument('--no_clipseg', action='store_true',
                        help='Skip CLIPSeg mask generation; use pre-existing mask .npy files only')
    parser.add_argument('--save_attn', action='store_true',
                        help='Save a self-attention map for debugging (content image).')
    parser.add_argument('--attn_step', type=int, default=None,
                        help='Diffusion step index to save attention map (default START_STEP).')
    parser.add_argument('--attn_layer', type=int, default=None,
                        help='Attention layer index to save (default first layer in LAYER_INDEX).')
    parser.add_argument('--attn_size', type=int, default=TARGET_IMG_SIZE,
                        help='Output size for attention map image.')
    parser.add_argument('--attn_max_res', type=int, default=64,
                        help='Max attention grid resolution to save (default 64).')
    parser.add_argument('--attn_focus_mask', type=str, default=None,
                        help='Optional focus mask .npy for attention map (e.g., *_eyes_mask.npy).')
    args = parser.parse_args()
    return args


def main():
    global TARGET_IMG_SIZE, RESIZE_MODE, TOTAL_STEP, NUM_DDIM_STEPS, LAYER_INDEX
    args = parse_args()
    TARGET_IMG_SIZE = args.target_size
    RESIZE_MODE = args.resize_mode
    TOTAL_STEP = args.total_step
    NUM_DDIM_STEPS = args.total_step
    LAYER_INDEX = parse_int_list(args.layer_index) or LAYER_INDEX
    SUB_EXP_NAME = args.sub_exp_name
    CONTENT_IMG_FOLDER = args.content_img_folder
    STYLE_IMG_FOLDER = args.style_img_folder
    attn_step = args.attn_step if args.attn_step is not None else args.start_step
    attn_layer = args.attn_layer if args.attn_layer is not None else LAYER_INDEX[0]
    print(
        f"Runtime config: size={TARGET_IMG_SIZE}, resize_mode={RESIZE_MODE}, "
        f"steps=[{args.start_step}, {args.end_step})/{TOTAL_STEP}, "
        f"layers={LAYER_INDEX}, scales=({args.content_style_scale}, {args.style_content_scale}), "
        f"eye_protect={args.eye_protect_strength}, face_protect={args.face_protect}, "
        f"internal_attention_protect={args.internal_attention_protect}, "
        f"spatial_adaptive_scale={args.spatial_adaptive_scale}"
    )

    null_inversion = NullInversion(model)

    if os.path.exists(SUB_EXP_NAME):
        shutil.rmtree(SUB_EXP_NAME)
    os.makedirs(SUB_EXP_NAME, exist_ok=True)
    attn_dir = os.path.join(SUB_EXP_NAME, "attn")
    if args.save_attn:
        os.makedirs(attn_dir, exist_ok=True)

    content_list = get_image(CONTENT_IMG_FOLDER)
    style_list = get_image(STYLE_IMG_FOLDER)

    def resolve_eye_mask(content):
        if args.eye_mask:
            return args.eye_mask
        candidate = sidecar_path(content, "_eyes_mask.npy")
        if os.path.exists(candidate):
            return candidate
        return None

    def resolve_mouth_mask(content):
        if args.mouth_mask:
            return args.mouth_mask
        return first_existing_sidecar(content, ["_mouth_mask.npy", "_lips_mask.npy"])

    def resolve_nose_mask(content):
        if args.nose_mask:
            return args.nose_mask
        return first_existing_sidecar(content, ["_nose_mask.npy"])

    def build_protect_masks(content, eye_protect_strength, enable_protection=True, internal_masks=None):
        if not enable_protection:
            return []
        default_dilate = args.protect_mask_dilate
        if default_dilate is None:
            default_dilate = args.eye_mask_dilate
        default_blur_sigma = args.protect_mask_blur_sigma
        if default_blur_sigma is None:
            default_blur_sigma = args.eye_mask_blur_sigma

        masks = parse_protect_mask_specs(
            args.protect_mask,
            default_dilate,
            default_blur_sigma,
        )
        if internal_masks:
            for name, mask_path, strength in internal_masks:
                masks.append({
                    "name": f"internal_{name}",
                    "path": mask_path,
                    "suffixes": [],
                    "strength": clamp01(strength),
                    "dilate": default_dilate,
                    "blur_sigma": default_blur_sigma,
                })

        use_face_parts = args.face_protect

        eye_path = resolve_eye_mask(content)
        mouth_path = resolve_mouth_mask(content)
        nose_path = resolve_nose_mask(content)
        if (
            use_face_parts
            and not nose_path
            and not args.no_infer_nose_mask
            and eye_path
            and mouth_path
        ):
            nose_path = infer_nose_mask_from_parts(content, eye_path, mouth_path)
            if nose_path:
                print(f"[FaceProtect] Inferred nose mask: {nose_path}")

        eye_strength = eye_protect_strength
        if use_face_parts and eye_strength <= 0:
            eye_strength = args.face_eye_protect_strength
        if eye_strength > 0 or args.eye_mask:
            masks.append({
                "name": "eyes",
                "path": eye_path,
                "suffixes": ["_eyes_mask.npy"],
                "strength": clamp01(eye_strength),
                "dilate": default_dilate,
                "blur_sigma": default_blur_sigma,
            })
        if use_face_parts or args.mouth_mask:
            masks.append({
                "name": "mouth",
                "path": mouth_path,
                "suffixes": ["_mouth_mask.npy", "_lips_mask.npy"],
                "strength": clamp01(args.mouth_protect_strength),
                "dilate": default_dilate,
                "blur_sigma": default_blur_sigma,
            })
        if use_face_parts or args.nose_mask:
            masks.append({
                "name": "nose",
                "path": nose_path,
                "suffixes": ["_nose_mask.npy"],
                "strength": clamp01(args.nose_protect_strength),
                "dilate": default_dilate,
                "blur_sigma": default_blur_sigma,
            })

        active = [mask for mask in masks if mask.get("strength", 0.0) > 0]
        if active:
            print(
                "[FaceProtect] "
                + ", ".join(
                    f"{mask['name']}={mask.get('path') or 'sidecar'}:{mask['strength']:.2f}"
                    for mask in active
                )
            )
        return active

    def collect_internal_attention_masks(content, x_t):
        if not args.internal_attention_protect:
            return []

        concepts = parse_concept_strengths(args.semantic_protect_concepts)
        token_indices = concept_token_indices(model.tokenizer, args.semantic_prompt, concepts)
        if not token_indices:
            print("[InternalAttn] No concept tokens found; internal protection disabled.")
            return []

        semantic_layers = parse_int_list(args.semantic_collect_layers)
        semantic_self_layers = parse_int_list(args.semantic_self_refine_layers)
        semantic_steps = list(range(args.semantic_collect_start, args.semantic_collect_end))
        extractor = CrossAttentionSemanticMaskExtractor(
            token_indices,
            step_idx=semantic_steps,
            layer_idx=semantic_layers,
            max_res=args.semantic_attn_max_res,
            output_size=TARGET_IMG_SIZE,
            quantile=args.semantic_mask_quantile,
            gamma=args.semantic_mask_gamma,
            step_weight_mode=args.semantic_step_weight_mode,
            layer_weight_mode=args.semantic_layer_weight_mode,
            cross_layer_count=args.semantic_cross_layer_count,
            self_refine=args.semantic_self_refine,
            self_refine_weight=args.semantic_self_refine_weight,
            self_refine_iters=args.semantic_self_refine_iters,
            self_refine_max_res=args.semantic_self_refine_max_res,
            self_layer_idx=semantic_self_layers,
            self_layer_count=args.semantic_self_layer_count,
        )
        regiter_attention_editor_diffusers(model, extractor)
        print(
            "[InternalAttn] Collecting semantic masks: "
            f"prompt='{args.semantic_prompt}', concepts={list(token_indices.keys())}, "
            f"step_weight={args.semantic_step_weight_mode}, "
            f"layer_weight={args.semantic_layer_weight_mode}, "
            f"self_refine={args.semantic_self_refine}"
        )
        model(
            [args.semantic_prompt],
            latents=x_t,
            guidance_scale=7.5,
            uncond_embeddings=None,
            num_inference_steps=TOTAL_STEP,
            height=TARGET_IMG_SIZE,
            width=TARGET_IMG_SIZE,
        )

        prefix = os.path.splitext(os.path.basename(content))[0]
        mask_dir = os.path.join(SUB_EXP_NAME, "internal_attention_masks")
        paths = extractor.export_masks(mask_dir, prefix)
        internal_masks = []
        strength_by_concept = dict(concepts)
        for concept, mask_path in paths.items():
            internal_masks.append((concept, mask_path, strength_by_concept.get(concept, 1.0)))
        if internal_masks:
            print(
                "[InternalAttn] Exported masks: "
                + ", ".join(f"{name}={path}:{strength:.2f}" for name, path, strength in internal_masks)
            )
        else:
            print("[InternalAttn] No masks exported; internal protection disabled.")
        return internal_masks

    def save_reconstruction_only(content, x_t, uncond_embeddings, content_image, full_image_name):
        editor = AttentionBase()
        regiter_attention_editor_diffusers(model, editor)
        recon = model(
            [""],
            latents=x_t,
            guidance_scale=7.5,
            uncond_embeddings=uncond_embeddings,
            num_inference_steps=TOTAL_STEP,
            height=TARGET_IMG_SIZE,
            width=TARGET_IMG_SIZE,
        )
        recon_display = torch.nan_to_num(recon, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        save_image(
            torch.cat([content_image * 0.5 + 0.5, recon_display], dim=0),
            os.path.join(SUB_EXP_NAME, full_image_name + "_diagnostic_reconstruction_only.png"),
        )
        save_image(
            recon_display,
            os.path.join(SUB_EXP_NAME, full_image_name + "_content_reconstruction_only.png"),
        )

    def run_stylization(
        content,
        style,
        content_image,
        style_image,
        content_latent_list,
        style_latent_list,
        x_t,
        uncond_embeddings,
        full_image_name,
        output_suffix,
        start_step,
        end_step,
        content_style_scale,
        style_content_scale,
        eye_protect_strength,
        enable_protection=True,
        internal_masks=None,
        save_panel=True,
    ):
        prompts = ["", ""]
        start_code_content = x_t.expand(len(prompts), -1, -1, -1)
        focus_mask_path = args.attn_focus_mask
        if args.save_attn and not focus_mask_path:
            focus_mask_path = resolve_eye_mask(content) or sidecar_path(content, "_mask.npy")
        if args.save_attn and focus_mask_path:
            print(f"[Attn] Focus mask: {focus_mask_path}")

        editor = ReweightCrossAttentionControl(
            start_step,
            end_step,
            layer_idx=LAYER_INDEX,
            total_steps=TOTAL_STEP,
            content_img_name=content,
            save_attn_dir=attn_dir if args.save_attn else None,
            save_attn_name=f"{full_image_name}_{output_suffix}",
            save_attn_step=attn_step,
            save_attn_layer=attn_layer,
            save_attn_size=args.attn_size,
            save_attn_focus_mask=focus_mask_path,
            attn_max_res=args.attn_max_res,
            content_style_scale=content_style_scale,
            style_content_scale=style_content_scale,
            eye_mask_path=resolve_eye_mask(content),
            eye_protect_strength=eye_protect_strength,
            eye_mask_dilate=args.eye_mask_dilate,
            eye_mask_blur_sigma=args.eye_mask_blur_sigma,
            protect_masks=build_protect_masks(
                content,
                eye_protect_strength,
                enable_protection,
                internal_masks=internal_masks,
            ),
            spatial_adaptive_scale=args.spatial_adaptive_scale,
            spatial_adaptive_strength=args.spatial_adaptive_strength,
            spatial_adaptive_min_scale=args.spatial_adaptive_min_scale,
            spatial_adaptive_entropy_weight=args.spatial_adaptive_entropy_weight,
            spatial_adaptive_edge_weight=args.spatial_adaptive_edge_weight,
            spatial_adaptive_local_weight=args.spatial_adaptive_local_weight,
            spatial_adaptive_blur_sigma=args.spatial_adaptive_blur_sigma,
            spatial_adaptive_save_dir=(
                os.path.join(SUB_EXP_NAME, "spatial_adaptive_maps")
                if args.spatial_adaptive_save_maps else None
            ),
            spatial_adaptive_save_name=f"{full_image_name}_{output_suffix or 'main'}",
        )
        regiter_attention_editor_diffusers(model, editor)
        image_stylized = model(
            prompts,
            latents=start_code_content,
            guidance_scale=7.5,
            uncond_embeddings=uncond_embeddings,
            num_inference_steps=TOTAL_STEP,
            height=TARGET_IMG_SIZE,
            width=TARGET_IMG_SIZE,
            ref_intermediate_latents=[content_latent_list, style_latent_list],
        )
        style_display = torch.nan_to_num(
            image_stylized[0:1], nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        if style_display.max() < 1e-4:
            style_display = (style_image * 0.5 + 0.5).clamp(0.0, 1.0)
        recon_display = torch.nan_to_num(
            image_stylized[-1:], nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)

        if save_panel:
            out_image = torch.cat(
                [content_image * 0.5 + 0.5, style_display, recon_display],
                dim=0,
            )
            save_image(out_image, os.path.join(SUB_EXP_NAME, full_image_name + output_suffix + ".png"))
        save_image(
            recon_display,
            os.path.join(SUB_EXP_NAME, full_image_name + output_suffix + "_reconstructed.png"),
        )
        return style_display, recon_display

    for content in content_list:
        content_image = load_image(content, device)

        # --- CLIPSeg: generate localized mask from text ---
        if args.region_text and not args.no_clipseg:
            mask_path = sidecar_path(content, "_mask.npy")
            if not os.path.exists(mask_path):
                print(f"\n[CLIPSeg] Generating mask for text: '{args.region_text}' ...")
                prompts = segment_utils.parse_region_text(args.region_text)
                print(f"[CLIPSeg] Parsed prompts: {prompts}")
                mask = segment_utils.generate_mask_from_text(
                    image_path=content,
                    text_prompts=prompts,
                    target_size=TARGET_IMG_SIZE,
                    threshold=args.mask_threshold,
                    blur_sigma=args.blur_sigma,
                    blur_kernel=args.blur_kernel,
                    device=device,
                )
                segment_utils.save_mask_npy(mask, mask_path)
                # Visualize mask for inspection
                vis_path = mask_path.replace(".npy", "_vis.png")
                segment_utils.visualize_mask(mask, vis_path)
                pos_ratio = (mask > 0).sum() / mask.size
                print(f"[CLIPSeg] Mask saved: style_region={pos_ratio:.2%}")
            else:
                print(f"[CLIPSeg] Mask already cached: {mask_path}")
        # -------------------------------------------------

        source_prompt = ""
        target_prompt = ""
        prompts = [source_prompt, target_prompt]
        pickle_file_name = inversion_cache_path(content)
        if os.path.isfile(pickle_file_name) and not args.refresh_inversion_cache:
            with open(pickle_file_name, "rb") as f:
                pre_computed_data = pickle.load(f)
            content_latent_list = pre_computed_data[0]
            x_t = pre_computed_data[1]
            uncond_embeddings = pre_computed_data[2]
            if has_nan_embeddings(uncond_embeddings):
                print("[WARN] NaNs in cached uncond embeddings; falling back to default unconditional embeddings.")
                uncond_embeddings = None
        else:
            _, content_latent_list, x_t, uncond_embeddings = null_inversion.invert(
                content, prompts, verbose=True
            )
            if has_nan_embeddings(uncond_embeddings):
                print("[WARN] NaNs in uncond embeddings; falling back to default unconditional embeddings.")
                uncond_embeddings = None
            with open(pickle_file_name, "wb") as f:
                pickle.dump([content_latent_list, x_t, uncond_embeddings], f)
        internal_masks = collect_internal_attention_masks(content, x_t)
        for style in style_list:
            style_image = load_image(style, device)
            editor = AttentionBase()
            regiter_attention_editor_diffusers(model, editor)
            _, style_latent_list = model.invert(
                style_image,
                source_prompt,
                guidance_scale=7.5,
                num_inference_steps=TOTAL_STEP,
                return_intermediates=True,
            )
            full_image_name = (
                content.split("/")[-1][:-4] + "_" + style.split("/")[-1][:-4]
            )
            if args.diagnostic_triplet:
                save_reconstruction_only(content, x_t, uncond_embeddings, content_image, full_image_name)
                _, official_recon = run_stylization(
                    content, style, content_image, style_image, content_latent_list,
                    style_latent_list, x_t, uncond_embeddings, full_image_name,
                    "_diagnostic_full_zstar", 5, 30, 1.2, 1.5, 0.0, False,
                )
                _, late_recon = run_stylization(
                    content, style, content_image, style_image, content_latent_list,
                    style_latent_list, x_t, uncond_embeddings, full_image_name,
                    "_diagnostic_late_weak", args.diagnostic_late_start,
                    args.diagnostic_late_end, args.diagnostic_content_style_scale,
                    args.diagnostic_style_content_scale, args.eye_protect_strength, True,
                    internal_masks=internal_masks,
                )
                save_image(
                    torch.cat([content_image * 0.5 + 0.5, official_recon, late_recon], dim=0),
                    os.path.join(SUB_EXP_NAME, full_image_name + "_diagnostic_triplet.png"),
                )
                print("Diagnostic images are saved for ", full_image_name)
                continue

            style_display, recon_display = run_stylization(
                content, style, content_image, style_image, content_latent_list,
                style_latent_list, x_t, uncond_embeddings, full_image_name,
                "", args.start_step, args.end_step, args.content_style_scale,
                args.style_content_scale, args.eye_protect_strength,
                internal_masks=internal_masks,
            )
            save_image(
                content_image * 0.5 + 0.5,
                os.path.join(SUB_EXP_NAME, full_image_name + "_source.png"),
            )
            save_image(
                style_display, os.path.join(
                    SUB_EXP_NAME, full_image_name + "_style.png")
            )
            save_image(
                recon_display,
                os.path.join(SUB_EXP_NAME, full_image_name +
                             "_reconstructed.png"),
            )
            print("Syntheiszed images are saved in ", os.path.join(SUB_EXP_NAME, full_image_name + ".png"))


if __name__ == "__main__":
    main()
