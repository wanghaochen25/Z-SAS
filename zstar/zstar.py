import os
from math import sqrt

import torch
import torch.nn.functional as F
import numpy as np
import shutil

from PIL import Image

from einops import rearrange

from .zstar_utils import AttentionBase


class ReweightCrossAttentionControl(AttentionBase):
    def __init__(
        self,
        start_step=4,
        end_step=1000,
        start_layer=10,
        end_layer=1000,
        layer_idx=None,
        step_idx=None,
        total_steps=50,
        content_img_name=None,
        save_attn_dir=None,
        save_attn_name=None,
        save_attn_step=None,
        save_attn_layer=None,
        save_attn_size=None,
        save_attn_focus_mask=None,
        attn_max_res=64,
        attn_content_index=-1,
        content_style_scale=1.2,
        style_content_scale=1.5,
        eye_mask_path=None,
        eye_protect_strength=0.0,
        eye_mask_dilate=3,
        eye_mask_blur_sigma=1.5,
        protect_masks=None,
        spatial_adaptive_scale=False,
        spatial_adaptive_strength=0.35,
        spatial_adaptive_min_scale=0.65,
        spatial_adaptive_entropy_weight=0.35,
        spatial_adaptive_edge_weight=0.55,
        spatial_adaptive_local_weight=0.10,
        spatial_adaptive_blur_sigma=1.0,
        spatial_adaptive_save_dir=None,
        spatial_adaptive_save_name=None,
    ):
        """
        Args:
            start_step: the step to start Cross-attention Reweighting
            start_layer: the layer to start Cross-attention Reweighting
            layer_idx: list of the layers to apply Cross-attention Reweighting
            step_idx: list the steps to apply Cross-attention Reweighting
            total_steps: the total number of steps
        """
        super().__init__()
        self.total_layers = 16
        self.total_steps = total_steps
        self.start_step = max(0, start_step)
        self.end_step = min(end_step, total_steps)
        self.start_layer = max(0, start_layer)
        self.end_layer = min(end_layer, self.total_layers)
        self.layer_idx = layer_idx if layer_idx is not None else list(
            range(self.start_layer, self.end_layer))
        self.step_idx = step_idx if step_idx is not None else list(
            range(self.start_step, self.end_step))
        self.content_img_name = content_img_name
        self.save_attn_dir = save_attn_dir
        self.save_attn_name = save_attn_name
        self.save_attn_step = save_attn_step
        self.save_attn_layer = save_attn_layer
        self.save_attn_size = save_attn_size
        self.save_attn_focus_mask = save_attn_focus_mask
        self.attn_max_res = attn_max_res
        self.attn_content_index = attn_content_index
        self.content_style_scale = content_style_scale
        self.style_content_scale = style_content_scale
        self.eye_mask_path = eye_mask_path
        self.eye_protect_strength = float(max(0.0, min(1.0, eye_protect_strength)))
        self.eye_mask_dilate = max(0, int(eye_mask_dilate))
        self.eye_mask_blur_sigma = max(0.0, float(eye_mask_blur_sigma))
        self.protect_masks = self._normalize_protect_masks(protect_masks)
        self.spatial_adaptive_scale = spatial_adaptive_scale
        self.spatial_adaptive_strength = float(max(0.0, min(1.0, spatial_adaptive_strength)))
        self.spatial_adaptive_min_scale = float(max(0.0, min(1.0, spatial_adaptive_min_scale)))
        self.spatial_adaptive_entropy_weight = max(0.0, float(spatial_adaptive_entropy_weight))
        self.spatial_adaptive_edge_weight = max(0.0, float(spatial_adaptive_edge_weight))
        self.spatial_adaptive_local_weight = max(0.0, float(spatial_adaptive_local_weight))
        self.spatial_adaptive_blur_sigma = max(0.0, float(spatial_adaptive_blur_sigma))
        self.spatial_adaptive_save_dir = spatial_adaptive_save_dir
        self.spatial_adaptive_save_name = spatial_adaptive_save_name
        self._spatial_risk_saved = set()
        self._local_window_cache = {}
        self._attn_saved = False
        self._focus_mask_cache = None
        self._protect_mask_cache = {}
        print("step_idx: ", self.step_idx)
        print("layer_idx: ", self.layer_idx)
        print(
            "attention scales: "
            f"content_style={self.content_style_scale}, "
            f"style_content={self.style_content_scale}, "
            f"protect_regions={self._protect_region_summary()}, "
            f"spatial_adaptive={self.spatial_adaptive_scale}"
        )

    def _normalize_protect_masks(self, protect_masks):
        normalized = []
        if protect_masks:
            normalized.extend(protect_masks)
        elif self.eye_protect_strength > 0:
            normalized.append({
                "name": "eyes",
                "path": self.eye_mask_path,
                "suffixes": ["_eyes_mask.npy"],
                "strength": self.eye_protect_strength,
                "dilate": self.eye_mask_dilate,
                "blur_sigma": self.eye_mask_blur_sigma,
            })

        output = []
        for item in normalized:
            strength = float(item.get("strength", 0.0))
            if strength <= 0:
                continue
            output.append({
                "name": item.get("name", "region"),
                "path": item.get("path"),
                "suffixes": item.get("suffixes", []),
                "strength": float(max(0.0, min(1.0, strength))),
                "dilate": max(0, int(item.get("dilate", self.eye_mask_dilate))),
                "blur_sigma": max(0.0, float(item.get("blur_sigma", self.eye_mask_blur_sigma))),
            })
        return output

    def _protect_region_summary(self):
        if not self.protect_masks:
            return "none"
        return ",".join(
            f"{item['name']}:{item['strength']:.2f}"
            for item in self.protect_masks
        )

    def _mask_path_for_suffix(self, suffix):
        if self.content_img_name is None:
            return None
        return (
            self.content_img_name
            .replace(".jpg", suffix)
            .replace(".png", suffix)
            .replace(".jpeg", suffix)
            .replace(".bmp", suffix)
        )

    def _get_focus_mask(self, spatial, device):
        if not self.save_attn_focus_mask:
            return None
        if self._focus_mask_cache is not None:
            cached = self._focus_mask_cache
            if cached.shape[-1] == spatial and cached.shape[-2] == spatial:
                return cached.to(device)
        if not os.path.exists(self.save_attn_focus_mask):
            return None
        if not self.save_attn_focus_mask.lower().endswith(".npy"):
            return None

        mask_np = np.load(self.save_attn_focus_mask).astype(np.float32)
        mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, size=(spatial, spatial), mode="bilinear", align_corners=False)
        mask = (mask > 0).float()
        self._focus_mask_cache = mask.cpu()
        return mask.to(device)

    def _resolve_mask_path(self, region):
        if region.get("path"):
            return region["path"]
        for suffix in region.get("suffixes", []):
            candidate = self._mask_path_for_suffix(suffix)
            if candidate and os.path.exists(candidate):
                return candidate
        return None

    def _load_soft_mask(self, mask_path, spatial, device, dilate, blur_sigma):
        if not mask_path or not os.path.exists(mask_path) or not mask_path.lower().endswith(".npy"):
            return None
        cache_key = (mask_path, spatial, dilate, blur_sigma)
        if cache_key in self._protect_mask_cache:
            return self._protect_mask_cache[cache_key].to(device)

        mask_np = np.load(mask_path).astype(np.float32)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        if mask_np.max() > 1.0 or mask_np.min() < 0.0:
            mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-8)
        mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float()

        if dilate > 0:
            kernel = dilate * 2 + 1
            mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=dilate)
        mask = F.interpolate(mask, size=(spatial, spatial), mode="bilinear", align_corners=False)
        if blur_sigma > 0:
            radius = max(1, int(blur_sigma * 3))
            kernel_size = radius * 2 + 1
            coords = torch.arange(kernel_size, dtype=mask.dtype) - radius
            kernel_1d = torch.exp(-(coords ** 2) / (2 * blur_sigma ** 2))
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_y = kernel_1d.view(1, 1, kernel_size, 1)
            kernel_x = kernel_1d.view(1, 1, 1, kernel_size)
            mask = F.pad(mask, (radius, radius, 0, 0), mode="reflect")
            mask = F.conv2d(mask, kernel_x)
            mask = F.pad(mask, (0, 0, radius, radius), mode="reflect")
            mask = F.conv2d(mask, kernel_y)
        mask = mask.clamp(0.0, 1.0).cpu()
        self._protect_mask_cache[cache_key] = mask
        return mask.to(device)

    def _get_protect_mask(self, pixel_size, device):
        if not self.protect_masks:
            return None
        spatial = int(sqrt(pixel_size))
        if spatial * spatial != pixel_size:
            return None

        combined_mask = None
        for region in self.protect_masks:
            mask_path = self._resolve_mask_path(region)
            mask = self._load_soft_mask(
                mask_path,
                spatial,
                device,
                region["dilate"],
                region["blur_sigma"],
            )
            if mask is None:
                continue
            mask = mask * region["strength"]
            combined_mask = mask if combined_mask is None else torch.maximum(combined_mask, mask)

        if combined_mask is None:
            return None
        return combined_mask.reshape(1, pixel_size, 1).clamp(0.0, 1.0)

    def _normalize_token_map(self, token_map):
        token_map = token_map.float()
        token_map = token_map - token_map.min()
        max_value = token_map.max()
        if max_value > 0:
            token_map = token_map / (max_value + 1e-8)
        return token_map.clamp(0.0, 1.0)

    def _blur_token_map(self, token_map, spatial):
        if self.spatial_adaptive_blur_sigma <= 0:
            return token_map
        sigma = self.spatial_adaptive_blur_sigma
        radius = max(1, int(sigma * 3))
        kernel_size = radius * 2 + 1
        token_dtype = token_map.dtype
        token_device = token_map.device
        coords = torch.arange(kernel_size, dtype=token_dtype, device=token_device) - radius
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_y = kernel_1d.view(1, 1, kernel_size, 1)
        kernel_x = kernel_1d.view(1, 1, 1, kernel_size)
        mask = token_map.reshape(1, 1, spatial, spatial)
        mask = F.pad(mask, (radius, radius, 0, 0), mode="reflect")
        mask = F.conv2d(mask, kernel_x)
        mask = F.pad(mask, (0, 0, radius, radius), mode="reflect")
        mask = F.conv2d(mask, kernel_y)
        return mask.reshape(spatial * spatial)

    def _local_window_mask(self, spatial, device):
        cache_key = (spatial, device)
        if cache_key in self._local_window_cache:
            return self._local_window_cache[cache_key]
        coords_y, coords_x = torch.meshgrid(
            torch.arange(spatial, device=device),
            torch.arange(spatial, device=device),
            indexing="ij",
        )
        coords = torch.stack([coords_y.reshape(-1), coords_x.reshape(-1)], dim=-1)
        delta = coords[:, None, :] - coords[None, :, :]
        mask = (delta.abs().amax(dim=-1) <= 1).float()
        self._local_window_cache[cache_key] = mask
        return mask

    def _feature_edge_map(self, q_content, spatial):
        feat = q_content.detach().mean(dim=0)
        feat = F.normalize(feat.float(), dim=-1)
        feat = feat.reshape(spatial, spatial, -1)
        dx = torch.zeros(spatial, spatial, device=feat.device, dtype=feat.dtype)
        dy = torch.zeros_like(dx)
        dx[:, 1:] = (feat[:, 1:] - feat[:, :-1]).pow(2).mean(dim=-1).sqrt()
        dy[1:, :] = (feat[1:, :] - feat[:-1, :]).pow(2).mean(dim=-1).sqrt()
        return self._normalize_token_map((dx + dy).reshape(spatial * spatial))

    def _spatial_structure_risk(self, content_sim, q_content):
        if not self.spatial_adaptive_scale or self.spatial_adaptive_strength <= 0:
            return None
        pixel_size = content_sim.shape[1]
        spatial = int(sqrt(pixel_size))
        if spatial * spatial != pixel_size:
            return None

        with torch.no_grad():
            attn_prob = content_sim.detach().float().softmax(dim=-1)
            entropy = -(attn_prob * (attn_prob + 1e-8).log()).sum(dim=-1)
            entropy = entropy / np.log(max(2, pixel_size))
            entropy_focus = self._normalize_token_map(1.0 - entropy.mean(dim=0))

            local_mask = self._local_window_mask(spatial, content_sim.device).to(attn_prob.dtype)
            local_mass = (attn_prob * local_mask.unsqueeze(0)).sum(dim=-1).mean(dim=0)
            local_risk = self._normalize_token_map(local_mass)

            edge_risk = self._feature_edge_map(q_content, spatial)

            total_weight = (
                self.spatial_adaptive_entropy_weight
                + self.spatial_adaptive_edge_weight
                + self.spatial_adaptive_local_weight
            )
            if total_weight <= 0:
                return None
            risk = (
                self.spatial_adaptive_entropy_weight * entropy_focus
                + self.spatial_adaptive_edge_weight * edge_risk
                + self.spatial_adaptive_local_weight * local_risk
            ) / total_weight
            risk = self._normalize_token_map(risk)
            risk = self._blur_token_map(risk, spatial)
            risk = self._normalize_token_map(risk)
            self._maybe_save_spatial_risk(risk, spatial)
            return risk.to(content_sim.device)

    def _spatial_scale_map(self, structure_risk):
        if structure_risk is None:
            return None
        scale = 1.0 - self.spatial_adaptive_strength * structure_risk
        scale = scale.clamp(self.spatial_adaptive_min_scale, 1.0)
        return scale.reshape(1, -1, 1)

    def _maybe_save_spatial_risk(self, risk, spatial):
        if not self.spatial_adaptive_save_dir:
            return
        if len(self._spatial_risk_saved) >= 12:
            return
        save_key = (self.cur_step, self.cur_att_layer, spatial)
        if save_key in self._spatial_risk_saved:
            return
        self._spatial_risk_saved.add(save_key)
        os.makedirs(self.spatial_adaptive_save_dir, exist_ok=True)
        risk_map = risk.reshape(spatial, spatial)
        risk_map = self._normalize_token_map(risk_map)
        risk_img = (risk_map.float().cpu().numpy() * 255.0).astype(np.uint8)
        image = Image.fromarray(risk_img).resize((560, 560), resample=Image.BILINEAR)
        name = self.spatial_adaptive_save_name or "structure_risk"
        image.save(
            os.path.join(
                self.spatial_adaptive_save_dir,
                f"{name}_step{self.cur_step}_layer{self.cur_att_layer}_res{spatial}.png",
            )
        )

    def _maybe_save_attention(self, attn, num_heads):
        if self._attn_saved or self.save_attn_dir is None:
            return
        if self.save_attn_step is not None and self.cur_step != self.save_attn_step:
            return
        if self.save_attn_layer is not None and self.cur_att_layer != self.save_attn_layer:
            return
        if attn.shape[1] > self.attn_max_res ** 2:
            return

        with torch.no_grad():
            b = attn.shape[0] // num_heads
            attn_view = attn.detach().reshape(b, num_heads, attn.shape[1], attn.shape[2])
            content_index = self.attn_content_index
            if content_index < 0 or content_index >= b:
                content_index = b - 1
            content_attn = attn_view[content_index]
            attn_map = content_attn.mean(dim=0).mean(dim=0)
            spatial = int(sqrt(attn_map.shape[0]))
            if spatial * spatial != attn_map.shape[0]:
                return
            attn_map = attn_map.reshape(spatial, spatial)
            attn_map = attn_map - attn_map.min()
            max_val = attn_map.max()
            if max_val > 0:
                attn_map = attn_map / max_val
            attn_img = (attn_map.float().cpu().numpy() * 255.0).astype(np.uint8)

            focus_img = None
            focus_mask = self._get_focus_mask(spatial, attn.device)
            if focus_mask is not None:
                mask_flat = focus_mask.reshape(-1)
                mask_sum = mask_flat.sum()
                if mask_sum > 0:
                    focus_map = (content_attn * mask_flat[None, :, None]).sum(dim=1) / mask_sum
                    focus_map = focus_map.mean(dim=0)
                    focus_map = focus_map.reshape(spatial, spatial)
                    focus_map = focus_map - focus_map.min()
                    focus_max = focus_map.max()
                    if focus_max > 0:
                        focus_map = focus_map / focus_max
                    focus_img = (focus_map.float().cpu().numpy() * 255.0).astype(np.uint8)

        os.makedirs(self.save_attn_dir, exist_ok=True)
        image = Image.fromarray(attn_img)
        if self.save_attn_size is not None:
            image = image.resize((self.save_attn_size, self.save_attn_size), resample=Image.BILINEAR)
        name = self.save_attn_name or "attn"
        out_path = os.path.join(
            self.save_attn_dir,
            f"{name}_step{self.cur_step}_layer{self.cur_att_layer}.png",
        )
        image.save(out_path)
        if focus_img is not None:
            focus_image = Image.fromarray(focus_img)
            if self.save_attn_size is not None:
                focus_image = focus_image.resize(
                    (self.save_attn_size, self.save_attn_size), resample=Image.BILINEAR
                )
            focus_path = os.path.join(
                self.save_attn_dir,
                f"{name}_step{self.cur_step}_layer{self.cur_att_layer}_focus.png",
            )
            focus_image.save(focus_path)
        self._attn_saved = True

    def get_batch_sim(self, type, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        return sim

    def get_batch_sim_with_mask(self, type, cc_sim, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        head_num = sim.shape[0]
        pixel_size = sim.shape[1]
        h = w = int(sqrt(pixel_size))
        sim_reshaped = sim.reshape(head_num, h, w, pixel_size)
        cc_sim_reshaped = cc_sim.reshape(head_num, h, w, pixel_size)
        min_cc_sim_reshaped, _ = torch.min(
            cc_sim_reshaped, dim=3, keepdim=True)
        max_sim_reshaped, _ = torch.max(sim_reshaped, dim=3, keepdim=True)
        start = 0.5
        end = -0.5
        length = w
        if self.content_img_name is not None:
            mask_path = self.content_img_name.replace(".jpg", "_mask.npy").replace(".png", "_mask.npy").replace(".jpeg", "_mask.npy").replace(".bmp", "_mask.npy")
            mask = torch.tensor(np.load(mask_path), dtype=torch.float32).cuda()
            mask = torch.clamp(mask, -1.0, 1.0)
            mask *= -1.0
        else:
            print("ERROR: mask npy not found!!!")
            mask = torch.tensor([[1., 1., 1., 1.],
                                [1., -1., -1., 1.],
                                [1., -1., -1., 1.],
                                [-1., -1., -1., -1.]], dtype=torch.float32).cuda()
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, size=(
            h, w), mode='bilinear', align_corners=False)
        gradual_vanished_array = mask.reshape(1, h, w, 1).to(sim.device)
        gradual_vanished_mask = (
            min_cc_sim_reshaped - max_sim_reshaped)[:, :, :, :] * gradual_vanished_array
        sim_reshaped[:, :length, :, :] += gradual_vanished_mask
        sim = sim_reshaped.reshape(head_num, pixel_size, pixel_size)
        return sim

    # latest
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if not is_cross:
            self._maybe_save_attention(attn, num_heads)
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        style_style_out_u_sim = self.get_batch_sim(
            "ss", qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        style_style_out_c_sim = self.get_batch_sim(
            "ss", qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        content_content_out_u_sim = self.get_batch_sim(
            "cc", qu[-num_heads:], ku[-num_heads:], vu[-num_heads:], sim[-num_heads:], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        content_content_out_c_sim = self.get_batch_sim(
            "cc", qc[-num_heads:], kc[-num_heads:], vc[-num_heads:], sim[-num_heads:], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        style_content_out_u_sim = self.get_batch_sim(
            "sc", qu[:num_heads], ku[-num_heads:], vu[-num_heads:], sim[-num_heads:], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        style_content_out_c_sim = self.get_batch_sim(
            "sc", qc[:num_heads], kc[-num_heads:], vc[-num_heads:], sim[-num_heads:], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        mask_path = None
        if self.content_img_name is not None:
            mask_path = (
                self.content_img_name
                .replace(".jpg", "_mask.npy")
                .replace(".png", "_mask.npy")
                .replace(".jpeg", "_mask.npy")
                .replace(".bmp", "_mask.npy")
            )
        if mask_path and os.path.exists(mask_path):
            # Apply spatial mask to content->style attention for localized transfer.
            content_style_out_u_sim = self.get_batch_sim_with_mask(
                "cs",
                content_content_out_u_sim,
                qu[-num_heads:],
                ku[:num_heads],
                vu[:num_heads],
                sim[:num_heads],
                attnu,
                is_cross,
                place_in_unet,
                num_heads,
                **kwargs,
            )
            content_style_out_c_sim = self.get_batch_sim_with_mask(
                "cs",
                content_content_out_c_sim,
                qc[-num_heads:],
                kc[:num_heads],
                vc[:num_heads],
                sim[:num_heads],
                attnc,
                is_cross,
                place_in_unet,
                num_heads,
                **kwargs,
            )
        else:
            content_style_out_u_sim = self.get_batch_sim(
                "cs", qu[-num_heads:], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            content_style_out_c_sim = self.get_batch_sim(
                "cs", qc[-num_heads:], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        structure_risk_u = self._spatial_structure_risk(
            content_content_out_u_sim,
            qu[-num_heads:],
        )
        structure_risk_c = self._spatial_structure_risk(
            content_content_out_c_sim,
            qc[-num_heads:],
        )
        spatial_scale_u = self._spatial_scale_map(structure_risk_u)
        spatial_scale_c = self._spatial_scale_map(structure_risk_c)
        if spatial_scale_u is not None:
            content_style_out_u_sim *= spatial_scale_u.to(content_style_out_u_sim.dtype)
        if spatial_scale_c is not None:
            content_style_out_c_sim *= spatial_scale_c.to(content_style_out_c_sim.dtype)

        content_style_out_u_sim *= self.content_style_scale
        content_style_out_c_sim *= self.content_style_scale

        style_content_out_u_sim *= self.style_content_scale
        style_content_out_c_sim *= self.style_content_scale

        b = qu[-num_heads:].shape[0] // num_heads
        pixel_size = content_content_out_u_sim.shape[1]
        protect_mask = self._get_protect_mask(pixel_size, content_content_out_u_sim.device)

        content_style_content_content_out_u_sim = torch.cat(
            (content_style_out_u_sim, content_content_out_u_sim), 2)
        content_style_content_content_out_c_sim = torch.cat(
            (content_style_out_c_sim, content_content_out_c_sim), 2)
        vu_cscc_concat = torch.cat((vu[:num_heads], vu[-num_heads:]), 1)
        vc_cscc_concat = torch.cat((vc[:num_heads], vc[-num_heads:]), 1)

        style_content_style_style_out_u_sim = torch.cat(
            (style_content_out_u_sim, style_style_out_u_sim), 2)
        style_content_style_style_out_c_sim = torch.cat(
            (style_content_out_c_sim, style_style_out_c_sim), 2)
        vu_scss_concat = torch.cat((vu[-num_heads:], vu[:num_heads]), 1)
        vc_scss_concat = torch.cat((vc[-num_heads:], vc[:num_heads]), 1)
        content_style_content_content_out_u_sim = content_style_content_content_out_u_sim.softmax(
            -1)
        content_style_content_content_out_c_sim = content_style_content_content_out_c_sim.softmax(
            -1)

        mixup_out_u = torch.einsum(
            "h i j, h j d -> h i d", content_style_content_content_out_u_sim, vu_cscc_concat)
        mixup_out_u = rearrange(mixup_out_u, "h (b n) d -> b n (h d)", b=b)
        mixup_out_c = torch.einsum(
            "h i j, h j d -> h i d", content_style_content_content_out_c_sim, vc_cscc_concat)
        mixup_out_c = rearrange(mixup_out_c, "h (b n) d -> b n (h d)", b=b)

        if protect_mask is not None:
            content_preserve_out_u_sim = content_content_out_u_sim.softmax(-1)
            content_preserve_out_c_sim = content_content_out_c_sim.softmax(-1)
            preserve_out_u = torch.einsum(
                "h i j, h j d -> h i d", content_preserve_out_u_sim, vu[-num_heads:])
            preserve_out_u = rearrange(preserve_out_u, "h (b n) d -> b n (h d)", b=b)
            preserve_out_c = torch.einsum(
                "h i j, h j d -> h i d", content_preserve_out_c_sim, vc[-num_heads:])
            preserve_out_c = rearrange(preserve_out_c, "h (b n) d -> b n (h d)", b=b)
            protect_mask = protect_mask.to(dtype=mixup_out_u.dtype)
            mixup_out_u = protect_mask * preserve_out_u + (1.0 - protect_mask) * mixup_out_u
            mixup_out_c = protect_mask * preserve_out_c + (1.0 - protect_mask) * mixup_out_c

        style_content_style_style_out_u_sim = style_content_style_style_out_u_sim.softmax(
            -1)
        style_content_style_style_out_c_sim = style_content_style_style_out_c_sim.softmax(
            -1)
        original_out_u = torch.einsum(
            "h i j, h j d -> h i d", style_content_style_style_out_u_sim, vu_scss_concat)
        original_out_u = rearrange(
            original_out_u, "h (b n) d -> b n (h d)", b=b)
        original_out_c = torch.einsum(
            "h i j, h j d -> h i d", style_content_style_style_out_c_sim, vc_scss_concat)
        original_out_c = rearrange(
            original_out_c, "h (b n) d -> b n (h d)", b=b)

        out = torch.cat([original_out_u, mixup_out_u,
                        original_out_c, mixup_out_c], dim=0)
        return out
