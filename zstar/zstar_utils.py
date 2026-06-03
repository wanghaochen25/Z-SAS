import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from PIL import Image


class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(
        self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs
    ):
        out = self.forward(
            q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs
        )
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class AttentionStore(AttentionBase):
    def __init__(self, res=[32], min_step=0, max_step=1000):
        super().__init__()
        self.res = res
        self.min_step = min_step
        self.max_step = max_step
        self.valid_steps = 0

        self.self_attns = []  # store the all attns
        self.cross_attns = []

        self.self_attns_step = []  # store the attns in each step
        self.cross_attns_step = []

    def after_step(self):
        if self.cur_step > self.min_step and self.cur_step < self.max_step:
            self.valid_steps += 1
            if len(self.self_attns) == 0:
                self.self_attns = self.self_attns_step
                self.cross_attns = self.cross_attns_step
            else:
                for i in range(len(self.self_attns)):
                    self.self_attns[i] += self.self_attns_step[i]
                    self.cross_attns[i] += self.cross_attns_step[i]
        self.self_attns_step.clear()
        self.cross_attns_step.clear()

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        if attn.shape[1] <= 64**2:  # avoid OOM
            if is_cross:
                self.cross_attns_step.append(attn)
            else:
                self.self_attns_step.append(attn)
        return super().forward(
            q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs
        )


class CrossAttentionSemanticMaskExtractor(AttentionBase):
    def __init__(
        self,
        concept_token_indices,
        step_idx=None,
        layer_idx=None,
        max_res=64,
        output_size=560,
        content_index=-1,
        quantile=0.80,
        gamma=1.0,
        step_weight_mode="semantic",
        layer_weight_mode="semantic",
        cross_layer_count=16,
        self_refine=False,
        self_refine_weight=0.15,
        self_refine_iters=2,
        self_refine_max_res=35,
        self_layer_idx=None,
        self_layer_count=16,
    ):
        super().__init__()
        self.concept_token_indices = concept_token_indices
        self.step_idx = step_idx
        self.layer_idx = layer_idx
        self.max_res = max_res
        self.output_size = output_size
        self.content_index = content_index
        self.quantile = quantile
        self.gamma = gamma
        self.step_weight_mode = step_weight_mode
        self.layer_weight_mode = layer_weight_mode
        self.cross_layer_count = cross_layer_count
        self.cur_cross_layer = 0
        self.cur_self_layer = 0
        self.self_refine = self_refine
        self.self_refine_weight = float(max(0.0, min(1.0, self_refine_weight)))
        self.self_refine_iters = max(0, int(self_refine_iters))
        self.self_refine_max_res = max(1, int(self_refine_max_res))
        self.self_layer_idx = self_layer_idx
        self.self_layer_count = max(1, int(self_layer_count))
        self.maps = {name: [] for name in concept_token_indices}
        self.self_affinity_sums = {}

    def after_step(self):
        self.cur_cross_layer = 0
        self.cur_self_layer = 0

    def _step_weight(self):
        if self.step_weight_mode == "uniform" or not self.step_idx:
            return 1.0
        ordered_steps = sorted(self.step_idx)
        if self.cur_step not in ordered_steps:
            return 1.0
        denom = max(1, len(ordered_steps) - 1)
        progress = ordered_steps.index(self.cur_step) / denom
        if progress <= 0.33:
            return 1.0
        if progress <= 0.66:
            return 0.8
        return 0.45

    def _layer_weight(self, cross_layer):
        if self.layer_weight_mode == "uniform":
            return 1.0
        denom = max(1, self.cross_layer_count - 1)
        progress = cross_layer / denom
        if progress < 0.50:
            return 0.35
        if progress < 0.70:
            return 0.75
        return 1.0

    def _self_layer_weight(self, self_layer):
        if self.layer_weight_mode == "uniform":
            return 1.0
        denom = max(1, self.self_layer_count - 1)
        progress = self_layer / denom
        if progress < 0.50:
            return 0.35
        if progress < 0.70:
            return 0.75
        return 1.0

    def _normalize_map(self, mask):
        mask = mask - mask.min()
        max_value = mask.max()
        if max_value > 0:
            mask = mask / max_value
        if self.quantile is not None and self.quantile > 0:
            threshold = torch.quantile(mask.float(), min(max(self.quantile, 0.0), 0.99))
            if threshold < 1:
                mask = (mask - threshold) / (1.0 - threshold + 1e-8)
                mask = mask.clamp(0.0, 1.0)
        if self.gamma is not None and self.gamma > 0 and self.gamma != 1.0:
            mask = mask.clamp(0.0, 1.0) ** self.gamma
        return mask.clamp(0.0, 1.0)

    def _rescale_map(self, mask):
        mask = mask - mask.min()
        max_value = mask.max()
        if max_value > 0:
            mask = mask / max_value
        return mask.clamp(0.0, 1.0)

    def _store_self_affinity(self, affinity, spatial, weight):
        affinity = affinity.float()
        affinity = 0.5 * (affinity + affinity.transpose(0, 1))
        affinity = affinity / (affinity.sum(dim=-1, keepdim=True) + 1e-8)
        affinity = affinity.cpu() * weight
        if spatial not in self.self_affinity_sums:
            self.self_affinity_sums[spatial] = [affinity, float(weight)]
        else:
            self.self_affinity_sums[spatial][0] += affinity
            self.self_affinity_sums[spatial][1] += float(weight)

    def _refine_with_self_attention(self, seed_mask):
        if (
            not self.self_refine
            or self.self_refine_weight <= 0
            or self.self_refine_iters <= 0
            or not self.self_affinity_sums
        ):
            return seed_mask

        refined_masks = []
        weights = []
        for spatial, (affinity_sum, weight_sum) in self.self_affinity_sums.items():
            if weight_sum <= 0:
                continue
            affinity = affinity_sum / weight_sum
            low_mask = F.interpolate(
                seed_mask.reshape(1, 1, self.output_size, self.output_size).float(),
                size=(spatial, spatial),
                mode="bilinear",
                align_corners=False,
            ).reshape(spatial * spatial, 1)
            propagated = low_mask
            for _ in range(self.self_refine_iters):
                propagated = affinity.matmul(propagated)
                propagated = self._rescale_map(propagated)
                propagated = 0.5 * low_mask + 0.5 * propagated
            high_mask = F.interpolate(
                propagated.reshape(1, 1, spatial, spatial),
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            refined_masks.append(high_mask * weight_sum)
            weights.append(weight_sum)

        if not refined_masks:
            return seed_mask
        refined = torch.stack(refined_masks, dim=0).sum(dim=0) / max(1e-8, float(sum(weights)))
        refined = self._rescale_map(refined)
        return (
            (1.0 - self.self_refine_weight) * seed_mask
            + self.self_refine_weight * refined
        ).clamp(0.0, 1.0)

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        cross_layer = self.cur_cross_layer
        self_layer = self.cur_self_layer
        if (
            is_cross
            and (self.step_idx is None or self.cur_step in self.step_idx)
            and (self.layer_idx is None or cross_layer in self.layer_idx)
            and attn.shape[1] <= self.max_res ** 2
        ):
            pixel_size = attn.shape[1]
            spatial = int(pixel_size ** 0.5)
            if spatial * spatial == pixel_size:
                with torch.no_grad():
                    batch = attn.shape[0] // num_heads
                    content_index = self.content_index
                    if content_index < 0 or content_index >= batch:
                        content_index = batch - 1
                    attn_view = attn.detach().reshape(
                        batch, num_heads, pixel_size, attn.shape[-1]
                    )
                    token_attn = attn_view[content_index].mean(dim=0)
                    for name, indices in self.concept_token_indices.items():
                        valid_indices = [idx for idx in indices if idx < token_attn.shape[-1]]
                        if not valid_indices:
                            continue
                        mask = token_attn[:, valid_indices].mean(dim=-1).reshape(1, 1, spatial, spatial)
                        mask = F.interpolate(
                            mask.float(),
                            size=(self.output_size, self.output_size),
                            mode="bilinear",
                            align_corners=False,
                        )[0, 0]
                        weight = self._step_weight() * self._layer_weight(cross_layer)
                        self.maps[name].append((mask.cpu(), float(weight)))

        if is_cross:
            self.cur_cross_layer += 1
        else:
            if (
                self.self_refine
                and (self.step_idx is None or self.cur_step in self.step_idx)
                and (self.self_layer_idx is None or self_layer in self.self_layer_idx)
                and attn.shape[1] <= self.self_refine_max_res ** 2
            ):
                pixel_size = attn.shape[1]
                spatial = int(pixel_size ** 0.5)
                if spatial * spatial == pixel_size:
                    with torch.no_grad():
                        batch = attn.shape[0] // num_heads
                        content_index = self.content_index
                        if content_index < 0 or content_index >= batch:
                            content_index = batch - 1
                        attn_view = attn.detach().reshape(
                            batch, num_heads, pixel_size, pixel_size
                        )
                        affinity = attn_view[content_index].mean(dim=0)
                        weight = self._step_weight() * self._self_layer_weight(self_layer)
                        self._store_self_affinity(affinity, spatial, weight)
            self.cur_self_layer += 1
        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

    def export_masks(self, output_dir, prefix):
        os.makedirs(output_dir, exist_ok=True)
        paths = {}
        for name, masks in self.maps.items():
            if not masks:
                continue
            weighted_masks = []
            weights = []
            for item in masks:
                if isinstance(item, tuple):
                    item_mask, item_weight = item
                else:
                    item_mask, item_weight = item, 1.0
                weighted_masks.append(item_mask * item_weight)
                weights.append(item_weight)
            weight_sum = max(1e-8, float(sum(weights)))
            mask = torch.stack(weighted_masks, dim=0).sum(dim=0) / weight_sum
            mask = self._normalize_map(mask)
            mask = self._refine_with_self_attention(mask)
            mask = self._rescale_map(mask).numpy().astype(np.float32)
            npy_path = os.path.join(output_dir, f"{prefix}_{name}_internal_attn_mask.npy")
            png_path = os.path.join(output_dir, f"{prefix}_{name}_internal_attn_mask.png")
            np.save(npy_path, mask)
            Image.fromarray((mask * 255.0).astype(np.uint8)).save(png_path)
            paths[name] = npy_path
        return paths


def regiter_attention_editor_diffusers(model, editor: AttentionBase):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """

    def ca_forward(self, place_in_unet):
        def forward(
            x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None
        ):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            context = context.to(dtype=q.dtype)
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(
                lambda t: rearrange(
                    t, "b n (h d) -> (b h) n d", h=h), (q, k, v)
            )

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, "b ... -> b (...)")
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, "b j -> (b h) () j", h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # the only difference
            out = editor(
                q,
                k,
                v,
                sim,
                attn,
                is_cross,
                place_in_unet,
                self.heads,
                scale=self.scale,
            )

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == "Attention":  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, "children"):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count


def regiter_attention_editor_ldm(model, editor: AttentionBase):
    """
    Register a attention editor to Stable Diffusion model, refer from [Prompt-to-Prompt]
    """

    def ca_forward(self, place_in_unet):
        def forward(
            x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None
        ):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            context = context.to(dtype=q.dtype)
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(
                lambda t: rearrange(
                    t, "b n (h d) -> (b h) n d", h=h), (q, k, v)
            )

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, "b ... -> b (...)")
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, "b j -> (b h) () j", h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # the only difference
            out = editor(
                q,
                k,
                v,
                sim,
                attn,
                is_cross,
                place_in_unet,
                self.heads,
                scale=self.scale,
            )

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == "CrossAttention":  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, "children"):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.model.diffusion_model.named_children():
        if "input" in net_name:
            cross_att_count += register_editor(net, 0, "input")
        elif "middle" in net_name:
            cross_att_count += register_editor(net, 0, "middle")
        elif "output" in net_name:
            cross_att_count += register_editor(net, 0, "output")
    editor.num_att_layers = cross_att_count
