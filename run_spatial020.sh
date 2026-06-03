#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/home/whc/.conda/envs/zstar310/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON_BIN" demo.py \
  --content_img_folder workdir/face_part_inputs/content \
  --style_img_folder workdir/eye_inputs/styles \
  --sub_exp_name workdir/reproduce_spatial020 \
  --start_step 5 \
  --end_step 30 \
  --layer_index 20,22,24,26,28,30 \
  --content_style_scale 1.50 \
  --style_content_scale 1.50 \
  --internal_attention_protect \
  --semantic_prompt 'a portrait photo with face eyes nose mouth lips hair background' \
  --semantic_protect_concepts 'eyes:0.85,mouth:0.75,lips:0.75,nose:0.55' \
  --semantic_collect_start 5 \
  --semantic_collect_end 20 \
  --semantic_mask_quantile 0.90 \
  --semantic_mask_gamma 1.0 \
  --semantic_step_weight_mode semantic \
  --semantic_layer_weight_mode semantic \
  --semantic_cross_layer_count 16 \
  --semantic_self_refine \
  --semantic_self_refine_weight 0.15 \
  --semantic_self_refine_iters 2 \
  --semantic_self_refine_max_res 35 \
  --semantic_self_layer_count 16 \
  --spatial_adaptive_scale \
  --spatial_adaptive_strength 0.20 \
  --spatial_adaptive_min_scale 0.75 \
  --eye_mask_dilate 3 \
  --eye_mask_blur_sigma 1.5
