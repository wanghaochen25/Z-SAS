#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/home/whc/.conda/envs/zstar310/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STYLE_DIR="workdir/ablation_v2/inputs/styles_demo1_wave"
PORTRAIT_DIR="workdir/ablation_v2/inputs/portrait_content"
LANDSCAPE_DIR="workdir/ablation_v2/inputs/landscape_content"
OUT_ROOT="workdir/ablation_v2/runs"

COMMON_ARGS=(
  --style_img_folder "$STYLE_DIR"
  --start_step 5
  --end_step 30
  --layer_index 20,22,24,26,28,30
  --content_style_scale 1.50
  --style_content_scale 1.50
  --eye_mask_dilate 3
  --eye_mask_blur_sigma 1.5
  --save_attn
  --attn_step 5
  --attn_layer 20
  --attn_size 560
)

INTERNAL_ARGS=(
  --internal_attention_protect
  --semantic_prompt "a portrait photo with face eyes nose mouth lips hair background"
  --semantic_protect_concepts "eyes:0.85,mouth:0.75,lips:0.75,nose:0.55"
  --semantic_collect_start 5
  --semantic_collect_end 20
  --semantic_mask_quantile 0.90
  --semantic_mask_gamma 1.0
  --semantic_step_weight_mode semantic
  --semantic_layer_weight_mode semantic
  --semantic_cross_layer_count 16
  --semantic_self_refine
  --semantic_self_refine_weight 0.15
  --semantic_self_refine_iters 2
  --semantic_self_refine_max_res 35
  --semantic_self_layer_count 16
)

SPATIAL020_ARGS=(
  --spatial_adaptive_scale
  --spatial_adaptive_strength 0.20
  --spatial_adaptive_min_scale 0.75
  --spatial_adaptive_save_maps
)

run_case() {
  local group="$1"
  local name="$2"
  local content_dir="$3"
  shift 3
  echo "===== Running ${group}: ${name} ====="
  "$PYTHON_BIN" demo.py \
    "${COMMON_ARGS[@]}" \
    --content_img_folder "$content_dir" \
    --sub_exp_name "${OUT_ROOT}/${group}/${name}" \
    "$@"
}

# Portrait: compare all face-aware variants.
run_case "portrait" "P0_direct_zstar" "$PORTRAIT_DIR"

run_case "portrait" "P1_internal_attention" "$PORTRAIT_DIR" \
  "${INTERNAL_ARGS[@]}"

run_case "portrait" "P2_spatial020_only" "$PORTRAIT_DIR" \
  "${SPATIAL020_ARGS[@]}"

run_case "portrait" "P3_internal_spatial020" "$PORTRAIT_DIR" \
  "${INTERNAL_ARGS[@]}" \
  "${SPATIAL020_ARGS[@]}"

run_case "portrait" "P4_mediapipe_masks" "$PORTRAIT_DIR" \
  --face_protect \
  --face_eye_protect_strength 0.85 \
  --mouth_protect_strength 0.75 \
  --nose_protect_strength 0.55

run_case "portrait" "P5_mediapipe_spatial020" "$PORTRAIT_DIR" \
  --face_protect \
  --face_eye_protect_strength 0.85 \
  --mouth_protect_strength 0.75 \
  --nose_protect_strength 0.55 \
  "${SPATIAL020_ARGS[@]}"

# Landscape: face semantic protection is not applicable; isolate spatial-only effect.
run_case "landscape" "L0_direct_zstar" "$LANDSCAPE_DIR"

run_case "landscape" "L1_spatial020_only" "$LANDSCAPE_DIR" \
  "${SPATIAL020_ARGS[@]}"

"$PYTHON_BIN" analyze_ablation_v2.py
