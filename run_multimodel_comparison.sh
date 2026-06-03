#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/whc/零样本风格迁移"
THIS_DIR="$ROOT/ZSTAR_spatial020_minimal"
PYTHON_BIN="${PYTHON_BIN:-/home/whc/.conda/envs/zstar310/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONTENT_SRC="$ROOT/official_repos/ZSTAR/content_images/demo_0.jpg"
STYLE_IMAGE="${STYLE_IMAGE:-Composition-VII.jpg}"
STYLE_SRC="${STYLE_SRC:-$ROOT/official_repos/ZSTAR/style_images/$STYLE_IMAGE}"
STYLE_FILE="$(basename "$STYLE_SRC")"
STYLE_STEM="${STYLE_FILE%.*}"
if [ "$STYLE_STEM" = "Composition-VII" ]; then
  WORK="${WORK:-$THIS_DIR/workdir/multimodel_compare}"
else
  WORK="${WORK:-$THIS_DIR/workdir/multimodel_compare_${STYLE_STEM}}"
fi
INPUT_CONTENT="$WORK/inputs/content"
INPUT_STYLE="$WORK/inputs/style"
OUT="$WORK/outputs"
OURS_MODE="${OURS_MODE:-full}"

rm -rf "$WORK"
mkdir -p "$INPUT_CONTENT" "$INPUT_STYLE" "$OUT"

cp "$CONTENT_SRC" "$INPUT_CONTENT/demo_0.jpg"
cp "$ROOT/official_repos/ZSTAR/content_images/demo_0.pkl" "$INPUT_CONTENT/demo_0.pkl"
for suffix in _eyes_mask.npy _lips_mask.npy _nose_mask.npy _mask.npy; do
  if [ -f "$ROOT/official_repos/ZSTAR/content_images/demo_0${suffix}" ]; then
    cp "$ROOT/official_repos/ZSTAR/content_images/demo_0${suffix}" "$INPUT_CONTENT/demo_0${suffix}"
  fi
done
cp "$STYLE_SRC" "$INPUT_STYLE/$STYLE_FILE"

echo "===== AdaIN ====="
(
  cd "$ROOT/official_repos/AdaIN"
  "$PYTHON_BIN" test.py \
    --content "$INPUT_CONTENT/demo_0.jpg" \
    --style "$INPUT_STYLE/$STYLE_FILE" \
    --content_size 560 \
    --style_size 560 \
    --output "$OUT/AdaIN"
)

echo "===== SANet ====="
(
  cd "$ROOT/official_repos/SANet"
  "$PYTHON_BIN" Eval.py \
    --content "$INPUT_CONTENT/demo_0.jpg" \
    --style "$INPUT_STYLE/$STYLE_FILE" \
    --output "$OUT/SANet"
)

echo "===== StyTR2 ====="
(
  cd "$ROOT/official_repos/StyTR2"
  "$PYTHON_BIN" test.py \
    --content "$INPUT_CONTENT/demo_0.jpg" \
    --style "$INPUT_STYLE/$STYLE_FILE" \
    --output "$OUT/StyTR2"
)

echo "===== Direct ZSTAR ====="
(
  cd "$ROOT/official_repos/ZSTAR"
  "$PYTHON_BIN" demo.py \
    --content_img_folder "$INPUT_CONTENT" \
    --style_img_folder "$INPUT_STYLE" \
    --sub_exp_name "$OUT/ZSTAR_direct" \
    --start_step 5 \
    --end_step 30 \
    --layer_index 20,22,24,26,28,30 \
    --content_style_scale 1.20 \
    --style_content_scale 1.50
)

if [ "$OURS_MODE" = "weighted_only" ]; then
  OURS_CONTENT_STYLE_SCALE="${OURS_CONTENT_STYLE_SCALE:-6.00}"
  OURS_STYLE_CONTENT_SCALE="${OURS_STYLE_CONTENT_SCALE:-8.00}"
  OURS_SPATIAL_STRENGTH="${OURS_SPATIAL_STRENGTH:-0.02}"
  OURS_SPATIAL_MIN_SCALE="${OURS_SPATIAL_MIN_SCALE:-0.98}"
  echo "===== Ours: weighted-only strong spatial injection ====="
  (
    cd "$THIS_DIR"
    "$PYTHON_BIN" demo.py \
      --content_img_folder "$INPUT_CONTENT" \
      --style_img_folder "$INPUT_STYLE" \
      --sub_exp_name "$OUT/Ours_weighted_only" \
      --start_step 5 \
      --end_step 30 \
      --layer_index 20,22,24,26,28,30 \
      --content_style_scale "$OURS_CONTENT_STYLE_SCALE" \
      --style_content_scale "$OURS_STYLE_CONTENT_SCALE" \
      --spatial_adaptive_scale \
      --spatial_adaptive_strength "$OURS_SPATIAL_STRENGTH" \
      --spatial_adaptive_min_scale "$OURS_SPATIAL_MIN_SCALE" \
      --eye_mask_dilate 3 \
      --eye_mask_blur_sigma 1.5
  )
else
  echo "===== Ours: internal attention + strong-balanced spatial injection ====="
  (
    cd "$THIS_DIR"
    "$PYTHON_BIN" demo.py \
      --content_img_folder "$INPUT_CONTENT" \
      --style_img_folder "$INPUT_STYLE" \
      --sub_exp_name "$OUT/Ours_internal_balanced" \
      --start_step 5 \
      --end_step 30 \
      --layer_index 20,22,24,26,28,30 \
      --content_style_scale 4.00 \
      --style_content_scale 5.00 \
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
      --spatial_adaptive_strength 0.05 \
      --spatial_adaptive_min_scale 0.95 \
      --eye_mask_dilate 3 \
      --eye_mask_blur_sigma 1.5
  )
fi

MULTIMODEL_WORK="$WORK" OURS_MODE="$OURS_MODE" "$PYTHON_BIN" "$THIS_DIR/make_multimodel_comparison.py"
