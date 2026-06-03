# Face-Aware ZSTAR via Internal Attention and Spatially Adaptive Injection

本目录保存的是基于 ZSTAR 的人像零样本风格迁移改进版本。它不是一个新的训练模型，而是在 ZSTAR 的 inference-time attention rearrangement 上加入语义区域保护与结构自适应调制，使模型在保持明显风格迁移的同时，减少眼睛、鼻子、嘴巴和脸部轮廓的结构崩坏。

## Motivation

ZSTAR 的核心思想是利用 diffusion UNet 中的 self-attention rearrangement，在 content latent 与 style latent 之间重组 attention，从而实现零样本风格迁移。原始方法对强风格非常有效，但在人像上存在一个明显问题：风格注入是全局、固定强度的，不区分背景、头发、衣服和五官等区域。

这会导致一个矛盾：

```text
strong global style injection  ->  good style strength
strong global style injection  ->  fragile facial details collapse
```

人脸五官尤其敏感。眼睛、嘴角、鼻梁这类区域在 latent resolution 中只占少量 token，一旦 attention 中混入过强的 style key/value，局部几何结构很难在后续 denoising 中恢复。相反，头发、背景和衣服即使产生较大纹理变化，也通常不会被感知为结构错误。

因此，本方法的核心问题不是“如何整体减弱风格”，而是：

```text
Can we keep strong stylization where it is safe,
while adaptively preserving content structure where it is fragile?
```

## Overview

我们在 ZSTAR 的基础上引入三个 inference-time 模块：

```text
1. Internal Cross-Attention Semantic Protection
2. Self-Attention Mask Refinement
3. Structure-Aware Spatially Adaptive Scaling
```

三个模块都利用 Stable Diffusion UNet 内部 attention 或 feature 表征，不需要额外训练。前两个模块提供语义级五官保护，第三个模块提供空间 token 级结构保护。

整体效果可以理解为：

```text
semantic protection:
  protect explicit facial parts such as eyes, nose, mouth, lips

spatial adaptive scaling:
  protect generic fragile structures such as contours, edges, hairline, eyelids

ZSTAR attention rearrangement:
  keep style transfer active in background, hair, clothing, and texture-rich regions
```

当前固定版本是 `spatial 0.20`，对应较平衡的人像结果。

## Baseline: ZSTAR Attention Rearrangement

在每个被选中的 UNet self-attention layer 中，ZSTAR 将 style/content 的 query、key、value 重新组合，构造 style-style、content-content、content-style 和 style-content attention logits。

简化表示为：

```text
A_cc = softmax(Q_c K_c^T)
A_cs = softmax(Q_c K_s^T)
A_sc = softmax(Q_s K_c^T)
A_ss = softmax(Q_s K_s^T)
```

其中 `c` 表示 content latent，`s` 表示 style latent。原始 ZSTAR 对 cross-domain logits 使用全局固定增强：

```text
L_cs <- lambda_cs * L_cs
L_sc <- lambda_sc * L_sc
```

这个设计能够增强风格迁移，但它隐含了一个不适合人像的假设：

```text
all spatial tokens can tolerate the same style injection strength
```

我们的改进正是围绕这个假设展开：不同空间位置应该有不同的 style injection policy。

## Module 1: Internal Cross-Attention Semantic Protection

第一步是让模型知道“哪里是眼睛、鼻子、嘴巴”。我们没有使用外部 face parser、MediaPipe 或 CLIPSeg，而是直接利用 Stable Diffusion 自身的 cross-attention map。

给定一个语义 prompt：

```text
a portrait photo with face eyes nose mouth lips hair background
```

在 denoising 过程中，UNet cross-attention 会建立图像空间 token 与文本 token 之间的响应关系。对于概念词 `eyes`、`nose`、`mouth`、`lips`，我们聚合其对应 token 的 cross-attention activation，得到每个概念的 soft semantic mask：

```text
M_k(x) = Avg_{t,l,h,i in T_k} CrossAttn_{t,l,h}(x, i)
```

其中：

- `k` 是语义概念，例如 `eyes`；
- `x` 是图像空间 token；
- `i` 是该概念对应的 text token index；
- `t,l,h` 分别表示 diffusion step、attention layer 和 attention head。

为了避免浅层噪声和 late-step 细节扰动稀释语义，我们使用 step/layer weighting，而不是简单均匀平均：

```text
M_k(x) = Sum w_t w_l CrossAttn_{t,l}(x, k) / Sum w_t w_l
```

直觉是：

- 较早 step 通常有更稳定的全局语义；
- 较深 cross-attention layer 通常有更明确的概念定位；
- 过晚 step 的 attention 可能更细碎，也更容易受局部纹理影响。

得到的 soft mask 会经过 quantile sharpening，仅保留高响应区域作为保护先验。

## Module 2: Self-Attention Mask Refinement

cross-attention 可以回答“哪里和某个文本概念相关”，但它的空间定位往往比较粗。self-attention 不包含文本 token 维度，因此不能直接判断哪里是眼睛或鼻子；但它包含图像 token 之间的相似性关系，适合做边界平滑和区域补全。

我们将 cross-attention mask 视为 seed mask，将 self-attention 视为图上的 affinity matrix：

```text
P_self(x, y) = SelfAttn(x, y)
```

然后对 seed mask 做少量传播：

```text
M'_k = Propagate(P_self, M_k)
M_refined_k = (1 - alpha) M_k + alpha M'_k
```

当前固定：

```text
alpha = 0.15
```

这个权重是有意保持较小的。实验中 `alpha = 0.35` 会让鼻子和嘴部 mask 扩散过多，导致脸部中心过度保护，风格变弱。`0.15` 更像是轻量 CRF-like refinement：它改善 mask 连贯性，但不让 mask 主导整个脸部。

## Module 3: Structure-Aware Spatially Adaptive Scaling

语义 mask 只能保护被显式命名的器官，但人像结构脆弱区域并不止眼睛、鼻子和嘴。例如脸轮廓、发际线、眼睑边界、下巴边缘同样容易被过强风格注入破坏。因此我们进一步引入一个无需语义标签的结构风险图。

原始 ZSTAR 使用全局常数：

```text
L_cs <- lambda_cs * L_cs
```

我们将它改写为空间自适应形式：

```text
L_cs(x, :) <- lambda_cs * S(x) * L_cs(x, :)
```

其中：

```text
S(x) = clamp(1 - beta R(x), s_min, 1)
```

`R(x)` 是 structure-risk map。风险越高，`S(x)` 越小，对应位置的 content-to-style injection 越弱；风险低的位置保持原始风格注入强度。

当前固定版本：

```text
beta = 0.20
s_min = 0.75
```

### Structure Risk Estimation

我们使用三个来自 UNet 内部的信号构造 `R(x)`：

```text
R(x) = Normalize(
    w_e R_entropy(x)
  + w_l R_local(x)
  + w_g R_edge(x)
)
```

### 1. Low-Entropy Self-Attention Focus

如果某个 token 的 self-attention 分布非常集中，说明它更可能是一个结构敏感或语义突出的 token。我们使用 attention entropy 的反向量作为风险：

```text
R_entropy(x) = 1 - H(SelfAttn(x, :)) / log N
```

### 2. Local Self-Attention Mass

结构边界和局部部件通常更依赖邻域 token 的一致性。我们统计 token 对局部窗口的 attention mass：

```text
R_local(x) = Sum_{y in N(x)} SelfAttn(x, y)
```

### 3. Content Feature Edge Strength

仅依赖 attention 仍可能不稳定，因此我们从当前 attention layer 的 content query feature 中估计局部特征梯度：

```text
R_edge(x) = ||F(x) - F(x + dx)|| + ||F(x) - F(x + dy)||
```

这个项对五官边缘、脸部轮廓、发丝边界特别有帮助。

三个信号合并后经过归一化和轻量平滑，得到最终风险图。它不是语义分割图，而是结构脆弱性估计图。

## Region-Protected Output Blending

语义保护区域中，最终输出不是简单丢弃风格，而是在 stylized attention output 和 content-preserving attention output 之间做 soft blend：

```text
O_final(x) = M(x) O_content(x) + (1 - M(x)) O_zstar(x)
```

其中 `M(x)` 是融合后的保护 mask。这样做的好处是：五官区域保留更多 content attention，非保护区域仍然走 ZSTAR 风格化路径。

## Why This Is Different From Simply Weakening Style

一个直接做法是降低全局 `style_content_scale` 或延后 injection step。但这会让整张图都变弱，尤其背景和头发的风格会下降。

本方法的核心区别是空间差异化：

```text
fragile facial structures -> weaker injection / stronger preservation
texture-tolerant regions  -> keep strong ZSTAR stylization
```

因此它不是“让风格变弱”，而是“把风格注入从全局常数改成空间自适应策略”。

## Practical Findings

当前小集实验中：

- internal cross-attention masks 能在不使用外部模型的情况下定位 eyes/nose/mouth/lips；
- self-attention refinement 有用，但必须轻量；
- spatial adaptive scaling 对脸部结构稳定性有明显帮助；
- `spatial_adaptive_strength = 0.20` 是较好的默认平衡点；
- `spatial_adaptive_strength = 0.35` 更保守，但脸部风格会被压低。

参考对比图：

```text
workdir/reference_outputs/spatial_adaptive_strength_comparison.png
workdir/reference_outputs/spatial_adaptive_result_comparison.png
```

## Ablation Study

为了支撑论文中的模块有效性分析，本目录额外提供一组 ablation：

```text
A0: baseline global ZSTAR
A1: internal attention only
A2: internal attention + spatial adaptive strength 0.20
A3: internal attention + spatial adaptive strength 0.35
A4: MediaPipe/sidecar external face-part masks only
A5: MediaPipe/sidecar external face-part masks + spatial adaptive strength 0.20
```

运行：

```bash
bash run_ablation_experiments.sh
```

输出目录：

```text
workdir/ablations
```

主要产物：

```text
workdir/ablations/ablation_report.md
workdir/ablations/ablation_result_contact_sheet.png
workdir/ablations/ablation_face_crop_contact_sheet.png
workdir/ablations/ablation_eye_crop_contact_sheet.png
workdir/ablations/ablation_self_attention_contact_sheet.png
workdir/ablations/ablation_mask_contact_sheet.png
workdir/ablations/ablation_spatial_risk_contact_sheet.png
```

这些图分别用于展示整体风格迁移、脸部局部结构、眼部细节、self-attention map、internal/external mask 差异，以及 spatial adaptive risk map。`ablation_report.md` 中包含简单 proxy metrics 和文字分析。注意这些 proxy metrics 只用于快速诊断；正式论文应补充 landmark error、identity similarity、masked LPIPS 和 style-strength metric。

## Ablation V2: Stylized-Only Portrait and Landscape Comparison

为了更贴近论文图表，本目录还提供第二组消融。它和上一组的主要区别是：横向对比图只放最终风格化结果，不再把 content/style/result panel 混在一起；同时补充了一个完全无保护模块的 direct ZSTAR baseline，并加入风景图作为非人脸场景验证。

这组实验回答三个问题：

```text
1. Direct ZSTAR 在相同强风格注入设置下会产生什么结果？
2. internal attention、spatial adaptive scaling、external face-part masks 各自贡献是什么？
3. 当图像中没有眼睛、鼻子、嘴巴等人脸语义时，spatial adaptive scaling 是否仍然有作用？
```

运行：

```bash
bash run_ablation_v2_landscape_portrait.sh
```

使用的 style 图像是：

```text
/home/whc/零样本风格迁移/official_repos/ZSTAR/style_images/demo_1.jpg
/home/whc/零样本风格迁移/official_repos/ZSTAR/style_images/wave.jpg
```

portrait 组包含六个变体：

```text
P0 Direct ZSTAR
P1 Internal attention
P2 Spatial 0.20 only
P3 Internal attention + spatial 0.20
P4 MediaPipe/sidecar masks
P5 MediaPipe/sidecar masks + spatial 0.20
```

landscape 组只保留两个变体：

```text
L0 Direct ZSTAR
L1 Spatial 0.20 only
```

这个设计是有意的：风景图没有眼睛、鼻子和嘴巴，所以不应该引入 face-part masks。若 L1 相比 L0 更稳定，证据应归因于 structure-aware spatial scaling，而不是人脸语义保护。

主要输出：

```text
workdir/ablation_v2/summary/ablation_v2_report.md
workdir/ablation_v2/summary/portrait_stylized_only_comparison.png
workdir/ablation_v2/summary/portrait_eye_crop_stylized_only.png
workdir/ablation_v2/summary/landscape_stylized_only_comparison.png
workdir/ablation_v2/summary/portrait_attention_maps.png
workdir/ablation_v2/summary/landscape_attention_maps.png
workdir/ablation_v2/summary/portrait_spatial_risk_maps.png
workdir/ablation_v2/summary/landscape_spatial_risk_maps.png
```

读图方式：

```text
portrait_stylized_only_comparison.png:
  观察整体风格强度、脸部身份保持、嘴鼻眼结构稳定性。

portrait_eye_crop_stylized_only.png:
  专门检查眼睛、瞳孔、眼睑和眉毛附近是否被 style pattern 改写。

landscape_stylized_only_comparison.png:
  检查建筑轮廓和天空笔触之间的 trade-off。这里 P/P face-mask 类方法不适用。

portrait_spatial_risk_maps.png / landscape_spatial_risk_maps.png:
  检查 spatial adaptive scaling 是否确实在结构敏感区域产生更高风险响应。
```

当前运行中，portrait direct baseline 的风格最强，但眼部和面部细节更容易被改写；internal attention 和 external masks 能更明确地保护五官；spatial 0.20 在 portrait 和 landscape 上都提供了结构级调制。风景图上没有人脸语义，因此更适合说明 spatial adaptive scaling 不是一个只依赖人脸 mask 的特殊处理。

## Multi-Model Comparison

为了和 `official_repos` 中的其他 image-style transfer 方法做公平对比，本目录提供一组统一 content/style 的多模型横向图。它使用同一张人脸 content 和同一张抽象画 style，而不是沿用各 repo 自带的不同示例。

运行：

```bash
bash run_multimodel_comparison.sh
```

也可以指定其他 style 图像，例如：

```bash
STYLE_IMAGE=demo_1.jpg bash run_multimodel_comparison.sh
STYLE_IMAGE=candy.jpg bash run_multimodel_comparison.sh
```

如果需要更强风格注入，并关闭 internal face-part masks，只使用 spatial weighted injection：

```bash
STYLE_IMAGE=candy.jpg \
OURS_MODE=weighted_only \
OURS_CONTENT_STYLE_SCALE=6.00 \
OURS_STYLE_CONTENT_SCALE=8.00 \
OURS_SPATIAL_STRENGTH=0.02 \
OURS_SPATIAL_MIN_SCALE=0.98 \
WORK="$PWD/workdir/multimodel_compare_candy_ours_weighted_w2" \
bash run_multimodel_comparison.sh
```

同一高风格模式也可用于其他 style：

```bash
STYLE_IMAGE=demo_1.jpg OURS_MODE=weighted_only \
WORK="$PWD/workdir/multimodel_compare_demo_1_ours_weighted_w2" \
bash run_multimodel_comparison.sh

STYLE_IMAGE=Starry.jpg OURS_MODE=weighted_only \
WORK="$PWD/workdir/multimodel_compare_Starry_ours_weighted_w2" \
bash run_multimodel_comparison.sh
```

如果头发区域风格过强，可以切回 internal attention 版本，使用语义保护和较保守的 spatial 0.20：

```bash
STYLE_IMAGE=demo_1.jpg OURS_MODE=full \
WORK="$PWD/workdir/multimodel_compare_demo_1_internal_attention" \
bash run_multimodel_comparison.sh

STYLE_IMAGE=Starry.jpg OURS_MODE=full \
WORK="$PWD/workdir/multimodel_compare_Starry_internal_attention" \
bash run_multimodel_comparison.sh
```

固定输入：

```text
content: official_repos/ZSTAR/content_images/demo_0.jpg
style:   official_repos/ZSTAR/style_images/Composition-VII.jpg
```

纳入方法：

```text
AdaIN
SANet
StyTR2
ZSTAR direct baseline
Ours: internal attention protection + spatial adaptive scaling 0.20
```

`CLIPstyler` 当前没有纳入这张 image-style comparison，因为本地可用入口是 text-condition 或 decoder-condition style transfer，不接受同一张 reference style image；强行放入会破坏对比公平性。

输出：

```text
workdir/multimodel_compare/summary/multimodel_face_abstract_comparison.png
workdir/multimodel_compare/summary/multimodel_comparison_report.md
```

当指定 `STYLE_IMAGE=demo_1.jpg` 时，输出目录为：

```text
workdir/multimodel_compare_demo_1/summary/multimodel_face_demo_1_comparison.png
workdir/multimodel_compare_demo_1/summary/multimodel_comparison_report.md
```

当指定 `STYLE_IMAGE=candy.jpg` 时，输出目录为：

```text
workdir/multimodel_compare_candy/summary/multimodel_face_candy_comparison.png
workdir/multimodel_compare_candy/summary/multimodel_comparison_report.md
```

强风格 weighted-only 版本输出目录为：

```text
workdir/multimodel_compare_candy_ours_weighted_w2/summary/multimodel_face_candy_comparison.png
workdir/multimodel_compare_candy_ours_weighted_w2/summary/multimodel_comparison_report.md
```

`demo_1.jpg` 和 `Starry.jpg` 的高风格版本输出目录为：

```text
workdir/multimodel_compare_demo_1_ours_weighted_w2/summary/multimodel_face_demo_1_comparison.png
workdir/multimodel_compare_demo_1_ours_weighted_w2/summary/multimodel_comparison_report.md
workdir/multimodel_compare_Starry_ours_weighted_w2/summary/multimodel_face_Starry_comparison.png
workdir/multimodel_compare_Starry_ours_weighted_w2/summary/multimodel_comparison_report.md
```

对应的 internal attention 版本输出目录为：

```text
workdir/multimodel_compare_demo_1_internal_attention/summary/multimodel_face_demo_1_comparison.png
workdir/multimodel_compare_demo_1_internal_attention/summary/multimodel_comparison_report.md
workdir/multimodel_compare_Starry_internal_attention/summary/multimodel_face_Starry_comparison.png
workdir/multimodel_compare_Starry_internal_attention/summary/multimodel_comparison_report.md
```

读图时建议同时看三点：风格强度、身份保持、眼睛/鼻子/嘴巴的局部稳定性。feed-forward baselines 往往风格更强，但在人脸局部更容易产生颜色或纹理覆盖；ZSTAR 系列更保结构，但需要通过本文的 face-aware 和 spatial-adaptive 调制改善五官稳定性。

## Implementation Mapping

主要实现位置：

```text
zstar/zstar.py
  ReweightCrossAttentionControl
  _spatial_structure_risk()
  _spatial_scale_map()
  forward()

zstar/zstar_utils.py
  CrossAttentionSemanticMaskExtractor
  _refine_with_self_attention()
  export_masks()

demo.py
  collect_internal_attention_masks()
  run_stylization()
```

关键计算路径：

```text
1. demo.py collects internal semantic masks
2. zstar_utils.py exports refined soft masks
3. demo.py passes masks into ReweightCrossAttentionControl
4. zstar.py performs ZSTAR attention rearrangement
5. zstar.py applies spatial adaptive scaling to content-style logits
6. zstar.py blends protected regions back toward content-preserving attention
```

## Reproduction

进入目录：

```bash
cd /home/whc/零样本风格迁移/ZSTAR_spatial020_minimal
```

运行：

```bash
bash run_spatial020.sh
```

默认使用：

```text
/home/whc/.conda/envs/zstar310
CUDA_VISIBLE_DEVICES=0
```

切换 GPU：

```bash
CUDA_VISIBLE_DEVICES=1 bash run_spatial020.sh
```

输出目录：

```text
workdir/reproduce_spatial020
```

参考结果：

```text
workdir/reference_outputs/spatial020
```

## Fixed Experimental Setting

当前固定的核心配置：

```text
start_step = 5
end_step = 30
layer_index = 20,22,24,26,28,30
content_style_scale = 1.50
style_content_scale = 1.50
semantic_self_refine_weight = 0.15
spatial_adaptive_strength = 0.20
spatial_adaptive_min_scale = 0.75
```

对应命令已写入：

```text
run_spatial020.sh
```

## Expected Outputs

成功运行后会生成：

```text
workdir/reproduce_spatial020/
  demo_0_Starry.png
  demo_0_candy.png
  demo_0_Composition-VII.png
  demo_0_Starry_reconstructed.png
  demo_0_candy_reconstructed.png
  demo_0_Composition-VII_reconstructed.png
  internal_attention_masks/
```

其中 `demo_0_*.png` 是横向 panel：

```text
content | style | stylized result
```

评价时建议优先观察：

- 眼睛和瞳孔是否稳定；
- 嘴唇边界是否扭曲；
- 鼻梁和脸轮廓是否保持；
- 头发、背景、衣服是否仍有明显风格。

## Package Contents

```text
demo.py
run_spatial020.sh
requirements.txt
stable-diffusion-v1-5/
zstar/
workdir/face_part_inputs/content/
workdir/eye_inputs/styles/
workdir/reference_outputs/
```

`stable-diffusion-v1-5` 是包内 diffusers 格式 SD v1.5 模型目录。它使用 hardlink 从当前已验证模型整理而来，因此本机不会额外占用第二份 15G 存储；普通拷贝到其他机器时，模型文件会作为实体文件一起带走。

## Suggested Paper Framing

如果将该方法写成论文，可以将贡献组织为：

```text
We propose a face-aware extension of ZSTAR that transforms global attention
rearrangement into semantic- and structure-adaptive attention modulation.
```

可能的贡献点：

1. 提出 internal cross-attention semantic protection，不依赖外部 face parser。
2. 使用 self-attention affinity 对 semantic masks 做 graph-like refinement。
3. 提出 structure-aware spatially adaptive scaling，将 ZSTAR 的全局 style injection scale 推广为空间自适应形式。
4. 在人像风格迁移中实现更好的 content-structure preservation 与 style strength trade-off。

## Quick Check

检查模型文件：

```bash
test -f stable-diffusion-v1-5/model_index.json
test -f stable-diffusion-v1-5/unet/diffusion_pytorch_model.safetensors
test -f stable-diffusion-v1-5/vae/diffusion_pytorch_model.safetensors
test -f stable-diffusion-v1-5/text_encoder/model.safetensors
```

检查代码语法：

```bash
/home/whc/.conda/envs/zstar310/bin/python -m py_compile demo.py ptp_utils.py segment_utils.py zstar/*.py
```
