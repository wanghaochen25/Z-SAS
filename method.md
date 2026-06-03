# Method

## Motivation: why global ZSTAR injection damages portrait content

ZSTAR performs zero-shot style transfer by rearranging attention between the
content branch and the style branch inside a pretrained diffusion model. This
mechanism is effective for transferring strong style patterns, but it also
introduces a failure mode on portrait images: the content structure can be
damaged when style information is injected uniformly across all spatial tokens.

The key issue is that different facial regions have very different tolerance to
style injection. Facial components such as eyes, mouth corners, and the nose
bridge occupy only a small number of tokens at latent resolution. Once these
tokens attend too strongly to style keys or values, their local geometry is hard
to recover in later denoising steps. In contrast, regions such as hair,
background, or clothing can usually absorb stronger style changes without being
perceived as structural failures.

This suggests that the problem is not simply that ZSTAR injects too much style.
Instead, the problem is that the same style injection policy is applied
everywhere:

```text
fragile facial structures -> need weaker style injection
texture-tolerant regions  -> can keep strong style injection
```

Our goal is therefore to turn ZSTAR's global attention injection into a
spatially adaptive policy.

## Baseline: ZSTAR attention rearrangement

At a selected self-attention layer and denoising step, let
`Q_c, K_c, V_c` denote the query, key, and value tensors from the content
branch, and let `Q_s, K_s, V_s` denote the corresponding tensors from the style
branch. ZSTAR constructs within-domain and cross-domain attention logits:

$$
L_{cc}=Q_cK_c^\top,\quad
L_{ss}=Q_sK_s^\top,\quad
L_{cs}=Q_cK_s^\top,\quad
L_{sc}=Q_sK_c^\top.
$$

The content-to-style and style-to-content logits are then strengthened by global
constants:

$$
L_{cs} \leftarrow \lambda_{cs} L_{cs},\qquad
L_{sc} \leftarrow \lambda_{sc} L_{sc}.
$$

This global scaling is the source of the portrait-specific trade-off. A larger
scale improves visible stylization, but it can also distort identity-critical
facial structures. A smaller scale preserves the face better, but weakens style
transfer across the whole image. We therefore keep the ZSTAR rearrangement as
the base mechanism, but replace its global spatial policy with semantic and
structure-aware modulation.

## From external spatial masks to internal semantic masks

A direct solution is to tell the model where the fragile regions are. In an
early design, this can be done with external region estimators such as
MediaPipe or CLIPSeg. These tools provide spatial masks for regions such as
eyes, nose, and mouth, allowing the attention controller to reduce style
injection on protected facial parts while keeping stronger stylization
elsewhere.

However, external masks introduce extra dependencies and make the method less
self-contained. Stable Diffusion already contains spatial-semantic information
inside its own cross-attention maps: image tokens attend to text tokens, and
concept words such as `eyes`, `nose`, `mouth`, and `lips` often produce
localized responses. We therefore replace external region masks with internal
cross-attention semantic masks.

Given a semantic prompt such as:

```text
a portrait photo with face eyes nose mouth lips hair background
```

we collect cross-attention activations between image tokens and the text tokens
corresponding to each protected concept. For a concept `k`, we aggregate its
cross-attention response over selected denoising steps, attention layers, and
heads:

$$
\tilde{M}_k(x)
=
\frac{
\sum_{t,l} w_t w_l
\operatorname{Avg}_{h,i\in T_k}
A^{\mathrm{cross}}_{t,l,h}(x,i)
}{
\sum_{t,l} w_t w_l
}.
$$

Here `x` is an image-space token, `T_k` is the set of text-token indices for
concept `k`, and `w_t, w_l` are deterministic step and layer weights. The
result is a soft semantic mask for each concept:

```text
eyes -> soft eye protection mask
nose -> soft nose protection mask
mouth/lips -> soft mouth protection mask
```

The masks are normalized and sharpened with quantile thresholding so that only
high-response regions become strong protection candidates.

## Self-attention refinement for semantic masks

Cross-attention tells us which spatial tokens are related to a concept, but its
boundaries can be coarse. To improve spatial continuity without using an
external parser, we refine the semantic masks with the model's self-attention.

We view self-attention as an affinity graph over image tokens:

$$
P(x,y)=A^{\mathrm{self}}(x,y).
$$

Starting from the cross-attention seed mask `M_k`, we propagate the mask through
this affinity graph:

$$
\hat{M}_k^{(r+1)}
=
\operatorname{Norm}(P\hat{M}_k^{(r)}),
\qquad
\hat{M}_k^{(0)}=M_k.
$$

The propagated mask is then blended back with the original seed:

$$
M'_k=(1-\alpha)M_k+\alpha\hat{M}_k.
$$

In the fixed reproduction setting, we use a small refinement weight
`alpha = 0.15`. This is intentionally conservative: the self-attention
refinement improves mask continuity, but does not allow the mask to spread over
the whole face and suppress stylization too broadly.

The final semantic protection mask combines the refined concept masks with
concept-specific strengths:

$$
M(x)=\max_k \rho_k M'_k(x).
$$

This mask is used to blend protected facial regions toward content-preserving
attention output.

## Structure-aware spatially adaptive scaling

Semantic masks protect explicitly named facial organs, but fragile portrait
structures are not limited to eyes, nose, and mouth. Face contours, hairlines,
eyelid boundaries, chin edges, and other local structures can also be damaged by
overly strong style injection. These regions may not correspond to a single
semantic token, so a purely semantic mask is insufficient.

We therefore introduce a label-free structure risk map `R(x)` that estimates
where style injection is likely to damage content geometry. Instead of applying
a single global content-to-style scale, we modulate the ZSTAR content-to-style
logits per spatial token:

$$
L_{cs}(x,:) \leftarrow \lambda_{cs} S(x) L_{cs}(x,:),
$$

where:

$$
S(x)=\operatorname{clamp}(1-\beta R(x), s_{\min}, 1).
$$

High-risk locations receive a smaller multiplier, while low-risk locations keep
the original ZSTAR style strength. In the fixed `spatial 0.20` setting, we use
`beta = 0.20` and `s_min = 0.75`.

We compute the risk map from three internal signals.

### Low-Entropy Self-Attention Focus

If a token has a highly concentrated self-attention distribution, it is likely
to represent a structurally important or semantically salient region. We use
the inverse normalized entropy as one risk signal:

$$
R_{\mathrm{entropy}}(x)
=
1-\frac{H(A^{\mathrm{self}}(x,:))}{\log N}.
$$

Low entropy means that the token depends strongly on a small set of other
tokens, so disrupting its attention pattern may more easily damage local
geometry.

### Local Self-Attention Mass

Local structures often depend on nearby tokens. We therefore measure how much
self-attention mass a token assigns to its local neighborhood:

$$
R_{\mathrm{local}}(x)
=
\sum_{y\in\mathcal{N}(x)}
A^{\mathrm{self}}(x,y).
$$

This term helps identify regions where local consistency matters, such as
eyelids, mouth boundaries, and facial edges.

### Content Feature Edge Strength

Attention alone can be noisy, so we also use the content query feature as a
local structure signal. Let `F(x)` be the normalized content query feature at
spatial token `x`. We estimate edge strength by local feature differences:

$$
R_{\mathrm{edge}}(x)
=
\|F(x)-F(x+\Delta_x)\|
+
\|F(x)-F(x+\Delta_y)\|.
$$

This term highlights sharp changes in the content representation, including
facial contours, hair boundaries, and fine facial details.

The final risk map is a normalized weighted combination:

$$
R(x)
=
\operatorname{Norm}
\left(
w_e R_{\mathrm{entropy}}(x)
+
w_l R_{\mathrm{local}}(x)
+
w_g R_{\mathrm{edge}}(x)
\right).
$$

After light smoothing and re-normalization, `R(x)` visualizes where the content
is most likely to be damaged by style injection. It is not a semantic
segmentation mask; it is a structure-fragility estimate.

## Protected attention output

After applying spatial adaptive scaling, the stylized content output is
computed from the ZSTAR content-to-style and content-to-content logits:

$$
O_{\mathrm{zstar}}
=
\operatorname{softmax}([L_{cs},L_{cc}])[V_s,V_c].
$$

For semantically protected regions, we also compute a content-preserving output:

$$
O_{\mathrm{content}}
=
\operatorname{softmax}(L_{cc})V_c.
$$

The final output is a soft blend:

$$
O(x)
=
M(x)O_{\mathrm{content}}(x)
+
(1-M(x))O_{\mathrm{zstar}}(x).
$$

This gives the method two complementary protection mechanisms:

- semantic masks protect explicitly named facial parts such as eyes, nose,
  mouth, and lips;
- structure risk maps reduce style injection on fragile unnamed regions such as
  face boundaries, hairlines, eyelids, and local contours.

As a result, style transfer remains active in texture-tolerant regions, while
identity-critical and geometry-sensitive regions receive weaker or more
content-preserving attention.

## Fixed inference setting

The current minimal reproduction uses the following fixed configuration:

```text
start_step = 5
end_step = 30
layer_index = 20,22,24,26,28,30
content_style_scale = 1.50
style_content_scale = 1.50
semantic_prompt = "a portrait photo with face eyes nose mouth lips hair background"
semantic_mask_quantile = 0.90
semantic_self_refine_weight = 0.15
semantic_self_refine_iters = 2
spatial_adaptive_strength = 0.20
spatial_adaptive_min_scale = 0.75
```

All components are applied at inference time. No additional training or
fine-tuning is introduced in this minimal version.
