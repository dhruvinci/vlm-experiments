# Armbar exploratory decoder result

## Decision

The local decoder campaign completed successfully and established two separate findings:

1. Frozen Qwen spatial features contain strong actor-ownership information. A compact decoder
   using full-resolution vision layer 11 plus Qwen's final merged visual embedding reached
   **0.8746 ± 0.0070 macro actor IoU** on eight held-out armbar frames.
2. The cached clip-level semantic marker states did **not** repair the trapped-arm ownership
   error. Action conditioning reduced mean macro IoU by 0.0083 versus the paired static decoder,
   assigned 0/27 trapped-arm cells correctly in every seed, and made the wrong ownership more
   confident.

This is a scientifically useful negative semantic result. It does not pass the exploratory
gate, is not eligible for the multi-clip north star, and does not justify LoRA training yet.

## Immutable provenance

- Campaign result:
  `decoder-runs/armbar-exploratory-v1/campaign-result.json`
- Campaign-result SHA-256:
  `f68727ced2434a104fb35fb28436333bea79794907928ca3abdd903c8b129445`
- Campaign source commit: `d0faa70`
- Supervision status: `exploratory_legacy_pseudo_labels`
- Training frames: 20 for screening, 24 for final refits
- Validation frames: 4, used only for architecture/layer selection
- Test frames: 8, including the sole manually corrected trapped-arm contact frame
- Test contact truth: 27 grid cells in three connected regions on frame 143
- Seeds: 7, 71, 701

The legacy actor interiors are conservative SAM-derived pseudo-labels, not fully human-reviewed
masks. Only the final trapped-arm contact evidence has manual ownership truth. Consequently,
none of these metrics may be presented as the project's multi-clip north-star result.

## Decoder constructed

The selected static model is a small three-class ownership decoder:

- independent 1x1 adapters for Qwen full-resolution layer 11 (1,152 channels) and the final
  merged visual embedding (5,120 channels);
- learned softmax fusion at the larger spatial grid;
- width 96 with two spatial residual blocks and FP32 GroupNorm;
- mutually exclusive `background`, `A1`, and `A2` logits.

The semantic variant adds a 5,120-dimensional actor-state projection. It conditions the shared
spatial field on the actor-pair mean and compares normalized per-pixel embeddings with the two
actor residuals. Its construction is exactly A1/A2-swap equivariant: swapping only the actor
states swaps the actor logits while preserving the background logit.

Training used batch size one, four-step gradient accumulation, AdamW at `3e-4`, weight decay
`1e-4`, at most 80 epochs, patience 10, mixed precision, FP32 loss calculation, class-balanced
cross-entropy plus soft Dice, contact weighting, and semantic swap-equivariance regularization.

## Layer screen

Selection was performed on the four-frame validation split before evaluating the test split.

| Spatial input | Validation macro actor IoU |
|---|---:|
| RGB | 0.6771 |
| Full vision layer 11 | 0.7675 |
| Pooled vision layer 12 | 0.7782 |
| Final merged visual embedding | 0.8546 |
| Full layer 11 + merged | **0.8622** |
| Full layers 5 + 11 + 18 + 26 | 0.7900 |

The merged embedding contains most of the usable ownership signal. Full-resolution layer 11 adds
a small contour benefit. Naively concatenating more full-resolution layers makes the decoder
worse rather than better.

With the selected spatial input, semantic language layer 25 scored 0.8656 on validation versus
0.8632 for layer 60, so layer 25 was frozen for final evaluation. The difference is too small to
claim a substantive language-layer effect.

## Held-out result

Values are the mean and population standard deviation across seeds 7, 71, and 701.

| Condition | Macro actor IoU | Contact accuracy | Contact margin | Background stability |
|---|---:|---:|---:|---:|
| Static L11 + merged | **0.8746 ± 0.0070** | 0.0988 | -0.7882 | 0.9982 |
| Action states | 0.8663 ± 0.0078 | 0.0000 | -0.9789 | 0.9983 |
| Norm-matched random states | 0.8566 ± 0.0067 | 0.0000 | -0.9708 | 0.9978 |
| Identity states, seed 7 | 0.8580 | 0.0000 | -0.9940 | 0.9982 |
| Contact-specific states, seed 7 | 0.8508 | 0.2222 | -0.6107 | 0.9978 |

Paired action-minus-static IoU deltas were `+0.0057`, `-0.0184`, and `-0.0123`, for a mean of
`-0.0083`. Paired action-minus-random deltas were `+0.0195`, `-0.0107`, and `+0.0203`, for a mean
of `+0.0097`, narrowly below the preregistered +0.01 control gate. Zero and pair-mean actor states
collapse the decoder to 0.0671 and 0.0945 IoU respectively, confirming that the semantic pathway
is active but not that it uses the desired meaning.

All exploratory gates failed except background stability:

- action beats norm-matched random: fail;
- action beats static: fail;
- trapped-arm margin improves: fail;
- ordered temporal context beats reversed/shuffled nulls: fail;
- background remains stable: pass.

## Temporal controls

Substituting cached marker pairs into each already-trained action decoder produced:

| Marker-state context | Macro actor IoU | Contact accuracy | Contact margin |
|---|---:|---:|---:|
| Ordered 4 fps | 0.866277 | 0.0000 | -0.978922 |
| Reversed 4 fps | 0.866507 | 0.0000 | -0.979366 |
| Shuffled 4 fps | 0.866247 | 0.0000 | -0.979098 |
| Single frame | 0.866127 | 0.0000 | -0.977941 |
| 8 fps | 0.866280 | 0.0000 | -0.978846 |
| Thinking `xhigh` | 0.866266 | 0.0000 | -0.978741 |

Ordered context is not better than the temporal nulls. A post-hoc state-geometry check shows that
Qwen did not literally emit identical states: at selected layer 25, the ordered actor residual's
cosine similarity is 0.8804 with reversed and 0.9202 with shuffled. However, training supplied
one constant marker pair for the entire clip, so the decoder had no cross-clip or per-frame
examples from which to learn what those temporal directions mean. It maps the distinct states to
effectively the same ownership field.

The 2 fps and five-frame artifacts require a known physical-actor remap. Their apparent 100%
contact accuracy is invalid as a fix: they reverse the entire actor field, reduce global IoU to
about 0.032, and only make the trapped arm correct by assigning almost every athlete pixel to the
opposite identity.

## Contact-frame visual audit

The checksum-bound diagnostics are local under
`diagnostics/armbar-exploratory-v1/`. Static and action-conditioned models trace the two athletes
well globally. On frame 143:

- static seed 7: 0.9324 macro IoU, 0 contact accuracy, -0.8332 margin;
- static seed 71: 0.9511 macro IoU, 0.2963 contact accuracy, -0.5532 margin;
- action ordered: 0.9161 macro IoU, 0 contact accuracy, -0.9940 margin;
- action reversed and shuffled: visually and numerically indistinguishable from ordered;
- contact-specific: 0.8981 macro IoU, 0.2222 contact accuracy, -0.6107 margin;
- 2 fps remapped: 0.0129 macro IoU despite 1.0 contact accuracy, exposing the global actor flip.

The failure is localized but systematic: good overall segmentation can hide confident ownership
errors at the exact body-contact region the project cares about.

## Local resource record

Only cached tensors and the small decoder were loaded locally; Qwen and SAM weights were not.
All 21 CUDA jobs ran serially in isolated `systemd` scopes with:

- 4 GiB hard RAM and 3.6 GiB high-water cgroup limits;
- swap disabled inside each child scope;
- 60% PyTorch per-process allocator ceiling;
- parent guards requiring at least 4 GiB host availability and 1.5 GiB free VRAM;
- global GPU-use ceiling of 75%;
- 30-minute job ceiling and 10-second terminate-to-kill grace period.

Every job completed on its first attempt. There were zero OOMs, retries, limit kills, or resource
violations. The longest job took 498.5 seconds. Across guard observations, host available memory
never fell below 8,610,635,776 bytes and GPU free memory never fell below 5,375,000,576 bytes.
The peak observed global GPU-used fraction was 0.1657. The completed campaign occupies about
802 MiB and is fully resumable/checksum-bound.

## What the cache can and cannot support next

The local cache also contains four high-contact breadth clips (`back_seatbelt`,
`guard_scramble`, `half_guard`, and `mount`), 24 frames each. For every clip it has final merged
visual embeddings, pooled copies of all 27 vision layers, full-resolution layers 5/11/18/26, and
three video-level semantic actor pairs. It does **not** contain per-frame image semantic states or
human-reviewed actor-ownership masks for these clips.

The next decisive decoder experiment is therefore data-gated, not compute-gated: finalize
human-reviewed mutually exclusive actor masks and explicit contact ownership for the four cached
breadth clips. Then run clip-held-out training using the same guarded local executor and compare:

1. static L11 + merged;
2. real action-conditioned states;
3. norm-matched random and wrong-clip semantic controls;
4. ordered versus reversed/shuffled temporal states, if those counterfactual states are cached per
   breadth clip.

That experiment teaches the semantic adapter from several independent query/clip pairs and tests
generalization to an unseen clip. If it passes the preregistered north-star gate, decoder-only
training has extracted useful relational ownership and LoRA becomes a justified next phase. If it
fails, the likely missing representation is frame-varying or locally aligned semantic state—not
decoder capacity—and the next extraction should target per-frame/query-conditioned spatial
features before fine-tuning the backbone.
