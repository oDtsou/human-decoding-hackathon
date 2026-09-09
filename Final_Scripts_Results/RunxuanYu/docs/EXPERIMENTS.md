# Python Decoding Experiments

A short map of the Python decoding notebooks. Every result below is **exploratory**: a channel-resample bootstrap on one pseudo-population scheme, with approximate task windows and **no cluster-based permutation test or subject-level validation**.

Shared pipeline (`scripts/decoding.py`, `scripts/spectral_exp.py`): 70–150 Hz high-gamma amplitude envelope → ~488 ms moving average → per-trial z-score to the 0.25–0.75 s fixation window → downsample → per-channel pseudo-population tensor `[trials × channels × time]` → per-time-bin linear SVM → Stratified Leave-One-Per-Class-Out CV, averaged over random channel subsamples. Reference: Laplacian. 3 trials/class (n_train ≈ 16–18 per fold). Chance = 1/9 (spatial) or 1/8 (shape). `SEED = 0` drives the channel subsample, trial draw, and CV folds.

The four experiments below each hold this pipeline fixed and vary one thing: the frequency band, the classifier, the channel set, or the fixation normalisation. The metric of record is the **delay-window mean** — the global "peak accuracy" over the time-resolved curve is a noise ceiling.

---

## Baseline

`notebooks/decode_spatial_baseline.ipynb`, `notebooks/decode_shape_baseline.ipynb`

- **Spatial (decode cued location, 9 classes):** delay-window decoding is **Dorsal ≈ 0.13–0.14 > Ventral ≈ 0.06–0.08** (chance 0.111) — directionally the dorsal advantage described in `DecodingLogic_Hackathon.pdf`, but weak.
- **Shape (decode cued identity, 8 classes):** ≈ 0.17 in both subdivisions (chance 0.125), i.e. only slightly above chance — identity is not strongly decodable. Match vs non-match is a more promising reframing.

## Experiment 1 — frequency bands

`notebooks/decode_spatial_multiband_v2.ipynb`, `notebooks/decode_shape_multiband_v2.ipynb`
(paired: one 4-band tensor per channel subsample, every condition a column subset)

- **Question:** does adding θ (4–8), α (8–12), β (13–30 Hz) amplitude to the HG feature help, and do the lower bands decode on their own?
- **Spatial:** for **Dorsal**, HG alone is the ceiling — no band, HG+band, or full fusion beats it. For **Ventral**, HG *fails* (≤ chance) and lower-frequency amplitude carries the signal (θ alone ≈ 0.15). Adding bands to HG only helps where HG itself is uninformative.
- **Shape:** HG is the strongest single feature in Dorsal (delay ≈ 0.17); β amplitude is informative in Ventral (≈ 0.19). Full 4-band fusion does not beat the best one/two bands — with p up to 312 features vs n_train ≈ 16–20, the extra dimensions dilute rather than add.
- **Takeaway:** keep HG as the feature; there is no free lunch from naive band concatenation.

## Experiment 2 — model comparison

`notebooks/decode_spatial_hg_models.ipynb`, `notebooks/decode_shape_hg_models.ipynb`

- **Question:** with the HG representation and all 80 channels fixed, do Linear SVM, L2 Logistic Regression, and Shrinkage LDA (`lsqr`, `shrinkage="auto"`) change the decoding conclusion?
- **Delay-window mean** (10 resamples):

  | | Spatial Dorsal | Spatial Ventral | Shape Dorsal | Shape Ventral |
  |---|---|---|---|---|
  | Linear SVM | 0.127 | 0.064 | 0.172 | 0.168 |
  | L2 LogReg | 0.119 | 0.069 | 0.171 | 0.156 |
  | Shrinkage LDA | 0.096 | 0.086 | 0.142 | 0.125 |

- **Linear SVM and L2 LogReg are interchangeable** (within ≈ 0.01 everywhere).
- **Shrinkage LDA on all 80 channels is worse** on the delay mean in 3 of 4 cells — it only helps the one genuinely over-dimensioned case (Spatial Ventral, 0.064 → 0.086). It earns its place only when paired with channel selection (Experiment 3).
- **Takeaway:** classifier choice does not change the qualitative pattern (Spatial Dorsal > Ventral; Shape ≈ chance in both). Linear SVM is a fine default.

## Experiment 3 — channel selection + low-N classifier

`notebooks/decode_spatial_hg_channel_models.ipynb`, `notebooks/decode_shape_hg_channel_models.ipynb`

- **Question:** does removing noisy channels (fold-wise ANOVA-F Top-10/20/30, computed on training data only) or a classifier better suited to n_train ≪ p (shrinkage LDA) improve HG decoding?
- **Spatial:** Top-K selection rescues the over-dimensioned **Ventral** case (SVM ≈ 0.06 → ≈ 0.10 at Top-10/20); **shrinkage LDA + Top-10/20** gives the best delay-window mean at roughly half the resample SD of SVM-on-all-channels. It does not help Dorsal SVM.
- **Shape:** the opposite — channel selection **hurts** (all-channels + linear SVM is best); shrinkage LDA does not beat SVM. HG information is distributed across channels here.
- **Takeaway:** whether pruning dimensions helps is **regime-dependent** (task and region); check `p / n_train` before selecting channels. There is no single "best HG pipeline" that transfers across tasks.

## Experiment 4 — fixation-window normalisation

`notebooks/decode_spatial_hg_normalization.ipynb`, `notebooks/decode_shape_hg_normalization.ipynb`

- **Question:** the baseline z-scores each trial/channel to the fixation window, but the 488 ms moving average precedes this, so `σ_fix` is estimated from ≈ 1 effective sample. Is the σ-division helping or just adding outliers?
- **Result:** `relchange` = `(env − μ_fix) / μ_fix` removes the extreme outlier features (`|x| > 10`: ~4–7 % → ~0 %; max feature value ~180–1900 → ~4–90) **at no accuracy cost** — delay-window mean ties or slightly beats the current z-score on both tasks and regions. `subtract` (mean only) is ~0.02–0.03 worse on the Dorsal delay; `none` is the least stable.
- **Takeaway:** a per-trial baseline + per-channel gain correction *is* needed (it is not detrending); `relchange` (`build_pseudopopulation(baseline_mode="relchange")`) is the best-behaved choice.
