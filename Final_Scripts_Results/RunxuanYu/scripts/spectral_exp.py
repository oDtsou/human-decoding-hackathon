"""
Utilities for multi-band, classifier, channel-selection, and normalization
experiments using paired pseudo-population comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from decoding import (
    HIGH_GAMMA_HZ, PseudoPopulation, build_pseudopopulation, hg_channel_features,
    sloo_folds, time_resolved_scores, window_score,
)

# Approximate task windows -- the files have no epoch markers, so these are best-effort.
# Spatial trial 0-6 s, feature window starts at sample 513 == 1.0 s ("remove ITI").
SPATIAL_CUE_WINDOW: tuple[float, float] = (1.0, 2.0)
SPATIAL_DELAY_WINDOW: tuple[float, float] = (2.5, 5.5)
# Shape (MNM) trial 0-9.5 s: sample -> long delay -> probe -> response.
SHAPE_CUE_WINDOW: tuple[float, float] = (1.0, 2.0)     # sample / cue period
SHAPE_DELAY_WINDOW: tuple[float, float] = (2.5, 7.5)   # maintenance period

# backwards-compatible aliases (used by the multi-band exploration notebooks)
CUE_WINDOW = SPATIAL_CUE_WINDOW
DELAY_WINDOW = SPATIAL_DELAY_WINDOW


def viable_region_pool(
    ds, subdiv: str, *, required_classes: int, min_trials_per_class: int = 3
) -> list[int]:
    """Viable channel indices for one prefrontal subdivision ('Dorsal'/'Ventral')."""
    summary = ds.summary()
    viable = set(
        ds.viable_channels(min_trials_per_class, required_classes=required_classes)
    )
    return [
        int(i)
        for i in summary.loc[summary.prefrontal_subdiv == subdiv, "idx"]
        if i in viable
    ]


def build_band_tensors(
    ds,
    pool: list[int],
    *,
    n_resamples: int,
    n_channels: int,
    ref: str,
    bands: list[tuple[float, float]],
    downsample: int,
    smooth_ms: float = 488.0,
    feature_transform=None,
    baseline_mode: str = "zscore",
    n_trials_per_class: int = 3,
    seed: int = 0,
) -> list[PseudoPopulation]:
    """Build one reproducible pseudo-population per random channel subsample."""
    tensors = []
    for r in range(n_resamples):
        sub = list(
            np.random.default_rng(seed + r).choice(
                pool, size=min(n_channels, len(pool)), replace=False
            )
        )
        tensors.append(
            build_pseudopopulation(
                ds, sub, ref=ref, bands=bands,
                n_trials_per_class=n_trials_per_class,
                smooth_ms=smooth_ms, feature_transform=feature_transform,
                baseline_mode=baseline_mode,
                downsample=downsample, seed=seed + r,
            )
        )
    return tensors


def build_hg_tensors(
    ds,
    pool: list[int],
    *,
    n_resamples: int,
    n_channels: int,
    ref: str = "Laplacian",
    downsample: int,
    smooth_ms: float = 488.0,
    feature_transform=None,
    baseline_mode: str = "zscore",
    n_trials_per_class: int = 3,
    seed: int = 0,
) -> list[PseudoPopulation]:
    """HG-only pseudo-population tensors (feature axis == channels)."""
    return build_band_tensors(
        ds, pool, n_resamples=n_resamples, n_channels=n_channels, ref=ref,
        bands=[HIGH_GAMMA_HZ], downsample=downsample, smooth_ms=smooth_ms,
        feature_transform=feature_transform, baseline_mode=baseline_mode,
        n_trials_per_class=n_trials_per_class, seed=seed,
    )


def subset_bands(pop: PseudoPopulation, band_idx: list[int]) -> PseudoPopulation:
    """Return a PseudoPopulation with only the requested bands on the feature axis."""
    nb = pop.n_bands
    cols = np.sort(
        np.concatenate([np.arange(i, pop.X.shape[1], nb) for i in band_idx])
    )
    return replace(pop, X=pop.X[:, cols, :], bands=[pop.bands[i] for i in band_idx])


# metrics and per-condition evaluation
def _window_stats(curves: np.ndarray, time_s: np.ndarray, win: tuple[float, float]):
    """peak-in-window per resample (max of the per-bin decoding curve)."""
    m = (time_s >= win[0]) & (time_s < win[1])
    if not m.any():
        return np.full(curves.shape[0], np.nan)
    return curves[:, m].max(axis=1)


def _pack(curves, cue_ws, del_ws, time_s, *, n_features, n_trials, n_classes,
          cue_window, delay_window, extra=None) -> dict:
    curves = np.asarray(curves)
    n_train = n_trials - n_classes
    peak_i = curves.argmax(axis=1)
    out = {
        "curves": curves,
        "curve_mean": curves.mean(0),
        "curve_sem": curves.std(0) / np.sqrt(len(curves)),
        "time_s": np.asarray(time_s),
        # feature-averaged window score (decode once from window-averaged features)
        "cue_mean": float(np.mean(cue_ws)), "cue_std": float(np.std(cue_ws)),
        "delay_mean": float(np.mean(del_ws)), "delay_std": float(np.std(del_ws)),
        # peak of the per-bin time-resolved accuracy, within each window
        "cue_peak_mean": float(np.nanmean(_window_stats(curves, time_s, cue_window))),
        "cue_peak_std": float(np.nanstd(_window_stats(curves, time_s, cue_window))),
        "delay_peak_mean": float(np.nanmean(_window_stats(curves, time_s, delay_window))),
        "delay_peak_std": float(np.nanstd(_window_stats(curves, time_s, delay_window))),
        # global peak over the whole trial
        "peak_mean": float(curves.max(axis=1).mean()),
        "peak_std": float(curves.max(axis=1).std()),
        "peak_t_mean": float(np.mean([time_s[i] for i in peak_i])),
        "n_features": int(n_features),
        "n_trials": int(n_trials),
        "n_train": int(n_train),
        "p_over_n": n_features / n_train,
        "chance": 1.0 / n_classes,
    }
    if extra:
        out.update(extra)
    return out


def run_condition(
    tensors: list[PseudoPopulation],
    band_idx: list[int],
    make_estimator,
    *,
    n_cv_repeats: int,
    cue_window: tuple[float, float] = CUE_WINDOW,
    delay_window: tuple[float, float] = DELAY_WINDOW,
    seed: int = 0,
) -> dict:
    """Evaluate one feature/model condition across channel resamples using paired SLOO folds."""
    curves, cue_ws, del_ws = [], [], []
    for r, pop in enumerate(tensors):
        sub = subset_bands(pop, band_idx)
        folds = sloo_folds(sub.y, n_repeats=n_cv_repeats, seed=seed + r)
        curves.append(time_resolved_scores(sub, folds, make_estimator))
        cue_ws.append(window_score(sub, cue_window, folds, make_estimator))
        del_ws.append(window_score(sub, delay_window, folds, make_estimator))
    sub0 = subset_bands(tensors[0], band_idx)
    return _pack(curves, cue_ws, del_ws, tensors[0].time_s,
                 n_features=sub0.X.shape[1], n_trials=sub0.X.shape[0],
                 n_classes=int(np.unique(sub0.y).size),
                 cue_window=cue_window, delay_window=delay_window)


# fold-wise Top-K channel selection by univariate ANOVA F (channel-selection experiment)
def _topk_fclassif(X_win: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k channels with the highest ANOVA F-score (training data only)."""
    from sklearn.feature_selection import f_classif

    F, _ = f_classif(X_win, y)
    F = np.nan_to_num(F, nan=-np.inf, posinf=-np.inf)
    return np.sort(np.argsort(F)[::-1][:k])


def run_channel_select_condition(
    tensors: list[PseudoPopulation],
    make_estimator,
    *,
    K: int | None,
    rank_window: tuple[float, float],
    n_cv_repeats: int,
    cue_window: tuple[float, float],
    delay_window: tuple[float, float],
    seed: int = 0,
    collect_selected: bool = False,
) -> dict:
    """Evaluate one feature/model condition across channel resamples using paired SLOO folds."""
    curves, cue_ws, del_ws = [], [], []
    sel_count: dict[int, int] = {}
    sel_total = 0
    for r, pop in enumerate(tensors):
        X, y, t = pop.X, pop.y, pop.time_s
        n_ch = X.shape[1]
        k = n_ch if K is None else min(K, n_ch)
        folds = sloo_folds(y, n_repeats=n_cv_repeats, seed=seed + r)
        rw = (t >= rank_window[0]) & (t < rank_window[1])

        fold_sel = []
        for tr, _te in folds:
            if K is None:
                sel = np.arange(n_ch)
            else:
                sel = _topk_fclassif(X[np.ix_(tr, np.arange(n_ch))][:, :, rw].mean(2),
                                     y[tr], k)
            fold_sel.append(sel)
            if collect_selected and K is not None:
                for ci in sel:
                    g = pop.channel_indices[int(ci)]
                    sel_count[g] = sel_count.get(g, 0) + 1
                sel_total += 1

        # time-resolved
        c = np.empty(X.shape[2])
        for ti in range(X.shape[2]):
            accs = []
            for (tr, te), sel in zip(folds, fold_sel):
                Xt = X[:, sel, ti]
                clf = make_estimator()
                clf.fit(Xt[tr], y[tr])
                accs.append(np.mean(clf.predict(Xt[te]) == y[te]))
            c[ti] = float(np.mean(accs))
        curves.append(c)

        # feature-averaged window scores (same fold selections)
        for win, store in ((cue_window, cue_ws), (delay_window, del_ws)):
            wm = (t >= win[0]) & (t < win[1])
            accs = []
            for (tr, te), sel in zip(folds, fold_sel):
                Xw = X[:, sel][:, :, wm].mean(2)
                clf = make_estimator()
                clf.fit(Xw[tr], y[tr])
                accs.append(np.mean(clf.predict(Xw[te]) == y[te]))
            store.append(float(np.mean(accs)))

    sub0 = tensors[0]
    k0 = sub0.X.shape[1] if K is None else min(K, sub0.X.shape[1])
    extra = {"K": (None if K is None else k0)}
    if collect_selected and sel_total:
        freq = {g: n / sel_total for g, n in sel_count.items()}
        extra["selection_freq"] = dict(sorted(freq.items(), key=lambda kv: -kv[1]))
    return _pack(curves, cue_ws, del_ws, sub0.time_s,
                 n_features=k0, n_trials=sub0.X.shape[0],
                 n_classes=int(np.unique(sub0.y).size),
                 cue_window=cue_window, delay_window=delay_window, extra=extra)


# plotting / results-table helpers
def viz_smooth(curves: np.ndarray, npts: int = 5):
    """Per-resample centred moving average, then mean + SEM. FOR PLOTTING ONLY."""
    curves = np.asarray(curves)
    k = np.ones(npts) / npts
    pad = npts // 2
    sm = np.stack([
        np.convolve(np.pad(row, pad, mode="edge"), k, mode="valid")[: row.size]
        for row in curves
    ])
    return sm.mean(0), sm.std(0) / np.sqrt(len(sm))


def region_diff(res_dorsal: dict, res_ventral: dict) -> dict:
    """Descriptive Dorsal - Ventral (per-bin curve mean, and window means)."""
    d = res_dorsal["curve_mean"] - res_ventral["curve_mean"]
    return {
        "time_s": res_dorsal["time_s"],
        "diff_curve": d,
        "cue_diff": res_dorsal["cue_mean"] - res_ventral["cue_mean"],
        "delay_diff": res_dorsal["delay_mean"] - res_ventral["delay_mean"],
    }


def summary_row(name: str, res: dict) -> str:
    """One formatted line for a results table."""
    return (
        f"{name:26s}  p={res['n_features']:4d}  p/n_train={res['p_over_n']:6.2f}  "
        f"cue={res['cue_mean']:.3f}±{res['cue_std']:.3f}  "
        f"delay={res['delay_mean']:.3f}±{res['delay_std']:.3f}  "
        f"delayPk={res['delay_peak_mean']:.3f}  "
        f"peak={res['peak_mean']:.3f}@{res['peak_t_mean']:.2f}s"
    )


# Training-only HG augmentation using recombination, mixup, or temporal jitter.
# Augmentation is applied within each SLOO training fold to avoid leakage.

@dataclass
class BankResample:
    """One channel-resample: the original pseudo-population + a per-channel bank
    of *all* eligible real-trial feature trajectories (for recombination)."""

    pop: PseudoPopulation                 # original, carries .source_trials
    bank_X: list                          # per channel: (n_elig_c, n_time) features
    bank_tidx: list                       # per channel: (n_elig_c,) real trial idx
    bank_y: list                          # per channel: (n_elig_c,) class label
    sub: list                             # the channel subsample (global indices)


def build_hg_bank(
    ds,
    pool: list[int],
    *,
    n_resamples: int,
    n_channels: int,
    ref: str = "Laplacian",
    downsample: int,
    smooth_ms: float = 488.0,
    baseline_mode: str = "zscore",
    feature_transform=None,
    n_trials_per_class: int = 3,
    seed: int = 0,
    verbose: bool = False,
) -> list[BankResample]:
    """Build HG pseudo-populations and per-channel trial banks for augmentation."""
    out: list[BankResample] = []
    fs = ds.fs
    band = HIGH_GAMMA_HZ
    fkw = dict(band=band, smooth_ms=smooth_ms, feature_transform=feature_transform,
               baseline_mode=baseline_mode, downsample=downsample)
    for r in range(n_resamples):
        sub = list(np.random.default_rng(seed + r).choice(
            pool, size=min(n_channels, len(pool)), replace=False))
        pop = build_pseudopopulation(
            ds, sub, ref=ref, bands=[band], n_trials_per_class=n_trials_per_class,
            smooth_ms=smooth_ms, feature_transform=feature_transform,
            baseline_mode=baseline_mode, downsample=downsample, seed=seed + r)
        target = np.unique(pop.y)
        bX, bT, bY = [], [], []
        for c_pos, ci in enumerate(pop.channel_indices):
            ch = ds.channel(ci)
            cls_all = ch.classes()
            idx, lab = [], []
            for cl in target:
                e = [i for i in np.where(cls_all == cl)[0] if ch.has_signal(i, ref=ref)]
                idx.extend(e); lab.extend([int(cl)] * len(e))
            sigs = np.vstack([ch.signal(i, ref=ref) for i in idx])
            feats, _t = hg_channel_features(sigs, fs, **fkw)
            bX.append(feats); bT.append(np.asarray(idx, dtype=int)); bY.append(np.asarray(lab, dtype=int))
        # consistency QC: reconstruct pop.X from the bank via source_trials
        st = pop.source_trials
        recon = np.empty_like(pop.X)
        for c_pos in range(pop.X.shape[1]):
            lut = {int(t): j for j, t in enumerate(bT[c_pos])}
            for k in range(pop.X.shape[0]):
                recon[k, c_pos] = bX[c_pos][lut[int(st[k, c_pos])]]
        assert np.allclose(recon, pop.X, atol=1e-9), "bank <-> pseudo-population mismatch"
        out.append(BankResample(pop=pop, bank_X=bX, bank_tidx=bT, bank_y=bY, sub=sub))
        if verbose:
            print(f"  bank resample {r}: {pop.X.shape}  bank sizes "
                  f"{[len(v) for v in bY][:5]}... (min {min(len(v) for v in bY)}, "
                  f"max {max(len(v) for v in bY)})")
    return out


# augmentation ops -- operate on the training rows only
def augment_recombination(br: BankResample, y: np.ndarray, train_rows, test_rows,
                          *, factor: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate same-class pseudo-populations by recombining channel trials while
    excluding trials used in held-out test rows."""
    n_ch = len(br.bank_X)
    n_time = br.bank_X[0].shape[1]
    st = br.pop.source_trials
    forbidden = [set(int(v) for v in st[test_rows, c]) for c in range(n_ch)]
    # eligible bank rows per (channel, class), test-source trials removed
    classes, counts = np.unique(y[train_rows], return_counts=True)
    per_class = {int(c): factor * int(n) for c, n in zip(classes, counts)}
    elig = {int(c): [np.array([j for j in np.where(br.bank_y[ch] == c)[0]
                               if int(br.bank_tidx[ch][j]) not in forbidden[ch]])
                     for ch in range(n_ch)] for c in classes}
    rows_X, rows_y, rows_prov = [], [], []
    for cls, k in per_class.items():
        ec = elig[cls]
        for _ in range(k):
            xr = np.empty((n_ch, n_time)); pr = np.empty(n_ch, dtype=int)
            for ch in range(n_ch):
                j = int(rng.choice(ec[ch]))
                xr[ch] = br.bank_X[ch][j]
                pr[ch] = int(br.bank_tidx[ch][j])
            rows_X.append(xr); rows_y.append(cls); rows_prov.append(pr)
    return (np.asarray(rows_X), np.asarray(rows_y, dtype=int),
            np.asarray(rows_prov, dtype=int))


def augment_mixup(X: np.ndarray, y: np.ndarray, train_rows,
                  *, factor: int, alpha: float, rng):
    """Generate within-class training samples using mixup."""
    tr = np.asarray(train_rows)
    classes, counts = np.unique(y[tr], return_counts=True)
    per_class = {int(c): factor * int(n) for c, n in zip(classes, counts)}
    Xs, ys, lams, pairs = [], [], [], []
    for cls, k in per_class.items():
        poolc = tr[y[tr] == cls]
        if poolc.size < 2:
            continue
        for _ in range(k):
            a, b = (int(v) for v in rng.choice(poolc, size=2, replace=False))
            # Beta(0.4, 0.4) has real mass at 0/1; clip to [0.1, 0.9] so every
            # synthetic sample is a genuine interpolation, not a near-duplicate.
            lam = float(np.clip(rng.beta(alpha, alpha), 0.1, 0.9))
            Xs.append(lam * X[a] + (1.0 - lam) * X[b])
            ys.append(cls); lams.append(lam); pairs.append((a, b))
    if not Xs:
        return (np.empty((0,) + X.shape[1:]), np.empty(0, dtype=int),
                np.empty(0), [])
    return np.asarray(Xs), np.asarray(ys, dtype=int), np.asarray(lams), pairs


def _shift_bin(a: np.ndarray, k: int) -> np.ndarray:
    """Non-circular ±k-bin shift along the last axis; edge values are replicated."""
    out = np.empty_like(a)
    if k > 0:
        out[..., :k] = a[..., :1]
        out[..., k:] = a[..., :-k]
    elif k < 0:
        out[..., k:] = a[..., -1:]
        out[..., :k] = a[..., -k:]
    else:
        out[...] = a
    return out


def augment_jitter(X_train: np.ndarray, y_train: np.ndarray):
    """Augment training data with ±1-bin temporal shifts."""
    m1 = _shift_bin(X_train, -1)
    p1 = _shift_bin(X_train, +1)
    return (np.concatenate([m1, p1], axis=0),
            np.concatenate([y_train, y_train]))


# augmented time-resolved evaluation
def run_augmented_condition(
    banks: list[BankResample],
    *,
    method: str,               # "none" | "recomb" | "mixup" | "jitter"
    factor: int,               # recomb/mixup: 1 or 3 (added = factor * n_train); ignored otherwise
    make_estimator,
    K: int | None,             # None -> all channels; else fold-wise ANOVA-F Top-K
    rank_window: tuple[float, float],
    n_cv_repeats: int,
    cue_window: tuple[float, float],
    delay_window: tuple[float, float],
    alpha: float = 0.4,        # mixup Beta(alpha, alpha)
    seed: int = 0,
    collect_diag: bool = False,
) -> dict:
    """Evaluate one training-data augmentation condition across channel resamples."""
    from sklearn.feature_selection import f_classif

    curves, cue_ws, del_ws = [], [], []
    diag: dict = {}
    for r, br in enumerate(banks):
        pop = br.pop
        X, y, t = pop.X, pop.y, pop.time_s
        n_ch, n_time = X.shape[1], X.shape[2]
        k = n_ch if K is None else min(K, n_ch)
        folds = sloo_folds(y, n_repeats=n_cv_repeats, seed=seed + r)
        rw = (t >= rank_window[0]) & (t < rank_window[1])
        arng = np.random.default_rng(10_000 + seed + r)     # augmentation RNG stream

        c = np.zeros(n_time)
        cue_fold, del_fold = [], []
        for fi, (tr, te) in enumerate(folds):
            # channel selection: original training rows only, kept identical across
            # augmentation conditions so this stays a paired comparison
            if K is None:
                sel = np.arange(n_ch)
            else:
                F, _ = f_classif(X[np.ix_(tr, np.arange(n_ch))][:, :, rw].mean(2), y[tr])
                F = np.nan_to_num(F, nan=-np.inf, posinf=-np.inf)
                sel = np.sort(np.argsort(F)[::-1][:k])

            # augment training rows only
            Xtr, ytr = X[tr], y[tr]
            if method == "none":
                Xa, ya = Xtr, ytr
            elif method == "recomb":
                Xs, ys, prov = augment_recombination(br, y, tr, te, factor=factor, rng=arng)
                Xa = np.concatenate([Xtr, Xs]); ya = np.concatenate([ytr, ys])
                if collect_diag and r == 0 and fi == 0:
                    forb = [set(int(v) for v in br.pop.source_trials[te, cc])
                            for cc in range(n_ch)]
                    assert all(int(prov[s, cc]) not in forb[cc]
                               for s in range(prov.shape[0]) for cc in range(n_ch)), \
                        "recomb leakage: synthetic row reused a test-source trial"
                    diag["recomb_prov_examples"] = prov[:2, :8].tolist()
                    diag["recomb_unique_sources_per_channel"] = [
                        int(np.unique(prov[:, cc]).size) for cc in range(min(n_ch, 8))]
                    diag["recomb_all_same_class"] = bool(np.all(
                        [ys[i] in set(np.unique(y)) for i in range(len(ys))]))
            elif method == "mixup":
                Xs, ys, lams, pairs = augment_mixup(X, y, tr, factor=factor, alpha=alpha, rng=arng)
                Xa = np.concatenate([Xtr, Xs]) if len(Xs) else Xtr
                ya = np.concatenate([ytr, ys]) if len(ys) else ytr
                if collect_diag and r == 0 and fi == 0 and len(Xs):
                    assert all(a in set(tr.tolist()) and b in set(tr.tolist())
                               for a, b in pairs), "mixup pair not in training"
                    assert all(y[a] == y[b] for a, b in pairs), "mixup pair cross-class"
                    da = np.array([np.linalg.norm(Xs[i] - X[pairs[i][0]]) for i in range(len(Xs))])
                    db = np.array([np.linalg.norm(Xs[i] - X[pairs[i][1]]) for i in range(len(Xs))])
                    diag["mixup_lambda"] = lams.tolist()
                    diag["mixup_dist_to_a_over_b"] = (da / (db + 1e-12)).round(3).tolist()
                    diag["mixup_min_dist"] = float(min(da.min(), db.min()))  # >0 -> not a duplicate
            elif method == "jitter":
                Xs, ys = augment_jitter(Xtr, ytr)
                Xa = np.concatenate([Xtr, Xs]); ya = np.concatenate([ytr, ys])
                if collect_diag and r == 0 and fi == 0:
                    ex = Xtr[0, 0]
                    diag["jitter_example_orig"] = ex[:6].round(3).tolist()
                    diag["jitter_example_minus1"] = _shift_bin(ex[None, None], -1)[0, 0][:6].round(3).tolist()
                    diag["jitter_no_wrap"] = bool(_shift_bin(ex[None, None], 1)[0, 0, 0] == ex[0])
            else:
                raise ValueError(method)

            # time-resolved decode; predict on the original test rows
            Xte = X[te]
            for ti in range(n_time):
                clf = make_estimator()
                clf.fit(Xa[:, sel, ti], ya)
                c[ti] += float(np.mean(clf.predict(Xte[:, sel, ti]) == y[te])) / len(folds)
            for win, store in ((cue_window, cue_fold), (delay_window, del_fold)):
                wm = (t >= win[0]) & (t < win[1])
                clf = make_estimator()
                clf.fit(Xa[:, sel][:, :, wm].mean(2), ya)
                store.append(float(np.mean(clf.predict(Xte[:, sel][:, :, wm].mean(2)) == y[te])))

            if collect_diag and r == 0 and fi == 0:
                diag["n_train_orig"] = int(len(tr))
                diag["n_train_aug"] = int(len(ya))
                diag["n_test"] = int(len(te))
                diag["p"] = int(k)
                diag["p_over_n_train_orig"] = k / len(tr)
                diag["p_over_n_train_aug"] = k / len(ya)

        curves.append(c)
        cue_ws.append(float(np.mean(cue_fold)))
        del_ws.append(float(np.mean(del_fold)))

    res = _pack(curves, cue_ws, del_ws, banks[0].pop.time_s,
                n_features=(banks[0].pop.X.shape[1] if K is None else min(K, banks[0].pop.X.shape[1])),
                n_trials=banks[0].pop.X.shape[0],
                n_classes=int(np.unique(banks[0].pop.y).size),
                cue_window=cue_window, delay_window=delay_window,
                extra={"method": method, "factor": factor, "diag": diag})
    return res
