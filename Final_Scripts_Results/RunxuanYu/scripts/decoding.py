"""
Baseline high-gamma SVM decoding
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from seeg_io import SEEGDataset

HIGH_GAMMA_HZ: tuple[float, float] = (70.0, 150.0)

# Canonical bands used for the multi-band comparison (docs/DecodingLogic_Hackathon.pdf)
CANONICAL_BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "high_gamma": (70.0, 150.0),
}


# feature extraction: high-gamma amplitude envelope
def high_gamma_envelope(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float] = HIGH_GAMMA_HZ,
    env_lpf_hz: float = 10.0,
) -> np.ndarray:
    """Bandpass -> |Hilbert| -> low-pass smoothed amplitude envelope. Last axis = time."""
    x = np.asarray(x, dtype=np.float64)
    nyq = 0.5 * fs
    b, a = butter(4, [band[0] / nyq, band[1] / nyq], btype="bandpass")
    xf = filtfilt(b, a, x, axis=-1)
    env = np.abs(hilbert(xf, axis=-1))
    bl, al = butter(2, env_lpf_hz / nyq, btype="low")
    return filtfilt(bl, al, env, axis=-1)


def hg_channel_features(
    sigs: np.ndarray,
    fs: float,
    *,
    band: tuple[float, float] = HIGH_GAMMA_HZ,
    smooth_ms: float = 488.0,
    feature_transform=None,
    baseline_mode: str = "zscore",
    zscore_window_s: tuple[float, float] = (0.25, 0.75),
    downsample: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel HG feature pipeline. """
    win = max(1, int(round(smooth_ms / 1000.0 * fs)))
    env = high_gamma_envelope(sigs, fs, band=band)
    env = moving_average(env, win)
    if feature_transform is not None:
        env = feature_transform(env)
    n_samp = env.shape[1]
    t = np.arange(n_samp) / fs
    m = (t >= zscore_window_s[0]) & (t < zscore_window_s[1])
    mu = env[:, m].mean(axis=1, keepdims=True)
    if baseline_mode == "zscore":
        sd = env[:, m].std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        env = (env - mu) / sd
    elif baseline_mode == "subtract":
        env = env - mu
    elif baseline_mode == "relchange":
        env = (env - mu) / np.where(np.abs(mu) < 1e-8, 1.0, mu)
    elif baseline_mode == "none":
        pass
    else:
        raise ValueError(f"unknown baseline_mode {baseline_mode!r}")
    return env[:, ::downsample], t[::downsample]


def moving_average(x: np.ndarray, win_samples: int) -> np.ndarray:
    """Centered moving average along the last axis (edge-padded)."""
    if win_samples <= 1:
        return x
    kernel = np.ones(win_samples) / win_samples
    pad = win_samples // 2
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="edge")
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), -1, xp)[
        ..., : x.shape[-1]
    ]


# pseudo-population tensor
@dataclass
class PseudoPopulation:
    """Build a trial × feature × time band-power tensor for one region.
    For multiple bands, features correspond to channel-band pairs."""

    X: np.ndarray          # (n_trials, n_channels * n_bands, n_time)
    y: np.ndarray          # (n_trials,) class labels
    time_s: np.ndarray     # (n_time,) seconds
    channel_indices: list[int] = field(default_factory=list)
    bands: list[tuple[float, float]] = field(default_factory=list)
    # Source trial indices used for leakage-safe pseudo-population recombination.
    source_trials: "np.ndarray | None" = None

    @property
    def n_classes(self) -> int:
        return int(np.unique(self.y).size)

    @property
    def chance(self) -> float:
        return 1.0 / self.n_classes

    @property
    def n_bands(self) -> int:
        return max(1, len(self.bands))


def build_pseudopopulation(
    ds: SEEGDataset,
    channel_indices: list[int],
    *,
    ref: str = "Laplacian",
    n_trials_per_class: int = 3,
    smooth_ms: float = 488.0,
    downsample: int = 10,
    zscore_window_s: tuple[float, float] = (0.25, 0.75),
    baseline_mode: str = "zscore",
    band: tuple[float, float] = HIGH_GAMMA_HZ,
    bands: list[tuple[float, float]] | None = None,
    feature_transform=None,
    classes: np.ndarray | None = None,
    seed: int = 0,
) -> PseudoPopulation:
    """Build a pseudo-population tensor from the selected channels and trials."""
    rng = np.random.default_rng(seed)
    fs = ds.fs
    win = max(1, int(round(smooth_ms / 1000.0 * fs)))
    use_bands = list(bands) if bands is not None else [band]

    if classes is None:
        class_sets = []
        for ci in channel_indices:
            cc = ds.channel(ci).classes()
            enough = [c for c in np.unique(cc) if np.sum(cc == c) >= n_trials_per_class]
            if enough:
                class_sets.append(set(enough))
        if not class_sets:
            raise ValueError("no channel has enough trials in any class")
        classes = np.array(sorted(set.intersection(*class_sets)))
    classes = np.asarray(classes)

    per_channel: list[np.ndarray] = []
    kept: list[int] = []
    src_per_channel: list[np.ndarray] = []      # real trial idx per (row) for each kept channel
    labels_ref: np.ndarray | None = None
    time_s: np.ndarray | None = None

    for ci in channel_indices:
        ch = ds.channel(ci)
        cls = ch.classes()
        if cls.size == 0:
            continue

        # pick trial indices for this channel, class by class
        picks: list[int] = []
        labels: list[int] = []
        ok = True
        for c in classes:
            idx = np.where(cls == c)[0]
            # keep only trials with a valid signal for this reference
            idx = np.array([i for i in idx if ch.has_signal(i, ref=ref)])
            if idx.size < n_trials_per_class:
                ok = False
                break
            sel = rng.choice(idx, size=n_trials_per_class, replace=False)
            picks.extend(sel.tolist())
            labels.extend([int(c)] * n_trials_per_class)
        if not ok:
            continue

        sigs = np.vstack([ch.signal(i, ref=ref) for i in picks])   # (n_pick, n_samp)

        # one amplitude envelope per band, identical preprocessing for each
        for bd in use_bands:
            env = high_gamma_envelope(sigs, fs, band=bd)
            env = moving_average(env, win)
            if feature_transform is not None:
                env = feature_transform(env)           # e.g. log(env + eps)

            n_samp = env.shape[1]
            if time_s is None:
                full_t = np.arange(n_samp) / fs
            # per (trial x channel) normalisation to the fixation window
            z0, z1 = zscore_window_s
            m = (np.arange(n_samp) / fs >= z0) & (np.arange(n_samp) / fs < z1)
            base_mean = env[:, m].mean(axis=1, keepdims=True)
            if baseline_mode == "zscore":
                base_std = env[:, m].std(axis=1, keepdims=True)
                base_std[base_std == 0] = 1.0
                env = (env - base_mean) / base_std
            elif baseline_mode == "subtract":
                env = env - base_mean
            elif baseline_mode == "relchange":
                denom = np.where(np.abs(base_mean) < 1e-8, 1.0, base_mean)
                env = (env - base_mean) / denom
            elif baseline_mode == "none":
                pass
            else:
                raise ValueError(f"unknown baseline_mode {baseline_mode!r}")

            env = env[:, ::downsample]                              # downsample time
            if time_s is None:
                time_s = full_t[::downsample]
            per_channel.append(env)

        kept.append(ci)
        src_per_channel.append(np.asarray(picks, dtype=int))   # (n_trials,) real trial idx
        if labels_ref is None:
            labels_ref = np.array(labels, dtype=int)

    if not per_channel:
        raise ValueError("no channel could supply the requested trials/class")

    # per_channel is [ch0·b0, ch0·b1, …, ch1·b0, …]  ->  (n_trials, n_kept*n_bands, n_time)
    X = np.stack(per_channel, axis=1)
    return PseudoPopulation(
        X=X, y=labels_ref, time_s=time_s, channel_indices=kept, bands=use_bands,
        source_trials=np.stack(src_per_channel, axis=1),       # (n_trials, n_kept)
    )


# cross-validation + classifier
def sloo_folds(
    y: np.ndarray, n_repeats: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate stratified leave-one-per-class-out cross-validation folds."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    by_class = {c: np.where(y == c)[0] for c in classes}
    k = min(len(v) for v in by_class.values())

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_repeats):
        order = {c: rng.permutation(v) for c, v in by_class.items()}
        for f in range(k):
            test = np.array([order[c][f] for c in classes])
            train = np.array([i for i in range(len(y)) if i not in set(test.tolist())])
            folds.append((train, test))
    return folds


def _svm(C: float) -> "object":
    # LinearSVC = linear kernel, One-vs-Rest (== One-vs-All), L2 penalty.
    return make_pipeline(
        StandardScaler(),
        LinearSVC(C=C, dual="auto", max_iter=5000),
    )


# Fit a fresh estimator within each fold to avoid data leakage.
def cv_score(X_t: np.ndarray, y: np.ndarray, folds, make_estimator) -> float:
    """Mean CV accuracy on one feature matrix X_t (n_trials, n_features)."""
    accs = []
    for train, test in folds:
        clf = make_estimator()
        clf.fit(X_t[train], y[train])
        accs.append(float(np.mean(clf.predict(X_t[test]) == y[test])))
    return float(np.mean(accs))


def time_resolved_scores(pop: "PseudoPopulation", folds, make_estimator) -> np.ndarray:
    """Per-time-bin CV accuracy for an arbitrary estimator factory. Returns (n_time,)."""
    return np.array(
        [cv_score(pop.X[:, :, t], pop.y, folds, make_estimator) for t in range(pop.X.shape[2])]
    )


def window_score(
    pop: "PseudoPopulation", window_s: tuple[float, float], folds, make_estimator
) -> float:
    """CV accuracy from features averaged over a time window, arbitrary estimator."""
    m = (pop.time_s >= window_s[0]) & (pop.time_s < window_s[1])
    return cv_score(pop.X[:, :, m].mean(axis=2), pop.y, folds, make_estimator)


# linear-SVM convenience wrappers (used by the baseline notebooks)
def decode_timepoint(X_t: np.ndarray, y: np.ndarray, folds, C: float = 0.1) -> float:
    """Mean CV accuracy at one time bin with the linear-SVM baseline."""
    return cv_score(X_t, y, folds, lambda: _svm(C))


def time_resolved_accuracy(
    pop: PseudoPopulation, *, C: float = 0.1, n_cv_repeats: int = 5, seed: int = 0
) -> np.ndarray:
    """Linear-SVM decoding accuracy at every time bin. Returns (n_time,)."""
    folds = sloo_folds(pop.y, n_repeats=n_cv_repeats, seed=seed)
    return time_resolved_scores(pop, folds, lambda: _svm(C))


def static_accuracy(
    pop: PseudoPopulation,
    window_s: tuple[float, float],
    *,
    C: float = 0.1,
    n_cv_repeats: int = 5,
    seed: int = 0,
) -> float:
    """Linear-SVM decoding accuracy from features averaged over a time window."""
    folds = sloo_folds(pop.y, n_repeats=n_cv_repeats, seed=seed)
    return window_score(pop, window_s, folds, lambda: _svm(C))


# region comparison with channel resampling
def region_decoding_curve(
    ds: SEEGDataset,
    region_channels: dict[str, list[int]],
    *,
    ref: str = "Laplacian",
    n_channels: int | None = None,
    n_channel_resamples: int = 5,
    n_trials_per_class: int = 3,
    C: float = 0.1,
    n_cv_repeats: int = 5,
    downsample: int = 10,
    smooth_ms: float = 488.0,
    bands: list[tuple[float, float]] | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute time-resolved decoding accuracy across random channel subsamples."""
    if n_channels is None:
        n_channels = min(len(v) for v in region_channels.values())

    # one shared class set for every region so chance levels match
    all_chans = [c for v in region_channels.values() for c in v]
    shared_classes: set[int] = set()
    for ci in all_chans:
        cc = ds.channel(ci).classes()
        enough = {int(c) for c in np.unique(cc) if np.sum(cc == c) >= n_trials_per_class}
        shared_classes = enough if not shared_classes else (shared_classes & enough)
    classes = np.array(sorted(shared_classes))
    if verbose:
        print(f"shared classes: {classes.tolist()}  (chance = {1/len(classes):.3f})")

    out: dict[str, dict[str, np.ndarray]] = {}
    for region, chans in region_channels.items():
        rng = np.random.default_rng(seed)
        curves = []
        time_s = chance = None
        for r in range(n_channel_resamples):
            sub = rng.choice(chans, size=min(n_channels, len(chans)), replace=False)
            pop = build_pseudopopulation(
                ds, sub.tolist(), ref=ref, n_trials_per_class=n_trials_per_class,
                smooth_ms=smooth_ms, downsample=downsample, bands=bands,
                classes=classes, seed=seed + r,
            )
            curves.append(time_resolved_accuracy(pop, C=C, n_cv_repeats=n_cv_repeats, seed=seed + r))
            time_s, chance = pop.time_s, pop.chance
            if verbose:
                print(f"  {region}: resample {r + 1}/{n_channel_resamples} "
                      f"(peak acc {curves[-1].max():.2f})")
        curves = np.array(curves)
        out[region] = {
            "mean": curves.mean(axis=0),
            "sem": curves.std(axis=0) / np.sqrt(curves.shape[0]),
            "time_s": time_s,
            "chance": chance,
        }
    return out
