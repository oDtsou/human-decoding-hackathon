"""
extract_bandpass_features.py
Extracts bandpass amplitude envelopes across trials and channels,
separating features by Prefrontal Subdivision (Dorsal vs. Ventral).
"""

import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, hilbert

# =============================================================================
# 1. Helper Signal Processing Functions
# =============================================================================

def bandpass_amplitude_envelope(data, fs, locutoff, hicutoff, env_lpf=10.0):
    """
    Computes the smoothed analytic amplitude envelope:
    1. Bandpass filters the signal (Butterworth 4th-order forward-backward).
    2. Computes the analytic amplitude via Hilbert transform (|hilbert(x)|).
    3. Smooths the envelope with a low-pass filter (Butterworth 2nd-order at 10 Hz).
    """
    data = np.asarray(data, dtype=np.float64)
    nyq = 0.5 * fs

    # 1. Bandpass filter
    b_bp, a_bp = butter(4, [locutoff / nyq, hicutoff / nyq], btype='bandpass')
    bp_data = filtfilt(b_bp, a_bp, data, axis=-1)

    # 2. Analytic amplitude envelope
    envelope = np.abs(hilbert(bp_data, axis=-1))

    # 3. Low-pass smoothing
    b_lp, a_lp = butter(2, env_lpf / nyq, btype='low')
    smooth_envelope = filtfilt(b_lp, a_lp, envelope, axis=-1)

    return smooth_envelope


# =============================================================================
# 2. Parameters & Configuration
# =============================================================================

LO_FREQ = 70.0      # High-Gamma lower cutoff (Hz)
HI_FREQ = 150.0     # High-Gamma upper cutoff (Hz)
FS = 512           # Sampling rate (Hz)

# Python 0-indexed slice matching MATLAB's 513:2500 (inclusive)
TIME_START = 513
TIME_END = 2500

REF_SCHEME = 'Laplacian'  # 'TrialData', 'Common', or 'Laplacian'
DATA_PATH = 'data/Spatial_Task_SEEG.mat'
OUT_PATH = 'data/Extracted_Features_Dorsal_Ventral.npz'


# =============================================================================
# 3. Load & Process Dataset
# =============================================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at '{DATA_PATH}'. Place it in data/ folder.")

print(f"Loading dataset from {DATA_PATH}...")
# squeeze_me=True and struct_as_record=False unpacks structs into accessible objects
mat_data = sio.loadmat(DATA_PATH, squeeze_me=True, struct_as_record=False)
seeg = mat_data['SEEG']

dorsal_data, dorsal_class = [], []
ventral_data, ventral_class = [], []

print(f"Processing {len(seeg)} channels...")

for chan in seeg:
    # Resolve trials field variation
    trials = None
    if hasattr(chan, 'CorrectTrials') and np.size(chan.CorrectTrials) > 0:
        trials = chan.CorrectTrials
    elif hasattr(chan, 'Correct') and np.size(chan.Correct) > 0:
        trials = chan.Correct
    else:
        continue

    # Ensure trials is an iterable array
    trials = np.atleast_1d(trials)
    num_trials = len(trials)
    if num_trials == 0:
        continue

    chan_features = np.zeros((num_trials, TIME_END - TIME_START))
    chan_labels = np.zeros(num_trials, dtype=int)

    for tr_idx, tr in enumerate(trials):
        sig = getattr(tr, REF_SCHEME)
        env = bandpass_amplitude_envelope(sig, FS, LO_FREQ, HI_FREQ)
        
        chan_features[tr_idx, :] = env[TIME_START:TIME_END]
        chan_labels[tr_idx] = int(tr.Class)

    subdiv = str(getattr(chan, 'Prefrontal_subdiv', '')).strip().lower()

    if subdiv == 'dorsal':
        dorsal_data.append(chan_features)
        dorsal_class.append(chan_labels)
    elif subdiv == 'ventral':
        ventral_data.append(chan_features)
        ventral_class.append(chan_labels)

print(f"Done! Processed {len(dorsal_data)} Dorsal and {len(ventral_data)} Ventral channels.")


# =============================================================================
# 4. Save Extracted Features
# =============================================================================

np.savez_compressed(
    OUT_PATH,
    DorsalData=np.array(dorsal_data, dtype=object),
    DorsalClass=np.array(dorsal_class, dtype=object),
    VentralData=np.array(ventral_data, dtype=object),
    VentralClass=np.array(ventral_class, dtype=object)
)
print(f"Features successfully saved to '{OUT_PATH}'.")
