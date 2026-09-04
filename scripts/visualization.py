"""
BrainHack Vandy 2026: Milestone 1 - Signal Inspection & Visualization
This script loads a sample sEEG dataset, inspects channel metadata,
and compares Raw, Common-Average, and Laplacian referenced signals.
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import welch

# =============================================================================
# 1. Configuration & Data Loading
# =============================================================================
data_path = 'data/Spatial_Task_SEEG.mat'

if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Dataset not found at '{data_path}'! Please download from Box and place it in the data/ folder."
    )

print(f"Loading dataset from {data_path}...")
mat_data = sio.loadmat(data_path, squeeze_me=True, struct_as_record=False)
seeg = mat_data['SEEG']
print(f"Loaded {len(seeg)} channels successfully.")

# =============================================================================
# 2. Select a Channel and Trial for Inspection
# =============================================================================
channel_idx = 0  # 0-indexed in Python (corresponds to channel 1 in MATLAB)
trial_idx = 0    # 0-indexed trial within channel

chan_data = seeg[channel_idx]

# Safely extract string labels whether they are arrays or strings
sub = str(chan_data.Sub)
label = str(chan_data.Channel_Label)
subregion = str(np.atleast_1d(chan_data.subRegion)[0])
subdiv = str(np.atleast_1d(chan_data.Prefrontal_subdiv)[0])

print("\n--- Channel Information ---")
print(f"Subject: {sub} | Contact: {label} | Subregion: {subregion} ({subdiv})")

# Extract trials struct
trials = None
if hasattr(chan_data, 'CorrectTrials') and np.size(chan_data.CorrectTrials) > 0:
    trials = chan_data.CorrectTrials
elif hasattr(chan_data, 'Correct') and np.size(chan_data.Correct) > 0:
    trials = chan_data.Correct
else:
    raise ValueError(f"No correct trials found for channel index {channel_idx}")

trials = np.atleast_1d(trials)
trial_struct = trials[trial_idx]

raw_sig = np.asarray(trial_struct.TrialData, dtype=np.float64)
car_sig = np.asarray(trial_struct.Common, dtype=np.float64)
laplacian_sig = np.asarray(trial_struct.Laplacian, dtype=np.float64)
stim_class = int(trial_struct.Class)

# Sampling rate configuration
fs = 512  # Hz
time_vector = np.arange(len(raw_sig)) / fs

# =============================================================================
# 3. Plot Raw vs. CAR vs. Laplacian Waveforms
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.canvas.manager.set_window_title('sEEG LFP Referencing Comparison')

axes[0].plot(time_vector, raw_sig, color='black', linewidth=1)
axes[0].set_title(f"Raw sEEG LFP (Subject: {sub}, Contact: {label}, Class: {stim_class})")
axes[0].set_ylabel(r"Amplitude ($\mu$V)")
axes[0].grid(True, linestyle='--', alpha=0.6)

axes[1].plot(time_vector, car_sig, color='tab:blue', linewidth=1)
axes[1].set_title("Common-Average-Referenced (CAR) LFP")
axes[1].set_ylabel(r"Amplitude ($\mu$V)")
axes[1].grid(True, linestyle='--', alpha=0.6)

axes[2].plot(time_vector, laplacian_sig, color='tab:red', linewidth=1)
axes[2].set_title("Laplacian-Referenced LFP")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel(r"Amplitude ($\mu$V)")
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# =============================================================================
# 4. TODO Tasks: Power Spectrum & Channel Filtering
# =============================================================================
# [TASK 1.1]: Compute and plot the Power Spectral Density (PSD) using scipy.signal.welch
#             for raw_sig, car_sig, and laplacian_sig to compare high-gamma (70-150 Hz) power.
# -------------------------------------------------------------------------
# Hint: freqs, psd = welch(raw_sig, fs=fs, nperseg=256)
# >> YOUR CODE HERE <<



# [TASK 1.2]: Iterate across all channels in seeg, count the number of correct
#             trials per class, and return a list of "viable channels" that meet
#             a minimum threshold (e.g., at least 3 trials per class).
# -------------------------------------------------------------------------
# >> YOUR CODE HERE <<
