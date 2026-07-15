# Decoding the Human Brain 🧠✨

Welcome to the official repository for the **BrainHack Vanderbilt 2026** project! 

This project explores how the human prefrontal cortex dynamically encodes and maintains environmental architecture during spatial and feature-based working memory tasks using rare human intracranial stereo-EEG (sEEG) recordings.

---

## 📂 The Data
Because the raw data files are incredibly large (~2 GB each), **do not upload them to GitHub**. 

* **Full Datasets (Download Here):** 
  * *Spatial Task Dataset* (https://vanderbilt.box.com/s/gfh6lnjqqzm0n31gty8c4h3jj82rbnyn)
  * *Feature Task Dataset* (https://vanderbilt.box.com/s/jw50ka2281j2dh95it49dle5qbdvftpb)

### Data Format & Schema
The data are stored in `.mat` files containing a single variable `SEEG`, which is a `1 × 462` struct array. Each element in the array represents a unique recording channel pooled across different subjects. 

Each channel struct has the following fields:
*   `Sub`: Subject identifier (string/character array)
*   `Task`: Task type (e.g., `'Delay3s'` or `'MNM'`)
*   `Condition`: Experimental condition metadata
*   `Channel`: Numeric channel ID
*   `Channel_Label`: Clinical label of the electrode contact
*   `subRegion`: Specific anatomical subregion location
*   `Prefrontal_subdiv`: Prefrontal subdivision (e.g., `'Dorsal'`, `'Ventral'`) used for our primary comparative analysis
*   `Hemisphere`: Hemisphere of the implant
*   `CorrectTrials` or `Correct`: A struct array containing trials that were performed correctly. Each element/trial in this array has the following fields for which we care about:
    *   `Class`: The stimulus class location (1-9 or 1-8)
    *   `TrialData`: The raw time-series voltage data for the trial
    *   `Common`: Common-average reference (CAR) filtered time-series data
    *   `Laplacian`: Local Laplacian-filtered time-series data
    *   *(Note: CueShape, MatchShape etc are task-phase metadata which can be bypassed for standard decoding)*

---

### Quickstart: Accessing Trials and Filter Variants

#### In MATLAB:
```matlab
% Load the dataset
load('... .mat'); % Loads the SEEG struct array

% Get the class and Laplacian-filtered data for the first correct trial of Channel 1
first_trial = SEEG(1).CorrectTrials(1);
trial_class = first_trial.Class;
trial_signal = first_trial.Laplacian; % You can also use .TrialData or .Common

% Quick check: Count how many correct trials exist for this channel
num_correct = length(SEEG(1).Correct);
```
---

## 🚀 BrainHack Milestones

### 🟢 Milestone 1: Data Cleaning & Visualization (Low Complexity)
* **Goal:** Parse the raw `.mat` files. Identify and filter out noisy clinical channels that lack a sufficient trial count per class.
* **Deliverable:** A script (`scripts/...`) that returns "viable" channels and plots their power spectra.

### 🟡 Milestone 2: Feature Engineering (Medium Complexity)
* **Goal:** Benchmark alternative features against our baseline 70–150 Hz high-gamma moving average.
* **Deliverable:** Implement alternative features in `scripts/...`.

### 🔴 Milestone 3: Time-Resolved Low-Sample Decoding (High Complexity)
* **Goal:** Build time-resolved SVM classifiers (which handle small sample sizes like our 3 trials/class brilliantly) to decode classes.
* **Deliverable:** Map and compare decoding accuracies over time between the **Dorsal** and **Ventral** subregions.

---

## 📖 Scientific Background & Baseline Logic
Before diving into code, please read our comprehensive decoding draft document:
👉 **[Read the Decoding Logic & Existing Results Guide](docs/DecodingLogic_Hackathon.pdf)**

This document details the logic behind our current high-gamma SVM approach and shows our existing Non-Human Primate (NHP) and human decoding results.

---

## 🛠️ Getting Started (MATLAB Setup)
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/oDtsou/human-decoding-hackathon.git](https://github.com/oDtsou/human-decoding-hackathon.git)
cd human-decoding-hackathon
```
```matlab
% Add project directories to path
addpath(genpath(pwd));
savepath;
```


