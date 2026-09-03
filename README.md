# Decoding the Human Brain 🧠✨

Welcome to the official repository for the **BrainHack Vanderbilt 2026** project! 

This project explores how the human prefrontal cortex dynamically encodes and maintains environmental architecture during spatial and feature-based working memory tasks using human intracranial stereo-EEG (sEEG) recordings.

---

## 📂 The Data & Sharing Permissions

### Data-Use and Sharing Policy
The intracranial sEEG recordings provided for this hackathon contain de-identified human electrophysiological research data. 
* **Permitted Use:** These data are provided exclusively for academic, educational, and benchmarking purposes within the scope of BrainHack Vanderbilt 2026.
* **Redistribution:** Do not redistribute, publicly host, or share these download links outside of the event without prior consent from the lab.

Because raw data files are large (~2 GB each), **they are not tracked in GitHub**.

* **Full Datasets (Download Here):** 
  * [Spatial Task Dataset](https://vanderbilt.box.com/s/gfh6lnjqqzm0n31gty8c4h3jj82rbnyn)
  * [Feature Task Dataset](https://vanderbilt.box.com/s/jw50ka2281j2dh95it49dle5qbdvftpb)

---

## 📋 Data Format & Schema

The data are stored in `.mat` files containing a single variable `SEEG`, which is a `1 × N` struct array where `N` is the number of channels pooled across subjects.

Each channel struct includes:
* `Sub`: Subject identifier (e.g., `'sub-01'`)
* `Task`: Task type (`'Delay3s'` or `'MNM'`)
* `Condition`: Experimental condition metadata
* `Channel`: Numeric channel ID
* `Channel_Label`: Clinical label of the electrode contact
* `subRegion`: Specific anatomical subregion
* `Prefrontal_subdiv`: Prefrontal subdivision (`'Dorsal'`, `'Ventral'`) used for primary comparative analysis
* `Hemisphere`: Hemisphere of the implant (`'L'` or `'R'`)
* `CorrectTrials` / `Correct`: Struct array containing successfully completed trials with fields:
  * `Class`: Stimulus class label (`1–9` or `'1–8'`)
  * `TrialData`: Raw time-series voltage data
  * `Common`: Common-average reference (CAR) filtered time-series data
  * `Laplacian`: Local Laplacian-filtered time-series data

---

## 🔬 Signal Referencing Schemes

* **Raw sEEG LFP (`TrialData`):** Unreferenced contact voltage $V_i(t)$.

* **Common-Average Reference (`Common`):** Subtracts the mean signal across all selected shaft contacts:

$$V^{\text{CAR}}_i(t) = V_i(t) - \frac{1}{N} \sum_{j=1}^{N} V_j(t)$$

*Reduces global/shared noise across the array; may attenuate broadly distributed activity.*

* **Laplacian Reference (`Laplacian`):** Subtracts the average of the two immediately adjacent contacts along the same shaft:

$$V^{\text{LAP}}_i(t) = V_i(t) - \frac{V_{i-1}(t) + V_{i+1}(t)}{2}$$

*Suppresses volume-conducted signals and isolates spatially focal neural activity.*

> **Research Opportunity:** Attendees are encouraged to benchmark decoding accuracy across raw (`TrialData`), `Common`, and `Laplacian` signals to evaluate how spatial filtering impacts classification performance.

---

## 🚀 BrainHack Milestones

### 🟢 Milestone 1: Data Cleaning & Visualization (Low Complexity)
* **Goal:** Parse `.mat` files, evaluate trial balances across classes, and remove noisy/sparse channels.
* **Deliverable:** Run and extend `scripts/visualization.m` to plot raw vs. CAR vs. Laplacian traces and power spectral densities (PSD).
* To run it:
       1. Download the dataset and place it in the `data/` folder.
       2. Open and run `scripts/visualization.m` in MATLAB.

### 🟡 Milestone 2: Feature Engineering & Dimensionality Reduction (Medium Complexity)
* **Goal:** Benchmark alternative feature representations against the baseline 70–150 Hz high-gamma moving-average envelope.
* **Deliverables & Tasks:**
  * **Continuous vs. Binned Signal:** Test feature extraction across the entire continuous trial time course versus static epoch/bin averages.
  * **Multi-Band Spectral Fusion:** Extract and combine features from multiple canonical frequency bands simultaneously (e.g., theta, beta, and high-gamma).
  * Implement these pipelines in `scripts/feature_extraction.m`.

---

### 🔴 Milestone 3: Time-Resolved Decoding & Subregion Dynamics (High Complexity)
* **Goal:** Build classifiers (e.g., Support Vector Machines with RBF kernels, regularized LDA) resilient to low-sample regimes (~3 trials per class) to decode spatial and feature targets over time.
* **Deliverables:**
  * Compute and contrast time-resolved classification trajectories between **Dorsal** and **Ventral** prefrontal subdivisions.
  * Identify latency peaks to determine which subregion encodes task variables earlier during the cue presentation and maintains them throughout the delay period.
  * Implement in `scripts/decode_subregions.m`.

---

### 🟣 Milestone 4: Advanced Modeling & Benchmarking (Wish List / Extension)
* **Goal:** Optimize pipeline robustness and evaluate core neuroengineering hypotheses.
* **Deliverables & Tasks:**
  * **Spatial Referencing Benchmark:** Systematically compare decoding accuracy across **Raw-ONLY**, **Common-ONLY**, and **Laplacian-ONLY** signals to determine which reference optimizes neural discriminability.
  * **Resampling & Synthetic Trials:** Develop strategies to overcome sample-size constraints (e.g., trial smoothing, jittering, or synthetic interpolation) to train on >3 trials per class.

---

## 📖 Scientific Background & Baseline Logic
Before diving into code, please read our comprehensive decoding draft document:
👉 **[Read the Decoding Logic & Existing Results Guide](docs/DecodingLogic_Hackathon.pdf)**

This document details the logic behind our current high-gamma SVM approach and shows our existing Non-Human Primate (NHP) and human decoding results.

---

## 🛠️ Getting Started & Prerequisites

### 1. MATLAB Installation & Licensing
This project requires MATLAB (R2022b or later recommended) with the Statistics and Machine Learning Toolbox and Signal Processing Toolbox.
* Download & installation instructions: [Vanderbilt Software Store - MATLAB](https://engineering.vanderbilt.edu/sds/matlab)
* *Note:* A valid Vanderbilt University MATLAB license is required ([License Portal](https://engineering.vanderbilt.edu/sds/matlab/)).

### 2. Repository Setup
```bash
git clone [https://github.com/oDtsou/human-decoding-hackathon.git](https://github.com/oDtsou/human-decoding-hackathon.git)
cd human-decoding-hackathon
```
Then, open MATLAB, navigate to the cloned directory, and run the following in the MATLAB Command Window:

```Matlab
% Add all subfolders to the MATLAB search path
addpath(genpath(pwd));
savepath;
