%% BrainHack Vandy 2026: 
% For example:
% Milestone 1 - Signal Inspection & Visualization
% This script loads a sample sEEG dataset, inspects channel trial counts,
% and compares Raw, Common-Average, and Laplacian referenced signals.

clear; close all; clc;

%% 1. Configuration & Data Loading
% =========================================================================
% UPDATE THIS PATH to your downloaded .mat file
dataPath = 'data/Spatial_Task_SEEG.mat'; 

if ~exist(dataPath, 'file')
    error('Dataset not found! Please download from Box and place it in the /data folder.');
end

fprintf('Loading dataset from %s...\n', dataPath);
loadedData = load(dataPath);
SEEG = loadedData.SEEG;
fprintf('Loaded %d channels successfully.\n', length(SEEG));

%% 2. Select a Channel and Trial for Inspection
% =========================================================================
channelIdx = 1; % Index in SEEG struct array
trialIdx   = 1; % Trial index within channel

chanData = SEEG(channelIdx);
fprintf('\n--- Channel Information ---\n');
fprintf('Subject: %s | Contact: %s | Subregion: %s (%s)\n', ...
    chanData.Sub, chanData.Channel_Label, chanData.subRegion{:}, chanData.Prefrontal_subdiv{:});

% Extract trial data across filter schemes
if isfield(chanData, 'CorrectTrials') && ~isempty(chanData.CorrectTrials)
    trialStruct = chanData.CorrectTrials(trialIdx);
elseif isfield(chanData, 'Correct') && ~isempty(chanData.Correct)
    trialStruct = chanData.Correct(trialIdx);
else
    error('No correct trials found for channel index %d', channelIdx);
end

rawSig       = trialStruct.TrialData;
carSig       = trialStruct.Common;
laplacianSig = trialStruct.Laplacian;
stimClass    = trialStruct.Class;

% Sampling rate configuration 
Fs = 512; % Hz
timeVector = (0:length(rawSig)-1) / Fs;

%% 3. Plot Raw vs. CAR vs. Laplacian Waveforms
% =========================================================================
figure('Name', 'sEEG LFP Referencing Comparison', 'Color', 'w', 'Position', [100 100 900 600]);

subplot(3, 1, 1);
plot(timeVector, rawSig, 'k', 'LineWidth', 1);
title(sprintf('Raw sEEG LFP (Subject: %s, Contact: %s, Class: %d)', ...
    chanData.Sub, chanData.Channel_Label, stimClass));
xlabel('Time (s)'); ylabel('Amplitude (\muV)');
grid on;

subplot(3, 1, 2);
plot(timeVector, carSig, 'b', 'LineWidth', 1);
title('Common-Average-Referenced (CAR) LFP');
xlabel('Time (s)'); ylabel('Amplitude (\muV)');
grid on;

subplot(3, 1, 3);
plot(timeVector, laplacianSig, 'r', 'LineWidth', 1);
title('Laplacian-Referenced LFP');
xlabel('Time (s)'); ylabel('Amplitude (\muV)');
grid on;

%% 4. TODO Tasks: Power Spectrum & Channel Filtering
% =========================================================================
% [TASK 1.1]: Compute and plot the Power Spectral Density (PSD) using pwelch
%             for rawSig, carSig, and laplacianSig to compare high-gamma (70-150 Hz) power.
% -------------------------------------------------------------------------
% >> YOUR CODE HERE <<



% [TASK 1.2]: Iterate across all channels in SEEG, count the number of correct
%             trials per class, and return a list of "viable channels" that meet
%             a minimum threshold (e.g., at least 3 trials per class).
% -------------------------------------------------------------------------
% >> YOUR CODE HERE <<