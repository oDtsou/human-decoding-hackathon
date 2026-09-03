%% extract_bandpass_features.m
% Extracts bandpass amplitude envelopes across trials and channels,
% separating features by Prefrontal Subdivision (Dorsal vs. Ventral).

clear; clc;

%% 1. Parameters
% Frequency band of interest (e.g., High-Gamma: 70-150 Hz, Beta: 15-30 Hz)
LOFreq = 70;
HIFreq = 150;
Fs     = 512;

% Time window of interest (in samples, e.g., delay epoch)
% Adjust indices according to your task timing
timeWindow = 513:3072; % Spatial: 513:3072 / Shape: 513:4864 (remove Inter-Trial Period)

% Spatial reference scheme to use: 'TrialData', 'Common', or 'Laplacian'
refScheme = 'Laplacian';

%% 2. Load Dataset
dataPath = 'data/Spatial_Task_SEEG.mat';
if ~exist(dataPath, 'file')
    error('File %s not found. Please verify the path.', dataPath);
end
load(dataPath, 'SEEG');

%% 3. Extract Features for Dorsal and Ventral Channels
DorsalData   = {};
DorsalClass  = {};
VentralData  = {};
VentralClass = {};

d_idx = 1;
v_idx = 1;

fprintf('Processing %d channels...\n', length(SEEG));

for ch = 1:length(SEEG)
    chan = SEEG(ch);
    
    % Access trials struct
    if isfield(chan, 'CorrectTrials') && ~isempty(chan.CorrectTrials)
        trials = chan.CorrectTrials;
    elseif isfield(chan, 'Correct') && ~isempty(chan.Correct)
        trials = chan.Correct;
    else
        continue;
    end
    
    numTrials = length(trials);
    if numTrials == 0; continue; end
    
    chanFeatures = zeros(numTrials, length(timeWindow));
    chanLabels   = zeros(numTrials, 1);
    
    for tr = 1:numTrials
        sig = trials(tr).(refScheme);
        env = BandpassAmplitudeEnvelope(sig, Fs, LOFreq, HIFreq);
        
        chanFeatures(tr, :) = env(timeWindow);
        chanLabels(tr)      = trials(tr).Class;
    end
    
    % Split based on Prefrontal Subdivision
    if strcmpi(chan.Prefrontal_subdiv, 'Dorsal')
        DorsalData{d_idx}   = chanFeatures;
        DorsalClass{d_idx}  = chanLabels;
        d_idx = d_idx + 1;
    elseif strcmpi(chan.Prefrontal_subdiv, 'Ventral')
        VentralData{v_idx}  = chanFeatures;
        VentralClass{v_idx} = chanLabels;
        v_idx = v_idx + 1;
    end
end

fprintf('Done! Processed %d Dorsal and %d Ventral channels.\n', d_idx-1, v_idx-1);

%% 4. Save Extracted Features
save('data/Extracted_Features_Dorsal_Ventral.mat', ...
    'DorsalData', 'DorsalClass', 'VentralData', 'VentralClass', '-v7.3');