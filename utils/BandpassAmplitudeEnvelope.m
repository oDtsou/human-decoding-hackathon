function [BPAE] = BandpassAmplitudeEnvelope(DATA,FS,locutoff,hicutoff)

%%% parameter
n     = 2; %filtorder
fhi   = 10;
Wn    = [fhi/(FS/2)]; % butterworth bandpass non-dimensional 
[b,a] = butter(n,Wn,'low');

if ~isempty(DATA)
    TR = size(DATA,1);
    if TR==1
        temp           = DATA;
        [smoothdata,~] = eegfilt(temp,FS,locutoff,hicutoff);
        % Absolute value of the Hilbert transform of a band-pass filtered signal corresponds with the amplitude envelope.
        tem  = ft_preproc_hilbert(smoothdata);  
        BPAE = filtfilt(b,a,tem);
        clear tem smoothdata temp; 
    else
        for tr=1:TR
            temp           = DATA(tr,:);
            [smoothdata,~] = eegfilt(temp,FS,locutoff,hicutoff);
            tem            = ft_preproc_hilbert(smoothdata); 
            ff(tr,:)       = filtfilt(b,a,tem);
            clear tem smoothdata temp; 
        end
        BPAE = ff;
        clear ff;
    end    
else
    BPAE =[];
end
