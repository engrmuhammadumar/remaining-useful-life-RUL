%% ============================================
% Author: Umar's Assistant
% Purpose: Split large .wfs file into 1-second chunks
% Output: segment_0000.mat, ..., segment_1037.mat
%% ============================================

clc; clear;

% === USER SETTINGS ===
filePath = "D:\Concrete RUL\STREAM20191203-105108-756.wfs";
outputFolder = "D:\Concrete RUL\Chunks\";  % Ensure this folder exists
mkdir(outputFolder);  % create if not exists

total_duration_sec = 1038;   % Total estimated seconds (update if exact)
fs = 5000000;                % Sampling rate (Hz)
nch = 8;                     % Number of channels
save_as_csv = false;         % Change to true if you want CSV files

% === LOOP TO READ AND SAVE CHUNKS ===
for seg = 0:total_duration_sec-1
    fprintf("Processing segment %d / %d\n", seg, total_duration_sec-1);
    
    % Read 1-sec chunk
    try
        [signals, t, ~, ~] = wfsread_exp(filePath, seg, seg + 1);
    catch
        warning("Failed to read segment %d. Skipping...", seg);
        continue;
    end

    % Create filename
    filename = sprintf("segment_%04d", seg);
    
    % Save as .mat or .csv
    if save_as_csv
        csvwrite(fullfile(outputFolder, filename + ".csv"), signals);
    else
        save(fullfile(outputFolder, filename + ".mat"), 'signals', 't');
    end
end

fprintf("✅ All chunks saved to: %s\n", outputFolder);
