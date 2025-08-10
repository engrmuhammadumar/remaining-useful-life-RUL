%% ======================================================
%  Author: Umar's Assistant
%  Purpose: Read AE data from PCI .wfs files and print full info
%% ======================================================

clc; clear;

%% === USER SETTINGS ===
filePath   = "D:\Concrete RUL\STREAM20191203-105108-756.wfs";  % AE File
start_time = 0;       % in seconds
end_time   = 1;       % read from 0 to 1 second

%% === READ DATA ===
[signals, t, fs, nch] = wfsread_exp(filePath, start_time, end_time);

%% === FILE SIZE & ESTIMATION ===
fileInfo = dir(filePath);
fileSizeBytes = fileInfo.bytes;
bytes_per_sample = 2; % int16
samples_per_sec_all = fs * nch;
bytes_per_sec = samples_per_sec_all * bytes_per_sample;
est_duration_sec = fileSizeBytes / bytes_per_sec;
total_samples_per_channel = round(fs * est_duration_sec);

%% === PRINT COMPLETE INFORMATION ===
fprintf('\n============= FILE INFORMATION =============\n');
fprintf('File Path        : %s\n', filePath);
fprintf('Channels         : %d AE sensors\n', nch);
fprintf('Sampling Rate    : %.0f Hz (%.1f MHz)\n', fs, fs/1e6);
fprintf('Total Duration   : %.2f seconds (estimated full file)\n', est_duration_sec);
fprintf('Signal Shape     : %d samples x %d channels\n', size(signals,1), size(signals,2));
fprintf('Time Vector Size : %d values\n', length(t));
fprintf('Time Range       : %.6f sec to %.6f sec\n', t(1), t(end));
fprintf('Amplitude Unit   : Volts (V)\n');
fprintf('Signal Unit      : AE Voltage Signals\n');
fprintf('Data Type        : double (scaled to voltage)\n');
fprintf('=============================================\n');

fprintf('------------ SUMMARY TABLE ------------\n');
fprintf('File Name                 : %s\n', filePath);
fprintf('Start Time (sec)          : %.2f\n', start_time);
fprintf('End Time (sec)            : %.2f\n', end_time);
fprintf('Duration (sec)            : %.2f\n', end_time - start_time);
fprintf('Sampling Frequency (Hz)   : %.0f\n', fs);
fprintf('Sampling Frequency (MHz)  : %.1f\n', fs/1e6);
fprintf('Total Channels            : %d\n', nch);
fprintf('Samples per Channel       : %d\n', size(signals,1));
fprintf('Total Samples             : %d\n', numel(signals));
fprintf('Signal Matrix Shape       : %d x %d\n', size(signals,1), size(signals,2));
fprintf('Time Vector Length        : %d\n', length(t));
fprintf('Signal Unit               : Volts (V)\n');
fprintf('Estimated File Duration   : %.2f sec (from file size)\n', est_duration_sec);
fprintf('---------------------------------------\n');

%% === OPTIONAL: PLOT FIRST CHANNEL ===
figure;
plot(t, signals(:,1), 'b');
xlabel('Time (s)'); ylabel('Amplitude (V)');
title('AE Signal - Channel 1');
grid on;
