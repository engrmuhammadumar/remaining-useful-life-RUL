clc; clear; close all;

% Set file path
file_path = 'D:\Pipeline RUL Data\Test20190731-200233-899.wfs'; % Update with actual path

% Read file metadata
[Number_of_channels, Sample_rate, Max_voltage, Header_length, delay_idx, pretrigger] = PCI2ReadHeader(file_path);

% Display metadata
disp(['Channels: ', num2str(Number_of_channels)]);
disp(['Sampling Rate: ', num2str(Sample_rate), ' kHz']);
disp(['Max Voltage: ±', num2str(Max_voltage), ' V']);
disp(['Pretrigger Time: ', num2str(pretrigger / Sample_rate), ' sec']);

% Read AE signals (Extract 5 seconds from 10 to 15 sec)
start_time = 0;  
end_time = 1;   

[signals, t, fs, nch] = wfsread_exp(file_path, start_time, end_time);

% Check if signals were read successfully
if isempty(signals)
    error('Error reading .wfs file! Ensure correct path and file format.');
end

% Plot first 4 channels
figure;
for ch = 1:min(4, nch)
    subplot(4,1,ch);
    plot(t, signals(:,ch));
    title(['Channel ', num2str(ch)]);
    xlabel('Time (s)');
    ylabel('Amplitude (V)');
end
sgtitle('Extracted AE Signals from Test20190731-200233-899.wfs');
