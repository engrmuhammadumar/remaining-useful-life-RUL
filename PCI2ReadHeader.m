function [Number_of_channels, Sample_rate, Max_voltage, Header_length, delay_idx, Pretrigger] = PCI2ReadHeader(filename)
% Reads metadata from the .wfs file header
% Outputs:
%   Number_of_channels - Total AE channels in the file
%   Sample_rate - Sampling rate in kHz
%   Max_voltage - Maximum voltage scaling
%   Header_length - Number of bytes in the header
%   delay_idx - Index delay for synchronization
%   Pretrigger - Pretrigger time

fid = fopen(filename, 'rb');
if fid == -1
    error('Error opening file: %s', filename);
end

% Read header
Header.Size_table = fread(fid, 1, 'short');
fseek(fid, Header.Size_table, 0);

% Read number of channels
Header.Size_table = fread(fid, 1, 'short');
fseek(fid, 3, 0);
Header.Number_of_channels = fread(fid, 1, 'int8');
fseek(fid, -4, 0);
fseek(fid, Header.Size_table, 0);

% Read sample rate, max voltage, and pretrigger
for i = 1:Header.Number_of_channels
    Header.Size_table = fread(fid, 1, 'short');
    fseek(fid, 12, 0);
    Header.sample_rate = fread(fid, 1, 'short');
    fread(fid, 2, 'short'); % Skip trigger mode & source
    Header.pretrigger = fread(fid, 1, 'short');
    fseek(fid, 2, 0);
    Header.maxvoltage = fread(fid, 1, 'short');
    fseek(fid, -24, 0);
    fseek(fid, Header.Size_table, 0);
end

% Find header length
Header.Length_of_header = ftell(fid);
fclose(fid);

% Assign outputs
Number_of_channels = Header.Number_of_channels;
Sample_rate = Header.sample_rate;
Max_voltage = Header.maxvoltage;
Header_length = Header.Length_of_header;
Pretrigger = Header.pretrigger;
delay_idx = 0; % Default value (can be adjusted if needed)
end
