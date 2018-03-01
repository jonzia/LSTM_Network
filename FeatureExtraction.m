% -------------------------------------------------------------------------
% Dataset Feature Extraction Rev 02
% Created by: Jonathan Zia
% Last Edited: Tuesday, Feb 27 2018
% Georgia Institute of Technology
% -------------------------------------------------------------------------

% This program imports and processes .csv file data files for use in LSTM
% network prediction and classification tasks.
clear;clc

% The data may be left in time-series format or may have one or more of the
% following applied: low-pass filtering, RMS calculation, and Fourier
% Analysis. Select one or more of these options:
LPF = false; % Enable/Disable low-pass filtering
RMS = false; % Enable/Disable RMS calculation
FA = false; % Enable/Disable Fourier Analysis

% Normalize LPF data?
norm_LPF = false;

% Window size for analysis
window_size = 100;

% Fourier analysis results in values for frequency ranges being divided
% into frequency bins and averaged to a discrete value, which is then
% passed as an input feature. Select number of frequency bins:
freq_bins = 9;
% Then select points per fourier bin OR auto-calculate
auto = false;
if auto
    % Auto-calculate points per bin
    points_bin = floor((window_size/2-1)/freq_bins);
else
    % Select points per bin
    points_bin = 2;
end

% Specify number of features, classes, and number of timestamp columns
feature_num = 9;
class_num = 3;
timestamp = 1;

% Specify output filenames for writing processed data to .csv files:
output_filenames = ["datafile_processed.csv"];


%% ------------------------------------------------------------------------
% Import CSV Data
% -------------------------------------------------------------------------
% Insert list of filenames from which to import data
filename_list = ["datafile.csv"];
% Obtain number of data files
num_files = size(filename_list);
% Import data into array of size [num_files x data_points x data_columns]
data = cell(num_files(2), 1);
for i = 1:num_files(2)
    disp("Importing File " + i);
    data{i} = importdata(filename_list(i));
    % If the data is imported as a character array, append timestamps
    if isfield(data{i},'rowheaders')
        num_elem = size(data{i}.data);
        t_stamps = 1:num_elem(1);
        data{i} = [transpose(t_stamps) data{i}.data];
    end
end


%% ------------------------------------------------------------------------
% Data Processing
% -------------------------------------------------------------------------

% Custom pre-processing code here

%% ------------------------------------------------------------------------
% Low-Pass Time-Series Filtering
% -------------------------------------------------------------------------
if LPF
    % FIR low-pass filter designed by Filter Designer
    % Specifications: Fs = 67Hz (estimated); Fpass = 5Hz; Fstop = 10Hz
    coeff = [0.000513597053186222,0.00143953792902160,0.00268728781999995,0.00348885290937029,0.00255573006377796,-0.00151369188523023,-0.00942351770093767,-0.0202580500631260,-0.0309209563782769,-0.0364032357005344,-0.0310787936017955,-0.0107642471154336,0.0251755590489557,0.0725937065908270,0.122913328882096,0.165199949276606,0.189363597050808,0.189363597050808,0.165199949276606,0.122913328882096,0.0725937065908270,0.0251755590489557,-0.0107642471154336,-0.0310787936017955,-0.0364032357005344,-0.0309209563782769,-0.0202580500631260,-0.00942351770093767,-0.00151369188523023,0.00255573006377796,0.00348885290937029,0.00268728781999995,0.00143953792902160,0.000513597053186222];
    % Specifications: Fs = 67Hz (estimated); Fpass = 1Hz; Fstop = 5Hz
    coeff2 = [4.79327748470482e-06,0.000263147414205478,0.000563196897790071,0.00112175527826935,0.00198999107314703,0.00325452633957827,0.00499929894323254,0.00729772496385976,0.0102035785723195,0.0137415157978823,0.0178995587752836,0.0226229466606835,0.0278114170707222,0.0333199221973773,0.0389640757832623,0.0445284393677857,0.0497795305685804,0.0544804071594470,0.0584072222157913,0.0613654721249311,0.0632042118363297,0.0638280730984335,0.0632042118363297,0.0613654721249311,0.0584072222157913,0.0544804071594470,0.0497795305685804,0.0445284393677857,0.0389640757832623,0.0333199221973773,0.0278114170707222,0.0226229466606835,0.0178995587752836,0.0137415157978823,0.0102035785723195,0.00729772496385976,0.00499929894323254,0.00325452633957827,0.00198999107314703,0.00112175527826935,0.000563196897790071,0.000263147414205478,4.79327748470482e-06];
    % Filtering data columns
    for file = 1:num_files(2) % For each file...
        disp("Filtering datapoints for File " + file);
        % Filter columns in each file
        data{file}(:,1+timestamp:feature_num+timestamp) = filter(coeff,1,data{file}(:,1+timestamp:feature_num+timestamp),[],1);
    end
    
    % If normalize LPF is selected...
    if norm_LPF
        % For each file...
        for file = 1:num_files(2)
            disp("Normalizing filtered data for File " + file);
            num_elem = size(data{file});   % Obtain number of datapoints
            % Obtain normalized values for each datapoint
%             for i = 1:num_elem(1)
%                 temp = data{file}(i,1+timestamp:feature_num+timestamp);
%                 data{file}(i,1+timestamp:feature_num+timestamp) = 100*(temp - mean(temp))./var(temp);
%             end
            for i = 1+timestamp:feature_num+timestamp
                temp = data{file}(:,i);
                data{file}(:,i) = 100*(temp - mean(temp))./var(temp);
            end
        end
    end
end

%% ------------------------------------------------------------------------
% RMS Calculation
% -------------------------------------------------------------------------
if RMS
    % Create placeholder for data output
    data_rms = cell(num_files(2),1);

    % For each data file...
    for file = 1:num_files(2)
        disp("Calculating RMS for File " + file);

        % Obtain number of samples in file
        num_elem = size(data{file});
        % Initialize placeholders
        root_mean_square = zeros(num_elem(1)-window_size,num_elem(2));

        % For each sample, obtain RMS columnwise
        for sample = 1:num_elem(1)-window_size
            % Obtain the RMS over bin_size samples
            root_mean_square(sample,1+timestamp:feature_num+timestamp) = rms(data{file}(sample:sample+window_size,1+timestamp:feature_num+1),1);
            % Add timestamps to root_mean_square
            root_mean_square(sample,1) = data{file}(sample+window_size,1);
            % Add labels to root_mean_square
            root_mean_square(sample,end-class_num+1:end) = data{file}(sample+window_size,end-class_num+1:end);
        end
        % Update data file for exporting
        data_rms{file} = root_mean_square;
    end
end


%% ------------------------------------------------------------------------
% Fourier Analysis
% -------------------------------------------------------------------------
if FA
    % Create placeholder for data output
    data_fa = cell(num_files(2),1);

    % For each data file...
    for file = 1:num_files(2)
        disp("Performing Fourier Analysis on File " + file);

        % Obtain number of samples in file
        num_elem = size(data{file});
        % Initialize placeholders
        fourier = cell(num_elem(1)-window_size,1);
        fourier_av = zeros(num_elem(1)-window_size,freq_bins+class_num+timestamp);

        % For each sample, obtain FFT columnwise
        for sample = 1:num_elem(1)-window_size
            % Obtain the FFT over bin_size samples
            raw_freq = fft(data{file}(sample:sample+window_size,1+timestamp:feature_num+timestamp),[],1);
            raw_freq = abs(raw_freq/window_size); % Convert raw_freq to scalar
            one_freq = raw_freq(1:(window_size/2)+1,:); % Obtain one-sided FFT
            one_freq = 2*one_freq(2:end-1,:); % Scale one-sied FFT by 2
            fourier{sample} = one_freq;
            % Adding timestamp to fourier_av
            if timestamp == 1
                fourier_av(sample,1) = data{file}(sample+window_size,1);
            end
            % Adding label to fourier_av
            fourier_av(sample,end-class_num+1:end) = data{file}(sample+window_size,end-class_num+1:end);
            % Adding frequency bins to fourier_av
            % For each bin...
            for i = 1:freq_bins
                % For all bins before the final one...
                if i < freq_bins && auto
                    % Calculate mean of all points in bin
                    fourier_av(sample,i+timestamp) = mean(mean(fourier{sample}((i-1)*points_bin+1:(i-1)*points_bin+points_bin,:)));
                elseif auto % Else, for the last bin...
                    % Average over the remaining number of points
                    fourier_av(sample,i+timestamp) = mean(mean(fourier{sample}((i-1)*points_bin+1:end,:)));
                else % If points_bin was specified
                    fourier_av(sample,i+timestamp) = mean(mean(fourier{sample}((i-1)*points_bin+1:(i-1)*points_bin+points_bin,:)));
                end
            end
        end
        % Update data file for exporting
        data_fa{file} = fourier_av;
    end
end


%% ------------------------------------------------------------------------
% Create Output CSV Files
% -------------------------------------------------------------------------
% Write updated arrays to .csv file
for file = 1:num_files(2)   % For each file...
    if FA % If Fourier analysis was performed
        % Write data to name.csv
        csvwrite(output_filenames(file),data_fa{file});
    elseif RMS % If RMS analysis was performed
        csvwrite(output_filenames(file),data_rms{file});
    else % If LPF or no analysis was performed
        csvwrite(output_filenames(file),data{file});
    end
end