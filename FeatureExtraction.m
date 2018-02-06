% -------------------------------------------------------------------------
% Dataset Feature Extraction
% Created by: Jonathan Zia
% Last Edited: Monday, Feb 5 2018
% Georgia Institute of Technology
% -------------------------------------------------------------------------

% This program imports and processes .csv file data files for use in LSTM
% network prediction and classification tasks.
clear;clc

% Specify output filenames for writing processed data to .csv files:
output_filenames = ["S01R01_t.csv", "S01R02_t.csv"];


%% ------------------------------------------------------------------------
% Import CSV Data
% -------------------------------------------------------------------------
% Insert list of filenames from which to import data
filename_list = ["S01R01.csv", "S01R02.csv"];
% Obtain number of data files
num_files = size(filename_list);
% Import data into array of size [num_files x data_points x data_columns]
data = cell(num_files(2), 1);
for i = 1:num_files(2)
    data{i} = importdata(filename_list(i));
end


%% ------------------------------------------------------------------------
% Data Processing
% -------------------------------------------------------------------------
% Remove datapoints with "0" labels and decrement nonzero labels
for file = 1:num_files(2)   % For each file...
    num_elem = size(data{file});   % Obtain number of datapoints
    num_removed = 0; % Number of datapoints removed during processing
    for i = 1:num_elem(1)   % For each datapoint in file...
        loop = i - num_removed; % Adjust loop counter for removed points
        if data{file}(loop,end) == 0   % If the label is "0"...
            data{file}(loop,:) = [];   % Remove the row
            num_removed = num_removed + 1; % Increment number of datapoints
        else    % Else...
            data{file}(loop,end) = data{file}(loop,end) - 1;  % Decrement label
        end
    end
end

% Append datafiles with two extra label columns for one-hot three-category
% classification: non-FoG, pre-FoG, and FoG.
for file = 1:num_files(2)   % For each file...
    num_elem = size(data{file});   % Obtain number of datapoints
    % Append two zero-columns
    data{file} = [data{file}, zeros(num_elem(1),1), zeros(num_elem(1),1)];
end

% Update labels into one-hot three-category classification:
% non-FoG:  [1 0 0]
% pre-FoG:  [0 1 0] "non-FoG" datapoints occuring within 'n' steps of FoG
n = 500; % Size of pre-FoG window
% FoG:      [0 0 1]
for file = 1:num_files(2)   % For each file...
    num_elem = size(data{file});   % Obtain number of datapoints
    
    % Update non-FoG and FoG labels
    for i = 1:num_elem(1)   % For each datapoint in file...
        if isequal(data{file}(i,11:13),[0 0 0])    % If it is a non-FoG event...
            data{file}(i,11:13) = [1 0 0];  % Update labels
        elseif isequal(data{file}(i,11:13),[1 0 0])  % If it is an FoG event...
            data{file}(i,11:13) = [0 0 1];  % Update labels
        end
    end
    
    % Update pre-FoG labels
    for i = 1:num_elem(1)-1   % For each datapoint in file...
        % If FoG occurs
        if isequal(data{file}(i,11:13),[1 0 0]) && isequal(data{file}(i+1,11:13),[0 0 1])
            % Update pre-FoG labels
            for j = i:-1:i-n % For the n timesteps before the FoG event
                data{file}(j,11:13) = [0 1 0];  % Update labels
            end
        end
    end
end

% Update labels into one-hot three-category classification:
% non-FoG:  [1 0 0]
% pre-FoG:  [0 1 0] "non-FoG" datapoints occuring within 'n' steps of FoG
% FoG:      [0 0 1]
for file = 1:num_files(2)   % For each file...
    num_elem = size(data{file});   % Obtain number of datapoints
    for i = 1:num_elem(1)   % For each datapoint in file...
        
    end
end

%% ------------------------------------------------------------------------
% Create Output CSV Files
% -------------------------------------------------------------------------
% Write updated arrays to .csv file
for file = 1:num_files(2)   % For each file...
    csvwrite(output_filenames(file),data{file}); % Write data to name.csv
end