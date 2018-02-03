% -------------------------------------------------------------------------
% Dataset Feature Extraction
% Created by: Jonathan Zia
% Last Edited: Saturday, Feb 3 2018
% Georgia Institute of Technology
% -------------------------------------------------------------------------

% This program imports and processes .csv file data files for use in LSTM
% network prediction and classification tasks.

% Specify output filenames for writing processed data to .csv files:
output_filenames = ["S01R01_t.csv","S01R02_t.csv"];


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


%% ------------------------------------------------------------------------
% Create Output CSV Files
% -------------------------------------------------------------------------
% Write updated arrays to .csv file
for file = 1:num_files(2)   % For each file...
    csvwrite(output_filenames(file),data{file}); % Write data to name.csv
end