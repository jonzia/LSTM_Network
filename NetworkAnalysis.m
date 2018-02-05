% -------------------------------------------------------------------------
% LSTM Network Analysis
% Created by: Jonathan Zia
% Last Edited: Monday, Feb 5 2018
% Georgia Institute of Technology
% -------------------------------------------------------------------------

% This program loads data from test_bench_.py and performs analysis on the
% trained LSTM network with the dataset specified in test_bench_.py.
clear;clc

%% ------------------------------------------------------------------------
% Load Data
% -------------------------------------------------------------------------
% Specify .txt files containing desired data
pred_filename = "predictions.txt";  % Prediction data
tar_filename = "targets.txt";       % Target data
% Write data to arrays
predictions = importdata(pred_filename);    % Prediction data
targets = importdata(tar_filename);         % Target data
% Obtain file sizes
filesize = size(targets);


%% ------------------------------------------------------------------------
% Data Analysis
% -------------------------------------------------------------------------
% Perform data analysis

% Round predictions to nearest integer for classification
round_predictions = round(predictions);


%% ------------------------------------------------------------------------
% Visualize Data
% -------------------------------------------------------------------------
% Plotting predictions and targets

% The following graph is designed for target vectors of length <= 3. These
% vectors may be one-hot or binary permutations. The graph plots correct
% predictions as green points and incorrect predictions as red points. The
% point is connected to the correct target via a line of green color if
% correct or red if it was misclassified.

% Prepare graph with desired format
figure(1); hold on; grid on
title('Prediction Analysis');
xlabel('Class 1'); ylabel('Class 2'); zlabel('Class 3');

% Plotting predictions
% Specify step size for plotting predictions vs targets
n = 1;  % Larger step sizes -> fewer points on graph
% For each nth prediction in the file...
for i = 1:n:filesize(1)
    % If the prediction matches the target...
    if round_predictions(i,:) == targets(i,:)
        % Plot predictions in green
        scatter3(predictions(i,1),predictions(i,2),predictions(i,3), 'MarkerEdgeColor',[0 1 0])
        plot3([predictions(i,1), targets(i,1)],[predictions(i,2), targets(i,2)],[predictions(i,3), targets(i,3)],'-g')
    else
        % Else, plot predictions in red
        scatter3(predictions(i,1),predictions(i,2),predictions(i,3), 'MarkerEdgeColor',[1 0 0])
        plot3([predictions(i,1), targets(i,1)],[predictions(i,2), targets(i,2)],[predictions(i,3), targets(i,3)],'-r')
    end
end

% Plotting targets in black
scatter3(targets(:,1),targets(:,2),targets(:,3),'MarkerEdgeColor',[0 0 0])

% Release graph
hold off
