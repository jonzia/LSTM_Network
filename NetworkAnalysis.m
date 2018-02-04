% -------------------------------------------------------------------------
% LSTM Network Analysis
% Created by: Jonathan Zia
% Last Edited: Sunday, Feb 4 2018
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

%% ------------------------------------------------------------------------
% Data Analysis
% -------------------------------------------------------------------------
% Perform data analysis

%% ------------------------------------------------------------------------
% Visualize Data
% -------------------------------------------------------------------------
% Plotting predictions and targets
figure(1); hold on; grid on;
plot(predictions, '-k'); plot(targets, '-r');
xlabel('Time'); ylabel('FoG Probability');
legend('Predictions', 'Targets','Location','northeast');
title('Predictions vs. Targets');
ylim([-0.1,1.1]);   % Setting y-axis limits
hold off