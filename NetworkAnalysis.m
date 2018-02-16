% -------------------------------------------------------------------------
% LSTM Network Analysis
% Created by: Jonathan Zia
% Last Edited: Friday, Feb 16 2018
% Georgia Institute of Technology
% -------------------------------------------------------------------------

% This program loads data from test_bench_.py and performs analysis on the
% trained LSTM network with the dataset specified in test_bench_.py.
clear;clc

% Specify whether to train a classifier or load a trained model
trainClass = true;  % true = train classifier, false = load model

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

% Convert targets to categories for classification learning
target_cat = zeros(filesize(1),1);
for i = 1:filesize(1)
    if isequal(targets(i,:),[1 0 0])
        target_cat(i) = 1;
    elseif isequal(targets(i,:),[0 1 0])
        target_cat(i) = 2;
    else
        target_cat(i) = 3;
    end
end

% Preparing data for classification learning
classificationData = [predictions, target_cat];

% Run classification model on predictions
if trainClass % Train new classifier
    [trainedModel, accuracy] = trainClassifier(classificationData);
    disp("Accuracy: " + accuracy); % Display classifier accuracy
    class_predictions = trainedModel.predictFcn(predictions); % Obtain predictions from classifier
    save('../trainedModel.mat','trainedModel'); % Save model
else % Load pre-trained classifier
    try
        % Load trained model if there is one
        load('../trainedModel.mat');
        class_predictions = trainedModel.predictFcn(predictions); % Obtain predictions from classifier
    catch
        % Else use generic decision rule
        disp("No trained model. Rounding predictions.");
        class_predictions = round(predictions);
    end
end

% Load classification learner to select appropriate model
%classificationLearner

%% ------------------------------------------------------------------------
% Visualize Data
% -------------------------------------------------------------------------
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
n = 10;  % Larger step sizes -> fewer points on graph
% For each nth prediction in the file...
for i = 1:n:filesize(1)
    % If the prediction matches the target...
    if class_predictions(i,:) == target_cat(i,:)
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


% The following figure plots a time series of target outputs and trained
% class predictions vs. time.

% Prepare graph with desired format
figure(2); hold on; grid on
title('Predictions/Targets vs. Time');
xlabel('time'); ylabel('Class');

% Plot data
plot(target_cat); plot(class_predictions);
legend('Target Categories','Class Predictions');
