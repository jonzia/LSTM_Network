% -------------------------------------------------------------------------
% LSTM Network Analysis
% Created by: Jonathan Zia
% Last Edited: Friday, Feb 16 2018
% Georgia Institute of Technology
% -------------------------------------------------------------------------

% This program loads data from test_bench_.py and performs analysis on the
% trained LSTM network with the dataset specified in test_bench_.py.
clear;clc

% Specify number of classes
numClasses = 3;

% Toggle data visualization
visualize = true;

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
one_hots = eye(numClasses);
for i = 1:filesize(1)
    for j = 1:numClasses
        if isequal(targets(i,:),one_hots(j,:))
            target_cat(i) = j;
        end
    end
end

% Obtain class predictions as integer
class_predictions = zeros(filesize(1),1);
for i = 1:filesize(1)
    [maximum, index] = max(predictions(i,:));
    for j = 1:numClasses
        if index == j
            class_predictions(i) = j;
        end
    end
end

% Obtain precision, recall, and F1 score w.r.t. class 1, 2, or 3
% Initialize counters
for class = 1:numClasses
    TP = 0; TN = 0; FP = 0; FN = 0;
    for i = 1:filesize(1)
        % For predictions in class [class]...
        if class_predictions(i) == class
            % If the sample was classified correctly, increment TP
            if target_cat(i) == class
                TP = TP + 1;
            else % Else, increment FP
                FP = FP + 1;
            end
        else % For predictions not in [class]
            % If the target was in [class], increment FN
            if target_cat(i) == class
                FN = FN + 1;
            else % If the targetwas not in [class] either
                TN = TN + 1;
            end
        end
    end

    % Display precision, recall, and F1 w.r.t. [class]
    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    disp("For class " + class)
    disp("-------------")
    disp("Precision: " + precision)
    disp("Recall: " + recall)
    disp("F1: " + 2*precision*recall/(precision+recall) + newline)
end

%% ------------------------------------------------------------------------
% Visualize Data
% -------------------------------------------------------------------------
% The following graph is designed for target vectors of length <= 3. These
% vectors may be one-hot or binary permutations. The graph plots correct
% predictions as green points and incorrect predictions as red points. The
% point is connected to the correct target via a line of green color if
% correct or red if it was misclassified.

if visualize
    % Prepare graph with desired format
    figure(1); hold on; grid on
    title('Prediction Analysis');
    xlabel('Class 1'); ylabel('Class 2'); zlabel('Class 3');

    % Plotting predictions
    % Specify step size for plotting predictions vs targets
    n = 20;  % Larger step sizes -> fewer points on graph
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

    % Release graph
    hold off
end