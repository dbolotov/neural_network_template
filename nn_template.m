% NN_TEMPLATE is a template neural network for classification.
%
% Perform classification for a multi-class dataset using a regularized 3-layer neural network.
% Learn parameters with fmincg.m.
%
% Features must be numerical.	
% Classes must be designated with consecutive integers, starting from 1 (Ex: {1,2,3,4} but not {1,2,4}). 
% Every class must be represented in training set.
% Due to Octave/MATLAB syntax, '0' cannot be used to designate a class.
%
% Overview:
%	Read in comma-delimited data file (without header)
%	Split into training and test set
%	Specify NN parameters (initialize random weights)
%	Minimize cost function with fmincg
%	Compute performance metrics:
%		prediction accuracy on training and test sets
%		confusion matrix, sensitivity, specificity
%
% Functions used: sigmoid.m, sigmoidGradient.m, randInitializeWeights.m, nnCostFunction.m, fmincg.m, predict.m
%
% Code based on ml-class.org Ex.4
%
% Datasets used:
% Fisher's Iris: http://archive.ics.uci.edu/ml/datasets/Iris
% Wine: http://archive.ics.uci.edu/ml/datasets/Wine
% Note: labels of "0" in these datasets have been replaced with non-zero values to accomodate MATLAB/Octave syntax
%
% To Do:
%	Test on dataset with added noise
%	randomize dataset before splitting into test/train sets
%	add confusion matrix for any number of classes; specificity, sensitivity
%	allow any number of hidden layers

%load data
data = load('wine.csv');

%randomize rows
order = randperm(size(data,1));
data = data(order,:);

%separate into features and class
X = data(:,1:end-1);
y = data(:,end);

%percentage of data to use for training
train_frac = 0.75;

%split into training and test sets:
test_rows = round(size(X,1)*(1-train_frac)); %number of rows to use in test set
X_test = X(1:test_rows,:); y_test = y(1:test_rows,:);%this is the test set
X = X(test_rows+1:end,:); y = y(test_rows+1:end,:);%this is the training set
m = size(X,1);

%NN layer sizes
input_layer_size = size(X,2);
hidden_layer_size = 20;
num_labels = size(unique(y),1); %output layer

%Initialize NN Parameters for the 3-layer NN
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% Implement backprop and train network using fmincg
fprintf('\nTraining Neural Network... \n')

% Set options for fmincg
options = optimset('MaxIter', 400);
lambda = 1.0;

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%Find training set accuracy
% pred = predict(Theta1, Theta2, X);
% fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

p_train = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(p_train == y)) * 100);

p_test = predict(Theta1, Theta2, X_test);
fprintf('\nTest Set Accuracy: %f\n', mean(double(p_test == y_test)) * 100);


