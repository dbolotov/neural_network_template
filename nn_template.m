% NN_TEMPLATE is a template neural network for classification for a multi-class dependent variable.
%
% Structure includes 1 hidden layer.
%
%
% Functions used:
%     sigmoidGradient.m
%     randInitializeWeights.m
%     nnCostFunction.m
%
%
% Code based on ml-class.org Ex.3

% Load data
load('ex4data1.mat');
m = size(X, 1);

%NN layer sizes
input_layer_size = size(X,2);
hidden_layer_size = 25;
num_labels = 10; %output layer


%load and unroll parameters
load('ex4weights.mat');
nn_params = [Theta1(:) ; Theta2(:)];
break

%compute feedforward cost with regularization
lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.383770)\n'], J);


%Initialize NN Parameters for the 3-layer NN
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


% Implement backprop
%  Check gradients by running checkNNGradients

lambda = 3;
checkNNGradients(lambda);

% Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = 10): %f ' ...
         '\n(this value should be about 0.576051)\n\n'], debug_J);


%Train NN using fmincg
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 400);
lambda = 0.5;

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
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


