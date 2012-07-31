function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%	Source: ml-class.org Ex.4


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
n = size(X, 2);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


%Forward pass:
y = eye(num_labels)(y,:);

a_1 = [ones(m,1) X];

z_2 = (Theta1*a_1')';

a_2 = sigmoid(Theta1*a_1')';
a_2 = [ones(size(a_2,1),1) a_2];
a_3 = sigmoid(Theta2*a_2')';

J_reg = lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
J = (1/m)*sum(sum(-y.*log(a_3) - (1-y).*log(1-a_3))) + J_reg; 


%Backprop:

d_3 = a_3 - y;
d_2 = ((d_3*Theta2(:,2:end)).*sigmoidGradient(z_2));

Theta1_grad = 1/m * d_2' * a_1;
Theta2_grad = 1/m * d_3' * a_2;

%add regularization:

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m*Theta2(:,2:end);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
