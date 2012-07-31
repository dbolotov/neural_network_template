function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z.

g = zeros(size(z));

f = 1.0 ./ (1.0 + exp(-z)); %sigmoid of z
g = f.*(ones(size(f))-f); %gradient of sigmoid

end
