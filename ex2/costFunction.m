function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%init sum
n_feature = size(X,2);
sum_J = 0; %zeros(n_feature,1)
sum_grad = zeros(n_feature,1);
for i = 1:m
    z = theta' * X'; %100 x 1 double
    h = sigmoid(z);
        sum_J = sum_J + (-y(i) * log(h(i)) - (1-y(i)) * log(1 - h(i)));
    for j = 1:n_feature
        sum_grad(j) = sum_grad(j) + (h(i) - y(i))*X(i,j); 
    end

end
J = sum_J/m;
grad = sum_grad ./m;
% =============================================================

end
