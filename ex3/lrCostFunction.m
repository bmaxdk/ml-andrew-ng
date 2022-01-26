function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%










% =============================================================

n = size(X,2); % number of features
m = size(X,1);
% m = length(y)
% fprintf('size theta = %f %f', size(theta)) % 4x1
% fprintf('\nsize X = %f %f', size(X)) %5x4
%% cost
z = X*theta; %5x1
h = sigmoid(z); %5x1
lr = (-y .* log(h)) - ((1-y) .* log(1-h));
sumLr = (ones(1,m)* lr)/m;

% regularization 
% *Note that you should not be regularizing  which is used for the bias term.
sumReg = (lambda/(2*m))*(theta(2:end,:)' * theta(2:end,:));

% cost function
J = sumLr + sumReg;

%% grad
% at j == 0
% grad(1) = ((h-y)' * X(:,1))/m
grad(1) = X(:,1)'*(h-y)/m;
% at j = 1:n
grad(2:end, :) = (X(:,2:end)'* (h-y))./m+ (lambda/m) .* theta(2:end,:);

grad = grad(:);


end

