function J = computeCost_HC(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

h = theta'*X';
h = h';
% jsum = 0;
% sum1 = 0;
% sum2 = 0;
% global alpha %, theta(1), theta(2)
for i = 1:m
%     h = theta'*X';
%     h = h';
    
%     sum1 = sum1 + (alpha/m) * ((h(i)-y(i)) * X(i,1));
%     sum2 = sum2 + (alpha/m) * ((h(i)-y(i)) * X(i,2));
    
    J = J + ((h(i)-y(i))^2)/(2*m);
end
%% 

% theta(1,1) = theta(1,1) - sum1;
% theta(2,1) = theta(2,1) - sum2;
% =========================================================================

end


