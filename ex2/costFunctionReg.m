function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

h = sigmoid(X * theta); % 118 x 1 which is == size(y)


% Cost J:
sum1 = 0;
sum2 = 0;
n = size(theta,1);
for i = 1:m
    temp1 = sum1 + (-y(i) * log(h(i)) - (1-y(i)) * log(1-h(i)))/m;
    sum1 = temp1;
    
end

for j = 1:n
    if j ~= 1
        temp2 = sum2 + (lambda*(theta(j)^2)/(2*m));
        sum2 = temp2; 
    end
end   

J = sum1 + sum2;


% Gradient Function:
for j = 1:n
    for i = 1:m
        if j == 1
            grad(j) = grad(j) + ((h(i)-y(i)) * X(i,j))/m;
        else
            grad(j) = grad(j) + ((h(i)-y(i)) * X(i,j) + (lambda * theta(j) / m))/m;
        end
    end
end

% =============================================================

end
