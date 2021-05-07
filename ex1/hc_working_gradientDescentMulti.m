function [theta, J_history] = hc_working_gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

temp = zeros(3);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    h = theta'*X';
    h = h';
    sum = zeros(3);
    
    for i = 1:m
        for j = 1:length(sum)
            sum(j) = sum(j) + (alpha/m) * ((h(i)-y(i)) * X(i,j));
        end
    end
    for j = 1: length(sum)
        theta(j) = theta(j) - sum(j);
        
    end
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end