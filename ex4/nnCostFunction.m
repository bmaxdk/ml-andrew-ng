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
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%




% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); % 25 x 401
Theta2_grad = zeros(size(Theta2)); % 10 x 26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




% size(X) %5000 X 400
% size(y) %5000 X 1
% size(nn_params) %10285 X 1
% hidden_layer_size %25
% num_labels %10
% -------------------------------------------------------------

% =========================================================================



%% Update y  
% original labels (in the variable y) were  for the purpose of training a neural network, we need to recode the labels as vectors containing only values 0 or 1,
ny = zeros(size(y,1),num_labels);
for i=1:m
  ny(i,y(i)) = 1;
end



%% forward
a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3); % 5000 x 10

%% Regularization
regularization = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));



%% cost
% sum(-ny .* log(h3)) %sum each row also can use sum(-ny .* log(h3),1)
J = (1/m) * sum(sum((-ny .* log(a3)) - ((1-ny).*log(1-a3)))) + regularization;


%% Backpropagation
a3; %5000x10
ny; %5000x10
y; %5000x1
delta_3 = a3 - ny; %5000 x 10
delta_2 = (Theta2(:,2:end))' * delta_3' .* sigmoidGradient(z2'); % 25x5000

%% change
D_1 = delta_2 * a1; %[25x5000] * [5000x401] = [25x401]
D_2 = delta_3' * a2;  %[5000x10] * [5000x26] = [10x26]
% Theta1_grad = (1/m)*(D_1);
% Theta2_grad = (1/m)*(D_2);
%% add reg in change
reg_1 = lambda*[zeros(size(Theta1,1),1) Theta1(:,2:end)]; %[25x401]
reg_2 = lambda*[zeros(size(Theta2,1),1) Theta2(:,2:end)]; %[10x26]


%% gradients
Theta1_grad = (1/m)*(D_1+reg_1);
Theta2_grad = (1/m)*(D_2+reg_2);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
