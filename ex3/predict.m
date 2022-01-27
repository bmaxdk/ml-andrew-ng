function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% =========================================================================

% Your goal is to implement the feedforward propagation algorithm to use our weights for prediction. 
% The  images are of size 20 x 20, this gives us 400 input layer units (excluding the extra bias unit which always outputs +1)
% Now you will implement feedforward propagation for the neural network. You will need to complete the code in predict.m to return the neural network's prediction. You should implement the feedforward computation that computes h_theta(x^_(i)) for every example  and returns the associated predictions. Similar to the one-vs-all classication strategy, the prediction from the neural network will be the label that has the largest output (h_theta(x))_k .
% Implementation Note: The matrix X contains the examples in rows. When you complete the code in predict.m, you will need to add the column of 1's to the matrix. The matrices Theta1 and Theta2 contain the parameters for each unit in rows. Specically, the first row of Theta1 corresponds to the first hidden unit in the second layer.

% layer2
% Add ones to the X data matrix
% X % 5000 x 400
% Theta1 %25 X 401
% Theta2 %10 X 26
X = [ones(m, 1) X]; % 5000 x 401
z2 = X*Theta1'; % 5000 X 25
h2 = sigmoid(z2);
h2 = [ones(m,1) h2];

% layer3
z3 = h2*Theta2';
h3 = sigmoid(z3);

[M, p] = max(h3, [], 2);



end
