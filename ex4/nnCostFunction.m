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
% Yehuan Li comments
% nn_paramsy原本为一个列向量，通过reshape函数，重新将其分成Theta1 & Theta2
% 三个参数分别代表在nn_params中的位置，输出矩阵的行数以及列数

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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

% Yehuan Li code

% Calculate h(x)
X = [ones(m,1) X];

a1 = X;
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);

h = a3;

% Calculate cost function (without regularization)
y_out = zeros(num_labels, m);
for i = 1:m
    y_out(y(i),i) = 1;
end

J_inner = 0;
for j = 1:m
    J_inner = J_inner - log(h(j,:))*y_out(:,j) - log(1-h(j,:))*(1-y_out(:,j));
end

% Calculate cost function (with regularization)
Regu = (lambda/(2*m))*(sum(sum(Theta1.*Theta1)) + sum(sum(Theta2.*Theta2)));
Regu = Regu - (lambda/(2*m))*((Theta1(:,1)'*Theta1(:,1)) + (Theta2(:,1)'*Theta2(:,1)));

J = (J_inner/m) + Regu;

% Calculate Backpropagation to have gradient
% Step1
% Have been done above

% Step 2-5
delta3 = h - y_out';
delta2_temp = delta3 * Theta2;
delta2 = delta2_temp(:,2:end) .* sigmoidGradient(z2);

Delta2 = (a2' * delta3)';
Delta1 = (a1' * delta2)';

Theta2_grad = Delta2/m;
Theta1_grad = Delta1/m;

Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
