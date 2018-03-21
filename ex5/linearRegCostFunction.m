function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% Yehuan Li code here
% Part1: Compuate the cost function of regularized linear regression

% X已经添加了bias，是一个12行，2列的矩阵，符合直线线性回归的要求
h = X * theta;
Predict_error = h -y;
Err = Predict_error .* Predict_error;

% 点乘之后要求sum，不然还是一个矩阵，我们要求的是一个singular value
J = sum(Err)/(2 * m);

% 进行Regularization
Regu = (theta' * theta) - theta(1)^2;
Regu = (Regu * lambda) / (2 * m);

J = J + Regu;

% Part2: Compuate the gradient of regularized linear regression

% grad是一个和theta具有同样size的矩阵。因为它就是cost function对theta求的偏导
grad_inner = (1 / m) * (Predict_error' * X);
Regu_grad = (lambda / m) * theta';
grad = grad_inner + Regu_grad;

% 要注意对于theta(1)，是不需要Regularization的
grad(1) = grad(1) - (lambda / m) * theta(1);


% =========================================================================

grad = grad(:);

end
