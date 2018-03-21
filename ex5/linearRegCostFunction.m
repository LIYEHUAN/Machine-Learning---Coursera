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

% X�Ѿ������bias����һ��12�У�2�еľ��󣬷���ֱ�����Իع��Ҫ��
h = X * theta;
Predict_error = h -y;
Err = Predict_error .* Predict_error;

% ���֮��Ҫ��sum����Ȼ����һ����������Ҫ�����һ��singular value
J = sum(Err)/(2 * m);

% ����Regularization
Regu = (theta' * theta) - theta(1)^2;
Regu = (Regu * lambda) / (2 * m);

J = J + Regu;

% Part2: Compuate the gradient of regularized linear regression

% grad��һ����theta����ͬ��size�ľ�����Ϊ������cost function��theta���ƫ��
grad_inner = (1 / m) * (Predict_error' * X);
Regu_grad = (lambda / m) * theta';
grad = grad_inner + Regu_grad;

% Ҫע�����theta(1)���ǲ���ҪRegularization��
grad(1) = grad(1) - (lambda / m) * theta(1);


% =========================================================================

grad = grad(:);

end
