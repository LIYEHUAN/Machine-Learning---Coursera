function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------
% Yehuan Li code here
% Introduction：该函数，是选取不同数量的example，利用trainLinearReg计算在此情况下的最优theta，
%               然后计算error train和error CV
% Computer error train and error CV
% 由于要选取不同数量的example，计算error train，需要for循环
for i = 1:m
    [theta] = trainLinearReg(X(1:i, :), y(1:i), lambda);
    Err_train_inner = X(1:i, :) * theta - y(1:i);
    % 对于error train，应该除以2*i，因为目前只选取了i个example
    error_train(i) = (Err_train_inner' * Err_train_inner) / (2 * i);
    
    % 虽然每次都是用全部的CV计算误差，但由于theta不一样，因此计算出的误差也不一样
    % 也正因为如此，才能体现不同algorithm条件的误差情况
    Err_CV_inner = (Xval * theta) - yval;
    
    % 对于error CV，应该除以2*size(Xval,1)，这才是CV的example数量
    error_val(i) = (Err_CV_inner' * Err_CV_inner) / (2 * size(Xval,1));
end



% -------------------------------------------------------------

% =========================================================================

end
