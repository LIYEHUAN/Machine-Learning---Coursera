function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
% Yehuan Li code here
for i = 1:K
    count = 0; 
    % 注意，这里需要计数，因为是除以有多少个x被分到的uk类中，不能除以总数K
    sum_x = zeros(1,n);
    % 求和是求的一个行矩阵，列数取决于X有多少features
    for j = 1:m
        if idx(j) == i
            count = count + 1;
            sum_x = sum_x + X(j,:);
        end
    end
    centroids(i,:) = (1/count) * sum_x;
end




% =============================================================


end

