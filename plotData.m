function plotData(X, y)

% Plots the data points X and y into a new figure 
% with + for the malignant and o for the benign. 
% X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% Find Indices 
pos_mal = find(y==1);
pos_ben = find(y==2);

% Plot
plot(X(pos_mal, 1), X(pos_mal, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(pos_ben, 1), X(pos_ben, 2), 'ko', 'MarkerFaceColor', 'r', 'MarkerSize', 7);
title('Breast Cancer Wisconsin (Diagnostic) Data Set');
legend('Malignant','Benign');

hold off;

end
