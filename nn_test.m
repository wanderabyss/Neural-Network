% Perform classification for a multi-class dataset 
% using a regularized 3-layer neural network.

% Initialization
clear all; clc; close all;

% total run time
time = cputime;

%load data
fprintf('Loading Data ...\n')

data = load('wdbc_data.csv');

%randomize rows
order = randperm(size(data,1));
data = data(order,:);

%separate into features and class
X = data(:,1:end-1);
y = data(:,end);

m = size(X,1);

% plot featuring data points
%plotData(X, y);

%percentage of data to use for training
train_frac = 0.75;

%split into training and test sets:
test_rows = round(size(X,1)*(1-train_frac)); %number of rows to use in test set

%this is the test set
X_test = X(1:test_rows,:); y_test = y(1:test_rows,:);

%plotData(X_test, y_test);

%this is the training set
X_train = X(test_rows+1:end,:); y_train = y(test_rows+1:end,:);

%plotData(X_train, y_train);

a = size(X_test,1);
b = size(X_train,1);


%Setup Parameters - NN layer sizes
input_layer_size = size(X_train,2);

hidden_layer_size = 40;

% For large nH, training error can become small because networks 
% have high expressive power and become tuned to the particular 
% training set. Test error is nevertheless high. 
% Also, networks with multiple hidden layers are more prone to 
% getting caught in undersireable local minima. 

num_labels = size(unique(y),1); %output layer

fprintf('\nInitializing Neural Network Parameters ...\n')


%Initialize NN Parameters for the 3-layer NN

% Randomly Initialize parameters for symmetry breaking
% weights to small values 
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% Implement backprop and train network 
fprintf('\nTraining Neural Network... \n\n')

% Set options for fmincg
options = optimset('MaxIter', 400);               
lambda = 1; % regulization parameter
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size,...
     num_labels, X_train, y_train, lambda);
    
% Get paramaters using fmincg						
 [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
 
 % Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
       
% predict output of nn on train data
predict_train = predict(Theta1, Theta2, X_train);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(predict_train == y_train)) * 100);

% predict output of nn on test data
predict_test = predict(Theta1, Theta2, X_test);
fprintf('\nTest Set Accuracy: %f\n', mean(double(predict_test == y_test)) * 100);

%confusion matrix, sensitivity, specificity
[cm,order] = confusionmat(y_test,predict_test);

sens = cm(1,1) / (cm(1,1) + cm(1,2));
%ability to identify positive class

spec = cm(2,2) / (cm(2,2) + cm(2,1));
%ability to identify negative class

fprintf('\nConfusion Matrix:\n');disp(cm)
fprintf('\nSensitivity: %g\n',sens);
fprintf('\nSpecificity: %g\n',spec);

theta = zeros(size(X_train,2), 1);
thresh = 0.5;
testError = misclassError(y_test,sigmoid(X_test*theta),thresh) %0/1 misclassification error on test set

e = cputime-time;

fprintf(['\nTotal run time: %f seconds ' '\n\n'], e);

% Test NN for random generated Data 

% Randomly permute examples
%rp = randperm(m);

%for i = 1:m
    
 %  pred = predict(Theta1, Theta2, X(rp(i),:));
  % fprintf('\nNeural Network Prediction: %d \n\n', pred);
  % plotData(X, pred)

 %Pause
   %fprintf('Program paused. Press enter to continue.\n');
   %pause;
%end


