function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                               
% Implements the neural network cost function which performs classification.  
% The returned parameter grad should be a "unrolled" vector of the
% partial derivatives of the neural network.


% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             
m = size(X, 1);
         
% return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Step 1
% Feedforward the neural network and return the cost in the variable J.

X = [ones(m,1) X];

% foward propagation
% a1 = X; 
a2 = sigmoid(Theta1 * X');
a2 = [ones(m,1) a2'];

h_theta = sigmoid(Theta2 * a2'); % h_theta equals z3

% y(k) - recode the labels as vectors containing only values 0 or 1 
yk = zeros(num_labels, m); 
for i=1:m,
  yk(y(i),i)=1;
end

% follow the form
J = (1/m) * sum ( sum (  (-yk) .* log(h_theta)  -  (1-yk) .* log(1-h_theta) ));


% should not be regularizing the terms that correspond to the bias. 
% For the matrices Theta1 and Theta2, this corresponds to the first column of each matrix.
t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));

% regularization formula
J_Reg = lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 ))) / (2*m);

% cost function + reg
J = J + J_Reg;


% step 2
%  Implement the backpropagation algorithm to compute the gradients
%  Theta1_grad and Theta2_grad. return the partial derivatives of
%  the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%  Theta2_grad, respectively.


% Backprop

for t=1:m,

	% dummie pass-by-pass
	% forward propag

	a1 = X(t,:); % X already have bias
	z2 = Theta1 * a1';

	a2 = sigmoid(z2);
	a2 = [1 ; a2]; % add bias

	z3 = Theta2 * a2;

	a3 = sigmoid(z3); % final activation layer a3 == h(theta)

	% back propag 	
	
	z2=[1; z2]; % bias

	delta_3 = a3 - yk(:,t); % y(k) trick - getting columns of t element
	delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2);

	% skipping sigma2(0) 
	delta_2 = delta_2(2:end); 

	Theta2_grad = Theta2_grad + delta_3 * a2';
	Theta1_grad = Theta1_grad + delta_2 * a1; 
   
end;

%step 3
%Implement regularization with the cost function and gradients.

% Regularization

	Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m;
	
	Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) ./ m + ((lambda/m) * Theta1(:, 2:end));
	
	Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m;
	
	Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) ./ m + ((lambda/m) * Theta2(:, 2:end));

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
