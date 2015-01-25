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

% Loop through features
for i=1:m % Loop through training examples
	% Forward prop
	input_layer = X(i,:)'; %'
	input_layer = [1; input_layer];
	hidden_layer = sigmoid(Theta1 * input_layer);
	hidden_layer = [1; hidden_layer];
	output_layer = sigmoid(Theta2 * hidden_layer);

	% Create 10-dimensional y-vector using logical array
	y_i = 1:num_labels;
	y_i = (y_i == y(i))'; %'

	% Compute cost for this feature and add it to total cost
	cost_i = sum(-1*y_i.*log(output_layer)-(1-y_i).*log(1-output_layer));
	J = J + cost_i;
end

J = J / m;

% Regularize cost function

% Remove bias weights from Theta1 and Theta2
unbiased_Theta1 = Theta1(:,2:end);
unbiased_Theta2 = Theta2(:,2:end);

J = J + (lambda/(2*m))*(sum(sum(unbiased_Theta1.^2)) + sum(sum(unbiased_Theta2.^2)));


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

% Initialize deltas
grad1 = 0;
grad2 = 0;
Y = eye(num_labels)(y,:); 

for t = 1:m
	% Step 1: Forward prop
	input_layer = [1; X(t,:)']; %'
	z_2 = Theta1 * input_layer;
	hidden_layer = [1; sigmoid(z_2);];
	z_3 = Theta2 * hidden_layer;
	output_layer = sigmoid(z_3);
    
    % Step 2: Cost for output_layer
	y_k = Y(t,:)'; %'
    delta_3 = output_layer - y_k;

    % Step 3: Cost for hidden_layer
    delta_2 = (unbiased_Theta2' * delta_3) .* sigmoidGradient(z_2); %'
    
    % Step 4: Accumulate gradient
    grad2 = grad2 + (delta_3 * hidden_layer'); % '
	grad1 = grad1 + (delta_2 * input_layer') ; % '
end 

% Obtain unregularized gradient
Theta1_grad = (1/m) * grad1;
Theta2_grad = (1/m) * grad2;

% Obtain regularized gradient
Theta1_grad = Theta1_grad + ((lambda/m)*[zeros(size(unbiased_Theta1, 1), 1) unbiased_Theta1]); 
Theta2_grad = Theta2_grad + ((lambda/m)*[zeros(size(unbiased_Theta2, 1), 1) unbiased_Theta2]); 

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
