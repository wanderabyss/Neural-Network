function W = randInitializeWeights(L_in, L_out)

% Randomly initialize the weights of a layer
% with L_in incoming connections and L_out outgoing connections

W = zeros(L_out, 1 + L_in);

%epsilon_init = 0.12;
epsilon_init = sqrt(6)/sqrt(L_in + L_out);

W = rand(L_out, 1+L_in) * 2 * epsilon_init - epsilon_init;

end
