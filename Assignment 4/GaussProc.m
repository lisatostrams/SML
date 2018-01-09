%% define some variables
thetas = [1 1 0 0;
          1 4 0 0;
          9 4 0 0;
          1 64 0 0;
          1 0.25 0 0;
          1 4 10 0;
          1 4 0 5];
theta = thetas(1,:);
N = 101;
X = linspace(-1,1,N);

%% compute the gram matrix for each value of theta
K = zeros(size(thetas,1),N,N);
for t=1:size(thetas,1)
    K(t,:,:) = gram_matrix(thetas(t,:), X, @kernel);
end

%% sample 5 random functions
mu = zeros(101,1)';
rng(42);%'default';

figure(); hold on;
samples = mvnrnd(mu, squeeze(K(1,:,:)), 5);
plot(X, samples, 'Linewidth', 2)
title(strcat('[', sprintf('%d ', fix(theta)), ']'))
hold off;

%% repeat for all thetas
figure(); hold on;
for t=2:size(thetas,1)
    sigma = squeeze(K(t,:,:));
    samples = mvnrnd(mu, sigma, 5);

    % plot the sampled functions
    subplot(2,3,t-1);
    plot(X, samples, 'Linewidth', 2)
    title(strcat('[', sprintf('%d ', fix(thetas(t,:))), ']'))
end
hold off;

%% compute the gram matrix for the training dataset
D = [-0.5 0.5; 0.2 -1; 0.3 3; -0.1 -2.5];
K_4 = gram_matrix(theta, D(:,1), @kernel);

% K_4
%% compute the predictive distribution
x_n_1 = 0;
beta = 1;

k = arrayfun(@(x) kernel(theta, x, x_n_1), D(:,1));
c = kernel(theta, x_n_1, x_n_1) + 1/beta;
C_N = K_4;
C_N(1:size(C_N,1)+1:end) = diag(C_N) + 1/beta;

% C_N

C_N_i = inv(C_N);
m = k'*C_N_i*D(:,2);
s_2 = c - k'*C_N_i*k;
