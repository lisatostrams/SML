thetas = [1 1 1 1;
          1 4 0 0;
          9 4 0 0;
          1 64 0 0;
          1 0.25 0 0;
          1 4 10 0;
          1 4 0 5];
N = 101;

X = linspace(-1,1,N);
K = zeros(size(thetas,1),101,101);

% populate the gram matrix
for t=1:size(thetas,1)
    for n=1:size(K,2)
        for m=1:size(K,3)
            K(t,n,m) = kernel(thetas(t,:), X(n), X(m));
        end
    end
end

mu = zeros(101,1)';
rng 'default';
% sample 5 random functions. Repeat for all thetas
for t=1:size(thetas,1)
    sigma = squeeze(K(t,:,:));
    samples = mvnrnd(mu, sigma, 5);

    % plot the sampled functions
    figure(t); hold on;
%     for i=1:5
        plot(X, samples, 'Linewidth', 2)
%     end
    hold off;
end

% % compute the gram matrix for the training dataset
% D = [-0.5 0.5; 0.2 -1; 0.3 3; -0.1 -2.5];
% K_5 = zeros(size(D,1),size(D,1))
% for i=1:size(K_5,1)
%     for j=1:size(K_5,2)
%         K_5(i,j) = kernel(thetas(1), D(i,2), D(j,2));
%     end
% end
% 
% K_5