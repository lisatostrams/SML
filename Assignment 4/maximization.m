function [mu_, pi_] = maximization(X, gamma)
    X = double(X);
    K = size(gamma,2);
    N = size(X,1);
    D = size(X,2);
    mu = zeros(K,D);
    pi = zeros(K,1);
    
    for k=1:K
        N_k = sum(gamma(:,k));
        for d=1:D
            mu(k,d) = (sum(gamma(:,k) .* double(X(:,d)))) / N_k;
        end
        pi(k) = N_k / N;
    end
    
    mu_ = mu; pi_ = pi;
end
