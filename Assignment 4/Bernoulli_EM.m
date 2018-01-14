function [f,gamma] = Bernoulli_EM(X, mu, pi, iterations)
    mu_ = mu; pi_ = pi;
    K = size(mu_,1);
    D = size(mu_,2);
    it = zeros(iterations+1,K,D+1);
    it(1,:,:) = [mu_, pi_];
    
    for i=2:iterations+1
        gamma = expectation(X,mu_,pi_);
        [mu_, pi_] = maximization(X,gamma);
                
        it(i,:,:) = [mu_, pi_];
    end
    
    f = it;
end

function f = expectation(X, mu, pi)
    X = double(X);
    N = size(X,1);
    K = size(pi,1);
        
    bern = @(x,p) (p.^x .* (1-p).^(1-x));
    gamma = zeros(N,K);
%     log_p_X_mu = zeros(N,K);
    for n=1:N
        for k=1:K
            log_p_xn_muk = sum(log(bern(X(n,:), mu(k,:))));
            gamma(n,k) = exp(log(pi(k)) + log_p_xn_muk);
%             log_p_X_mu(n,k) = log_p_xn_muk;
        end
    end
    
    normFactor = sum(gamma,2);
    for n=1:N
        gamma(n,:) = gamma(n,:) ./ normFactor(n);
    end
       
    f = gamma;
end

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
