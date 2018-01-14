function f = expectation(X, mu, pi)
    X = double(X);
    N = size(X,1);
    K = size(pi,1);
        
    bern = @(x,p) (p.^x .* (1-p).^(1-x));
    gamma = zeros(N,K);
    for n=1:N
        for k=1:K
            log_p_xn_muk = sum(log(bern(X(n,:), mu(k,:))));
            gamma(n,k) = exp(log(pi(k)) + log_p_xn_muk);
        end
    end
    
    normFactor = sum(gamma,2);
    for n=1:N
        gamma(n,:) = gamma(n,:) ./ normFactor(n);
    end
       
    f = gamma;
end