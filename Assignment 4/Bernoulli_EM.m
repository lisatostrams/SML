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