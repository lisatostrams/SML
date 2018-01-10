function f = Bernoulli_EM(X,mu,pi)
    bern = @(x,p) p^x .* (1-p)^(1-x);
    p_X_mu = arrayfun(bern, X, mu) 
    p_X_mu_pi = sum(pi*prod(p_X_mu));
end

