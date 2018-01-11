function f = Bernoulli_EM(X, mu, pi, iterations)
    mu_ = mu; pi_ = pi;
    K = size(mu_,1);
    D = size(mu_,2);
    it = zeros(iterations,K,D+1);
    
    for i=1:iterations
        gamma = expectation(X,mu_,pi_);
        [mu_, pi_] = maximization(X,gamma);
        
%         % map means to grayscale
%         mu__ = uint8(mu_ * 255);
%         
%         figure();
%         colormap(gray(255));
%         for k=1:K
%             subplot(1,K,k);
%             image(reshape(mu__(k,:),[28,28]));
%             title(sprintf('Class: %d', k));        
%         end
        
        it(i,:,:) = [mu_, pi_];
    end
    
    f = it;
end

function f = expectation(X, mu, pi)
    N = size(X,1);
    K = size(pi,1);
        
    bern = @(x,p) (p.^x .* (1-p).^(1-x));
    gamma = zeros(N,K);
    p_X_mu = zeros(N,K);
    for n=1:N
        for k=1:K
            p_xn_muk = exp(sum(log(bern(double(X(n,:)), mu(k,:)))));
            gamma(n,k) = (pi(k) .* p_xn_muk);
            p_X_mu(n,k) = p_xn_muk;
        end
    end
    
%     log_pi = log(pi);
%     log_mu = log(mu);    
%     % array of expectations for the log likelhoods of each image
%     C = X * log_mu' + (1-X) * (1-log_mu)'; % N x K
%     
%     expectations = zeros(N,1);
    normalizer = (p_X_mu * pi); % N x K, K x 1 -> N x 1
    for n=1:N
        gamma(n,:) = gamma(n,:) ./ normalizer(n);
%         expectations(n) = gamma(n,:)*(log_pi + C(n,:));
    end
       
    f = gamma;
end

function [mu_, pi_] = maximization(X, gamma)
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
