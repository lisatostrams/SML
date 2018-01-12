X = load('a011_mixdata.txt', '-ASCII');
%%
for i = 1:4
    for j = 1:4
        subplot(4,4,(i-1)*4+j)
        if (i==j)
            formatSpec = 'Histogram of %d.';
            str = sprintf(formatSpec,i);
            hist(X(:,i))
            title(str)
        else
            formatSpec = 'Plot of %d vs %d.';
            str = sprintf(formatSpec,i,j);
            scatter(X(:,i),X(:,j),'.')
            title(str)
            xlabel(i)
            ylabel(j)
        end
    end
end

%%
figure;
scatter3(X(:,1),X(:,2),X(:,3))
title('1 vs 2 vs 3')
xlabel(1)
ylabel(2)
zlabel(3)
figure;
scatter3(X(:,1),X(:,2),X(:,4))
title('1 vs 2 vs 4')
xlabel(1)
ylabel(2)
zlabel(4)
figure;
scatter3(X(:,1),X(:,3),X(:,4))
title('1 vs 3 vs 4')
xlabel(1)
ylabel(3)
zlabel(4)
figure; 
scatter3(X(:,2),X(:,3),X(:,4))
title('2 vs 3 vs 4')
xlabel(2)
ylabel(3)
zlabel(4)

%% EM

%init
K = 2;
N = size(X,1);
mu = ones(K,4);
for i = 1:K
    r = -1 + 2.*rand(1,4);
    mu(i,:) = mu(i,:) .* (mean(X,1) + r);
end

Sigma = zeros(K,4,4);
for i = 1:K
   r = 4*rand(1,4)+2;
   Sigma(i,:,:) = diag(r);
end

pi_k = ones(K,1);

%functions
gauss = @(x,mu,Sig)((1/((2*pi)^2))*(1/(det(Sig)^(0.5)))*exp(-0.5*(x-mu)*inv(Sig)*(x-mu)'));


loglikelihood = 0;
for n = 1:size(X,1)
    ll=0;
    for i = 1:K
        ll = ll +  pi_k(i)*gauss(X(n,:),mu(i,:),reshape(Sigma(i,:,:),[4,4]));
    end
    loglikelihood = loglikelihood + log(ll);
end

for it =1:100
    % E step
    gamma = zeros(N,K);
    for n = 1:N
        sum_K = 0;
        for i = 1:K
            sum_K = sum_K + pi_k(i)*gauss(X(n,:),mu(i,:),reshape(Sigma(i,:,:),[4,4]));
        end
        
        for i = 1:K
            gamma(n,i) = pi_k(i)*gauss(X(n,:),mu(i,:),reshape(Sigma(i,:,:),[4,4])) / sum_K;
        end
    end
    
    % M step
    Nk = sum(gamma,1)
    
    for i = 1:K
        for n = 1:N
        end
    end
            
end
    


