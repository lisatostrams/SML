X = load('a011_mixdata.txt', '-ASCII');
%%
% for i = 1:4
%     for j = 1:4
%         subplot(4,4,(i-1)*4+j)
%         if (i==j)
%             formatSpec = 'Histogram of %d.';
%             str = sprintf(formatSpec,i);
%             hist(X(:,i))
%             title(str)
%         else
%             formatSpec = 'Plot of %d vs %d.';
%             str = sprintf(formatSpec,i,j);
%             scatter(X(:,i),X(:,j),'.')
%             title(str)
%             xlabel(i)
%             ylabel(j)
%         end
%     end
% end

%%
% figure;
% scatter3(X(:,1),X(:,2),X(:,3))
% title('1 vs 2 vs 3')
% xlabel(1)
% ylabel(2)
% zlabel(3)
% figure;
% scatter3(X(:,1),X(:,2),X(:,4))
% title('1 vs 2 vs 4')
% xlabel(1)
% ylabel(2)
% zlabel(4)
% figure;
% scatter3(X(:,1),X(:,3),X(:,4))
% title('1 vs 3 vs 4')
% xlabel(1)
% ylabel(3)
% zlabel(4)
% figure; 
% scatter3(X(:,2),X(:,3),X(:,4))
% title('2 vs 3 vs 4')
% xlabel(2)
% ylabel(3)
% zlabel(4)

%% EM (BISHOP 9.2.2)

%init
K = 4;
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


loglikelihood = zeros(1,101);
for n = 1:size(X,1)
    ll=0;
    for i = 1:K
        ll = ll +  pi_k(i)*gauss(X(n,:),mu(i,:),reshape(Sigma(i,:,:),[4,4]));
    end
    loglikelihood(1) = loglikelihood(1) + log(ll);
end
fprintf('Log likelihood is %1.4f at initalisation.\n',loglikelihood(1))
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
    Nk = sum(gamma,1); 
    for i = 1:K
        sum_N = zeros(1,4);
        for n = 1:N
            sum_N = sum_N + (gamma(n,i).*X(n,:));
        end
        mu(i,:) = 1/Nk(i) .* sum_N;
        
        sum_N = zeros(4,4);
        for n = 1:N
            sum_N = sum_N + (gamma(n,i).*(X(n,:)-mu(i,:))'*(X(n,:)-mu(i,:)));
        end
        Sigma(i,:,:) = 1/Nk(i) .* sum_N; 
        
        pi_k(i) = Nk(i)/N;
    end
    

    for n = 1:size(X,1)
        ll=0;
        for i = 1:K
            ll = ll +  pi_k(i)*gauss(X(n,:),mu(i,:),reshape(Sigma(i,:,:),[4,4]));
        end
        loglikelihood(it+1) = loglikelihood(it+1) + log(ll);
    end
    
    fprintf('Log likelihood is %1.4f in iteration %d. \n',loglikelihood(it+1), it)
    %mu
    %Nk

end
    

% %%
% plot(loglikelihood)
% title('Log likelihood of the data over iterations')
% xlabel('Iteration')
% ylabel('Log likelihood')

%%

figure; hold on; 
[argvalue, argmax] = max(gamma');
for i = 1:K
    scatter(X(argmax==i,1),X(argmax==i,2),'.')
    Xk = X(argmax==i,:);
    rho = corrcoef(Xk(:,1),Xk(:,2));
    fprintf('In group %d  the correlation between x_1 and x_2 is %1.3f. \n',i,rho(1,2))
end
title('x_1 x_2 coordinates colored by most probable component')
xlabel('x_1')
ylabel('x_2')
legend('Group 1', 'Group 2','Group 3','Group 4')
hold off;


%% Calculate expectations for 4 samples
A = [11.85, 2.2, 0.5, 4.0];
B = [11.95, 3.1, 0.0, 1.0];
C = [12.00, 2.5, 0.0, 2.0];
D = [12.00, 3.0, 1.0, 6.3];
samples = [A;B;C;D];
gamma4 = zeros(4,K);
for n = 1:4
    sum_K = 0;
    for i = 1:K
        sum_K = sum_K + pi_k(i)*gauss(samples(n,:),mu(i,:),reshape(Sigma(i,:,:),[4,4]));
    end

    for i = 1:K
        gamma4(n,i) = pi_k(i)*gauss(samples(n,:),mu(i,:),reshape(Sigma(i,:,:),[4,4])) / sum_K;
    end
end

%%
figure; hold on; 
[argvalue, argmax] = max(gamma');
[argvalue, argmax4] = max(gamma4');
colors = {'b.','g.', 'r.','c.'} ;
colorf = {'b','g', 'r','c'} ;

for i = 1:K
    scatter(X(argmax==i,1),X(argmax==i,2), colors{i})
end
legend('Group 1', 'Group 2','Group 3','Group 4')
for i = 1:K
    scatter(samples(argmax4==i,1),samples(argmax4==i,2),100,'k','LineWidth',1,'MarkerFaceColor',colorf{i})
end  
text(A(1),A(2),'A','FontSize',18,'FontWeight','bold')
text(B(1),B(2),'B','FontSize',18,'FontWeight','bold')
text(C(1),C(2),'C','FontSize',18,'FontWeight','bold')
text(D(1),D(2),'D','FontSize',18,'FontWeight','bold')
title('x_1 x_2 coordinates colored by most probable component')
xlabel('x_1')
ylabel('x_2')
hold off;