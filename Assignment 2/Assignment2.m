%% Exercise 1.3
p_c0 = @(z)((1-z.^2)./((3.*z+1).*(z+1)));
p_c1 = @(z)(z./((3.*z+1)./4));
z = linspace(0,1,100);

figure; hold on; 
plot(z,p_c0(z),'r--')
plot(z,p_c1(z),'b--')
legend('p(c=0|z)','p(c=1|z)')
xlabel('z')
ylabel('p')


%% Exercise 3 part 1.4
D = [3.6 7.7 -2.6 4.9 -2.3 0.2 -7.3 4.4 7.3 -5.7];
alpha = linspace(-10,10,1001);
p = @(x,alpha)(2./(pi.*(4+(x-alpha).^2)));
p_a = zeros([1,1001]);
i=1;
for a = alpha
    p_a(i) = prod(p(D,a));
    i=i+1;
end
hold on; 
plot(alpha,p_a)
plot(D, zeros(size(D)),'r.', 'MarkerSize',20)
plot(mean(D), 0,'ks','Markersize',10,'MarkerFaceColor','b')
[y,i] = max(p_a); 
plot(alpha(i),0,'g*','Markersize',10)
legend('p(a|D,b=2)', 'Data','Mean data','alpha max posterior')
hold off; 
%% Exercise 3 part 2
alpha = 10.*rand;
beta = 2 + 2.*rand;
theta = -pi/2 + pi.*rand(500,1);

D = (beta*tan(theta))+alpha;
means = zeros(200,1);
j=1;
thetas = -pi/2 + pi.*rand(2000,1);
for i = 1:10:2000
    theta = datasample(thetas,i,'Replace',false);
    D = (beta*tan(theta))+alpha;
    means(j) = mean(D);
    j=j+1;
end

close all; hold on;
plot(10:10:2000,means,'b-','LineWidth',1.5)
xlim([-10,2000])
plot(10:10:2000,ones(200,1)*alpha,'r--','LineWidth',1.5)
legend('Sample means','alpha')
hold off;

%% Exercise 3 part 3
theta = -pi+0.01/2 + pi-0.01.*rand(500,1);
D = (beta*tan(theta))+alpha;

alphas = linspace(-10,10,100);
betas = linspace(0,5,100); 
LL = zeros(100,100);
L = zeros(100,100); 
for i = 1:100
    for j = 1:100
        LL(i,j) = loglikelihood(D(1:20),alphas(i),betas(j));
        L(i,j) = likelihood(D(1:20),alphas(i),betas(j));
    end
end
close all; 
figure; hold on; 
surf(alphas,betas,LL)
xlabel('Alpha')
ylabel('Beta')
title('loglikelihood')
hold off; 

figure; hold on; 
surf(alphas,betas,L)
xlabel('Alpha')
ylabel('Beta')
title('likelihood')
hold off; 

%%
data = D(1:3);
min = fminsearch(@(a,b) loglikelihood(data,a,b),[0,1]);
