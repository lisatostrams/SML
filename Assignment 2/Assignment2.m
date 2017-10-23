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


%%